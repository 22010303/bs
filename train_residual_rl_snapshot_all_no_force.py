"""
train_residual_rl_snapshot_all_no_force.py — Snapshot-based residual RL trainer
that supports SAC / PPO / TD3 in a single script without force feedback.

Compared with train_residual_rl_snapshot_all.py:
  - RL observation does not use force / torque.
  - Force-history input is replaced with zeros to keep trainer interfaces
    unchanged across SAC / PPO / TD3.
  - Reward is geometry-only.
  - Force-based early termination is removed.
  python train_residual_rl_snapshot_all_no_force.py --algo ppo --snapshot ./ckpt/residual_rl/snapshot_step315.pkl --no_render
"""

import argparse
import os
from collections import deque

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from mujoco_env.pih_env2 import PIHEnv2
from residual_rl.no_force_reward import NoForceFeedbackReward
from residual_rl.snapshot import load_snapshot, perturb_ee_pose, restore_snapshot

from residual_rl.config import ResidualRLConfig
from residual_rl.sac_trainer import SACTrainer

from residual_rl_ppo.config import ResidualPPOConfig
from residual_rl_ppo.ppo_trainer import PPOTrainer

from residual_rl_td3.config import ResidualTD3Config
from residual_rl_td3.td3_trainer import TD3Trainer

IMG_TRANSFORM = transforms.Compose([transforms.ToTensor()])

PEG_TIP_SITE = 'peg_tip_site'
HOLE_ENTRY_SITE = 'hole_entry_site'
HOLE_BOTTOM_SITE = 'hole_bottom_site'

ARM_JOINTS = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]
EEF_BODY = 'tool0_link'


def load_smolvla(cfg):
    from lerobot.common.datasets.factory import resolve_delta_timestamps
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.common.datasets.utils import dataset_to_policy_features
    from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.configs.types import FeatureType

    print(f"[smolvla] loading from {cfg.smolvla_pretrained}")
    meta = LeRobotDatasetMetadata(cfg.smolvla_repo_name, root=cfg.smolvla_dataset_root)
    feats = dataset_to_policy_features(meta.features)
    out_feats = {k: f for k, f in feats.items() if f.type is FeatureType.ACTION}
    in_feats = {k: f for k, f in feats.items() if k not in out_feats}

    smol_cfg = SmolVLAConfig(
        input_features=in_feats,
        output_features=out_feats,
        chunk_size=cfg.smolvla_chunk_size,
        n_action_steps=cfg.smolvla_n_action_steps,
    )
    resolve_delta_timestamps(smol_cfg, meta)

    policy = SmolVLAPolicy.from_pretrained(
        cfg.smolvla_pretrained,
        config=smol_cfg,
        dataset_stats=meta.stats,
    )
    policy.to(cfg.smolvla_device).eval()
    for p in policy.parameters():
        p.requires_grad = False
    print(f"[smolvla] {sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M params, frozen")
    return policy


def get_smolvla_action(policy, pih, device):
    state = pih.get_joint_state()[:6]
    a_img, w_img = pih.grab_image()
    a = IMG_TRANSFORM(Image.fromarray(a_img).resize((256, 256)))
    w = IMG_TRANSFORM(Image.fromarray(w_img).resize((256, 256)))
    data = {
        'observation.state': torch.from_numpy(np.array([state], dtype=np.float32)).to(device),
        'observation.image': a.unsqueeze(0).to(device),
        'observation.wrist_image': w.unsqueeze(0).to(device),
        'task': [pih.instruction],
    }
    with torch.no_grad():
        act = policy.select_action(data)
    act_np = act[0, :7].cpu().numpy()
    act_np[6] = 200.0
    return act_np


def check_contact(pih):
    pt = pih.env.get_p_site(PEG_TIP_SITE)
    he = pih.env.get_p_site(HOLE_ENTRY_SITE)
    xy = np.linalg.norm(pt[:2] - he[:2])
    return (pt[2] < he[2]) and (xy < 0.003)


def build_rl_obs_no_force(pih, cfg):
    ee = pih.get_ee_pose()
    pt = pih.env.get_p_site(PEG_TIP_SITE)
    he = pih.env.get_p_site(HOLE_ENTRY_SITE)
    rel = (he - pt).astype(np.float32)
    state_vec = np.concatenate(
        [np.zeros(cfg.ft_dim, dtype=np.float32), ee, rel],
        dtype=np.float32,
    )
    ft_hist = np.zeros((cfg.force_history_len, cfg.ft_dim), dtype=np.float32)
    return state_vec, ft_hist


def get_cart_limits(cfg):
    return np.array([
        cfg.max_residual_pos, cfg.max_residual_pos, cfg.max_residual_pos,
        cfg.max_residual_rot, cfg.max_residual_rot, cfg.max_residual_rot,
    ], dtype=np.float32)


def cart_to_joint_delta(pih, cart, cfg):
    _, _, j_full = pih.env.get_J_body(EEF_BODY)
    j_arm = j_full[:, pih.env.get_idxs_jac(ARM_JOINTS)]
    jjt = j_arm @ j_arm.T + cfg.ik_damping_eps * np.eye(6)
    j_pinv = j_arm.T @ np.linalg.solve(jjt, np.eye(6))
    dq = np.clip(j_pinv @ cart, -cfg.max_joint_delta, cfg.max_joint_delta)
    return dq.astype(np.float32)


class AlgoAdapter:
    name = ''
    checkpoint_filename = ''
    uses_warmup = False
    has_alpha = False

    @property
    def total_steps(self):
        return self.trainer.total_steps

    @total_steps.setter
    def total_steps(self, value):
        self.trainer.total_steps = value

    @property
    def alpha(self):
        return 0.0

    def save(self, path):
        self.trainer.save(path)

    def load(self, path):
        self.trainer.load(path)

    def act(self, state, ft_hist, in_warmup): ...
    def store_step(self, state, ft_hist, action, extras, reward, next_state, next_ft_hist, done): ...
    def step_updates(self): ...
    def on_episode_end(self, last_state, last_ft_hist, last_done): ...
    def log_episode(self, ep_reward, ep_len, ep_info, reward_info): ...


class SACAdapter(AlgoAdapter):
    name = 'sac'
    checkpoint_filename = 'sac_checkpoint.pt'
    uses_warmup = True
    has_alpha = True

    def __init__(self, cfg: ResidualRLConfig, device, use_wandb):
        self.cfg = cfg
        self.trainer = SACTrainer(config=cfg, device=device, use_wandb=use_wandb)
        self.device = self.trainer.device

    @property
    def alpha(self):
        return self.trainer.alpha

    def act(self, state, ft_hist, in_warmup):
        limits = get_cart_limits(self.cfg)
        if in_warmup:
            cart = np.random.uniform(-limits, limits).astype(np.float32)
        else:
            s_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            h_t = torch.FloatTensor(ft_hist).unsqueeze(0).to(self.device)
            with torch.no_grad():
                a, _ = self.trainer.actor.sample(s_t, h_t)
                cart = a.squeeze(0).cpu().numpy()
            cart = np.clip(cart, -limits, limits).astype(np.float32)
        return cart, {}

    def store_step(self, state, ft_hist, action, extras, reward, next_state, next_ft_hist, done):
        del extras
        self.trainer.store_transition(
            state=state,
            ft_history=ft_hist,
            action=action,
            reward=reward,
            next_state=next_state,
            next_ft_history=next_ft_hist,
            done=done,
        )
        self.trainer.total_steps += 1

    def step_updates(self):
        if self.trainer.total_steps >= self.cfg.warmup_steps and self.trainer.total_steps % self.cfg.update_every == 0:
            for _ in range(self.cfg.updates_per_step):
                self.trainer.update()

    def on_episode_end(self, last_state, last_ft_hist, last_done):
        del last_state, last_ft_hist, last_done

    def log_episode(self, ep_reward, ep_len, ep_info, reward_info):
        self.trainer.log_episode(ep_reward, ep_len, ep_info, reward_info=reward_info)


class TD3Adapter(AlgoAdapter):
    name = 'td3'
    checkpoint_filename = 'td3_checkpoint.pt'
    uses_warmup = True

    def __init__(self, cfg: ResidualTD3Config, device, use_wandb):
        self.cfg = cfg
        self.trainer = TD3Trainer(config=cfg, device=device, use_wandb=use_wandb)
        self.device = self.trainer.device

    def act(self, state, ft_hist, in_warmup):
        if in_warmup:
            limit = self.trainer.action_limit.cpu().numpy()
            cart = np.random.uniform(-limit, limit).astype(np.float32)
        else:
            cart = self.trainer.select_action(state, ft_hist, deterministic=False)
        return cart, {}

    def store_step(self, state, ft_hist, action, extras, reward, next_state, next_ft_hist, done):
        del extras
        self.trainer.store_transition(
            state=state,
            ft_history=ft_hist,
            action=action,
            reward=reward,
            next_state=next_state,
            next_ft_history=next_ft_hist,
            done=done,
        )
        self.trainer.total_steps += 1

    def step_updates(self):
        if self.trainer.total_steps >= self.cfg.warmup_steps and self.trainer.total_steps % self.cfg.update_every == 0:
            for _ in range(self.cfg.updates_per_step):
                self.trainer.update()

    def on_episode_end(self, last_state, last_ft_hist, last_done):
        del last_state, last_ft_hist, last_done

    def log_episode(self, ep_reward, ep_len, ep_info, reward_info):
        if not self.trainer.use_wandb:
            return
        import wandb

        log_dict = {
            'episode/reward': ep_reward,
            'episode/length': ep_len,
            'episode/success': float(ep_info.get('success', False)),
            'episode/contact_step': ep_info.get('contact_step', -1),
            'episode/rl_steps': ep_info.get('steps_after_contact', 0),
        }
        if reward_info:
            for k in ['insertion_progress', 'max_progress', 'xy_error', 'depth_to_entry']:
                if k in reward_info:
                    log_dict[f'episode/{k}'] = reward_info[k]
            for k in ['r_depth', 'r_progress', 'r_retract', 'r_align', 'r_in_hole', 'r_outside']:
                if k in reward_info:
                    log_dict[f'reward/{k}'] = reward_info[k]
        wandb.log(log_dict, step=self.trainer.total_steps)


class PPOAdapter(AlgoAdapter):
    name = 'ppo'
    checkpoint_filename = 'ppo_checkpoint.pt'

    def __init__(self, cfg: ResidualPPOConfig, device, use_wandb):
        self.cfg = cfg
        self.trainer = PPOTrainer(config=cfg, device=device, use_wandb=use_wandb)
        self.device = self.trainer.device
        self._last_next = (None, None, False)

    def act(self, state, ft_hist, in_warmup):
        del in_warmup
        action, log_prob, value = self.trainer.act(state, ft_hist)
        return action, {'log_prob': log_prob, 'value': value}

    def store_step(self, state, ft_hist, action, extras, reward, next_state, next_ft_hist, done):
        self.trainer.store_transition(
            state=state,
            ft_history=ft_hist,
            action=action,
            log_prob=extras['log_prob'],
            reward=reward * self.cfg.reward_scale,
            done=done,
            value=extras['value'],
        )
        self.trainer.total_steps += 1
        self._last_next = (next_state.copy(), next_ft_hist.copy(), bool(done))

    def step_updates(self):
        if self.trainer.buffer.full:
            ns, nh, nd = self._last_next
            if ns is None:
                ns = np.zeros(self.cfg.state_dim, dtype=np.float32)
                nh = np.zeros((self.cfg.force_history_len, self.cfg.ft_dim), dtype=np.float32)
                nd = False
            self.trainer.finish_rollout(ns, nh, nd)
            self.trainer.update()

    def on_episode_end(self, last_state, last_ft_hist, last_done):
        if self.trainer.buffer.size > 0:
            self.trainer.finish_rollout(last_state, last_ft_hist, last_done)
            self.trainer.update()
            self._last_next = (None, None, False)

    def log_episode(self, ep_reward, ep_len, ep_info, reward_info):
        if not self.trainer.use_wandb:
            return
        import wandb

        log_dict = {
            'episode/reward': ep_reward,
            'episode/length': ep_len,
            'episode/success': float(ep_info.get('success', False)),
            'episode/contact_step': ep_info.get('contact_step', -1),
            'episode/rl_steps': ep_info.get('steps_after_contact', 0),
        }
        if reward_info:
            for k in ['insertion_progress', 'max_progress', 'xy_error', 'depth_to_entry']:
                if k in reward_info:
                    log_dict[f'episode/{k}'] = reward_info[k]
        wandb.log(log_dict, step=self.trainer.total_steps)


ALGO_REGISTRY = {
    'sac': (SACAdapter, ResidualRLConfig, './ckpt/residual_rl_snapshot_no_force'),
    'ppo': (PPOAdapter, ResidualPPOConfig, './ckpt/residual_rl_snapshot_ppo_no_force'),
    'td3': (TD3Adapter, ResidualTD3Config, './ckpt/residual_rl_snapshot_td3_no_force'),
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algo', type=str, default='sac', choices=list(ALGO_REGISTRY.keys()))
    parser.add_argument('--snapshot', type=str, default='./ckpt/residual_rl/snapshot_step315.pkl')
    parser.add_argument('--xml_path', type=str, default='./asset/pih.xml')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--total_steps', type=int, default=200_000)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--no_smolvla', action='store_true')
    parser.add_argument('--smolvla_pretrained', type=str, default='./ckpt/smolvla_pih1/checkpoints/020000/pretrained_model')
    parser.add_argument('--smolvla_dataset_root', type=str, default='./demo_data_pih')
    parser.add_argument('--smolvla_repo_name', type=str, default='ur5e_pih_language')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--max_steps', type=int, default=650)
    parser.add_argument('--noise_pos', type=float, default=2e-4)
    parser.add_argument('--noise_rot_deg', type=float, default=0.2)
    parser.add_argument('--no_render', action='store_true')
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='residual-rl-pih')
    parser.add_argument('--wandb_name', type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    adapter_cls, config_cls, default_output_dir = ALGO_REGISTRY[args.algo]
    output_dir = args.output_dir or default_output_dir
    cfg = config_cls(
        xml_path=args.xml_path,
        seed=args.seed,
        total_steps=args.total_steps,
        output_dir=output_dir,
        log_dir=os.path.join(output_dir, 'logs'),
        smolvla_pretrained=args.smolvla_pretrained,
        smolvla_dataset_root=args.smolvla_dataset_root,
        smolvla_repo_name=args.smolvla_repo_name,
        smolvla_device=str(device),
    )

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    snap = load_snapshot(args.snapshot)
    base_step = snap['meta']['base_step']
    instruction_ref = snap['pih']['instruction']
    print(f"[snap] base_step={base_step}, peg={snap['pih']['active_peg']}, instruction='{instruction_ref}'")

    use_wandb = args.wandb
    if use_wandb:
        import wandb

        run_config = dict(vars(cfg)) if hasattr(cfg, '__dict__') else {}
        run_config.update({
            'algo': args.algo,
            'snapshot_path': args.snapshot,
            'snapshot_base_step': base_step,
            'snapshot_peg': snap['pih']['active_peg'],
            'snapshot_instruction': instruction_ref,
            'noise_pos_m': args.noise_pos,
            'noise_rot_deg': args.noise_rot_deg,
            'max_steps': args.max_steps,
            'no_smolvla': args.no_smolvla,
            'no_render': args.no_render,
            'force_feedback_input': False,
            'force_feedback_reward': False,
            'force_feedback_termination': False,
        })
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f'residual_rl_snap_{args.algo}_no_force_seed{cfg.seed}',
            config=run_config,
        )
        print("Wandb initialized.")

    smolvla = None if args.no_smolvla else load_smolvla(cfg)

    adapter = adapter_cls(cfg, device=str(device), use_wandb=use_wandb)
    if args.resume:
        adapter.load(args.resume)

    print(f"[env] creating PIHEnv2 from {cfg.xml_path}")
    pih = PIHEnv2(
        xml_path=cfg.xml_path,
        action_type='joint_angle',
        state_type='joint_angle',
    )

    reward_fn = NoForceFeedbackReward(cfg)
    ft_history = deque(maxlen=cfg.force_history_len)

    step = base_step
    episode = 0
    contact_detected = False
    contact_step = -1
    steps_after_contact = 0
    episode_reward = 0.0
    prev_obs = None
    last_rinfo = {}
    ft_offset = snap['ft']['offset'].copy()

    last_next_state = None
    last_next_ft_hist = None

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    log_file = open(os.path.join(cfg.log_dir, 'train_log.csv'), 'w')
    log_file.write('step,episode,reward,length,success,contact_step,rl_steps\n')

    def reset_from_snapshot():
        nonlocal step, episode, contact_detected, contact_step, steps_after_contact
        nonlocal episode_reward, prev_obs, last_rinfo, ft_offset
        nonlocal last_next_state, last_next_ft_hist

        if smolvla is not None:
            smolvla.reset()

        ft_offset = restore_snapshot(pih, snap)
        info = perturb_ee_pose(
            pih,
            pos_noise=args.noise_pos,
            rot_noise_deg=args.noise_rot_deg,
            rng=rng,
        )

        reward_fn.reset()
        ft_history.clear()
        for _ in range(cfg.force_history_len):
            ft_history.append(np.zeros(cfg.ft_dim, dtype=np.float32))

        step = base_step
        episode += 1
        contact_detected = False
        contact_step = -1
        steps_after_contact = 0
        episode_reward = 0.0
        prev_obs = None
        last_rinfo = {}
        last_next_state = None
        last_next_ft_hist = None

        print(f"\n--- Episode {episode} --- dp={info['dp']*1000}mm drpy={np.rad2deg(info['drpy'])}deg")

    def store_terminal(done_reward, is_success):
        nonlocal episode_reward, last_rinfo
        nonlocal last_next_state, last_next_ft_hist
        if prev_obs is None or not contact_detected:
            return

        ft_history.append(np.zeros(cfg.ft_dim, dtype=np.float32))
        state_vec, ft_hist = build_rl_obs_no_force(pih, cfg)
        pt = pih.env.get_p_site(PEG_TIP_SITE)
        he = pih.env.get_p_site(HOLE_ENTRY_SITE)
        hb = pih.env.get_p_site(HOLE_BOTTOM_SITE)
        reward, rinfo = reward_fn.compute(None, pt, he, hb, is_success)
        reward += done_reward
        episode_reward += reward
        last_rinfo = rinfo

        adapter.store_step(
            state=prev_obs[0],
            ft_hist=prev_obs[1],
            action=prev_obs[2],
            extras=prev_obs[3],
            reward=reward,
            next_state=state_vec,
            next_ft_hist=ft_hist,
            done=True,
        )
        last_next_state = state_vec
        last_next_ft_hist = ft_hist

    reset_from_snapshot()

    print(f"\n=== Snapshot Residual RL ({args.algo.upper()}, NO FORCE) ===")
    print(f"base_step        = {base_step}")
    print(f"instruction      = {pih.instruction}")
    print(f"active peg       = {pih.active_peg}")
    print(f"noise pos/rot    = {args.noise_pos*1000:.2f}mm / {args.noise_rot_deg:.2f}deg")
    print(f"max_residual_pos = {cfg.max_residual_pos} m")
    print(f"max_residual_rot = {cfg.max_residual_rot} rad")
    print(f"render           = {not args.no_render}")
    print(f"total RL target  = {cfg.total_steps}\n")

    while adapter.total_steps < cfg.total_steps:
        if not args.no_render and not pih.env.is_viewer_alive():
            break
        pih.step_env()

        if not pih.env.loop_every(HZ=20):
            continue

        success = pih.check_success()
        timeout = step >= args.max_steps

        out_of_hole = False
        if contact_detected:
            pt = pih.env.get_p_site(PEG_TIP_SITE)
            he = pih.env.get_p_site(HOLE_ENTRY_SITE)
            xy = np.linalg.norm(pt[:2] - he[:2])
            above = pt[2] > he[2] + 0.005
            if xy > cfg.out_of_hole_xy_th or above:
                out_of_hole = True

        if success or timeout or out_of_hole:
            if success:
                reason, terminal_r = 'SUCCESS', 0.0
            elif out_of_hole:
                reason, terminal_r = 'OUT_OF_HOLE', -cfg.penalty_early_stop
            else:
                reason, terminal_r = 'TIMEOUT', -cfg.penalty_early_stop * 0.5

            store_terminal(terminal_r, success)
            adapter.on_episode_end(
                last_state=last_next_state if last_next_state is not None else np.zeros(cfg.state_dim, dtype=np.float32),
                last_ft_hist=last_next_ft_hist if last_next_ft_hist is not None else np.zeros((cfg.force_history_len, cfg.ft_dim), dtype=np.float32),
                last_done=True,
            )

            c_tag = f'contact@{contact_step}' if contact_step >= 0 else 'no_contact'
            alpha_tag = f'alpha={adapter.alpha:.3f}' if adapter.has_alpha else ''
            print(
                f"[{args.algo.upper()} {adapter.total_steps:>7d} | Ep {episode:>4d}] "
                f"R={episode_reward:>8.2f} L={step-base_step:>4d} {reason} "
                f"{c_tag} rl_steps={steps_after_contact} {alpha_tag}"
            )
            log_file.write(
                f"{adapter.total_steps},{episode},{episode_reward:.4f},"
                f"{step-base_step},{int(success)},{contact_step},{steps_after_contact}\n"
            )
            log_file.flush()

            adapter.log_episode(
                episode_reward,
                step - base_step,
                {
                    'success': success,
                    'contact_step': contact_step,
                    'steps_after_contact': steps_after_contact,
                    'reason': reason,
                },
                reward_info=last_rinfo,
            )

            if adapter.total_steps > 0 and adapter.total_steps % cfg.save_every < max(steps_after_contact + 1, 1):
                adapter.save(os.path.join(cfg.output_dir, f'step_{adapter.total_steps}'))

            reset_from_snapshot()
            continue

        if smolvla is not None:
            base_action = get_smolvla_action(smolvla, pih, str(device))
        else:
            q = pih.get_joint_state()[:6]
            base_action = np.concatenate([q, [200.0]], dtype=np.float32)

        if not contact_detected and check_contact(pih):
            contact_detected = True
            contact_step = step
            pt = pih.env.get_p_site(PEG_TIP_SITE)
            he = pih.env.get_p_site(HOLE_ENTRY_SITE)
            print(f"  >> Insertion started at step {step} (depth={1000*(he[2]-pt[2]):.1f}mm)")

        cart_res = np.zeros(6, dtype=np.float32)
        joint_delta = np.zeros(6, dtype=np.float32)

        if contact_detected and cfg.max_residual_pos > 0:
            ft_history.append(np.zeros(cfg.ft_dim, dtype=np.float32))
            state_vec, ft_hist = build_rl_obs_no_force(pih, cfg)

            if prev_obs is not None:
                pt = pih.env.get_p_site(PEG_TIP_SITE)
                he = pih.env.get_p_site(HOLE_ENTRY_SITE)
                hb = pih.env.get_p_site(HOLE_BOTTOM_SITE)
                reward, rinfo = reward_fn.compute(None, pt, he, hb, False)
                episode_reward += reward
                last_rinfo = rinfo
                adapter.store_step(
                    state=prev_obs[0],
                    ft_hist=prev_obs[1],
                    action=prev_obs[2],
                    extras=prev_obs[3],
                    reward=reward,
                    next_state=state_vec,
                    next_ft_hist=ft_hist,
                    done=False,
                )
                last_next_state = state_vec
                last_next_ft_hist = ft_hist
                steps_after_contact += 1
                adapter.step_updates()

            in_warmup = adapter.uses_warmup and adapter.total_steps < cfg.warmup_steps
            cart_res, extras = adapter.act(state_vec, ft_hist, in_warmup=in_warmup)
            joint_delta = cart_to_joint_delta(pih, cart_res, cfg)
            prev_obs = (state_vec.copy(), ft_hist.copy(), cart_res.copy(), dict(extras))
        else:
            ft_history.append(np.zeros(cfg.ft_dim, dtype=np.float32))

        final_action = base_action.copy()
        final_action[:6] += joint_delta
        _ = pih.step(final_action)

        pt = pih.env.get_p_site(PEG_TIP_SITE)
        he = pih.env.get_p_site(HOLE_ENTRY_SITE)
        xy = np.linalg.norm(pt[:2] - he[:2])
        depth = he[2] - pt[2]

        if not args.no_render:
            pih.render(idx=episode)
            if contact_detected and prev_obs is not None:
                pih.env.viewer_text_overlay(
                    text1=f'Residual {args.algo.upper()} (no force)',
                    text2=(
                        f'R={episode_reward:.2f}  '
                        f'|cart|={np.linalg.norm(cart_res):.5f}  '
                        f'|dq|={np.linalg.norm(joint_delta):.5f}  '
                        f'rl_steps={steps_after_contact}'
                    ),
                )
            pih.env.viewer_text_overlay(
                text1='Geometry',
                text2=f'xy={xy*1000:.2f}mm depth={depth*1000:.2f}mm peg_z={pt[2]:.4f}',
            )
            pih.env.viewer_text_overlay(
                text1='Snapshot',
                text2=f'base_step={base_step}  step={step}  contact={contact_detected}',
            )

        if contact_detected and step % 5 == 0:
            r_str = f'R={episode_reward:>7.2f}' if prev_obs is not None else 'R=  n/a'
            print(
                f'  step={step:>4d} {r_str} '
                f'xy={xy*1000:>6.2f}mm depth={depth*1000:>6.2f}mm peg_z={pt[2]:.4f} '
                f'|cart|={np.linalg.norm(cart_res):.5f} |dq|={np.linalg.norm(joint_delta):.5f}'
            )

        step += 1

    adapter.save(os.path.join(cfg.output_dir, 'final'))
    log_file.close()
    if not args.no_render:
        pih.env.close_viewer()
    if use_wandb:
        import wandb
        wandb.finish()
    print(f"\n[done] {episode} episodes, {adapter.total_steps} RL steps.")


if __name__ == '__main__':
    main()
