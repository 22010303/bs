"""
train_residual_rl_ppo_aux.py — PPO residual RL with:
  1. CLS-token pooling for force temporal attention.
  2. Auxiliary next-step force prediction.
  3. Explicit 3-way insertion state classification with debounce.

The three ablation modules all share the same SharedEncoder in
residual_rl_ppo/aux_policy.py.
python train_residual_rl_ppo_aux.py --snapshot ./ckpt/residual_rl/snapshot_step315.pkl --wandb
"""

import argparse
import os
from collections import deque

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from mujoco_env.pih_env2 import PIHEnv2
from residual_rl.reward import HierarchicalReward
from residual_rl.snapshot import load_snapshot, perturb_ee_pose, restore_snapshot
from residual_rl_ppo.aux_config import ResidualPPOAuxConfig
from residual_rl_ppo.aux_trainer import PPOAuxTrainer


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

    meta = LeRobotDatasetMetadata(cfg.smolvla_repo_name, root=cfg.smolvla_dataset_root)
    features = dataset_to_policy_features(meta.features)
    out_feat = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
    in_feat = {k: f for k, f in features.items() if k not in out_feat}
    smolvla_cfg = SmolVLAConfig(
        input_features=in_feat,
        output_features=out_feat,
        chunk_size=cfg.smolvla_chunk_size,
        n_action_steps=cfg.smolvla_n_action_steps,
    )
    resolve_delta_timestamps(smolvla_cfg, meta)
    policy = SmolVLAPolicy.from_pretrained(
        cfg.smolvla_pretrained, config=smolvla_cfg, dataset_stats=meta.stats
    )
    policy.to(cfg.smolvla_device).eval()
    for p in policy.parameters():
        p.requires_grad = False
    return policy


def get_smolvla_action(policy, pih_env, device):
    state = pih_env.get_joint_state()[:6]
    agent_image, wrist_image = pih_env.grab_image()
    agent_img = IMG_TRANSFORM(Image.fromarray(agent_image).resize((256, 256)))
    wrist_img = IMG_TRANSFORM(Image.fromarray(wrist_image).resize((256, 256)))
    data = {
        'observation.state': torch.from_numpy(np.array([state], dtype=np.float32)).to(device),
        'observation.image': agent_img.unsqueeze(0).to(device),
        'observation.wrist_image': wrist_img.unsqueeze(0).to(device),
        'task': [pih_env.instruction],
    }
    with torch.no_grad():
        action = policy.select_action(data)
    action_np = action[0, :7].cpu().numpy()
    action_np[6] = 200.0
    return action_np


def check_contact(pih_env):
    peg_tip = pih_env.env.get_p_site(PEG_TIP_SITE)
    hole_entry = pih_env.env.get_p_site(HOLE_ENTRY_SITE)
    xy_dist = np.linalg.norm(peg_tip[:2] - hole_entry[:2])
    return (peg_tip[2] < hole_entry[2]) and (xy_dist < 0.03)


def build_rl_obs(ft, ee_pose, ee_vel, peg_tip, hole_entry, ft_history):
    relative = (hole_entry - peg_tip).astype(np.float32)
    state = np.concatenate([ft, ee_pose, relative, ee_vel], dtype=np.float32)
    ft_hist = np.array(list(ft_history), dtype=np.float32)
    return state, ft_hist


def cartesian_to_joint_delta(pih_env, cart_residual, cfg):
    _, _, j_full = pih_env.env.get_J_body(EEF_BODY)
    jac_idxs = pih_env.env.get_idxs_jac(ARM_JOINTS)
    j_arm = j_full[:, jac_idxs]
    jjt = j_arm @ j_arm.T + cfg.ik_damping_eps * np.eye(6)
    j_pinv = j_arm.T @ np.linalg.solve(jjt, np.eye(6))
    joint_delta = j_pinv @ cart_residual
    return np.clip(joint_delta, -cfg.max_joint_delta, cfg.max_joint_delta).astype(np.float32)


class DebouncedInsertionStateLabeler:
    """
    Explicit state recognizer with debounce counter.

    Labels:
      0 = smooth insertion
      1 = jamming
      2 = bottoming out
    """

    SMOOTH = 0
    JAMMING = 1
    BOTTOMING = 2

    def __init__(self, cfg: ResidualPPOAuxConfig):
        self.cfg = cfg
        self.stable_label = self.SMOOTH
        self.candidate_label = self.SMOOTH
        self.candidate_count = 0

    def reset(self):
        self.stable_label = self.SMOOTH
        self.candidate_label = self.SMOOTH
        self.candidate_count = 0

    def _raw_label(self, ft, ee_vel, peg_tip, hole_entry, hole_bottom):
        lateral_force = np.linalg.norm(ft[:2])
        axial_force = abs(ft[2])
        bending_torque = np.linalg.norm(ft[3:5])
        speed = np.linalg.norm(ee_vel[:3])
        xy_error = np.linalg.norm(peg_tip[:2] - hole_entry[:2])
        hole_depth = max(hole_entry[2] - hole_bottom[2], 0.01)
        progress = np.clip((hole_entry[2] - peg_tip[2]) / hole_depth, -1.0, 1.0)

        if (
            progress > self.cfg.bottom_progress_th and
            axial_force > self.cfg.bottom_axial_force_th and
            speed < self.cfg.low_speed_th
        ):
            return self.BOTTOMING

        # 1. 首先判定：如果力已经处于极端危险状态，无视速度和防抖，直接报卡滞！
        if lateral_force > 35.0 or bending_torque > 4.0:
            return self.JAMMING  # 一票否决，直接卡死

        if (
            progress > 0.0 and
            speed < self.cfg.low_speed_th and
            (
                lateral_force > self.cfg.jam_lateral_force_th or
                bending_torque > self.cfg.jam_torque_th or
                xy_error > self.cfg.jam_xy_error_th
            )
        ):
            return self.JAMMING

        return self.SMOOTH

    def update(self, ft, ee_vel, peg_tip, hole_entry, hole_bottom):
        raw = self._raw_label(ft, ee_vel, peg_tip, hole_entry, hole_bottom)
        if raw == self.stable_label:
            self.candidate_label = raw
            self.candidate_count = 0
            return self.stable_label

        if raw == self.candidate_label:
            self.candidate_count += 1
        else:
            self.candidate_label = raw
            self.candidate_count = 1

        if self.candidate_count >= self.cfg.state_label_debounce_steps:
            self.stable_label = self.candidate_label
            self.candidate_count = 0

        return self.stable_label


def compute_state_stats(state_counts):
    total = int(sum(state_counts))
    if total <= 0:
        return {
            'smooth': 0,
            'jamming': 0,
            'bottoming': 0,
            'ratio_smooth': 0.0,
            'ratio_jamming': 0.0,
            'ratio_bottoming': 0.0,
        }
    return {
        'smooth': int(state_counts[0]),
        'jamming': int(state_counts[1]),
        'bottoming': int(state_counts[2]),
        'ratio_smooth': float(state_counts[0] / total),
        'ratio_jamming': float(state_counts[1] / total),
        'ratio_bottoming': float(state_counts[2] / total),
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--snapshot', type=str,
                   default='./ckpt/residual_rl/snapshot_step315.pkl',
                   help='Path to a snapshot produced by create_snapshot.py')
    p.add_argument('--xml_path', type=str, default='./asset/pih.xml')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--total_steps', type=int, default=200_000)
    p.add_argument('--output_dir', type=str, default='./ckpt/residual_rl_ppo_aux')
    p.add_argument('--no_smolvla', action='store_true')
    p.add_argument('--smolvla_pretrained', type=str,
                   default='./ckpt/smolvla_pih1/checkpoints/020000/pretrained_model')
    p.add_argument('--smolvla_dataset_root', type=str, default='./demo_data_pih')
    p.add_argument('--smolvla_repo_name', type=str, default='ur5e_pih_language')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--max_steps', type=int, default=650,
                   help='Max ABSOLUTE step index (snapshot base_step counts)')
    p.add_argument('--noise_pos', type=float, default=2e-4,
                   help='EEF position noise half-range (m), per axis')
    p.add_argument('--noise_rot_deg', type=float, default=0.2,
                   help='EEF rotation noise half-range (deg), per axis')
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb_project', type=str, default='residual-rl-ppo-aux-pih')
    p.add_argument('--wandb_name', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    cfg = ResidualPPOAuxConfig(
        xml_path=args.xml_path,
        seed=args.seed,
        total_steps=args.total_steps,
        output_dir=args.output_dir,
        log_dir=os.path.join(args.output_dir, 'logs'),
        smolvla_pretrained=args.smolvla_pretrained,
        smolvla_dataset_root=args.smolvla_dataset_root,
        smolvla_repo_name=args.smolvla_repo_name,
        smolvla_device=str(device),
    )

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    dt = 1.0 / cfg.control_hz

    snap = load_snapshot(args.snapshot)
    base_step = snap['meta']['base_step']
    instruction_ref = snap['pih']['instruction']
    print(
        f"[snap] base_step={base_step}, peg={snap['pih']['active_peg']}, "
        f"instruction='{instruction_ref}'"
    )

    use_wandb = args.wandb
    if use_wandb:
        import wandb
        run_config = dict(vars(cfg)) if hasattr(cfg, '__dict__') else {}
        run_config.update({
            'snapshot_path': args.snapshot,
            'snapshot_base_step': base_step,
            'snapshot_peg': snap['pih']['active_peg'],
            'snapshot_instruction': instruction_ref,
            'noise_pos_m': args.noise_pos,
            'noise_rot_deg': args.noise_rot_deg,
            'max_steps': args.max_steps,
        })
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f'residual_rl_ppo_aux_seed{cfg.seed}',
            config=run_config,
        )

    smolvla = None if args.no_smolvla else load_smolvla(cfg)
    trainer = PPOAuxTrainer(config=cfg, device=str(device), use_wandb=use_wandb)
    if args.resume:
        trainer.load(args.resume)

    env = PIHEnv2(xml_path=cfg.xml_path, action_type='joint_angle', state_type='joint_angle')
    reward_fn = HierarchicalReward(cfg)
    state_labeler = DebouncedInsertionStateLabeler(cfg)

    ft_offset = snap['ft']['offset'].copy()
    ft_init = np.zeros(cfg.ft_dim, dtype=np.float32)
    reward_fn.reset(initial_ft=ft_init)
    ft_history = deque(maxlen=cfg.force_history_len)
    for _ in range(cfg.force_history_len):
        ft_history.append(ft_init.copy())

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    log_file = open(os.path.join(cfg.log_dir, 'train_log.csv'), 'w')
    log_file.write(
        'step,episode,reward,length,success,contact_step,rl_steps,reason,'
        'state_count_smooth,state_count_jamming,state_count_bottoming,'
        'state_ratio_smooth,state_ratio_jamming,state_ratio_bottoming\n'
    )

    step = base_step
    episode = 0
    episode_reward = 0.0
    contact_detected = False
    contact_step = -1
    steps_after_contact = 0
    excessive_force_counter = 0

    prev_ee_pose = env.get_ee_pose().copy()
    last_next_state = None
    last_next_ft_hist = None
    last_done = False
    last_state_label = 0
    last_rinfo = {}
    state_counts = np.zeros(cfg.num_state_classes, dtype=np.int64)

    def reset_episode():
        nonlocal step, episode, episode_reward, contact_detected, contact_step
        nonlocal excessive_force_counter, ft_offset, prev_ee_pose
        nonlocal last_next_state, last_next_ft_hist, last_done, last_state_label
        nonlocal steps_after_contact, last_rinfo, state_counts
        if smolvla is not None:
            smolvla.reset()
        ft_offset = restore_snapshot(env, snap)
        info = perturb_ee_pose(
            env,
            pos_noise=args.noise_pos,
            rot_noise_deg=args.noise_rot_deg,
            rng=rng,
        )
        ft_init_local = env.get_force_torque() - ft_offset
        reward_fn.reset(initial_ft=ft_init_local)
        ft_history.clear()
        for _ in range(cfg.force_history_len):
            ft_history.append(ft_init_local.copy())
        state_labeler.reset()
        prev_ee_pose = env.get_ee_pose().copy()
        step = base_step
        episode += 1
        episode_reward = 0.0
        contact_detected = False
        contact_step = -1
        steps_after_contact = 0
        excessive_force_counter = 0
        last_next_state = None
        last_next_ft_hist = None
        last_done = False
        last_state_label = 0
        last_rinfo = {}
        state_counts = np.zeros(cfg.num_state_classes, dtype=np.int64)
        print(
            f"\n--- Episode {episode} --- "
            f"dp={info['dp']*1000}mm drpy={np.rad2deg(info['drpy'])}deg  "
            f"ft_cal={ft_init_local}"
        )

    reset_episode()

    print(f"\n=== Snapshot PPO AUX ===")
    print(f"base_step        = {base_step}")
    print(f"instruction      = {env.instruction}")
    print(f"active peg       = {env.active_peg}")
    print(f"noise pos/rot    = {args.noise_pos*1000:.2f}mm / {args.noise_rot_deg:.2f}deg")
    print(f"max_residual_pos = {cfg.max_residual_pos} m")
    print(f"max_residual_rot = {cfg.max_residual_rot} rad")
    print(f"total RL target  = {cfg.total_steps}\n")

    while env.env.is_viewer_alive() and trainer.total_steps < cfg.total_steps:
        env.step_env()

        if not env.env.loop_every(HZ=20):
            continue

        success = env.check_success()
        timeout = step >= args.max_steps
        excessive_force = False
        if contact_detected:
            ft_cal = env.get_force_torque() - ft_offset
            if abs(ft_cal[2]) > cfg.excessive_force_th:
                excessive_force_counter += 1
            else:
                excessive_force_counter = 0
            excessive_force = excessive_force_counter >= cfg.excessive_force_steps

        out_of_hole = False
        if contact_detected:
            peg_tip = env.env.get_p_site(PEG_TIP_SITE)
            hole_entry = env.env.get_p_site(HOLE_ENTRY_SITE)
            xy_dist = np.linalg.norm(peg_tip[:2] - hole_entry[:2])
            peg_above = peg_tip[2] > hole_entry[2] + 0.005
            out_of_hole = xy_dist > cfg.out_of_hole_xy_th or peg_above

        done = success or timeout or excessive_force or out_of_hole

        if smolvla is not None:
            base_action = get_smolvla_action(smolvla, env, device)
        else:
            q = env.get_joint_state()[:6]
            base_action = np.concatenate([q, [200.0]], dtype=np.float32)

        if not contact_detected:
            contact_detected = check_contact(env)
            if contact_detected:
                contact_step = step
                print(f"  >> Insertion started at step {step}")

        cartesian_residual = np.zeros(6, dtype=np.float32)
        joint_delta = np.zeros(6, dtype=np.float32)
        log_prob = 0.0
        value = 0.0

        ft = env.get_force_torque() - ft_offset
        ee_pose = env.get_ee_pose()
        ee_vel = (ee_pose - prev_ee_pose) / dt
        ft_history.append(ft.copy())
        peg_tip = env.env.get_p_site(PEG_TIP_SITE)
        hole_entry = env.env.get_p_site(HOLE_ENTRY_SITE)
        hole_bottom = env.env.get_p_site(HOLE_BOTTOM_SITE)
        state_vec, ft_hist = build_rl_obs(ft, ee_pose, ee_vel, peg_tip, hole_entry, ft_history)

        state_label = 0
        if contact_detected:
            state_label = state_labeler.update(ft, ee_vel, peg_tip, hole_entry, hole_bottom)
            cartesian_residual, log_prob, value = trainer.act(state_vec, ft_hist)
            joint_delta = cartesian_to_joint_delta(env, cartesian_residual, cfg)

        final_action = base_action.copy()
        final_action[:6] += joint_delta
        env.step(final_action)

        next_ft = env.get_force_torque() - ft_offset
        next_ee_pose = env.get_ee_pose()
        next_ee_vel = (next_ee_pose - ee_pose) / dt
        next_peg_tip = env.env.get_p_site(PEG_TIP_SITE)
        next_hole_entry = env.env.get_p_site(HOLE_ENTRY_SITE)
        next_hole_bottom = env.env.get_p_site(HOLE_BOTTOM_SITE)
        next_hist_deque = deque(ft_history, maxlen=cfg.force_history_len)
        next_hist_deque.append(next_ft.copy())
        next_state_vec, next_ft_hist = build_rl_obs(
            next_ft, next_ee_pose, next_ee_vel, next_peg_tip, next_hole_entry, next_hist_deque
        )

        reward, _ = reward_fn.compute(next_ft, next_peg_tip, next_hole_entry, next_hole_bottom, success)
        if done and not success:
            reward -= cfg.penalty_early_stop
        episode_reward += reward
        last_rinfo = {
            'total_force': np.linalg.norm(next_ft[:3]),
            'lateral_force': np.linalg.norm(next_ft[:2]),
            'axial_force': abs(next_ft[2]),
            'bending_torque': np.linalg.norm(next_ft[3:5]),
            'insertion_progress': np.clip(
                (next_hole_entry[2] - next_peg_tip[2]) /
                max(next_hole_entry[2] - next_hole_bottom[2], 0.01),
                -1.0, 1.0,
            ),
        }

        if contact_detected:
            trainer.store_transition(
                state=state_vec,
                ft_history=ft_hist,
                action=cartesian_residual,
                log_prob=log_prob,
                reward=reward * cfg.reward_scale,
                done=done,
                value=value,
                next_force_target=next_ft,
                state_label=state_label,
            )
            trainer.total_steps += 1
            last_next_state = next_state_vec
            last_next_ft_hist = next_ft_hist
            last_done = done
            last_state_label = state_label
            state_counts[state_label] += 1
            steps_after_contact += 1

        if trainer.buffer.full:
            trainer.finish_rollout(last_next_state, last_next_ft_hist, last_done)
            trainer.update()

        ft_cal = next_ft
        env.render(idx=episode)
        env.env.viewer_text_overlay(
            text1='Residual PPO AUX',
            text2=(
                f'R={episode_reward:.2f} |cart|={np.linalg.norm(cartesian_residual):.5f} '
                f'|dq|={np.linalg.norm(joint_delta):.5f} '
                f'rl_steps={steps_after_contact} label={state_label}'
            ),
        )
        env.env.viewer_text_overlay(
            text1='F/T + EE vel',
            text2=(
                f'Fx={ft_cal[0]:.1f} Fy={ft_cal[1]:.1f} Fz={ft_cal[2]:.1f} '
                f'|v|={np.linalg.norm(next_ee_vel[:3]):.4f}'
            ),
        )
        env.env.viewer_text_overlay(
            text1='Snapshot',
            text2=f'base_step={base_step}  step={step}  contact={contact_detected}  excess={excessive_force_counter}',
        )

        if contact_detected and step % 5 == 0:
            state_stats = compute_state_stats(state_counts)
            print(
                f'  step={step:>4d} R={episode_reward:>7.2f} '
                f'|F|={np.linalg.norm(ft_cal[:3]):>5.1f}N '
                f'lat={np.linalg.norm(ft_cal[:2]):>4.1f} Fz={ft_cal[2]:>5.1f} '
                f'bend={np.linalg.norm(ft_cal[3:5]):.3f}Nm '
                f'|v|={np.linalg.norm(next_ee_vel[:3]):.4f} '
                f'label={state_label} '
                f'states[s/j/b]='
                f'{state_stats["ratio_smooth"]:.2f}/'
                f'{state_stats["ratio_jamming"]:.2f}/'
                f'{state_stats["ratio_bottoming"]:.2f}'
            )

        prev_ee_pose = next_ee_pose.copy()

        if done:
            if trainer.buffer.size > 0:
                trainer.finish_rollout(last_next_state, last_next_ft_hist, True)
                trainer.update()
            state_stats = compute_state_stats(state_counts)
            if success:
                reason = 'SUCCESS'
            elif excessive_force:
                reason = 'EXCESSIVE_FORCE'
            elif out_of_hole:
                reason = 'OUT_OF_HOLE'
            else:
                reason = 'TIMEOUT'
            c_tag = f'contact@{contact_step}' if contact_step >= 0 else 'no_contact'
            print(
                f"[PPO-AUX {trainer.total_steps:>7d} | Ep {episode:>4d}] "
                f"R={episode_reward:>8.2f} L={step-base_step:>4d} {reason} "
                f"{c_tag} rl_steps={steps_after_contact} "
                f"states[s/j/b]={state_stats['ratio_smooth']:.2f}/"
                f"{state_stats['ratio_jamming']:.2f}/{state_stats['ratio_bottoming']:.2f}"
            )
            log_file.write(
                f"{trainer.total_steps},{episode},{episode_reward:.4f},{step-base_step},"
                f"{int(success)},{contact_step},{steps_after_contact},{reason},"
                f"{state_stats['smooth']},{state_stats['jamming']},{state_stats['bottoming']},"
                f"{state_stats['ratio_smooth']:.6f},{state_stats['ratio_jamming']:.6f},"
                f"{state_stats['ratio_bottoming']:.6f}\n"
            )
            log_file.flush()
            trainer.log_episode(
                episode_reward,
                step - base_step,
                {
                    'success': success,
                    'contact_step': contact_step,
                    'steps_after_contact': steps_after_contact,
                    'reason': reason,
                    'state_count_smooth': state_stats['smooth'],
                    'state_count_jamming': state_stats['jamming'],
                    'state_count_bottoming': state_stats['bottoming'],
                    'state_ratio_smooth': state_stats['ratio_smooth'],
                    'state_ratio_jamming': state_stats['ratio_jamming'],
                    'state_ratio_bottoming': state_stats['ratio_bottoming'],
                },
                reward_info=last_rinfo,
            )
            if trainer.total_steps > 0 and trainer.total_steps % cfg.save_every < max(trainer.buffer.size, 1):
                trainer.save(os.path.join(cfg.output_dir, f'step_{trainer.total_steps}'))
            reset_episode()
            continue

        step += 1

    trainer.save(os.path.join(cfg.output_dir, 'final'))
    log_file.close()
    env.env.close_viewer()
    if use_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()
