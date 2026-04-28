"""
train_residual_rl_snapshot_admittance.py — Snapshot-based SAC + z-axis admittance.

Differences vs train_residual_rl_snapshot.py
--------------------------------------------
  • RL residual action is 4-D: [dx, dy, drx, dry]
      - NO z translation, NO z rotation from the policy.
  • Z-axis descent is governed by an analytic admittance law:
        dz_admit = k_admit * max(0, Fz_cal - fz_admit_threshold)
        dz_admit = clip(dz_admit, 0, dz_admit_max)
    Interpretation: the larger the axial resistance, the more upward
    residual we inject to cancel out the base policy's down-push.
    dz_admit is CLAMPED to be non-negative and bounded by dz_admit_max
    (< typical base descent per control step), so the peg never
    reverses out of the hole — it only descends more slowly.
  • The full Cartesian residual sent to IK is
        cart = [rl_dx, rl_dy, dz_admit, rl_drx, rl_dry, 0]
    so RL still fine-tunes lateral pose + bending, while admittance
    handles axial compliance.

Usage
-----
  python create_snapshot.py --snapshot_step 315
  python train_residual_rl_snapshot_admittance.py \
        --snapshot ./ckpt/residual_rl/snapshot_step315.pkl
  python train_residual_rl_snapshot_admittance.py --wandb \
        --snapshot ./ckpt/residual_rl/snapshot_step315.pkl
"""

import argparse
import os
from collections import deque

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from mujoco_env.pih_env2 import PIHEnv2
from residual_rl.config import ResidualRLConfig
from residual_rl.residual_policy import ResidualActor, ResidualCritic
from residual_rl.reward import HierarchicalReward
from residual_rl.sac_trainer import SACTrainer
from residual_rl.snapshot import load_snapshot, restore_snapshot, perturb_ee_pose


IMG_TRANSFORM = transforms.Compose([transforms.ToTensor()])

PEG_TIP_SITE     = 'peg_tip_site'
HOLE_ENTRY_SITE  = 'hole_entry_site'
HOLE_BOTTOM_SITE = 'hole_bottom_site'

ARM_JOINTS = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]
EEF_BODY = 'tool0_link'


# ======================================================================
# 4-D actor: [dx, dy, drx, dry]
# ======================================================================
class ResidualActor4D(ResidualActor):
    """
    Same encoder/head as ResidualActor, but the output is a 4-vector
    scaled by [pos, pos, rot, rot]. z-translation and z-rotation are
    removed from the RL action space — z-translation is handled by
    admittance control, and z-rotation is zeroed.
    """

    def __init__(self, cfg: ResidualRLConfig):
        assert cfg.action_dim == 4, \
            f"ResidualActor4D expects cfg.action_dim=4, got {cfg.action_dim}"
        super().__init__(cfg)
        # Parent registered a 6-vector action_scale; overwrite with 4-vector.
        self.register_buffer('action_scale', torch.tensor([
            cfg.max_residual_pos, cfg.max_residual_pos,
            cfg.max_residual_rot, cfg.max_residual_rot,
        ], dtype=torch.float32))


class SACTrainer4D(SACTrainer):
    """SAC trainer that uses ResidualActor4D instead of ResidualActor."""

    def __init__(self, config=None, device='cuda', use_wandb=False):
        super().__init__(config=config, device=device, use_wandb=use_wandb)
        # Replace the 6-D actor with the 4-D variant.
        import copy
        self.actor = ResidualActor4D(self.cfg).to(self.device)
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.cfg.lr_actor,
        )
        # Rebuild critic so that its action-dim input also matches 4.
        # ResidualCritic reads cfg.action_dim internally — it's already 4.
        self.critic = ResidualCritic(self.cfg).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad = False
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.cfg.lr_critic,
        )


# ======================================================================
# SmolVLA helpers (identical to the 6-D snapshot script)
# ======================================================================
def load_smolvla(cfg):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.common.datasets.utils import dataset_to_policy_features
    from lerobot.common.datasets.factory import resolve_delta_timestamps
    from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.configs.types import FeatureType

    print(f"[smolvla] loading from {cfg.smolvla_pretrained}")
    meta = LeRobotDatasetMetadata(cfg.smolvla_repo_name, root=cfg.smolvla_dataset_root)
    feats = dataset_to_policy_features(meta.features)
    out_feats = {k: f for k, f in feats.items() if f.type is FeatureType.ACTION}
    in_feats  = {k: f for k, f in feats.items() if k not in out_feats}

    smol_cfg = SmolVLAConfig(
        input_features=in_feats, output_features=out_feats,
        chunk_size=cfg.smolvla_chunk_size, n_action_steps=cfg.smolvla_n_action_steps,
    )
    resolve_delta_timestamps(smol_cfg, meta)

    policy = SmolVLAPolicy.from_pretrained(
        cfg.smolvla_pretrained, config=smol_cfg, dataset_stats=meta.stats,
    )
    policy.to(cfg.smolvla_device).eval()
    for p in policy.parameters():
        p.requires_grad = False
    print(f"[smolvla] {sum(p.numel() for p in policy.parameters())/1e6:.1f}M params, frozen")
    return policy


def get_smolvla_action(policy, pih, device):
    state = pih.get_joint_state()[:6]
    a_img, w_img = pih.grab_image()
    a = IMG_TRANSFORM(Image.fromarray(a_img).resize((256, 256)))
    w = IMG_TRANSFORM(Image.fromarray(w_img).resize((256, 256)))
    data = {
        'observation.state': torch.from_numpy(
            np.array([state], dtype=np.float32)).to(device),
        'observation.image': a.unsqueeze(0).to(device),
        'observation.wrist_image': w.unsqueeze(0).to(device),
        'task': [pih.instruction],
    }
    with torch.no_grad():
        act = policy.select_action(data)
    act_np = act[0, :7].cpu().numpy()
    act_np[6] = 200.0
    return act_np


# ======================================================================
# Geometry / observation helpers
# ======================================================================
def check_contact(pih, ft_offset):
    pt = pih.env.get_p_site(PEG_TIP_SITE)
    he = pih.env.get_p_site(HOLE_ENTRY_SITE)
    xy = np.linalg.norm(pt[:2] - he[:2])
    return (pt[2] < he[2]) and (xy < 0.03)


def build_rl_obs(pih, ft_offset, ft_history):
    ft = pih.get_force_torque() - ft_offset
    ee = pih.get_ee_pose()
    pt = pih.env.get_p_site(PEG_TIP_SITE)
    he = pih.env.get_p_site(HOLE_ENTRY_SITE)
    rel = (he - pt).astype(np.float32)
    return np.concatenate([ft, ee, rel], dtype=np.float32), \
           np.array(list(ft_history), dtype=np.float32)


def get_cart_limits_4d(cfg):
    """4-D action limits for [dx, dy, drx, dry]."""
    return np.array([
        cfg.max_residual_pos, cfg.max_residual_pos,
        cfg.max_residual_rot, cfg.max_residual_rot,
    ], dtype=np.float32)


def admittance_dz(fz_cal, cfg):
    """
    Z-axis admittance law.

    Args:
        fz_cal: scalar — calibrated axial force on the peg (world +z = up).
                Positive Fz ⇒ hole pushes peg upward ⇒ peg is resisting descent.
        cfg: ResidualRLConfig with fz_admit_threshold, k_admit, dz_admit_max.

    Returns:
        dz_admit: scalar in [0, dz_admit_max], world-frame +z (upward).
                  Added to cart[2] to cancel part of the base policy's down-push.
                  Never negative → never pushes peg down harder.
                  Capped at dz_admit_max (< base-policy descent/step) so the peg
                  can slow down but NEVER reverses out of the hole.
    """
    resistance = max(0.0, float(fz_cal) - cfg.fz_admit_threshold)
    dz = cfg.k_admit * resistance
    dz = min(dz, cfg.dz_admit_max)
    return max(0.0, dz)


def expand_rl_to_cart(rl_action4, dz_admit):
    """Map 4-D RL action + admittance dz into 6-D Cartesian residual."""
    cart = np.zeros(6, dtype=np.float32)
    cart[0] = rl_action4[0]      # dx
    cart[1] = rl_action4[1]      # dy
    cart[2] = dz_admit           # dz from admittance (always >=0, upward)
    cart[3] = rl_action4[2]      # drx
    cart[4] = rl_action4[3]      # dry
    cart[5] = 0.0                # drz pinned to zero
    return cart


def cart_to_joint_delta(pih, cart, cfg):
    _, _, J_full = pih.env.get_J_body(EEF_BODY)
    J = J_full[:, pih.env.get_idxs_jac(ARM_JOINTS)]
    JJT = J @ J.T + cfg.ik_damping_eps * np.eye(6)
    J_pinv = J.T @ np.linalg.solve(JJT, np.eye(6))
    dq = np.clip(J_pinv @ cart, -cfg.max_joint_delta, cfg.max_joint_delta)
    return dq.astype(np.float32)


# ======================================================================
# CLI
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--snapshot', type=str,
                   default='./ckpt/residual_rl/snapshot_step315.pkl')
    p.add_argument('--xml_path', type=str, default='./asset/pih.xml')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--total_steps', type=int, default=100_000)
    p.add_argument('--output_dir', type=str,
                   default='./ckpt/residual_rl_snapshot_admittance')
    p.add_argument('--no_smolvla', action='store_true')
    p.add_argument('--smolvla_pretrained', type=str,
                   default='./ckpt/smolvla_pih1/checkpoints/020000/pretrained_model')
    p.add_argument('--smolvla_dataset_root', type=str, default='./demo_data_pih')
    p.add_argument('--smolvla_repo_name', type=str, default='ur5e_pih_language')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--max_steps', type=int, default=650)
    p.add_argument('--noise_pos', type=float, default=1.5e-5)
    p.add_argument('--noise_rot_deg', type=float, default=0.2)
    # Admittance params (can be overridden on the CLI)
    p.add_argument('--fz_admit_threshold', type=float, default=15.0,
                   help='Fz (N) above which admittance kicks in')
    p.add_argument('--k_admit', type=float, default=1.5e-5,
                   help='Compliance: meters of upward residual per Newton of excess Fz')
    p.add_argument('--dz_admit_max', type=float, default=5.0e-4,
                   help='Hard cap on per-step admittance residual (m). '
                        'Must stay below typical base-policy descent/step to '
                        'guarantee no reverse pull-out.')
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb_project', type=str, default='residual-rl-pih')
    p.add_argument('--wandb_name', type=str, default=None)
    return p.parse_args()


# ======================================================================
# Main
# ======================================================================
def main():
    args = parse_args()
    device = torch.device(args.device)

    cfg = ResidualRLConfig(
        xml_path=args.xml_path, seed=args.seed,
        total_steps=args.total_steps,
        output_dir=args.output_dir,
        log_dir=os.path.join(args.output_dir, 'logs'),
        smolvla_pretrained=args.smolvla_pretrained,
        smolvla_dataset_root=args.smolvla_dataset_root,
        smolvla_repo_name=args.smolvla_repo_name,
        smolvla_device=str(device),
    )
    # 4-D residual action
    cfg.action_dim = 4
    # Admittance parameters (attached to cfg so trainer/actor can see them)
    cfg.fz_admit_threshold = args.fz_admit_threshold
    cfg.k_admit = args.k_admit
    cfg.dz_admit_max = args.dz_admit_max
    # Target entropy: -dim/2 is a reasonable default for bounded continuous action
    cfg.target_entropy = -2.0

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------
    snap = load_snapshot(args.snapshot)
    base_step = snap['meta']['base_step']
    instruction_ref = snap['pih']['instruction']
    print(f"[snap] base_step={base_step}, peg={snap['pih']['active_peg']}, "
          f"instruction='{instruction_ref}'")

    # ------------------------------------------------------------------
    # Wandb
    # ------------------------------------------------------------------
    use_wandb = args.wandb
    if use_wandb:
        import wandb
        run_config = vars(cfg) if hasattr(cfg, '__dict__') else {}
        run_config = dict(run_config)
        run_config.update({
            'snapshot_path':       args.snapshot,
            'snapshot_base_step':  base_step,
            'snapshot_peg':        snap['pih']['active_peg'],
            'snapshot_instruction': instruction_ref,
            'noise_pos_m':         args.noise_pos,
            'noise_rot_deg':       args.noise_rot_deg,
            'max_steps':           args.max_steps,
            'no_smolvla':          args.no_smolvla,
            'action_dim':          cfg.action_dim,
            'fz_admit_threshold':  cfg.fz_admit_threshold,
            'k_admit':             cfg.k_admit,
            'dz_admit_max':        cfg.dz_admit_max,
        })
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f'residual_rl_snap_admit_seed{cfg.seed}',
            config=run_config,
        )
        print("Wandb initialized.")

    # ------------------------------------------------------------------
    # SmolVLA (frozen)
    # ------------------------------------------------------------------
    smolvla = None if args.no_smolvla else load_smolvla(cfg)

    # ------------------------------------------------------------------
    # SAC trainer (4-D actor)
    # ------------------------------------------------------------------
    trainer = SACTrainer4D(config=cfg, device=str(device), use_wandb=use_wandb)
    if args.resume:
        trainer.load(args.resume)

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    print(f"[env] creating PIHEnv2 from {cfg.xml_path}")
    pih = PIHEnv2(
        xml_path=cfg.xml_path,
        action_type='joint_angle',
        state_type='joint_angle',
    )

    reward_fn = HierarchicalReward(cfg)
    ft_history = deque(maxlen=cfg.force_history_len)

    # ------------------------------------------------------------------
    # Episode state
    # ------------------------------------------------------------------
    step = base_step
    episode = 0
    contact_detected = False
    contact_step = -1
    steps_after_contact = 0
    episode_reward = 0.0
    prev_obs = None
    last_rinfo = {}
    excessive_force_counter = 0
    ft_offset = snap['ft']['offset'].copy()
    # Diagnostics
    ep_admit_trigger_steps = 0
    ep_admit_dz_sum = 0.0
    ep_max_fz = 0.0

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    log_file = open(os.path.join(cfg.log_dir, 'train_log.csv'), 'w')
    log_file.write(
        'step,episode,reward,length,success,contact_step,rl_steps,'
        'admit_triggered,admit_dz_sum,max_fz\n'
    )

    # ------------------------------------------------------------------
    # Reset helpers
    # ------------------------------------------------------------------
    def reset_from_snapshot():
        nonlocal step, episode, contact_detected, contact_step, steps_after_contact
        nonlocal episode_reward, prev_obs, last_rinfo, excessive_force_counter
        nonlocal ft_offset
        nonlocal ep_admit_trigger_steps, ep_admit_dz_sum, ep_max_fz

        if smolvla is not None:
            smolvla.reset()

        ft_offset = restore_snapshot(pih, snap)
        info = perturb_ee_pose(pih,
                               pos_noise=args.noise_pos,
                               rot_noise_deg=args.noise_rot_deg,
                               rng=rng)

        ft_cal = pih.get_force_torque() - ft_offset
        reward_fn.reset(initial_ft=ft_cal)
        ft_history.clear()
        for _ in range(cfg.force_history_len):
            ft_history.append(ft_cal.copy())

        step = base_step
        episode += 1
        contact_detected = False
        contact_step = -1
        steps_after_contact = 0
        episode_reward = 0.0
        prev_obs = None
        last_rinfo = {}
        excessive_force_counter = 0
        ep_admit_trigger_steps = 0
        ep_admit_dz_sum = 0.0
        ep_max_fz = 0.0

        print(f"\n--- Episode {episode} --- "
              f"dp={info['dp']*1000}mm drpy={np.rad2deg(info['drpy'])}deg  "
              f"ft_cal={ft_cal}")

    def store_terminal(done_reward, is_success):
        nonlocal episode_reward, last_rinfo
        if prev_obs is None or not contact_detected:
            return
        ft = pih.get_force_torque() - ft_offset
        ft_history.append(ft.copy())
        state_vec, ft_hist = build_rl_obs(pih, ft_offset, ft_history)
        pt = pih.env.get_p_site(PEG_TIP_SITE)
        he = pih.env.get_p_site(HOLE_ENTRY_SITE)
        hb = pih.env.get_p_site(HOLE_BOTTOM_SITE)
        r, rinfo = reward_fn.compute(ft, pt, he, hb, is_success)
        r += done_reward
        episode_reward += r
        last_rinfo = rinfo
        trainer.store_transition(
            state=prev_obs[0], ft_history=prev_obs[1],
            action=prev_obs[2], reward=r,
            next_state=state_vec, next_ft_history=ft_hist,
            done=True,
        )
        trainer.total_steps += 1

    # ------------------------------------------------------------------
    # First reset
    # ------------------------------------------------------------------
    reset_from_snapshot()

    print(f"\n=== Snapshot-based SAC + z-axis Admittance ===")
    print(f"base_step           = {base_step}")
    print(f"instruction         = {pih.instruction}")
    print(f"active peg          = {pih.active_peg}")
    print(f"noise pos/rot       = {args.noise_pos*1000:.2f}mm / {args.noise_rot_deg:.2f}deg")
    print(f"RL action dim       = {cfg.action_dim} [dx,dy,drx,dry]")
    print(f"max_residual_pos    = {cfg.max_residual_pos} m")
    print(f"max_residual_rot    = {cfg.max_residual_rot} rad")
    print(f"fz_admit_threshold  = {cfg.fz_admit_threshold} N")
    print(f"k_admit             = {cfg.k_admit} m/N")
    print(f"dz_admit_max        = {cfg.dz_admit_max} m (per step cap, no reverse)")
    print(f"total RL target     = {cfg.total_steps}\n")

    # ==================================================================
    # MAIN LOOP
    # ==================================================================
    while pih.env.is_viewer_alive() and trainer.total_steps < cfg.total_steps:
        pih.step_env()

        if not pih.env.loop_every(HZ=20):
            continue

        # ----- termination checks -----
        success = pih.check_success()
        timeout = (step >= args.max_steps)

        excessive_force = False
        if contact_detected:
            ft_cal_check = pih.get_force_torque() - ft_offset
            if abs(ft_cal_check[2]) > cfg.excessive_force_th:
                excessive_force_counter += 1
            else:
                excessive_force_counter = 0
            if excessive_force_counter >= cfg.excessive_force_steps:
                excessive_force = True

        out_of_hole = False
        if contact_detected:
            pt = pih.env.get_p_site(PEG_TIP_SITE)
            he = pih.env.get_p_site(HOLE_ENTRY_SITE)
            xy = np.linalg.norm(pt[:2] - he[:2])
            above = pt[2] > he[2] + 0.005
            if xy > cfg.out_of_hole_xy_th or above:
                out_of_hole = True

        if success or timeout or excessive_force or out_of_hole:
            if success:
                reason, terminal_r = 'SUCCESS', 0.0
            elif excessive_force:
                reason, terminal_r = 'EXCESSIVE_FORCE', -cfg.penalty_early_stop
            elif out_of_hole:
                reason, terminal_r = 'OUT_OF_HOLE', -cfg.penalty_early_stop
            else:
                reason, terminal_r = 'TIMEOUT', -cfg.penalty_early_stop * 0.5

            store_terminal(terminal_r, success)

            c_tag = f'contact@{contact_step}' if contact_step >= 0 else 'no_contact'
            print(f"[RL {trainer.total_steps:>7d} | Ep {episode:>4d}] "
                  f"R={episode_reward:>8.2f} L={step-base_step:>4d} {reason} "
                  f"{c_tag} rl_steps={steps_after_contact} "
                  f"admit={ep_admit_trigger_steps} dz_sum={ep_admit_dz_sum*1000:.2f}mm "
                  f"max|Fz|={ep_max_fz:.1f}N alpha={trainer.alpha:.3f}")
            log_file.write(
                f"{trainer.total_steps},{episode},{episode_reward:.4f},"
                f"{step-base_step},{int(success)},{contact_step},"
                f"{steps_after_contact},{ep_admit_trigger_steps},"
                f"{ep_admit_dz_sum:.6f},{ep_max_fz:.3f}\n"
            )
            log_file.flush()

            trainer.log_episode(
                episode_reward, step - base_step,
                {'success': success, 'contact_step': contact_step,
                 'steps_after_contact': steps_after_contact, 'reason': reason},
                reward_info=last_rinfo,
            )
            if use_wandb:
                import wandb
                wandb.log({
                    'admittance/trigger_steps': ep_admit_trigger_steps,
                    'admittance/dz_sum_m':      ep_admit_dz_sum,
                    'admittance/max_abs_fz':    ep_max_fz,
                }, step=trainer.total_steps)

            if trainer.total_steps > 0 and \
                    trainer.total_steps % cfg.save_every < steps_after_contact + 1:
                trainer.save(os.path.join(cfg.output_dir, f'step_{trainer.total_steps}'))

            reset_from_snapshot()
            continue

        # ----- SmolVLA base action -----
        if smolvla is not None:
            base_action = get_smolvla_action(smolvla, pih, str(device))
        else:
            q = pih.get_joint_state()[:6]
            base_action = np.concatenate([q, [200.0]], dtype=np.float32)

        # ----- contact detection -----
        if not contact_detected:
            if check_contact(pih, ft_offset):
                contact_detected = True
                contact_step = step
                pt = pih.env.get_p_site(PEG_TIP_SITE)
                he = pih.env.get_p_site(HOLE_ENTRY_SITE)
                print(f"  >> Insertion started at step {step} "
                      f"(depth={1000*(he[2]-pt[2]):.1f}mm)")

        # ----- residual: RL (4-D) + admittance (z) -----
        cart_res = np.zeros(6, dtype=np.float32)
        joint_delta = np.zeros(6, dtype=np.float32)
        dz_admit_this = 0.0

        if contact_detected and cfg.max_residual_pos > 0:
            ft = pih.get_force_torque() - ft_offset
            fz_cal = float(ft[2])
            ft_history.append(ft.copy())
            state_vec, ft_hist = build_rl_obs(pih, ft_offset, ft_history)

            if prev_obs is not None:
                pt = pih.env.get_p_site(PEG_TIP_SITE)
                he = pih.env.get_p_site(HOLE_ENTRY_SITE)
                hb = pih.env.get_p_site(HOLE_BOTTOM_SITE)
                r, rinfo = reward_fn.compute(ft, pt, he, hb, False)
                episode_reward += r
                last_rinfo = rinfo
                trainer.store_transition(
                    state=prev_obs[0], ft_history=prev_obs[1],
                    action=prev_obs[2], reward=r,
                    next_state=state_vec, next_ft_history=ft_hist,
                    done=False,
                )
                trainer.total_steps += 1
                steps_after_contact += 1

                if trainer.total_steps >= cfg.warmup_steps and \
                        trainer.total_steps % cfg.update_every == 0:
                    for _ in range(cfg.updates_per_step):
                        trainer.update()

            # RL samples 4-D residual
            s_t = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
            h_t = torch.FloatTensor(ft_hist).unsqueeze(0).to(device)
            limits4 = get_cart_limits_4d(cfg)
            if trainer.total_steps < cfg.warmup_steps:
                rl_action = np.random.uniform(-limits4, limits4).astype(np.float32)
            else:
                with torch.no_grad():
                    a, _ = trainer.actor.sample(s_t, h_t)
                    rl_action = a.squeeze(0).cpu().numpy()
                rl_action = np.clip(rl_action, -limits4, limits4)

            # Admittance: compute dz_admit from current Fz BEFORE stepping.
            dz_admit_this = admittance_dz(fz_cal, cfg)
            if dz_admit_this > 0.0:
                ep_admit_trigger_steps += 1
                ep_admit_dz_sum += dz_admit_this
            if abs(fz_cal) > ep_max_fz:
                ep_max_fz = abs(fz_cal)

            cart_res = expand_rl_to_cart(rl_action, dz_admit_this)
            joint_delta = cart_to_joint_delta(pih, cart_res, cfg)

            # Store the 4-D RL action (NOT the 6-D cart) in the replay buffer,
            # because the policy only controls 4 dims; admittance is deterministic.
            prev_obs = (state_vec.copy(), ft_hist.copy(), rl_action.copy())
        else:
            ft = pih.get_force_torque() - ft_offset
            ft_history.append(ft.copy())

        final_action = base_action.copy()
        final_action[:6] += joint_delta
        _ = pih.step(final_action)

        # ----- monitoring -----
        ft_cal = pih.get_force_torque() - ft_offset
        lat_f  = np.linalg.norm(ft_cal[:2])
        axial  = ft_cal[2]
        bend_t = np.linalg.norm(ft_cal[3:5])
        tot_f  = np.linalg.norm(ft_cal[:3])
        peg_z  = pih.env.get_p_site(PEG_TIP_SITE)[2]

        pih.render(idx=episode)
        if contact_detected and prev_obs is not None:
            pih.env.viewer_text_overlay(
                text1='Residual RL (4D) + Admittance',
                text2=f'R={episode_reward:.2f}  '
                      f'|cart|={np.linalg.norm(cart_res):.5f}  '
                      f'dz_admit={dz_admit_this*1000:.3f}mm  '
                      f'rl_steps={steps_after_contact}',
            )
        pih.env.viewer_text_overlay(
            text1='F/T Calibrated',
            text2=f'Fx={ft_cal[0]:.1f} Fy={ft_cal[1]:.1f} Fz={ft_cal[2]:.1f} '
                  f'|Flat|={lat_f:.1f} |F|={tot_f:.1f}N  '
                  f'Tx={ft_cal[3]:.3f} Ty={ft_cal[4]:.3f} |Tb|={bend_t:.3f}Nm',
        )
        pih.env.viewer_text_overlay(
            text1='Snapshot+Admit',
            text2=f'base_step={base_step}  step={step}  '
                  f'contact={contact_detected}  excess={excessive_force_counter}',
        )

        if contact_detected and step % 5 == 0:
            r_str = f'R={episode_reward:>7.2f}' if prev_obs is not None else 'R=  n/a'
            print(f'  step={step:>4d} {r_str} '
                  f'|F|={tot_f:>5.1f}N lat={lat_f:>4.1f} Fz={axial:>5.1f} '
                  f'bend={bend_t:.3f}Nm peg_z={peg_z:.4f} '
                  f'dz_admit={dz_admit_this*1e6:>5.1f}um '
                  f'|cart|={np.linalg.norm(cart_res):.5f} '
                  f'|dq|={np.linalg.norm(joint_delta):.5f}')

        step += 1

    # ------------------------------------------------------------------
    # Wrap up
    # ------------------------------------------------------------------
    trainer.save(os.path.join(cfg.output_dir, 'final'))
    log_file.close()
    pih.env.close_viewer()
    if use_wandb:
        import wandb
        wandb.finish()
    print(f"\n[done] {episode} episodes, {trainer.total_steps} RL steps.")


if __name__ == '__main__':
    main()
