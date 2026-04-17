"""
train_residual_rl.py — Train residual RL (SAC) on top of frozen SmolVLA.

Uses the EXACT same control loop as smolvla1.py to guarantee identical base
trajectory. Residual RL only adds tiny corrections after contact detection.

Usage:
  python train_residual_rl.py
  python train_residual_rl.py --no_smolvla
  python train_residual_rl.py --resume ./ckpt/residual_rl/step_50000
"""

import argparse
import os
import copy
import time
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from collections import deque

from residual_rl.config import ResidualRLConfig
from residual_rl.residual_policy import ResidualActor, ResidualCritic
from residual_rl.replay_buffer import ReplayBuffer
from residual_rl.reward import HierarchicalReward
from residual_rl.sac_trainer import SACTrainer
from mujoco_env.pih_env2 import PIHEnv2


IMG_TRANSFORM = transforms.Compose([transforms.ToTensor()])

# Site names
PEG_TIP_SITE = 'peg_tip_site'
HOLE_ENTRY_SITE = 'hole_entry_site'
HOLE_BOTTOM_SITE = 'hole_bottom_site'


def load_smolvla(cfg):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.common.datasets.utils import dataset_to_policy_features
    from lerobot.common.datasets.factory import resolve_delta_timestamps
    from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.configs.types import FeatureType

    print(f"Loading SmolVLA from {cfg.smolvla_pretrained} ...")
    meta = LeRobotDatasetMetadata(cfg.smolvla_repo_name, root=cfg.smolvla_dataset_root)
    features = dataset_to_policy_features(meta.features)
    out_feat = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
    in_feat = {k: f for k, f in features.items() if k not in out_feat}

    smolvla_cfg = SmolVLAConfig(
        input_features=in_feat, output_features=out_feat,
        chunk_size=cfg.smolvla_chunk_size, n_action_steps=cfg.smolvla_n_action_steps,
    )
    resolve_delta_timestamps(smolvla_cfg, meta)

    policy = SmolVLAPolicy.from_pretrained(
        cfg.smolvla_pretrained, config=smolvla_cfg, dataset_stats=meta.stats,
    )
    policy.to(cfg.smolvla_device)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad = False
    print(f"SmolVLA loaded ({sum(p.numel() for p in policy.parameters())/1e6:.1f}M params, frozen)")
    return policy


def get_smolvla_action(policy, pih_env, device):
    """Exactly the same inference as smolvla1.py."""
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


def check_contact(pih_env, ft_offset):
    """Check if peg tip is below hole entry (insertion started)."""
    peg_tip = pih_env.env.get_p_site(PEG_TIP_SITE)
    hole_entry = pih_env.env.get_p_site(HOLE_ENTRY_SITE)
    xy_dist = np.linalg.norm(peg_tip[:2] - hole_entry[:2])
    peg_below = peg_tip[2] < hole_entry[2]
    return peg_below and xy_dist < 0.03


def build_rl_obs(pih_env, ft_offset, ft_history):
    """Build observation: ft(6) + ee_pose(6) + relative(3) = 15D."""
    ft = pih_env.get_force_torque() - ft_offset
    ee = pih_env.get_ee_pose()
    peg_tip = pih_env.env.get_p_site(PEG_TIP_SITE)
    hole_entry = pih_env.env.get_p_site(HOLE_ENTRY_SITE)
    relative = (hole_entry - peg_tip).astype(np.float32)
    state = np.concatenate([ft, ee, relative], dtype=np.float32)
    ft_hist = np.array(list(ft_history), dtype=np.float32)
    return state, ft_hist


# Joint names for Jacobian computation
ARM_JOINTS = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]
EEF_BODY = 'tool0_link'


def get_cartesian_residual_limits(cfg):
    """Per-axis Cartesian residual bounds [m, m, m, rad, rad, rad]."""
    return np.array([
        cfg.max_residual_pos, cfg.max_residual_pos, cfg.max_residual_pos,
        cfg.max_residual_rot, cfg.max_residual_rot, cfg.max_residual_rot,
    ], dtype=np.float32)


def cartesian_to_joint_delta(pih_env, cart_residual, cfg):
    """
    Convert 6D Cartesian residual [dx,dy,dz,drx,dry,drz] to 6D joint angle delta
    using the Jacobian pseudo-inverse.

    Args:
        pih_env: PIHEnv2 instance
        cart_residual: np.ndarray (6,) — Cartesian residual
        cfg: ResidualRLConfig

    Returns:
        np.ndarray (6,) — joint angle delta, clamped to max_joint_delta
    """
    # Get Jacobian for the arm joints at EEF
    _, _, J_full = pih_env.env.get_J_body(EEF_BODY)
    jac_idxs = pih_env.env.get_idxs_jac(ARM_JOINTS)
    J_arm = J_full[:, jac_idxs]  # (6, 6)

    # Damped pseudo-inverse: J^+ = J^T (J J^T + εI)^{-1}
    eps = cfg.ik_damping_eps
    JJT = J_arm @ J_arm.T + eps * np.eye(6)
    J_pinv = J_arm.T @ np.linalg.solve(JJT, np.eye(6))

    # Joint delta = J^+ @ Cartesian residual
    joint_delta = J_pinv @ cart_residual

    # Safety clamp
    joint_delta = np.clip(joint_delta, -cfg.max_joint_delta, cfg.max_joint_delta)

    return joint_delta.astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--xml_path', type=str, default='./asset/pih.xml')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--total_steps', type=int, default=100_000)
    p.add_argument('--output_dir', type=str, default='./ckpt/residual_rl')
    p.add_argument('--no_smolvla', action='store_true')
    p.add_argument('--smolvla_pretrained', type=str,
                   default='./ckpt/smolvla_pih1/checkpoints/020000/pretrained_model')
    p.add_argument('--smolvla_dataset_root', type=str, default='./demo_data_pih')
    p.add_argument('--smolvla_repo_name', type=str, default='ur5e_pih_language')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--max_steps', type=int, default=650)
    p.add_argument('--wandb', action='store_true', help='Enable wandb logging')
    p.add_argument('--wandb_project', type=str, default='residual-rl-pih')
    p.add_argument('--wandb_name', type=str, default=None)
    return p.parse_args()


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

    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)

    # Wandb
    use_wandb = args.wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f'residual_rl_seed{cfg.seed}',
            config=vars(cfg) if hasattr(cfg, '__dict__') else {},
        )
        print("Wandb initialized.")

    # Load SmolVLA
    smolvla = None
    if not args.no_smolvla:
        smolvla = load_smolvla(cfg)

    # Create SAC trainer
    trainer = SACTrainer(config=cfg, device=str(device), use_wandb=use_wandb)
    if args.resume:
        trainer.load(args.resume)

    # Create environment — EXACTLY like smolvla1.py
    print("Creating PIH environment ...")
    PIHEnv = PIHEnv2(
        xml_path=cfg.xml_path,
        action_type='joint_angle',
        state_type='joint_angle',
    )
    # Force round peg
    PIHEnv.set_instruction('Insert the round peg into the blue round hole.')

    # Reward function
    reward_fn = HierarchicalReward(cfg)

    # F/T calibration (just read once, no extra physics)
    ft_offset = PIHEnv.get_force_torque().copy()
    reward_fn.reset(initial_ft=PIHEnv.get_force_torque() - ft_offset)

    # Force history
    ft_history = deque(maxlen=cfg.force_history_len)
    ft_init = PIHEnv.get_force_torque() - ft_offset
    for _ in range(cfg.force_history_len):
        ft_history.append(ft_init.copy())

    # Logging
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    log_file = open(os.path.join(cfg.log_dir, 'train_log.csv'), 'w')
    log_file.write('step,episode,reward,length,success,contact_step,rl_steps\n')

    # State
    step = 0
    episode = 0
    contact_detected = False
    contact_step = -1
    steps_after_contact = 0
    episode_reward = 0.0
    prev_obs = None  # for storing transitions
    last_rinfo = {}  # last reward info for logging

    print(f"\n=== Training: SmolVLA + Residual RL (same loop as smolvla1.py) ===")
    print(f"Instruction: {PIHEnv.instruction}")
    print(f"Active peg: {PIHEnv.active_peg}")
    print(f"max_residual_pos: {cfg.max_residual_pos} m")
    print(f"max_residual_rot: {cfg.max_residual_rot} rad")
    print(f"Total RL steps target: {cfg.total_steps}")
    print()

    # Early-stop state
    excessive_force_counter = 0

    def reset_episode():
        """Reset all episode state. Returns nothing, modifies nonlocal vars."""
        nonlocal step, episode, contact_detected, contact_step, steps_after_contact
        nonlocal episode_reward, prev_obs, last_rinfo, excessive_force_counter

        if smolvla is not None:
            smolvla.reset()
        PIHEnv.reset()
        PIHEnv.set_instruction('Insert the round peg into the blue round hole.')

        nonlocal ft_offset
        ft_offset = PIHEnv.get_force_torque().copy()
        reward_fn.reset(initial_ft=PIHEnv.get_force_torque() - ft_offset)
        ft_history.clear()
        ft_init = PIHEnv.get_force_torque() - ft_offset
        for _ in range(cfg.force_history_len):
            ft_history.append(ft_init.copy())

        step = 0
        episode += 1
        contact_detected = False
        contact_step = -1
        steps_after_contact = 0
        episode_reward = 0.0
        prev_obs = None
        last_rinfo = {}
        excessive_force_counter = 0

    def store_terminal_transition(done_reward, is_success):
        """Store the FINAL transition of the episode with done=True."""
        nonlocal episode_reward, last_rinfo
        if prev_obs is None or not contact_detected:
            return

        ft = PIHEnv.get_force_torque() - ft_offset
        ft_history.append(ft.copy())
        state_vec, ft_hist = build_rl_obs(PIHEnv, ft_offset, ft_history)

        peg_tip = PIHEnv.env.get_p_site(PEG_TIP_SITE)
        hole_entry = PIHEnv.env.get_p_site(HOLE_ENTRY_SITE)
        hole_bottom = PIHEnv.env.get_p_site(HOLE_BOTTOM_SITE)
        reward, rinfo = reward_fn.compute(
            ft, peg_tip, hole_entry, hole_bottom, is_success)
        reward += done_reward  # extra terminal bonus/penalty
        episode_reward += reward
        last_rinfo = rinfo

        trainer.store_transition(
            state=prev_obs[0], ft_history=prev_obs[1],
            action=prev_obs[2], reward=reward,
            next_state=state_vec, next_ft_history=ft_hist,
            done=True,  # TERMINAL
        )
        trainer.total_steps += 1

    # ================================================================
    # MAIN LOOP — identical structure to smolvla1.py
    # ================================================================
    while PIHEnv.env.is_viewer_alive() and trainer.total_steps < cfg.total_steps:
        PIHEnv.step_env()

        if PIHEnv.env.loop_every(HZ=20):

            # ==========================================================
            # TERMINATION CHECK — success / timeout / excessive force / out of hole
            # ==========================================================
            success = PIHEnv.check_success()
            timeout = (step >= args.max_steps)

            # Excessive force: |Fz| > threshold for N consecutive steps
            excessive_force = False
            if contact_detected:
                ft_cal = PIHEnv.get_force_torque() - ft_offset
                if abs(ft_cal[2]) > cfg.excessive_force_th:
                    excessive_force_counter += 1
                else:
                    excessive_force_counter = 0
                if excessive_force_counter >= cfg.excessive_force_steps:
                    excessive_force = True

            # Out of hole: peg was in hole but moved out
            out_of_hole = False
            if contact_detected:
                peg_tip = PIHEnv.env.get_p_site(PEG_TIP_SITE)
                hole_entry = PIHEnv.env.get_p_site(HOLE_ENTRY_SITE)
                xy_dist = np.linalg.norm(peg_tip[:2] - hole_entry[:2])
                peg_above = peg_tip[2] > hole_entry[2] + 0.005
                if xy_dist > cfg.out_of_hole_xy_th or peg_above:
                    out_of_hole = True

            is_done = success or timeout or excessive_force or out_of_hole

            if is_done:
                # Determine reason and terminal reward
                if success:
                    reason = "SUCCESS"
                    terminal_reward = 0.0  # w_success already in reward_fn
                elif excessive_force:
                    reason = "EXCESSIVE_FORCE"
                    terminal_reward = -cfg.penalty_early_stop
                elif out_of_hole:
                    reason = "OUT_OF_HOLE"
                    terminal_reward = -cfg.penalty_early_stop
                else:
                    reason = "TIMEOUT"
                    terminal_reward = -cfg.penalty_early_stop * 0.5

                # Store final transition with done=True
                store_terminal_transition(terminal_reward, success)

                # Log
                c_tag = f"contact@{contact_step}" if contact_step >= 0 else "no_contact"
                print(f"[RL {trainer.total_steps:>7d} | Ep {episode:>4d}] "
                      f"R={episode_reward:>8.2f} L={step:>4d} {reason} {c_tag} "
                      f"rl_steps={steps_after_contact} alpha={trainer.alpha:.3f}")

                log_file.write(f"{trainer.total_steps},{episode},{episode_reward:.4f},"
                               f"{step},{int(success)},{contact_step},{steps_after_contact}\n")
                log_file.flush()

                trainer.log_episode(
                    episode_reward, step,
                    {'success': success, 'contact_step': contact_step,
                     'steps_after_contact': steps_after_contact,
                     'reason': reason},
                    reward_info=last_rinfo,
                )

                # Save periodically
                if trainer.total_steps > 0 and trainer.total_steps % cfg.save_every < steps_after_contact + 1:
                    trainer.save(os.path.join(cfg.output_dir, f'step_{trainer.total_steps}'))

                reset_episode()
                continue

            # ==========================================================
            # SmolVLA inference (EXACTLY like smolvla1.py)
            # ==========================================================
            if smolvla is not None:
                action_np = get_smolvla_action(smolvla, PIHEnv, device)
            else:
                q = PIHEnv.get_joint_state()[:6]
                action_np = np.concatenate([q, [200.0]], dtype=np.float32)

            # --- Contact detection ---
            if not contact_detected:
                contact_detected = check_contact(PIHEnv, ft_offset)
                if contact_detected:
                    contact_step = step
                    peg_tip = PIHEnv.env.get_p_site(PEG_TIP_SITE)
                    hole_entry = PIHEnv.env.get_p_site(HOLE_ENTRY_SITE)
                    print(f"  >> Insertion started at step {step} "
                          f"(depth={1000*(hole_entry[2]-peg_tip[2]):.1f}mm)")

            # ==========================================================
            # Residual RL (only after contact)
            # ==========================================================
            cartesian_residual = np.zeros(6, dtype=np.float32)
            joint_delta = np.zeros(6, dtype=np.float32)

            if contact_detected and cfg.max_residual_pos > 0:
                ft = PIHEnv.get_force_torque() - ft_offset
                ft_history.append(ft.copy())
                state_vec, ft_hist = build_rl_obs(PIHEnv, ft_offset, ft_history)

                # Store PREVIOUS transition (done=False for mid-episode steps)
                if prev_obs is not None:
                    peg_tip = PIHEnv.env.get_p_site(PEG_TIP_SITE)
                    hole_entry = PIHEnv.env.get_p_site(HOLE_ENTRY_SITE)
                    hole_bottom = PIHEnv.env.get_p_site(HOLE_BOTTOM_SITE)
                    reward, rinfo = reward_fn.compute(
                        ft, peg_tip, hole_entry, hole_bottom, False)
                    episode_reward += reward
                    last_rinfo = rinfo

                    trainer.store_transition(
                        state=prev_obs[0], ft_history=prev_obs[1],
                        action=prev_obs[2], reward=reward,
                        next_state=state_vec, next_ft_history=ft_hist,
                        done=False,  # mid-episode: not terminal
                    )
                    trainer.total_steps += 1
                    steps_after_contact += 1

                    # SAC update
                    if trainer.total_steps >= cfg.warmup_steps:
                        if trainer.total_steps % cfg.update_every == 0:
                            for _ in range(cfg.updates_per_step):
                                trainer.update()

                # Select residual action
                s_t = torch.FloatTensor(state_vec).unsqueeze(0).to(device)
                h_t = torch.FloatTensor(ft_hist).unsqueeze(0).to(device)
                residual_limits = get_cartesian_residual_limits(cfg)

                if trainer.total_steps < cfg.warmup_steps:
                    cartesian_residual = np.random.uniform(
                        low=-residual_limits,
                        high=residual_limits,
                    ).astype(np.float32)
                else:
                    with torch.no_grad():
                        cartesian_residual, _ = trainer.actor.sample(s_t, h_t)
                        cartesian_residual = cartesian_residual.squeeze(0).cpu().numpy()
                    cartesian_residual = np.clip(
                        cartesian_residual, -residual_limits, residual_limits
                    )

                joint_delta = cartesian_to_joint_delta(PIHEnv, cartesian_residual, cfg)

                prev_obs = (state_vec.copy(), ft_hist.copy(), cartesian_residual.copy())
            else:
                ft = PIHEnv.get_force_torque() - ft_offset
                ft_history.append(ft.copy())

            # --- Apply action: base joint target + IK-converted Cartesian residual ---
            final_action = action_np.copy()
            final_action[:6] += joint_delta

            # --- Step environment ---
            _ = PIHEnv.step(final_action)

            # --- Real-time monitoring ---
            ft_cal = PIHEnv.get_force_torque() - ft_offset
            lateral_f = np.linalg.norm(ft_cal[:2])
            axial_f = ft_cal[2]
            bending_t = np.linalg.norm(ft_cal[3:5])
            total_f = np.linalg.norm(ft_cal[:3])
            peg_z = PIHEnv.env.get_p_site(PEG_TIP_SITE)[2]

            PIHEnv.render(idx=episode)
            if contact_detected and prev_obs is not None:
                PIHEnv.env.viewer_text_overlay(
                    text1='Residual RL',
                    text2=f'R={episode_reward:.2f}  |cart|={np.linalg.norm(cartesian_residual):.5f}  '
                          f'|dq|={np.linalg.norm(joint_delta):.5f}  rl_steps={steps_after_contact}',
                )
            PIHEnv.env.viewer_text_overlay(
                text1='F/T Calibrated',
                text2=f'Fx={ft_cal[0]:.1f} Fy={ft_cal[1]:.1f} Fz={ft_cal[2]:.1f} '
                      f'|Flat|={lateral_f:.1f} |F|={total_f:.1f}N  '
                      f'Tx={ft_cal[3]:.3f} Ty={ft_cal[4]:.3f} |Tb|={bending_t:.3f}Nm',
            )
            PIHEnv.env.viewer_text_overlay(
                text1='Peg Z',
                text2=f'{peg_z:.4f}  contact={contact_detected}  '
                      f'excess_cnt={excessive_force_counter}',
            )

            if contact_detected and step % 5 == 0:
                r_str = f'R={episode_reward:>7.2f}' if prev_obs is not None else 'R=  n/a'
                print(f'  step={step:>4d} {r_str} '
                      f'|F|={total_f:>5.1f}N lat={lateral_f:>4.1f} Fz={axial_f:>5.1f} '
                      f'bend={bending_t:.3f}Nm peg_z={peg_z:.4f} '
                      f'|cart|={np.linalg.norm(cartesian_residual):.5f} '
                      f'|dq|={np.linalg.norm(joint_delta):.5f}')

            step += 1

    # Final save
    trainer.save(os.path.join(cfg.output_dir, 'final'))
    log_file.close()
    PIHEnv.env.close_viewer()
    if use_wandb:
        import wandb
        wandb.finish()
    print(f"\nDone. {episode} episodes, {trainer.total_steps} RL steps.")


if __name__ == '__main__':
    main()
