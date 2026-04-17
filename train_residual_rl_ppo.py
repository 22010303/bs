"""
train_residual_rl_ppo.py — Train PPO residual RL on top of frozen SmolVLA.
"""

import argparse
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from collections import deque

from residual_rl.reward import HierarchicalReward
from residual_rl_ppo.config import ResidualPPOConfig
from residual_rl_ppo.ppo_trainer import PPOTrainer
from mujoco_env.pih_env2 import PIHEnv2


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
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.common.datasets.utils import dataset_to_policy_features
    from lerobot.common.datasets.factory import resolve_delta_timestamps
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
    policy.to(cfg.smolvla_device)
    policy.eval()
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
    peg_below = peg_tip[2] < hole_entry[2]
    return peg_below and xy_dist < 0.03


def build_rl_obs(pih_env, ft_offset, ft_history):
    ft = pih_env.get_force_torque() - ft_offset
    ee = pih_env.get_ee_pose()
    peg_tip = pih_env.env.get_p_site(PEG_TIP_SITE)
    hole_entry = pih_env.env.get_p_site(HOLE_ENTRY_SITE)
    relative = (hole_entry - peg_tip).astype(np.float32)
    state = np.concatenate([ft, ee, relative], dtype=np.float32)
    ft_hist = np.array(list(ft_history), dtype=np.float32)
    return state, ft_hist


def cartesian_to_joint_delta(pih_env, cart_residual, cfg):
    _, _, J_full = pih_env.env.get_J_body(EEF_BODY)
    jac_idxs = pih_env.env.get_idxs_jac(ARM_JOINTS)
    J_arm = J_full[:, jac_idxs]
    JJT = J_arm @ J_arm.T + cfg.ik_damping_eps * np.eye(6)
    J_pinv = J_arm.T @ np.linalg.solve(JJT, np.eye(6))
    joint_delta = J_pinv @ cart_residual
    return np.clip(joint_delta, -cfg.max_joint_delta, cfg.max_joint_delta).astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--xml_path', type=str, default='./asset/pih.xml')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--total_steps', type=int, default=100_000)
    p.add_argument('--output_dir', type=str, default='./ckpt/residual_rl_ppo')
    p.add_argument('--no_smolvla', action='store_true')
    p.add_argument('--smolvla_pretrained', type=str,
                   default='./ckpt/smolvla_pih1/checkpoints/020000/pretrained_model')
    p.add_argument('--smolvla_dataset_root', type=str, default='./demo_data_pih')
    p.add_argument('--smolvla_repo_name', type=str, default='ur5e_pih_language')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--max_steps', type=int, default=650)
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb_project', type=str, default='residual-rl-ppo-pih')
    p.add_argument('--wandb_name', type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    cfg = ResidualPPOConfig(
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

    use_wandb = args.wandb
    if use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f'residual_rl_ppo_seed{cfg.seed}',
            config=vars(cfg),
        )

    smolvla = None if args.no_smolvla else load_smolvla(cfg)
    trainer = PPOTrainer(config=cfg, device=str(device), use_wandb=use_wandb)
    if args.resume:
        trainer.load(args.resume)

    env = PIHEnv2(xml_path=cfg.xml_path, action_type='joint_angle', state_type='joint_angle')
    env.set_instruction('Insert the round peg into the blue round hole.')
    reward_fn = HierarchicalReward(cfg)

    ft_offset = env.get_force_torque().copy()
    reward_fn.reset(initial_ft=env.get_force_torque() - ft_offset)
    ft_history = deque(maxlen=cfg.force_history_len)
    ft_init = env.get_force_torque() - ft_offset
    for _ in range(cfg.force_history_len):
        ft_history.append(ft_init.copy())

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    log_file = open(os.path.join(cfg.log_dir, 'train_log.csv'), 'w')
    log_file.write('step,episode,reward,length,success,contact_step,ppo_updates\n')

    step = 0
    episode = 0
    episode_reward = 0.0
    contact_detected = False
    contact_step = -1
    ppo_updates = 0
    excessive_force_counter = 0

    last_state_vec = None
    last_ft_hist = None
    last_done = False

    def reset_episode():
        nonlocal step, episode, episode_reward, contact_detected, contact_step
        nonlocal excessive_force_counter, ft_offset, last_state_vec, last_ft_hist, last_done
        if smolvla is not None:
            smolvla.reset()
        env.reset()
        env.set_instruction('Insert the round peg into the blue round hole.')
        ft_offset = env.get_force_torque().copy()
        reward_fn.reset(initial_ft=env.get_force_torque() - ft_offset)
        ft_history.clear()
        ft_init_local = env.get_force_torque() - ft_offset
        for _ in range(cfg.force_history_len):
            ft_history.append(ft_init_local.copy())
        step = 0
        episode += 1
        episode_reward = 0.0
        contact_detected = False
        contact_step = -1
        excessive_force_counter = 0
        last_state_vec = None
        last_ft_hist = None
        last_done = False

    while env.env.is_viewer_alive() and trainer.total_steps < cfg.total_steps:
        env.step_env()

        if env.env.loop_every(HZ=20):
            success = env.check_success()
            timeout = (step >= args.max_steps)
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

            cartesian_residual = np.zeros(6, dtype=np.float32)
            joint_delta = np.zeros(6, dtype=np.float32)
            log_prob = 0.0
            value = 0.0

            ft = env.get_force_torque() - ft_offset
            ft_history.append(ft.copy())
            state_vec, ft_hist = build_rl_obs(env, ft_offset, ft_history)

            if contact_detected:
                cartesian_residual, log_prob, value = trainer.act(state_vec, ft_hist)
                joint_delta = cartesian_to_joint_delta(env, cartesian_residual, cfg)

            final_action = base_action.copy()
            final_action[:6] += joint_delta
            env.step(final_action)

            peg_tip = env.env.get_p_site(PEG_TIP_SITE)
            hole_entry = env.env.get_p_site(HOLE_ENTRY_SITE)
            hole_bottom = env.env.get_p_site(HOLE_BOTTOM_SITE)
            reward, _ = reward_fn.compute(ft, peg_tip, hole_entry, hole_bottom, success)
            if done and not success:
                reward -= cfg.penalty_early_stop
            episode_reward += reward

            if contact_detected:
                trainer.store_transition(
                    state=state_vec,
                    ft_history=ft_hist,
                    action=cartesian_residual,
                    log_prob=log_prob,
                    reward=reward * cfg.reward_scale,
                    done=done,
                    value=value,
                )
                trainer.total_steps += 1
                last_state_vec = state_vec
                last_ft_hist = ft_hist
                last_done = done

            if trainer.buffer.full:
                trainer.finish_rollout(last_state_vec, last_ft_hist, last_done)
                trainer.update()
                ppo_updates += 1

            ft_cal = env.get_force_torque() - ft_offset
            env.render(idx=episode)
            env.env.viewer_text_overlay(
                text1='Residual PPO',
                text2=f'R={episode_reward:.2f} |cart|={np.linalg.norm(cartesian_residual):.5f} '
                      f'|dq|={np.linalg.norm(joint_delta):.5f}',
            )
            env.env.viewer_text_overlay(
                text1='F/T Calibrated',
                text2=f'Fx={ft_cal[0]:.1f} Fy={ft_cal[1]:.1f} Fz={ft_cal[2]:.1f}',
            )

            if done:
                if trainer.buffer.size > 0:
                    trainer.finish_rollout(last_state_vec, last_ft_hist, True)
                    trainer.update()
                    ppo_updates += 1
                print(f"[PPO {trainer.total_steps:>7d} | Ep {episode:>4d}] "
                      f"R={episode_reward:>8.2f} L={step:>4d} success={int(success)}")
                log_file.write(
                    f"{trainer.total_steps},{episode},{episode_reward:.4f},{step},"
                    f"{int(success)},{contact_step},{ppo_updates}\n"
                )
                log_file.flush()
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

