"""
deploy_residual_rl.py — Deploy SmolVLA + trained residual RL policy.

SmolVLA runs throughout; residual RL activates after contact detection.

Usage:
  python deploy_residual_rl.py --rl_checkpoint ./ckpt/residual_rl/best
  python deploy_residual_rl.py --rl_checkpoint ./ckpt/residual_rl/best --no_smolvla
"""

import argparse
import os
import numpy as np
import torch
from collections import deque
from PIL import Image
from torchvision import transforms

from residual_rl.config import ResidualRLConfig
from residual_rl.residual_policy import ResidualActor
from residual_rl.reward import HierarchicalReward
from residual_rl.pih_gym_wrapper import PIHResidualEnv
from mujoco_env.pih_env2 import PIHEnv2


IMG_TRANSFORM = transforms.Compose([transforms.ToTensor()])

ARM_JOINTS = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]
EEF_BODY = 'tool0_link'


def cartesian_to_joint_delta(pih_env, cart_residual, cfg):
    """Convert 6D Cartesian residual to arm joint delta via damped Jacobian pseudo-inverse."""
    _, _, J_full = pih_env.env.get_J_body(EEF_BODY)
    jac_idxs = pih_env.env.get_idxs_jac(ARM_JOINTS)
    J_arm = J_full[:, jac_idxs]

    eps = cfg.ik_damping_eps
    JJT = J_arm @ J_arm.T + eps * np.eye(6)
    J_pinv = J_arm.T @ np.linalg.solve(JJT, np.eye(6))
    joint_delta = J_pinv @ cart_residual
    joint_delta = np.clip(joint_delta, -cfg.max_joint_delta, cfg.max_joint_delta)
    return joint_delta.astype(np.float32)


def load_smolvla(cfg):
    """Load frozen SmolVLA policy."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.common.datasets.utils import dataset_to_policy_features
    from lerobot.common.datasets.factory import resolve_delta_timestamps
    from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.configs.types import FeatureType

    print(f"Loading SmolVLA from {cfg.smolvla_pretrained} ...")
    dataset_metadata = LeRobotDatasetMetadata(
        cfg.smolvla_repo_name, root=cfg.smolvla_dataset_root
    )
    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
    input_features = {k: f for k, f in features.items() if k not in output_features}

    smolvla_cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=cfg.smolvla_chunk_size,
        n_action_steps=cfg.smolvla_n_action_steps,
    )
    _ = resolve_delta_timestamps(smolvla_cfg, dataset_metadata)

    policy = SmolVLAPolicy.from_pretrained(
        cfg.smolvla_pretrained, config=smolvla_cfg,
        dataset_stats=dataset_metadata.stats,
    )
    policy.to(cfg.smolvla_device)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad = False
    print(f"SmolVLA loaded ({sum(p.numel() for p in policy.parameters())/1e6:.1f}M params)")
    return policy


def parse_args():
    p = argparse.ArgumentParser(description='Deploy SmolVLA + Residual RL')
    p.add_argument('--rl_checkpoint', type=str, required=True)
    p.add_argument('--xml_path', type=str, default='./asset/pih.xml')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--no_smolvla', action='store_true')
    p.add_argument('--smolvla_pretrained', type=str,
                   default='./ckpt/smolvla_pih1/checkpoints/020000/pretrained_model')
    p.add_argument('--smolvla_dataset_root', type=str, default='./demo_data_pih')
    p.add_argument('--smolvla_repo_name', type=str, default='ur5e_pih_language')
    p.add_argument('--max_steps', type=int, default=650)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Load config from checkpoint
    ckpt_path = os.path.join(args.rl_checkpoint, 'sac_checkpoint.pt')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', ResidualRLConfig())
    cfg.smolvla_pretrained = args.smolvla_pretrained
    cfg.smolvla_dataset_root = args.smolvla_dataset_root
    cfg.smolvla_repo_name = args.smolvla_repo_name
    cfg.smolvla_device = args.device

    # Load residual actor
    print(f"Loading residual RL actor from {args.rl_checkpoint} ...")
    actor = ResidualActor(cfg).to(device)
    actor.load_state_dict(ckpt['actor'])
    actor.eval()
    print(f"Residual actor loaded (step {ckpt.get('total_steps', '?')})")

    # Load SmolVLA
    smolvla = None
    if not args.no_smolvla:
        smolvla = load_smolvla(cfg)

    # Create environment
    print("Creating PIH environment ...")
    pih_env = PIHEnv2(
        xml_path=args.xml_path,
        action_type='joint_angle',
        state_type='joint_angle',
    )
    env = PIHResidualEnv(pih_env=pih_env, config=cfg, render_mode='human')

    # SmolVLA action function
    def get_smolvla_action():
        if smolvla is None:
            q = pih_env.get_joint_state()[:6]
            return np.concatenate([q, [200.0]], dtype=np.float32)
        dev = next(smolvla.parameters()).device
        agent_img, wrist_img = pih_env.grab_image()
        a = IMG_TRANSFORM(Image.fromarray(agent_img).resize((256, 256)))
        w = IMG_TRANSFORM(Image.fromarray(wrist_img).resize((256, 256)))
        state = pih_env.get_joint_state()[:6]
        data = {
            'observation.state': torch.from_numpy(np.array([state], dtype=np.float32)).to(dev),
            'observation.image': a.unsqueeze(0).to(dev),
            'observation.wrist_image': w.unsqueeze(0).to(dev),
            'task': [pih_env.instruction],
        }
        with torch.no_grad():
            act = smolvla.select_action(data)
        act_np = act[0, :7].cpu().numpy()
        act_np[6] = 200.0
        return act_np

    # Rollout
    step = 0
    episode = 0

    obs, info = env.reset()
    if smolvla is not None:
        smolvla.reset()

    print(f"\n=== Deploy: SmolVLA + Residual RL (contact-triggered) ===")
    print(f"Instruction: {pih_env.instruction}")
    print(f"Active peg: {pih_env.active_peg}")
    print(f"Max residual pos: {cfg.max_residual_pos} m")
    print(f"Max residual rot: {cfg.max_residual_rot} rad")
    print("Press Ctrl+C to quit.\n")

    while pih_env.env.is_viewer_alive():
        pih_env.step_env()

        if pih_env.env.loop_every(HZ=20):
            # Check success / timeout
            success = info.get('success', False)
            if success or step >= args.max_steps:
                tag = "SUCCESS" if success else "TIMEOUT"
                print(f"\n{tag}! Episode {episode}, {step} steps.")
                if smolvla is not None:
                    smolvla.reset()
                obs, info = env.reset()
                step = 0
                episode += 1
                print(f"--- Episode {episode} ---")
                print(f"Instruction: {pih_env.instruction}")
                continue

            # SmolVLA base action (always)
            base_action = get_smolvla_action()

            # Contact check
            contact = env.contact_detected

            # Residual (only after contact)
            if contact:
                state_t = torch.FloatTensor(obs['state']).unsqueeze(0).to(device)
                ft_hist_t = torch.FloatTensor(obs['ft_history']).unsqueeze(0).to(device)
                with torch.no_grad():
                    residual = actor.deterministic_action(state_t, ft_hist_t)
                    residual_np = residual.squeeze(0).cpu().numpy()
                joint_delta = cartesian_to_joint_delta(pih_env, residual_np, cfg)
                combined = base_action[:6] + joint_delta
                action_7d = np.concatenate([combined, [200.0]], dtype=np.float32)
                cart_norm = np.linalg.norm(residual_np)
                dq_norm = np.linalg.norm(joint_delta)
            else:
                action_7d = base_action.copy()
                residual_np = np.zeros(6)
                joint_delta = np.zeros(6)
                cart_norm = 0.0
                dq_norm = 0.0

            # Step
            obs, reward, terminated, truncated, info = env.step(action_7d)

            # Display
            ft = env.get_ft_calibrated()
            phase = "RL_ACTIVE" if contact else "SMOLVLA_ONLY"
            print(f"Step {step:>4d} | {phase:>12s} | "
                  f"R={reward:>7.3f} | |cart|={cart_norm:.4f} | |dq|={dq_norm:.4f} | "
                  f"F=[{ft[0]:.1f},{ft[1]:.1f},{ft[2]:.1f}]N")

            env.render()
            step += 1

    env.close()
    print(f"\nDone. {episode} episodes.")


if __name__ == '__main__':
    main()
