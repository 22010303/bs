"""
deploy_residual_rl_ppo.py — Deploy SmolVLA + trained residual PPO policy.
"""

import argparse
import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from residual_rl_ppo.config import ResidualPPOConfig
from residual_rl_ppo.ppo_trainer import PPOTrainer
from residual_rl.pih_gym_wrapper import PIHResidualEnv
from mujoco_env.pih_env2 import PIHEnv2


IMG_TRANSFORM = transforms.Compose([transforms.ToTensor()])
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
        cfg.smolvla_pretrained, config=smolvla_cfg, dataset_stats=dataset_metadata.stats
    )
    policy.to(cfg.smolvla_device)
    policy.eval()
    for p in policy.parameters():
        p.requires_grad = False
    return policy


def cartesian_to_joint_delta(pih_env, cart_residual, cfg):
    _, _, J_full = pih_env.env.get_J_body(EEF_BODY)
    jac_idxs = pih_env.env.get_idxs_jac(ARM_JOINTS)
    J_arm = J_full[:, jac_idxs]
    JJT = J_arm @ J_arm.T + cfg.ik_damping_eps * np.eye(6)
    J_pinv = J_arm.T @ np.linalg.solve(JJT, np.eye(6))
    joint_delta = J_pinv @ cart_residual
    return np.clip(joint_delta, -cfg.max_joint_delta, cfg.max_joint_delta).astype(np.float32)


def parse_args():
    p = argparse.ArgumentParser(description='Deploy SmolVLA + Residual PPO')
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
    ckpt_path = os.path.join(args.rl_checkpoint, 'ppo_checkpoint.pt')
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ckpt.get('config', ResidualPPOConfig())
    cfg.smolvla_pretrained = args.smolvla_pretrained
    cfg.smolvla_dataset_root = args.smolvla_dataset_root
    cfg.smolvla_repo_name = args.smolvla_repo_name
    cfg.smolvla_device = args.device

    trainer = PPOTrainer(config=cfg, device=str(device))
    trainer.load(args.rl_checkpoint)
    smolvla = None if args.no_smolvla else load_smolvla(cfg)

    pih_env = PIHEnv2(xml_path=args.xml_path, action_type='joint_angle', state_type='joint_angle')
    env = PIHResidualEnv(pih_env=pih_env, config=cfg, render_mode='human')

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

    step = 0
    episode = 0
    obs, info = env.reset()
    if smolvla is not None:
        smolvla.reset()

    while pih_env.env.is_viewer_alive():
        pih_env.step_env()
        if pih_env.env.loop_every(HZ=20):
            success = info.get('success', False)
            if success or step >= args.max_steps:
                if smolvla is not None:
                    smolvla.reset()
                obs, info = env.reset()
                step = 0
                episode += 1
                continue

            base_action = get_smolvla_action()
            contact = env.contact_detected
            if contact:
                cart_residual = trainer.deterministic_action(obs['state'], obs['ft_history'])
                joint_delta = cartesian_to_joint_delta(pih_env, cart_residual, cfg)
                combined = base_action[:6] + joint_delta
                action_7d = np.concatenate([combined, [200.0]], dtype=np.float32)
                cart_norm = np.linalg.norm(cart_residual)
                dq_norm = np.linalg.norm(joint_delta)
            else:
                action_7d = base_action.copy()
                cart_norm = 0.0
                dq_norm = 0.0

            obs, reward, terminated, truncated, info = env.step(action_7d)
            ft = env.get_ft_calibrated()
            phase = "PPO_ACTIVE" if contact else "SMOLVLA_ONLY"
            print(f"Step {step:>4d} | {phase:>12s} | "
                  f"R={reward:>7.3f} | |cart|={cart_norm:.4f} | |dq|={dq_norm:.4f} | "
                  f"F=[{ft[0]:.1f},{ft[1]:.1f},{ft[2]:.1f}]N")
            env.render()
            step += 1

    env.close()


if __name__ == '__main__':
    main()
