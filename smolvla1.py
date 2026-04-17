"""
smolvla.py — Deploy SmolVLA base model (zero-shot, no fine-tuning) on the
UR5e peg-in-hole scene (pih.xml + PIHEnv2).

Usage:
python smolvla1.py [--dataset_root ./demo_data_pih] [--device cuda]

Requirements (install before running):
pip install transformers==4.50.3 num2words accelerate "safetensors>=0.4.3"
"""

import argparse
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.types import FeatureType

from mujoco_env.pih_env2 import PIHEnv2


def get_default_transform():
    return transforms.Compose([
        transforms.ToTensor(),  # [0-255] -> [0.0-1.0], H×W×C -> C×H×W
    ])


def parse_args():
    p = argparse.ArgumentParser(description='SmolVLA zero-shot deploy on PIH')
    p.add_argument('--dataset_root', type=str, default='./demo_data_pih',
                   help='Path to the PIH dataset (for metadata/stats only)')
    p.add_argument('--repo_name', type=str, default='ur5e_pih_language',
                   help='Dataset repo_id used when creating the dataset')
    p.add_argument('--xml_path', type=str, default='./asset/pih.xml')
    p.add_argument('--device', type=str, default='cuda')
    # p.add_argument('--pretrained', type=str, default='lerobot/smolvla_base',
    #                help='Pretrained model path (HuggingFace Hub or local)')
    p.add_argument('--pretrained', type=str, default='./ckpt/smolvla_pih1/checkpoints/020000/pretrained_model',
                   help='Local directory containing the trained model (./ckpt/smolvla_pih1/checkpoints/020000/pretrained_model)')
    p.add_argument('--chunk_size', type=int, default=5)
    p.add_argument('--n_action_steps', type=int, default=5)
    p.add_argument('--max_steps', type=int, default=750,
                   help='Max steps per episode before forced reset')
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device

    # ------------------------------------------------------------------
    # 1. Load dataset metadata (for feature schema + normalization stats)
    # ------------------------------------------------------------------
    print(f"Loading dataset metadata from {args.dataset_root} ...")
    dataset_metadata = LeRobotDatasetMetadata(
        args.repo_name, root=args.dataset_root
    )

    features = dataset_to_policy_features(dataset_metadata.features)
    output_features = {
        key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
    }
    input_features = {
        key: ft for key, ft in features.items() if key not in output_features
    }

    print(f"Input features: {list(input_features.keys())}")
    print(f"Output features: {list(output_features.keys())}")

    # ------------------------------------------------------------------
    # 2. Configure SmolVLA policy
    # ------------------------------------------------------------------
    cfg = SmolVLAConfig(
        input_features=input_features,
        output_features=output_features,
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
    )
    delta_timestamps = resolve_delta_timestamps(cfg, dataset_metadata)
    print(f"Delta timestamps: {delta_timestamps}")

    # ------------------------------------------------------------------
    # 3. Load pretrained SmolVLA model
    # ------------------------------------------------------------------
    print(f"Loading pretrained model from {args.pretrained} ...")
    policy = SmolVLAPolicy.from_pretrained(
        args.pretrained,
        config=cfg,
        dataset_stats=dataset_metadata.stats,
    )
    policy.to(device)
    policy.eval()
    print(f"Model loaded on {device}. Parameters: "
          f"{sum(p.numel() for p in policy.parameters()) / 1e6:.1f}M")

    # ------------------------------------------------------------------
    # 4. Create simulation environment
    # ------------------------------------------------------------------
    print("Creating PIH environment ...")
    PIHEnv = PIHEnv2(
        xml_path=args.xml_path,
        action_type='joint_angle',
        state_type='joint_angle',
    )

    IMG_TRANSFORM = get_default_transform()

    # ------------------------------------------------------------------
    # 5. Rollout loop
    # ------------------------------------------------------------------
    step = 0
    episode = 0

    print("\n=== Starting rollout ===")
    print(f"Instruction: {PIHEnv.instruction}")
    print(f"Active peg: {PIHEnv.active_peg}")
    print("Press Z to reset, Ctrl+C or close window to quit.\n")

    while PIHEnv.env.is_viewer_alive():
        PIHEnv.step_env()

        if PIHEnv.env.loop_every(HZ=20):
            # --- Check success ---
            success = PIHEnv.check_success()
            if success:
                print(f"SUCCESS! Episode {episode} completed in {step} steps.")
                policy.reset()
                PIHEnv.reset()
                step = 0
                episode += 1
                print(f"\n--- Episode {episode} ---")
                print(f"Instruction: {PIHEnv.instruction}")
                print(f"Active peg: {PIHEnv.active_peg}")
                continue

            # --- Forced reset on timeout ---
            if step >= args.max_steps:
                print(f"Timeout at {step} steps. Resetting.")
                policy.reset()
                PIHEnv.reset()
                step = 0
                episode += 1
                print(f"\n--- Episode {episode} ---")
                print(f"Instruction: {PIHEnv.instruction}")
                print(f"Active peg: {PIHEnv.active_peg}")
                continue

            # --- Get observations ---
            state = PIHEnv.get_joint_state()[:6]  # 6 joint angles
            # state = PIHEnv.get_ee_pose()
            agent_image, wrist_image = PIHEnv.grab_image()

            # Preprocess images: resize + to tensor [0,1]
            agent_img = IMG_TRANSFORM(
                Image.fromarray(agent_image).resize((256, 256))
            )
            wrist_img = IMG_TRANSFORM(
                Image.fromarray(wrist_image).resize((256, 256))
            )

            # Build observation dict for policy
            data = {
                'observation.state': torch.from_numpy(
                    np.array([state], dtype=np.float32)
                ).to(device),
                'observation.image': agent_img.unsqueeze(0).to(device),
                'observation.wrist_image': wrist_img.unsqueeze(0).to(device),
                'task': [PIHEnv.instruction],
            }

            # --- Select action ---
            with torch.no_grad():
                action = policy.select_action(data)

            # action shape: (n_action_steps, action_dim)
            # Take first action, first 7 dims = 6 joints + 1 gripper
            action_np = action[0, :7].cpu().numpy()
            # ====== 新增：打印 action 到控制台 ======
            print(f"Step {step}: action = {action_np}")

            # --- Step environment ---
            _ = PIHEnv.step(action_np)
            PIHEnv.render(idx=episode)
            step += 1

    PIHEnv.env.close_viewer()
    print(f"\nDone. Ran {episode} episodes.")


if __name__ == '__main__':
    main()