"""
create_snapshot.py — Run SmolVLA to step N and save a complete simulator snapshot.

Use this one-shot before residual-RL training.  A snapshot captures
MuJoCo qpos/qvel/act/ctrl/applied-forces/sensordata, PIHEnv2 internal
state (target joint command, EEF pose reference, active peg and
instruction, gripper state, obj_init_pose) and the F/T zero calibration
taken at the home pose.  RL episodes can then be bootstrapped from the
snapshot instead of replaying the first 315 SmolVLA steps each time.

Usage
-----
  python create_snapshot.py
  python create_snapshot.py --snapshot_step 315 \
        --out ./ckpt/residual_rl/snapshot_step315.pkl
  python create_snapshot.py --no_viewer   # headless

Notes
-----
* F/T offset is captured at the home pose (no contact → best gravity
  compensation).  We keep that offset and reuse it after the snapshot
  is restored.
* The control loop is a faithful copy of smolvla1.py — same 20 Hz
  inference cadence, same image pre-processing.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from mujoco_env.pih_env2 import PIHEnv2
from residual_rl.snapshot import capture_snapshot, save_snapshot


IMG_TRANSFORM = transforms.Compose([transforms.ToTensor()])


def load_smolvla(pretrained, dataset_root, repo_name, device,
                 chunk_size=5, n_action_steps=5):
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.common.datasets.utils import dataset_to_policy_features
    from lerobot.common.datasets.factory import resolve_delta_timestamps
    from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.configs.types import FeatureType

    print(f"[smolvla] loading from {pretrained}")
    meta = LeRobotDatasetMetadata(repo_name, root=dataset_root)
    feats = dataset_to_policy_features(meta.features)
    out_feats = {k: f for k, f in feats.items() if f.type is FeatureType.ACTION}
    in_feats  = {k: f for k, f in feats.items() if k not in out_feats}

    cfg = SmolVLAConfig(
        input_features=in_feats, output_features=out_feats,
        chunk_size=chunk_size, n_action_steps=n_action_steps,
    )
    resolve_delta_timestamps(cfg, meta)

    policy = SmolVLAPolicy.from_pretrained(
        pretrained, config=cfg, dataset_stats=meta.stats,
    )
    policy.to(device).eval()
    for p in policy.parameters():
        p.requires_grad = False
    print(f"[smolvla] loaded ({sum(p.numel() for p in policy.parameters())/1e6:.1f}M params)")
    return policy


def smolvla_action(policy, pih_env, device):
    state = pih_env.get_joint_state()[:6]
    agent_img_raw, wrist_img_raw = pih_env.grab_image()
    agent_img = IMG_TRANSFORM(Image.fromarray(agent_img_raw).resize((256, 256)))
    wrist_img = IMG_TRANSFORM(Image.fromarray(wrist_img_raw).resize((256, 256)))
    data = {
        'observation.state': torch.from_numpy(
            np.array([state], dtype=np.float32)).to(device),
        'observation.image': agent_img.unsqueeze(0).to(device),
        'observation.wrist_image': wrist_img.unsqueeze(0).to(device),
        'task': [pih_env.instruction],
    }
    with torch.no_grad():
        act = policy.select_action(data)
    act_np = act[0, :7].cpu().numpy()
    act_np[6] = 200.0
    return act_np


def parse_args():
    p = argparse.ArgumentParser(description='Create SmolVLA snapshot for residual RL')
    p.add_argument('--xml_path', type=str, default='./asset/pih.xml')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--snapshot_step', type=int, default=315,
                   help='RL-loop step at which to capture the snapshot')
    p.add_argument('--out', type=str,
                   default='./ckpt/residual_rl/snapshot_step315.pkl')
    p.add_argument('--instruction', type=str,
                   default='Insert the round peg into the blue round hole.')
    p.add_argument('--smolvla_pretrained', type=str,
                   default='./ckpt/smolvla_pih1/checkpoints/020000/pretrained_model')
    p.add_argument('--smolvla_dataset_root', type=str, default='./demo_data_pih')
    p.add_argument('--smolvla_repo_name', type=str, default='ur5e_pih_language')
    p.add_argument('--chunk_size', type=int, default=5)
    p.add_argument('--n_action_steps', type=int, default=5)
    p.add_argument('--no_viewer', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # ------------------------------------------------------------------
    # 1. Environment
    # ------------------------------------------------------------------
    print(f"[env] creating PIHEnv2 from {args.xml_path}")
    pih = PIHEnv2(
        xml_path=args.xml_path,
        action_type='joint_angle',
        state_type='joint_angle',
    )
    pih.set_instruction(args.instruction)

    # F/T offset at home — gravity + sensor bias, used throughout
    ft_offset = pih.get_force_torque().copy()
    print(f"[ft] home offset = {ft_offset}")

    # ------------------------------------------------------------------
    # 2. SmolVLA
    # ------------------------------------------------------------------
    policy = load_smolvla(
        args.smolvla_pretrained, args.smolvla_dataset_root,
        args.smolvla_repo_name, str(device),
        args.chunk_size, args.n_action_steps,
    )

    # ------------------------------------------------------------------
    # 3. Run base policy until snapshot_step
    # ------------------------------------------------------------------
    step = 0
    snapshot_taken = False
    print(f"\n[run] driving SmolVLA to step {args.snapshot_step}")
    print(f"[run] instruction: {pih.instruction}")
    print(f"[run] active peg:  {pih.active_peg}")

    while True:
        pih.step_env()

        # Headless-safe: don't depend on viewer being alive
        if not args.no_viewer and not pih.env.is_viewer_alive():
            print("[run] viewer closed before reaching snapshot step — abort")
            return

        if not pih.env.loop_every(HZ=20):
            continue

        if step >= args.snapshot_step and not snapshot_taken:
            extra = {
                'ee_pose':       pih.get_ee_pose(),
                'joint_state':   pih.get_joint_state(),
                'peg_tip_pos':   pih.env.get_p_site('peg_tip_site'),
                'hole_entry':    pih.env.get_p_site('hole_entry_site'),
                'hole_bottom':   pih.env.get_p_site('hole_bottom_site'),
                'instruction':   pih.instruction,
            }
            snap = capture_snapshot(pih, ft_offset, base_step=step, extra=extra)
            save_snapshot(snap, args.out)

            ft_raw = snap['ft']['raw']
            ft_cal = snap['ft']['calibrated']
            ee = extra['ee_pose']
            pt = extra['peg_tip_pos']
            he = extra['hole_entry']
            print("\n=== SNAPSHOT CAPTURED ===")
            print(f"  step           = {step}")
            print(f"  ee_pose        = {ee}")
            print(f"  peg_tip        = {pt}")
            print(f"  hole_entry     = {he}")
            print(f"  peg_tip - he   = {pt - he}  (xy={np.linalg.norm(pt[:2]-he[:2])*1000:.1f}mm, "
                  f"dz={(pt[2]-he[2])*1000:.1f}mm)")
            print(f"  ft raw         = {ft_raw}")
            print(f"  ft calibrated  = {ft_cal}")
            snapshot_taken = True
            break

        # Base action
        action = smolvla_action(policy, pih, str(device))
        pih.step(action)
        if not args.no_viewer:
            pih.render(idx=0)
        if step % 20 == 0:
            print(f"  step={step:>4d}  ee={pih.get_ee_pose()}")
        step += 1

    print("\n[done] snapshot saved. You can now train with train_residual_rl_snapshot.py.")
    pih.env.close_viewer()


if __name__ == '__main__':
    main()
