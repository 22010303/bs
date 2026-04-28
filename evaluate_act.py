"""
evaluate_act.py — Quantitative ACT deployment evaluation on PIHEnv2.

Produces:
  * per-step CSV of zero-calibrated F/T for every episode
  * per-episode CSV (success flag, step count at success, wall-time)
  * a small summary (success rate)
  * six line plots: Fx/Fy/Fz/Tx/Ty/Tz vs. step (one file each),
    overlaid across episodes with a mean line on top.

Based on 4.deployact.ipynb, extended with metrics + plotting.

Usage:
  python evaluate_act.py
  python evaluate_act.py --episodes 30 --max_steps 800 --no_viewer
"""

import argparse
import csv
import os
import time
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # headless-safe
import matplotlib.pyplot as plt
from PIL import Image
import torchvision

from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.configs.types import FeatureType

from mujoco_env.pih_env2 import PIHEnv2


# ----------------------------------------------------------------------
# Policy loading — mirrors 4.deployact.ipynb
# ----------------------------------------------------------------------
def load_act_policy(ckpt_path, dataset_root, repo_name, device,
                    chunk_size=30, n_action_steps=1, temporal_ensemble_coeff=0.9):
    print(f"[act] loading metadata from {dataset_root}")
    meta = LeRobotDatasetMetadata(repo_name, root=dataset_root)
    feats = dataset_to_policy_features(meta.features)
    out_feats = {k: f for k, f in feats.items() if f.type is FeatureType.ACTION}
    in_feats = {k: f for k, f in feats.items() if k not in out_feats}
    # 4.deployact.ipynb removes the wrist image input
    in_feats.pop('observation.wrist_image', None)

    cfg = ACTConfig(
        input_features=in_feats,
        output_features=out_feats,
        chunk_size=chunk_size,
        n_action_steps=n_action_steps,
        temporal_ensemble_coeff=temporal_ensemble_coeff,
    )
    resolve_delta_timestamps(cfg, meta)
    print(f"[act] loading weights from {ckpt_path}")
    policy = ACTPolicy.from_pretrained(
        ckpt_path, config=cfg, dataset_stats=meta.stats,
    )
    policy.to(device).eval()
    return policy


# ----------------------------------------------------------------------
# Episode rollout
# ----------------------------------------------------------------------
def run_episode(pih, policy, device, max_steps, seed, instruction,
                render=True):
    """Run one ACT episode; return per-step arrays + summary dict."""
    img_tf = torchvision.transforms.ToTensor()

    pih.reset(seed=seed)
    pih.set_instruction(instruction)
    policy.reset()

    # Zero-calibration: after pih.reset()'s 100-step settle, use the
    # current F/T reading as the gravity + bias offset.
    ft_offset = pih.get_force_torque().copy()

    step = 0
    records = {
        'step': [], 'fx': [], 'fy': [], 'fz': [],
        'tx': [], 'ty': [], 'tz': [],
    }
    success = False
    success_step = -1
    t0 = time.time()

    while True:
        pih.step_env()
        # Break early if viewer is closed by user
        if render and not pih.env.is_viewer_alive():
            break
        if not pih.env.loop_every(HZ=20):
            continue

        # Log calibrated F/T at this control tick
        ft_cal = (pih.get_force_torque() - ft_offset).astype(np.float32)
        records['step'].append(step)
        records['fx'].append(ft_cal[0]); records['fy'].append(ft_cal[1]); records['fz'].append(ft_cal[2])
        records['tx'].append(ft_cal[3]); records['ty'].append(ft_cal[4]); records['tz'].append(ft_cal[5])

        if pih.check_success():
            success = True
            success_step = step
            break
        if step >= max_steps:
            break

        # Observation
        state = pih.get_joint_state()[:6]
        agent_img, _ = pih.grab_image()
        img = img_tf(Image.fromarray(agent_img).resize((256, 256)))
        data = {
            'observation.state': torch.tensor([state]).to(device),
            'observation.image': img.unsqueeze(0).to(device),
            'task': [pih.instruction],
            'timestamp': torch.tensor([step / 20.0]).to(device),
        }

        with torch.no_grad():
            action = policy.select_action(data)
        action = action[0].cpu().numpy()
        pih.step(action)
        if render:
            pih.render(idx=seed)
        step += 1

    elapsed = time.time() - t0
    summary = {
        'seed': seed,
        'success': int(success),
        'success_step': success_step,
        'steps_taken': step,
        'elapsed_s': elapsed,
    }
    # Convert to arrays
    for k in records:
        records[k] = np.asarray(records[k], dtype=np.float32)
    return records, summary


# ----------------------------------------------------------------------
# CSV writers
# ----------------------------------------------------------------------
def write_perstep_csv(path, all_records):
    """all_records: list of (episode_idx, records_dict)."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['episode', 'step', 'fx', 'fy', 'fz', 'tx', 'ty', 'tz'])
        for ep_idx, rec in all_records:
            for i in range(len(rec['step'])):
                w.writerow([ep_idx, int(rec['step'][i]),
                            f"{rec['fx'][i]:.6f}", f"{rec['fy'][i]:.6f}", f"{rec['fz'][i]:.6f}",
                            f"{rec['tx'][i]:.6f}", f"{rec['ty'][i]:.6f}", f"{rec['tz'][i]:.6f}"])
    print(f"[csv] per-step -> {path}")


def write_summary_csv(path, summaries, success_rate):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['episode', 'seed', 'success', 'success_step',
                    'steps_taken', 'elapsed_s'])
        for i, s in enumerate(summaries):
            w.writerow([i, s['seed'], s['success'], s['success_step'],
                        s['steps_taken'], f"{s['elapsed_s']:.3f}"])
        w.writerow([])
        n = len(summaries)
        succ = sum(s['success'] for s in summaries)
        w.writerow(['success_rate', f"{success_rate:.4f}",
                    f"{succ}/{n}"])
        if succ > 0:
            avg_steps = np.mean([s['success_step'] for s in summaries if s['success']])
            avg_time  = np.mean([s['elapsed_s'] for s in summaries if s['success']])
            w.writerow(['mean_success_step', f"{avg_steps:.2f}"])
            w.writerow(['mean_success_time_s', f"{avg_time:.2f}"])
    print(f"[csv] summary  -> {path}")


# ----------------------------------------------------------------------
# Plotting: 6 files, one per axis
# ----------------------------------------------------------------------
def plot_per_axis(all_records, out_dir):
    """Overlay all episodes; bold line = mean on common step grid."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    channels = [
        ('fx', 'Fx [N]',   'force_Fx_vs_step.png'),
        ('fy', 'Fy [N]',   'force_Fy_vs_step.png'),
        ('fz', 'Fz [N]',   'force_Fz_vs_step.png'),
        ('tx', 'Tx [Nm]',  'torque_Tx_vs_step.png'),
        ('ty', 'Ty [Nm]',  'torque_Ty_vs_step.png'),
        ('tz', 'Tz [Nm]',  'torque_Tz_vs_step.png'),
    ]
    # Common step grid = longest episode
    max_len = max(len(rec['step']) for _, rec in all_records) if all_records else 0

    for key, ylabel, fname in channels:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        stacked = np.full((len(all_records), max_len), np.nan, dtype=np.float32)
        for row, (ep, rec) in enumerate(all_records):
            n = len(rec[key])
            stacked[row, :n] = rec[key]
            ax.plot(rec['step'], rec[key], alpha=0.35, linewidth=0.9,
                    label=f'ep{ep}' if len(all_records) <= 10 else None)
        # Mean across episodes at each step (ignoring NaN)
        if max_len > 0:
            mean = np.nanmean(stacked, axis=0)
            x = np.arange(max_len)
            ax.plot(x, mean, color='black', linewidth=2.0, label='mean')

        ax.set_xlabel('step (20 Hz)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'ACT deployment: {ylabel} vs step (zero-calibrated)')
        ax.grid(True, alpha=0.3)
        ax.axhline(0.0, color='gray', linewidth=0.5)
        if len(all_records) <= 10:
            ax.legend(loc='best', fontsize=8, ncol=2)
        else:
            ax.legend(['mean'], loc='best')
        fig.tight_layout()
        out_path = out_dir / fname
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
        print(f"[plot] {ylabel:8s} -> {out_path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description='ACT policy deployment evaluator')
    p.add_argument('--act_pretrained', type=str, default='./ckpt/act_y')
    p.add_argument('--dataset_root',   type=str, default='./demo_data_pih')
    p.add_argument('--repo_name',      type=str, default='pih')
    p.add_argument('--xml_path',       type=str, default='./asset/pih.xml')
    p.add_argument('--device',         type=str, default='cuda')
    p.add_argument('--episodes',       type=int, default=20)
    p.add_argument('--max_steps',      type=int, default=650)
    p.add_argument('--base_seed',      type=int, default=0)
    p.add_argument('--instruction',    type=str,
                   default='Insert the round peg into the blue round hole.')
    p.add_argument('--chunk_size',     type=int, default=30)
    p.add_argument('--n_action_steps', type=int, default=1)
    p.add_argument('--temporal_ensemble_coeff', type=float, default=0.9)
    p.add_argument('--output_dir',     type=str, default='./eval_results/act')
    p.add_argument('--no_viewer',      action='store_true')
    return p.parse_args()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main():
    args = parse_args()
    device = torch.device(args.device)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Policy
    policy = load_act_policy(
        args.act_pretrained, args.dataset_root, args.repo_name, str(device),
        chunk_size=args.chunk_size,
        n_action_steps=args.n_action_steps,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
    )

    # Env
    print(f"[env] creating PIHEnv2 from {args.xml_path}")
    pih = PIHEnv2(args.xml_path, action_type='joint_angle')
    pih.set_instruction(args.instruction)

    render = not args.no_viewer
    summaries = []
    all_records = []

    t_all = time.time()
    for ep in range(args.episodes):
        seed = args.base_seed + ep
        print(f"\n=== Episode {ep+1}/{args.episodes}  (seed={seed}) ===")
        rec, summ = run_episode(
            pih, policy, str(device),
            max_steps=args.max_steps, seed=seed,
            instruction=args.instruction, render=render,
        )
        tag = 'SUCCESS' if summ['success'] else 'FAIL'
        print(f"  [{tag}] success_step={summ['success_step']} "
              f"steps={summ['steps_taken']} time={summ['elapsed_s']:.2f}s")
        summaries.append(summ)
        all_records.append((ep, rec))

        if render and not pih.env.is_viewer_alive():
            print("  viewer closed; stopping early")
            break

    # Stats
    n = len(summaries)
    succ = sum(s['success'] for s in summaries)
    success_rate = succ / n if n > 0 else 0.0
    print(f"\n=== Summary ===")
    print(f"  episodes evaluated : {n}")
    print(f"  success rate       : {success_rate*100:.1f}%  ({succ}/{n})")
    if succ > 0:
        sst = [s['success_step'] for s in summaries if s['success']]
        sts = [s['elapsed_s']    for s in summaries if s['success']]
        print(f"  mean success step  : {np.mean(sst):.2f}")
        print(f"  mean success time  : {np.mean(sts):.2f}s")
    print(f"  total wall time    : {time.time()-t_all:.1f}s")

    # Write outputs
    write_perstep_csv(out_dir / 'perstep_force_torque.csv', all_records)
    write_summary_csv(out_dir / 'episode_summary.csv', summaries, success_rate)
    plot_per_axis(all_records, out_dir)

    pih.env.close_viewer()
    print(f"\n[done] results saved to {out_dir.resolve()}")


if __name__ == '__main__':
    main()
