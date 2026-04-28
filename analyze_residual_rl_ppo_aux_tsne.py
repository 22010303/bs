"""
analyze_residual_rl_ppo_aux_tsne.py — Compare mean pooling vs CLS pooling
encoder features with t-SNE.

By default this script can compare:
  - two different checkpoints, or
  - the same checkpoint under two pooling modes if only one checkpoint is given.
"""

import argparse
import os
from collections import deque

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from mujoco_env.pih_env2 import PIHEnv2
from residual_rl.snapshot import load_snapshot, perturb_ee_pose, restore_snapshot
from residual_rl_ppo.aux_config import ResidualPPOAuxConfig
from residual_rl_ppo.aux_trainer import PPOAuxTrainer


PEG_TIP_SITE = 'peg_tip_site'
HOLE_ENTRY_SITE = 'hole_entry_site'
HOLE_BOTTOM_SITE = 'hole_bottom_site'
STATE_NAMES = ['smooth', 'jamming', 'bottoming']
STATE_COLORS = ['#4c78a8', '#e45756', '#54a24b']


def resolve_checkpoint_dir(checkpoint_arg):
    checkpoint_arg = os.path.abspath(checkpoint_arg)
    if os.path.isfile(checkpoint_arg):
        if os.path.basename(checkpoint_arg) != 'ppo_aux_checkpoint.pt':
            raise FileNotFoundError(f'Expected ppo_aux_checkpoint.pt, got {checkpoint_arg}')
        return os.path.dirname(checkpoint_arg)
    direct = os.path.join(checkpoint_arg, 'ppo_aux_checkpoint.pt')
    nested = os.path.join(checkpoint_arg, 'final', 'ppo_aux_checkpoint.pt')
    if os.path.isfile(direct):
        return checkpoint_arg
    if os.path.isfile(nested):
        return os.path.join(checkpoint_arg, 'final')
    raise FileNotFoundError(f'Could not resolve checkpoint from {checkpoint_arg}')


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


class DebouncedInsertionStateLabeler:
    SMOOTH = 0
    JAMMING = 1
    BOTTOMING = 2

    def __init__(self, cfg):
        self.cfg = cfg
        self.reset()

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
        if progress > self.cfg.bottom_progress_th and axial_force > self.cfg.bottom_axial_force_th and speed < self.cfg.low_speed_th:
            return self.BOTTOMING
        if progress > 0.0 and speed < self.cfg.low_speed_th and (
            lateral_force > self.cfg.jam_lateral_force_th or
            bending_torque > self.cfg.jam_torque_th or
            xy_error > self.cfg.jam_xy_error_th
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


def load_trainer(checkpoint_arg, device, force_cls_pooling=None):
    checkpoint_dir = resolve_checkpoint_dir(checkpoint_arg)
    ckpt = torch.load(os.path.join(checkpoint_dir, 'ppo_aux_checkpoint.pt'),
                      map_location=device, weights_only=False)
    cfg = ckpt.get('config', ResidualPPOAuxConfig())
    if force_cls_pooling is not None:
        cfg.use_cls_pooling = bool(force_cls_pooling)
    trainer = PPOAuxTrainer(config=cfg, device=str(device), use_wandb=False)
    trainer.load(checkpoint_dir)
    trainer.policy.cfg.use_cls_pooling = cfg.use_cls_pooling
    trainer.policy.encoder.ft_attention.cfg.use_cls_pooling = cfg.use_cls_pooling
    return trainer, cfg


def collect_samples(trainer, cfg, snapshot_path, xml_path, seed, samples_target, noise_pos, noise_rot_deg):
    snap = load_snapshot(snapshot_path)
    rng = np.random.default_rng(seed)
    dt = 1.0 / cfg.control_hz
    env = PIHEnv2(xml_path=xml_path, action_type='joint_angle', state_type='joint_angle')
    labeler = DebouncedInsertionStateLabeler(cfg)
    samples = []
    labels = []

    episode = 0
    while len(labels) < samples_target:
        ft_offset = restore_snapshot(env, snap)
        perturb_ee_pose(env, pos_noise=noise_pos, rot_noise_deg=noise_rot_deg, rng=rng)
        ft_history = deque(maxlen=cfg.force_history_len)
        ft_init = env.get_force_torque() - ft_offset
        for _ in range(cfg.force_history_len):
            ft_history.append(ft_init.copy())
        labeler.reset()
        prev_ee_pose = env.get_ee_pose().copy()
        contact_detected = False
        step = snap['meta']['base_step']

        while step < snap['meta']['base_step'] + cfg.max_episode_steps and len(labels) < samples_target:
            env.step_env()
            if not env.env.loop_every(HZ=20):
                continue

            if not contact_detected:
                contact_detected = check_contact(env)

            ft = env.get_force_torque() - ft_offset
            ee_pose = env.get_ee_pose()
            ee_vel = (ee_pose - prev_ee_pose) / dt
            peg_tip = env.env.get_p_site(PEG_TIP_SITE)
            hole_entry = env.env.get_p_site(HOLE_ENTRY_SITE)
            hole_bottom = env.env.get_p_site(HOLE_BOTTOM_SITE)
            ft_history.append(ft.copy())
            state_vec, ft_hist = build_rl_obs(ft, ee_pose, ee_vel, peg_tip, hole_entry, ft_history)

            if contact_detected:
                label = labeler.update(ft, ee_vel, peg_tip, hole_entry, hole_bottom)
                state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(trainer.device)
                hist_t = torch.FloatTensor(ft_hist).unsqueeze(0).to(trainer.device)
                state_t = trainer._normalize_state(state_t)
                with torch.no_grad():
                    analysis = trainer.policy.analyze(state_t, hist_t)
                # We save both the fused feature and the force-attention feature
                # so the script can compare 128D and 64D spaces separately.
                samples.append({
                    'fused_feat': analysis['feat'].squeeze(0).cpu().numpy(),
                    'ft_feat': analysis['ft_feat'].squeeze(0).cpu().numpy(),
                })
                labels.append(int(label))

            prev_ee_pose = ee_pose.copy()
            step += 1

        episode += 1
        if episode > 2000:
            break

    if env.env.is_viewer_alive():
        env.env.close_viewer()
    return samples, np.array(labels, dtype=np.int32)


def run_tsne(feats, seed):
    tsne = TSNE(n_components=2, init='pca', learning_rate='auto', random_state=seed, perplexity=30)
    return tsne.fit_transform(feats)


def scatter_panel(ax, points, labels, title):
    for label_idx, name in enumerate(STATE_NAMES):
        mask = labels == label_idx
        if np.any(mask):
            ax.scatter(points[mask, 0], points[mask, 1], s=10, alpha=0.7,
                       color=STATE_COLORS[label_idx], label=name)
    ax.set_title(title)
    ax.grid(True, alpha=0.2)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--cls_checkpoint', type=str, default='./ckpt/residual_rl_ppo_aux/final')
    p.add_argument('--mean_checkpoint', type=str, default=None,
                   help='Optional checkpoint trained with mean pooling. '
                        'If omitted, reuse the CLS checkpoint but switch pooling off at analysis time.')
    p.add_argument('--snapshot', type=str, default='./ckpt/residual_rl/snapshot_step315.pkl')
    p.add_argument('--xml_path', type=str, default='./asset/pih.xml')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--samples', type=int, default=400)
    p.add_argument('--noise_pos', type=float, default=2e-4)
    p.add_argument('--noise_rot_deg', type=float, default=0.2)
    p.add_argument('--output_dir', type=str, default='./eval_results/ppo_aux_tsne')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    cls_trainer, cls_cfg = load_trainer(args.cls_checkpoint, device, force_cls_pooling=True)
    if args.mean_checkpoint is None:
        mean_trainer, mean_cfg = load_trainer(args.cls_checkpoint, device, force_cls_pooling=False)
    else:
        mean_trainer, mean_cfg = load_trainer(args.mean_checkpoint, device, force_cls_pooling=False)

    cls_samples, cls_labels = collect_samples(
        cls_trainer, cls_cfg, args.snapshot, args.xml_path, args.seed,
        args.samples, args.noise_pos, args.noise_rot_deg
    )
    mean_samples, mean_labels = collect_samples(
        mean_trainer, mean_cfg, args.snapshot, args.xml_path, args.seed,
        args.samples, args.noise_pos, args.noise_rot_deg
    )

    n = min(len(cls_samples), len(mean_samples), len(cls_labels), len(mean_labels))
    if n == 0:
        raise RuntimeError('No samples collected for t-SNE.')
    cls_labels = cls_labels[:n]
    mean_labels = mean_labels[:n]

    cls_fused = np.stack([s['fused_feat'] for s in cls_samples[:n]], axis=0)
    cls_ft = np.stack([s['ft_feat'] for s in cls_samples[:n]], axis=0)
    mean_fused = np.stack([s['fused_feat'] for s in mean_samples[:n]], axis=0)
    mean_ft = np.stack([s['ft_feat'] for s in mean_samples[:n]], axis=0)

    cls_fused_2d = run_tsne(cls_fused, args.seed)
    mean_fused_2d = run_tsne(mean_fused, args.seed)
    cls_ft_2d = run_tsne(cls_ft, args.seed)
    mean_ft_2d = run_tsne(mean_ft, args.seed)

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    scatter_panel(axes[0, 0], mean_ft_2d, mean_labels, 'Mean pooling: 64D force-temporal feature')
    scatter_panel(axes[0, 1], cls_ft_2d, cls_labels, 'CLS pooling: 64D force-temporal feature')
    scatter_panel(axes[1, 0], mean_fused_2d, mean_labels, 'Mean pooling: fused encoder feature')
    scatter_panel(axes[1, 1], cls_fused_2d, cls_labels, 'CLS pooling: fused encoder feature')
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=3)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(os.path.join(args.output_dir, 'tsne_mean_vs_cls.png'), dpi=180)
    plt.close(fig)

    np.savez(
        os.path.join(args.output_dir, 'tsne_mean_vs_cls_data.npz'),
        cls_fused=cls_fused_2d,
        mean_fused=mean_fused_2d,
        cls_ft=cls_ft_2d,
        mean_ft=mean_ft_2d,
        cls_labels=cls_labels,
        mean_labels=mean_labels,
    )


if __name__ == '__main__':
    main()
