"""
evaluate_residual_rl_ppo_aux.py — Evaluate snapshot-based PPO-AUX policy.

Outputs:
  1. Per-episode time-series plot:
     - x-axis: step
     - left y-axis: force metrics
     - right y-axis: predicted state {0,1,2}
     - background shading by predicted state:
         0 smooth       -> white / no fill
         1 jamming      -> light red
         2 bottoming    -> light green
  2. One confusion matrix over all collected true/pred labels.
  3. Per-step CSVs for later inspection.
"""

import argparse
import csv
import os
from collections import deque

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from mujoco_env.pih_env2 import PIHEnv2
from residual_rl.snapshot import load_snapshot, perturb_ee_pose, restore_snapshot
from residual_rl_ppo.aux_config import ResidualPPOAuxConfig
from residual_rl_ppo.aux_trainer import PPOAuxTrainer


IMG_TRANSFORM = transforms.Compose([transforms.ToTensor()])
PEG_TIP_SITE = 'peg_tip_site'
HOLE_ENTRY_SITE = 'hole_entry_site'
HOLE_BOTTOM_SITE = 'hole_bottom_site'
STATE_NAMES = ['smooth', 'jamming', 'bottoming']
STATE_COLORS = {
    0: None,
    1: '#f7caca',
    2: '#d9f2d9',
}


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def resolve_checkpoint_dir(checkpoint_arg):
    """
    Accept any of the following:
      - a directory containing `ppo_aux_checkpoint.pt`
      - a parent directory containing `final/ppo_aux_checkpoint.pt`
      - a direct path to `ppo_aux_checkpoint.pt`
    Return the checkpoint directory expected by PPOAuxTrainer.load(...).
    """
    checkpoint_arg = os.path.abspath(checkpoint_arg)

    if os.path.isfile(checkpoint_arg):
        if os.path.basename(checkpoint_arg) != 'ppo_aux_checkpoint.pt':
            raise FileNotFoundError(
                f'Unsupported checkpoint file: {checkpoint_arg}. '
                'Expected a `ppo_aux_checkpoint.pt` file.'
            )
        return os.path.dirname(checkpoint_arg)

    direct_ckpt = os.path.join(checkpoint_arg, 'ppo_aux_checkpoint.pt')
    if os.path.isfile(direct_ckpt):
        return checkpoint_arg

    final_ckpt = os.path.join(checkpoint_arg, 'final', 'ppo_aux_checkpoint.pt')
    if os.path.isfile(final_ckpt):
        return os.path.join(checkpoint_arg, 'final')

    raise FileNotFoundError(
        'Could not resolve PPO-AUX checkpoint from '
        f'`{checkpoint_arg}`. Tried:\n'
        f'  - {direct_ckpt}\n'
        f'  - {final_ckpt}'
    )


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
    return action_np.astype(np.float32)


def cartesian_to_joint_delta(pih_env, cart_residual, cfg):
    arm_joints = [
        'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
        'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
    ]
    _, _, j_full = pih_env.env.get_J_body('tool0_link')
    jac_idxs = pih_env.env.get_idxs_jac(arm_joints)
    j_arm = j_full[:, jac_idxs]
    jjt = j_arm @ j_arm.T + cfg.ik_damping_eps * np.eye(6)
    j_pinv = j_arm.T @ np.linalg.solve(jjt, np.eye(6))
    joint_delta = j_pinv @ cart_residual
    return np.clip(joint_delta, -cfg.max_joint_delta, cfg.max_joint_delta).astype(np.float32)


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


def predict_action_and_label(trainer, state, ft_history):
    state_t = torch.FloatTensor(state).unsqueeze(0).to(trainer.device)
    hist_t = torch.FloatTensor(ft_history).unsqueeze(0).to(trainer.device)
    state_t = trainer._normalize_state(state_t)
    with torch.no_grad():
        action = trainer.policy.deterministic_action(state_t, hist_t)
        _, _, _, state_logits = trainer.policy._forward_all(state_t, hist_t)
    pred_label = int(state_logits.argmax(dim=-1).item())
    return action.squeeze(0).cpu().numpy(), pred_label


def shade_predicted_states(ax, steps, pred_labels):
    if len(steps) == 0:
        return
    start_idx = 0
    current_label = pred_labels[0]
    for idx in range(1, len(pred_labels) + 1):
        if idx == len(pred_labels) or pred_labels[idx] != current_label:
            if current_label != 0:
                ax.axvspan(
                    steps[start_idx],
                    steps[idx - 1] + 1,
                    color=STATE_COLORS[current_label],
                    alpha=0.45,
                    zorder=0,
                )
            if idx < len(pred_labels):
                start_idx = idx
                current_label = pred_labels[idx]


def plot_force_state_timeline(episode_idx, rows, output_dir):
    steps = np.array([r['step'] for r in rows], dtype=np.int32)
    lateral = np.array([r['lateral_force'] for r in rows], dtype=np.float32)
    axial = np.array([r['axial_force'] for r in rows], dtype=np.float32)
    total = np.array([r['total_force'] for r in rows], dtype=np.float32)
    true_labels = np.array([r['true_label'] for r in rows], dtype=np.int32)
    pred_labels = np.array([r['pred_label'] for r in rows], dtype=np.int32)

    fig, ax_force = plt.subplots(figsize=(14, 6))
    shade_predicted_states(ax_force, steps, pred_labels.tolist())
    ax_force.plot(steps, lateral, label='Lateral |Fxy|', color='#d62728', linewidth=1.8)
    ax_force.plot(steps, axial, label='Axial |Fz|', color='#1f77b4', linewidth=1.8)
    ax_force.plot(steps, total, label='Total |F|', color='#2ca02c', linewidth=1.5, alpha=0.8)
    ax_force.set_xlabel('Step')
    ax_force.set_ylabel('Force')
    ax_force.grid(True, alpha=0.25)

    ax_state = ax_force.twinx()
    ax_state.step(steps, pred_labels, where='post', color='black', linewidth=1.8, label='Pred state')
    ax_state.step(steps, true_labels, where='post', color='#9467bd', linewidth=1.2,
                  linestyle='--', alpha=0.9, label='True state')
    ax_state.set_ylabel('State')
    ax_state.set_yticks([0, 1, 2])
    ax_state.set_yticklabels(['smooth', 'jamming', 'bottoming'])
    ax_state.set_ylim(-0.25, 2.25)

    handles1, labels1 = ax_force.get_legend_handles_labels()
    handles2, labels2 = ax_state.get_legend_handles_labels()
    ax_force.legend(handles1 + handles2, labels1 + labels2, loc='upper right')
    ax_force.set_title(
        f'Episode {episode_idx}: force trajectory with predicted-state shading'
    )
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'episode_{episode_idx:03d}_force_state_timeline.png'), dpi=180)
    plt.close(fig)


def plot_confusion_matrix(true_labels, pred_labels, output_path):
    cm = np.zeros((3, 3), dtype=np.int32)
    for t, p in zip(true_labels, pred_labels):
        cm[int(t), int(p)] += 1

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks([0, 1, 2])
    ax.set_yticks([0, 1, 2])
    ax.set_xticklabels(STATE_NAMES)
    ax.set_yticklabels(STATE_NAMES)
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title('PPO-AUX state recognition confusion matrix')

    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)
    return cm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, default='./ckpt/residual_rl_ppo_aux/final')
    p.add_argument('--snapshot', type=str, default='./ckpt/residual_rl/snapshot_step315.pkl')
    p.add_argument('--xml_path', type=str, default='./asset/pih.xml')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--episodes', type=int, default=10)
    p.add_argument('--max_steps', type=int, default=650)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--noise_pos', type=float, default=2e-4)
    p.add_argument('--noise_rot_deg', type=float, default=0.2)
    p.add_argument('--output_dir', type=str, default='./eval_results/ppo_aux')
    p.add_argument('--no_smolvla', action='store_true')
    p.add_argument('--smolvla_pretrained', type=str,
                   default='./ckpt/smolvla_pih1/checkpoints/020000/pretrained_model')
    p.add_argument('--smolvla_dataset_root', type=str, default='./demo_data_pih')
    p.add_argument('--smolvla_repo_name', type=str, default='ur5e_pih_language')
    p.add_argument('--no_viewer', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dir(args.output_dir)
    device = torch.device(args.device)
    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint)

    ckpt = torch.load(
        os.path.join(checkpoint_dir, 'ppo_aux_checkpoint.pt'),
        map_location=device,
        weights_only=False,
    )
    cfg = ckpt.get('config', ResidualPPOAuxConfig())
    cfg.xml_path = args.xml_path
    cfg.smolvla_pretrained = args.smolvla_pretrained
    cfg.smolvla_dataset_root = args.smolvla_dataset_root
    cfg.smolvla_repo_name = args.smolvla_repo_name
    cfg.smolvla_device = str(device)

    trainer = PPOAuxTrainer(config=cfg, device=str(device), use_wandb=False)
    trainer.load(checkpoint_dir)
    smolvla = None if args.no_smolvla else load_smolvla(cfg)

    snap = load_snapshot(args.snapshot)
    base_step = snap['meta']['base_step']
    rng = np.random.default_rng(args.seed)
    dt = 1.0 / cfg.control_hz

    env = PIHEnv2(xml_path=cfg.xml_path, action_type='joint_angle', state_type='joint_angle')
    labeler = DebouncedInsertionStateLabeler(cfg)

    all_true_labels = []
    all_pred_labels = []

    for episode_idx in range(args.episodes):
        if smolvla is not None:
            smolvla.reset()
        ft_offset = restore_snapshot(env, snap)
        perturb_ee_pose(
            env,
            pos_noise=args.noise_pos,
            rot_noise_deg=args.noise_rot_deg,
            rng=rng,
        )

        ft_history = deque(maxlen=cfg.force_history_len)
        ft_init = env.get_force_torque() - ft_offset
        for _ in range(cfg.force_history_len):
            ft_history.append(ft_init.copy())

        labeler.reset()
        prev_ee_pose = env.get_ee_pose().copy()
        contact_detected = False
        episode_rows = []

        step = base_step
        while step < args.max_steps:
            env.step_env()
            if not env.env.loop_every(HZ=20):
                continue

            success = env.check_success()
            if smolvla is not None:
                base_action = get_smolvla_action(smolvla, env, device)
            else:
                q = env.get_joint_state()[:6]
                base_action = np.concatenate([q, [200.0]], dtype=np.float32)

            if not contact_detected:
                contact_detected = check_contact(env)

            cartesian_residual = np.zeros(6, dtype=np.float32)
            joint_delta = np.zeros(6, dtype=np.float32)
            pred_label = 0
            true_label = 0

            ft = env.get_force_torque() - ft_offset
            ee_pose = env.get_ee_pose()
            ee_vel = (ee_pose - prev_ee_pose) / dt
            peg_tip = env.env.get_p_site(PEG_TIP_SITE)
            hole_entry = env.env.get_p_site(HOLE_ENTRY_SITE)
            hole_bottom = env.env.get_p_site(HOLE_BOTTOM_SITE)

            ft_history.append(ft.copy())
            state_vec, ft_hist = build_rl_obs(ft, ee_pose, ee_vel, peg_tip, hole_entry, ft_history)

            if contact_detected:
                true_label = labeler.update(ft, ee_vel, peg_tip, hole_entry, hole_bottom)
                cartesian_residual, pred_label = predict_action_and_label(trainer, state_vec, ft_hist)
                joint_delta = cartesian_to_joint_delta(env, cartesian_residual, cfg)

            final_action = base_action.copy()
            final_action[:6] += joint_delta
            env.step(final_action)

            next_ft = env.get_force_torque() - ft_offset
            total_force = float(np.linalg.norm(next_ft[:3]))
            lateral_force = float(np.linalg.norm(next_ft[:2]))
            axial_force = float(abs(next_ft[2]))

            rel_step = step - base_step
            episode_rows.append({
                'step': rel_step,
                'force_fx': float(next_ft[0]),
                'force_fy': float(next_ft[1]),
                'force_fz': float(next_ft[2]),
                'lateral_force': lateral_force,
                'axial_force': axial_force,
                'total_force': total_force,
                'true_label': int(true_label),
                'pred_label': int(pred_label),
                'contact_detected': int(contact_detected),
                'success': int(success),
            })
            if contact_detected:
                all_true_labels.append(int(true_label))
                all_pred_labels.append(int(pred_label))

            prev_ee_pose = env.get_ee_pose().copy()

            if not args.no_viewer:
                env.render(idx=episode_idx)
                env.env.viewer_text_overlay(
                    text1='PPO AUX Eval',
                    text2=(
                        f'step={rel_step} true={true_label} pred={pred_label} '
                        f'|Fxy|={lateral_force:.1f} |Fz|={axial_force:.1f}'
                    ),
                )

            if success:
                break
            step += 1

        csv_path = os.path.join(args.output_dir, f'episode_{episode_idx:03d}_timeline.csv')
        with open(csv_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(episode_rows[0].keys()) if episode_rows else [
                'step', 'force_fx', 'force_fy', 'force_fz', 'lateral_force',
                'axial_force', 'total_force', 'true_label', 'pred_label',
                'contact_detected', 'success',
            ])
            writer.writeheader()
            writer.writerows(episode_rows)

        plot_force_state_timeline(episode_idx, episode_rows, args.output_dir)

    if env.env.is_viewer_alive():
        env.env.close_viewer()

    cm = plot_confusion_matrix(
        all_true_labels,
        all_pred_labels,
        os.path.join(args.output_dir, 'state_confusion_matrix.png'),
    )

    summary_path = os.path.join(args.output_dir, 'state_confusion_matrix.csv')
    with open(summary_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['true/pred'] + STATE_NAMES)
        for i, name in enumerate(STATE_NAMES):
            writer.writerow([name] + cm[i].tolist())


if __name__ == '__main__':
    main()
