"""
analyze_residual_rl_ppo_aux_attention.py — Visualize Fz transition and the
CLS-token attention over the K-step force history for PPO-AUX.
"""

import argparse
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
    action_np = action[0, :7].cpu().numpy().astype(np.float32)
    action_np[6] = 200.0
    return action_np


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


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=str, default='./ckpt/residual_rl_ppo_aux/final')
    p.add_argument('--snapshot', type=str, default='./ckpt/residual_rl/snapshot_step315.pkl')
    p.add_argument('--xml_path', type=str, default='./asset/pih.xml')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--max_steps', type=int, default=650)
    p.add_argument('--noise_pos', type=float, default=2e-4)
    p.add_argument('--noise_rot_deg', type=float, default=0.2)
    p.add_argument('--output_dir', type=str, default='./eval_results/ppo_aux_attention')
    p.add_argument('--analysis_step', type=int, default=None,
                   help='Specific relative step to analyze. Default: largest positive Fz jump.')
    p.add_argument('--no_smolvla', action='store_true')
    p.add_argument('--smolvla_pretrained', type=str,
                   default='./ckpt/smolvla_pih1/checkpoints/020000/pretrained_model')
    p.add_argument('--smolvla_dataset_root', type=str, default='./demo_data_pih')
    p.add_argument('--smolvla_repo_name', type=str, default='ur5e_pih_language')
    p.add_argument('--no_viewer', action='store_true')
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)
    checkpoint_dir = resolve_checkpoint_dir(args.checkpoint)

    ckpt = torch.load(os.path.join(checkpoint_dir, 'ppo_aux_checkpoint.pt'),
                      map_location=device, weights_only=False)
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

    if smolvla is not None:
        smolvla.reset()
    ft_offset = restore_snapshot(env, snap)
    perturb_ee_pose(env, pos_noise=args.noise_pos, rot_noise_deg=args.noise_rot_deg, rng=rng)
    ft_history = deque(maxlen=cfg.force_history_len)
    ft_init = env.get_force_torque() - ft_offset
    for _ in range(cfg.force_history_len):
        ft_history.append(ft_init.copy())
    labeler.reset()
    prev_ee_pose = env.get_ee_pose().copy()
    contact_detected = False
    rows = []

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
        pred_label = 0
        true_label = 0
        attention_scores = np.zeros(cfg.force_history_len, dtype=np.float32)

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
            state_t = torch.FloatTensor(state_vec).unsqueeze(0).to(trainer.device)
            hist_t = torch.FloatTensor(ft_hist).unsqueeze(0).to(trainer.device)
            state_t = trainer._normalize_state(state_t)
            with torch.no_grad():
                analysis = trainer.policy.analyze(state_t, hist_t)
            cartesian_residual = analysis['action'].squeeze(0).cpu().numpy()
            pred_label = int(analysis['state_logits'].argmax(dim=-1).item())
            attention_scores = analysis['attention_scores'].squeeze(0).detach().cpu().numpy()

        joint_delta = cartesian_to_joint_delta(env, cartesian_residual, cfg)
        final_action = base_action.copy()
        final_action[:6] += joint_delta
        env.step(final_action)

        next_ft = env.get_force_torque() - ft_offset
        rel_step = step - base_step
        rows.append({
            'step': rel_step,
            'fz': float(next_ft[2]),
            'true_label': int(true_label),
            'pred_label': int(pred_label),
            'contact_detected': int(contact_detected),
            'attention_scores': attention_scores.copy(),
        })

        prev_ee_pose = env.get_ee_pose().copy()
        if not args.no_viewer:
            env.render(idx=0)
        if success:
            break
        step += 1

    if env.env.is_viewer_alive():
        env.env.close_viewer()

    fz = np.array([r['fz'] for r in rows], dtype=np.float32)
    steps = np.array([r['step'] for r in rows], dtype=np.int32)
    if len(rows) < 2:
        raise RuntimeError('Not enough rollout data to analyze attention.')
    fz_delta = np.diff(fz, prepend=fz[0])
    if args.analysis_step is None:
        analysis_idx = int(np.argmax(fz_delta))
    else:
        matches = np.where(steps == args.analysis_step)[0]
        if len(matches) == 0:
            raise ValueError(f'analysis_step={args.analysis_step} not found in collected steps.')
        analysis_idx = int(matches[0])

    analysis_step = int(steps[analysis_idx])
    analysis_fz = float(fz[analysis_idx])
    analysis_attn = rows[analysis_idx]['attention_scores']

    fig, (ax_top, ax_bottom) = plt.subplots(
        2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [2, 1]}, sharex=False
    )
    ax_top.plot(steps, fz, color='#1f77b4', linewidth=2.0)
    ax_top.axvline(analysis_step, color='#d62728', linestyle='--', linewidth=1.8,
                   label=f'Fz jump step={analysis_step}')
    ax_top.scatter([analysis_step], [analysis_fz], color='#d62728', zorder=5)
    ax_top.set_title('Fz over time with marked contact-force jump')
    ax_top.set_xlabel('Step')
    ax_top.set_ylabel('Fz')
    ax_top.grid(True, alpha=0.25)
    ax_top.legend(loc='upper right')

    hist_idx = np.arange(cfg.force_history_len)
    ax_bottom.bar(hist_idx, analysis_attn, color='#4c78a8', alpha=0.9)
    ax_bottom.set_ylim(0.0, max(1.0, float(np.max(analysis_attn) * 1.1)))
    ax_bottom.set_title(
        f'CLS attention over history window at step={analysis_step} '
        f'(true={rows[analysis_idx]["true_label"]}, pred={rows[analysis_idx]["pred_label"]})'
    )
    ax_bottom.set_xlabel('History frame index (old -> recent)')
    ax_bottom.set_ylabel('Attention score')
    ax_bottom.grid(True, axis='y', alpha=0.25)

    fig.tight_layout()
    fig.savefig(os.path.join(args.output_dir, 'fz_attention_analysis.png'), dpi=180)
    plt.close(fig)

    np.savez(
        os.path.join(args.output_dir, 'fz_attention_analysis_data.npz'),
        steps=steps,
        fz=fz,
        fz_delta=fz_delta,
        analysis_step=analysis_step,
        analysis_attention=analysis_attn,
    )


if __name__ == '__main__':
    main()
