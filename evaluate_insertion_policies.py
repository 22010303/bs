"""
evaluate_insertion_policies.py — Quantitative evaluation for insertion policies.

Evaluates:
  - act
  - smolvla
  - smolvla + residual_sac
  - smolvla + residual_ppo
  - smolvla + residual_td3

Outputs:
  - per-episode CSV
  - summary CSV
  - summary JSON
"""

import argparse
import csv
import json
import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.types import FeatureType

from mujoco_env.pih_env2 import PIHEnv2
from residual_rl.config import ResidualRLConfig
from residual_rl.residual_policy import ResidualActor
from residual_rl_ppo.config import ResidualPPOConfig
from residual_rl_ppo.ppo_trainer import PPOTrainer
from residual_rl_td3.config import ResidualTD3Config
from residual_rl_td3.td3_trainer import TD3Trainer


IMG_TRANSFORM = transforms.Compose([transforms.ToTensor()])
PEG_TIP_SITE = 'peg_tip_site'
HOLE_ENTRY_SITE = 'hole_entry_site'
HOLE_BOTTOM_SITE = 'hole_bottom_site'
ARM_JOINTS = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]
EEF_BODY = 'tool0_link'
ROUND_INSTRUCTION = 'Insert the round peg into the blue round hole.'


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate insertion policies quantitatively')
    p.add_argument('--methods', nargs='+', default=[
        'act', 'smolvla', 'smolvla+sac', 'smolvla+ppo', 'smolvla+td3'
    ])
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--max_steps', type=int, default=650)
    p.add_argument('--xml_path', type=str, default='./asset/pih.xml')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--dataset_root', type=str, default='./demo_data_pih')
    p.add_argument('--repo_name', type=str, default='ur5e_pih_language')
    p.add_argument('--smolvla_pretrained', type=str,
                   default='./ckpt/smolvla_pih1/checkpoints/020000/pretrained_model')
    p.add_argument('--act_pretrained', type=str, default='./ckpt/act_y')
    p.add_argument('--sac_checkpoint', type=str, default='./ckpt/residual_rl/final')
    p.add_argument('--ppo_checkpoint', type=str, default='./ckpt/residual_rl_ppo/final')
    p.add_argument('--td3_checkpoint', type=str, default='./ckpt/residual_rl_td3/final')
    p.add_argument('--output_dir', type=str, default='./eval_results')
    p.add_argument('--render', action='store_true')
    return p.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def check_contact(env):
    peg_tip = env.env.get_p_site(PEG_TIP_SITE)
    hole_entry = env.env.get_p_site(HOLE_ENTRY_SITE)
    xy_dist = np.linalg.norm(peg_tip[:2] - hole_entry[:2])
    peg_below = peg_tip[2] < hole_entry[2]
    return peg_below and xy_dist < 0.03


def cartesian_to_joint_delta(pih_env, cart_residual, cfg):
    _, _, j_full = pih_env.env.get_J_body(EEF_BODY)
    jac_idxs = pih_env.env.get_idxs_jac(ARM_JOINTS)
    j_arm = j_full[:, jac_idxs]
    jjt = j_arm @ j_arm.T + cfg.ik_damping_eps * np.eye(6)
    j_pinv = j_arm.T @ np.linalg.solve(jjt, np.eye(6))
    joint_delta = j_pinv @ cart_residual
    return np.clip(joint_delta, -cfg.max_joint_delta, cfg.max_joint_delta).astype(np.float32)


def load_dataset_features(repo_name, dataset_root):
    meta = LeRobotDatasetMetadata(repo_name, root=dataset_root)
    features = dataset_to_policy_features(meta.features)
    output_features = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}
    input_features = {k: f for k, f in features.items() if k not in output_features}
    return meta, input_features, output_features


def filter_input_features(input_features, allowed_keys):
    return {k: v for k, v in input_features.items() if k in allowed_keys}


def build_base_policy_obs(env, device, input_feature_keys):
    obs = {}
    state = env.get_joint_state()[:6]
    force_torque = env.get_force_torque()
    agent_image, wrist_image = env.grab_image()

    if 'observation.state' in input_feature_keys:
        obs['observation.state'] = torch.from_numpy(
            np.array([state], dtype=np.float32)
        ).to(device)
    if 'observation.force_torque' in input_feature_keys:
        obs['observation.force_torque'] = torch.from_numpy(
            np.array([force_torque], dtype=np.float32)
        ).to(device)
    if 'observation.image' in input_feature_keys:
        img = IMG_TRANSFORM(Image.fromarray(agent_image).resize((256, 256)))
        obs['observation.image'] = img.unsqueeze(0).to(device)
    if 'observation.wrist_image' in input_feature_keys:
        wrist = IMG_TRANSFORM(Image.fromarray(wrist_image).resize((256, 256)))
        obs['observation.wrist_image'] = wrist.unsqueeze(0).to(device)

    obs['task'] = [env.instruction]
    return obs


def build_residual_obs(env, ft_offset, ft_history):
    ft = env.get_force_torque() - ft_offset
    ee = env.get_ee_pose()
    peg_tip = env.env.get_p_site(PEG_TIP_SITE)
    hole_entry = env.env.get_p_site(HOLE_ENTRY_SITE)
    relative = (hole_entry - peg_tip).astype(np.float32)
    state = np.concatenate([ft, ee, relative], dtype=np.float32)
    ft_hist = np.array(ft_history, dtype=np.float32)
    return state, ft_hist


class ActController:
    def __init__(self, args, device):
        meta, input_features, output_features = load_dataset_features(args.repo_name, args.dataset_root)
        with open(os.path.join(args.act_pretrained, 'config.json'), 'r', encoding='utf-8') as f:
            cfg_json = json.load(f)
        input_features = filter_input_features(input_features, cfg_json['input_features'].keys())
        cfg = ACTConfig(
            input_features=input_features,
            output_features=output_features,
            chunk_size=cfg_json['chunk_size'],
            n_action_steps=cfg_json['n_action_steps'],
            temporal_ensemble_coeff=cfg_json.get('temporal_ensemble_coeff'),
        )
        resolve_delta_timestamps(cfg, meta)
        self.policy = ACTPolicy.from_pretrained(
            args.act_pretrained, config=cfg, dataset_stats=meta.stats
        )
        self.policy.to(device)
        self.policy.eval()
        self.device = device
        self.input_feature_keys = set(input_features.keys())

    def reset(self):
        if hasattr(self.policy, 'reset'):
            self.policy.reset()

    def act(self, env):
        data = build_base_policy_obs(env, self.device, self.input_feature_keys)
        with torch.no_grad():
            action = self.policy.select_action(data)
        action_np = action[0, :7].cpu().numpy().astype(np.float32)
        action_np[6] = 200.0
        return action_np, 0.0, 0.0


class SmolVLAController:
    def __init__(self, args, device):
        meta, input_features, output_features = load_dataset_features(args.repo_name, args.dataset_root)
        with open(os.path.join(args.smolvla_pretrained, 'config.json'), 'r', encoding='utf-8') as f:
            cfg_json = json.load(f)
        input_features = filter_input_features(input_features, cfg_json['input_features'].keys())
        cfg = SmolVLAConfig(
            input_features=input_features,
            output_features=output_features,
            chunk_size=cfg_json['chunk_size'],
            n_action_steps=cfg_json['n_action_steps'],
        )
        resolve_delta_timestamps(cfg, meta)
        self.policy = SmolVLAPolicy.from_pretrained(
            args.smolvla_pretrained, config=cfg, dataset_stats=meta.stats
        )
        self.policy.to(device)
        self.policy.eval()
        self.device = device
        self.input_feature_keys = set(input_features.keys())

    def reset(self):
        if hasattr(self.policy, 'reset'):
            self.policy.reset()

    def act(self, env):
        data = build_base_policy_obs(env, self.device, self.input_feature_keys)
        with torch.no_grad():
            action = self.policy.select_action(data)
        action_np = action[0, :7].cpu().numpy().astype(np.float32)
        action_np[6] = 200.0
        return action_np, 0.0, 0.0


class SmolVLAResidualSACController:
    def __init__(self, args, device):
        self.base = SmolVLAController(args, device)
        ckpt = torch.load(
            os.path.join(args.sac_checkpoint, 'sac_checkpoint.pt'),
            map_location=device,
            weights_only=False,
        )
        self.cfg = ckpt.get('config', ResidualRLConfig())
        self.actor = ResidualActor(self.cfg).to(device)
        self.actor.load_state_dict(ckpt['actor'])
        self.actor.eval()
        self.device = device

    def reset(self):
        self.base.reset()

    def act(self, env, residual_state):
        base_action, _, _ = self.base.act(env)
        if not residual_state['contact']:
            return base_action, 0.0, 0.0

        state_t = torch.FloatTensor(residual_state['state']).unsqueeze(0).to(self.device)
        hist_t = torch.FloatTensor(residual_state['ft_history']).unsqueeze(0).to(self.device)
        with torch.no_grad():
            cart = self.actor.deterministic_action(state_t, hist_t).squeeze(0).cpu().numpy()
        dq = cartesian_to_joint_delta(env, cart, self.cfg)
        final_action = base_action.copy()
        final_action[:6] += dq
        return final_action, float(np.linalg.norm(cart)), float(np.linalg.norm(dq))


class SmolVLAResidualPPOController:
    def __init__(self, args, device):
        self.base = SmolVLAController(args, device)
        ckpt = torch.load(
            os.path.join(args.ppo_checkpoint, 'ppo_checkpoint.pt'),
            map_location=device,
            weights_only=False,
        )
        self.cfg = ckpt.get('config', ResidualPPOConfig())
        self.trainer = PPOTrainer(config=self.cfg, device=str(device))
        self.trainer.load(args.ppo_checkpoint)

    def reset(self):
        self.base.reset()

    def act(self, env, residual_state):
        base_action, _, _ = self.base.act(env)
        if not residual_state['contact']:
            return base_action, 0.0, 0.0

        cart = self.trainer.deterministic_action(
            residual_state['state'], residual_state['ft_history']
        )
        dq = cartesian_to_joint_delta(env, cart, self.cfg)
        final_action = base_action.copy()
        final_action[:6] += dq
        return final_action, float(np.linalg.norm(cart)), float(np.linalg.norm(dq))


class SmolVLAResidualTD3Controller:
    def __init__(self, args, device):
        self.base = SmolVLAController(args, device)
        ckpt = torch.load(
            os.path.join(args.td3_checkpoint, 'td3_checkpoint.pt'),
            map_location=device,
            weights_only=False,
        )
        self.cfg = ckpt.get('config', ResidualTD3Config())
        self.trainer = TD3Trainer(config=self.cfg, device=str(device))
        self.trainer.load(args.td3_checkpoint)

    def reset(self):
        self.base.reset()

    def act(self, env, residual_state):
        base_action, _, _ = self.base.act(env)
        if not residual_state['contact']:
            return base_action, 0.0, 0.0

        cart = self.trainer.select_action(
            residual_state['state'], residual_state['ft_history'], deterministic=True
        )
        dq = cartesian_to_joint_delta(env, cart, self.cfg)
        final_action = base_action.copy()
        final_action[:6] += dq
        return final_action, float(np.linalg.norm(cart)), float(np.linalg.norm(dq))


def make_controller(method, args, device):
    if method == 'act':
        if not os.path.isdir(args.act_pretrained):
            return None, f'ACT checkpoint not found: {args.act_pretrained}'
        return ActController(args, device), None
    if method == 'smolvla':
        if not os.path.isdir(args.smolvla_pretrained):
            return None, f'SmolVLA checkpoint not found: {args.smolvla_pretrained}'
        return SmolVLAController(args, device), None
    if method == 'smolvla+sac':
        if not os.path.isfile(os.path.join(args.sac_checkpoint, 'sac_checkpoint.pt')):
            return None, f'SAC checkpoint not found: {args.sac_checkpoint}'
        return SmolVLAResidualSACController(args, device), None
    if method == 'smolvla+ppo':
        if not os.path.isfile(os.path.join(args.ppo_checkpoint, 'ppo_checkpoint.pt')):
            return None, f'PPO checkpoint not found: {args.ppo_checkpoint}'
        return SmolVLAResidualPPOController(args, device), None
    if method == 'smolvla+td3':
        if not os.path.isfile(os.path.join(args.td3_checkpoint, 'td3_checkpoint.pt')):
            return None, f'TD3 checkpoint not found: {args.td3_checkpoint}'
        return SmolVLAResidualTD3Controller(args, device), None
    return None, f'Unknown method: {method}'


def init_episode_metrics():
    return {
        'sum_force_norm_all': 0.0,
        'sum_torque_norm_all': 0.0,
        'sum_lateral_force_all': 0.0,
        'sum_axial_force_all': 0.0,
        'sum_bending_torque_all': 0.0,
        'sum_force_norm_contact': 0.0,
        'sum_torque_norm_contact': 0.0,
        'sum_lateral_force_contact': 0.0,
        'sum_axial_force_contact': 0.0,
        'sum_bending_torque_contact': 0.0,
        'peak_force_norm': 0.0,
        'peak_torque_norm': 0.0,
        'contact_steps': 0,
        'all_steps': 0,
        'contact_started': False,
        'contact_start_step': -1,
        'sum_cart_residual_norm': 0.0,
        'sum_joint_delta_norm': 0.0,
        'residual_steps': 0,
    }


def finalize_episode_metrics(metrics):
    def mean_or_zero(total, count):
        return float(total / count) if count > 0 else 0.0

    return {
        'mean_force_norm_all': mean_or_zero(metrics['sum_force_norm_all'], metrics['all_steps']),
        'mean_torque_norm_all': mean_or_zero(metrics['sum_torque_norm_all'], metrics['all_steps']),
        'mean_lateral_force_all': mean_or_zero(metrics['sum_lateral_force_all'], metrics['all_steps']),
        'mean_axial_force_all': mean_or_zero(metrics['sum_axial_force_all'], metrics['all_steps']),
        'mean_bending_torque_all': mean_or_zero(metrics['sum_bending_torque_all'], metrics['all_steps']),
        'mean_force_norm_contact': mean_or_zero(metrics['sum_force_norm_contact'], metrics['contact_steps']),
        'mean_torque_norm_contact': mean_or_zero(metrics['sum_torque_norm_contact'], metrics['contact_steps']),
        'mean_lateral_force_contact': mean_or_zero(metrics['sum_lateral_force_contact'], metrics['contact_steps']),
        'mean_axial_force_contact': mean_or_zero(metrics['sum_axial_force_contact'], metrics['contact_steps']),
        'mean_bending_torque_contact': mean_or_zero(metrics['sum_bending_torque_contact'], metrics['contact_steps']),
        'mean_cart_residual_norm': mean_or_zero(metrics['sum_cart_residual_norm'], metrics['residual_steps']),
        'mean_joint_delta_norm': mean_or_zero(metrics['sum_joint_delta_norm'], metrics['residual_steps']),
        'peak_force_norm': metrics['peak_force_norm'],
        'peak_torque_norm': metrics['peak_torque_norm'],
        'contact_steps': metrics['contact_steps'],
        'contact_start_step': metrics['contact_start_step'],
    }


def compute_summary(method, episode_rows):
    success_rows = [r for r in episode_rows if r['success'] == 1]
    contact_rows = [r for r in episode_rows if r['contact_steps'] > 0]

    def mean(key, rows=None):
        rows = episode_rows if rows is None else rows
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    return {
        'method': method,
        'episodes': len(episode_rows),
        'success_rate': mean('success'),
        'avg_episode_time_sec': mean('duration_sec'),
        'avg_success_time_sec': mean('duration_sec', success_rows),
        'avg_steps': mean('steps'),
        'avg_success_steps': mean('steps', success_rows),
        'avg_contact_start_sec': mean('contact_start_sec', contact_rows),
        'avg_contact_force_norm_N': mean('mean_force_norm_contact', contact_rows),
        'avg_contact_torque_norm_Nm': mean('mean_torque_norm_contact', contact_rows),
        'avg_contact_lateral_force_N': mean('mean_lateral_force_contact', contact_rows),
        'avg_contact_axial_force_N': mean('mean_axial_force_contact', contact_rows),
        'avg_contact_bending_torque_Nm': mean('mean_bending_torque_contact', contact_rows),
        'avg_peak_force_norm_N': mean('peak_force_norm'),
        'avg_peak_torque_norm_Nm': mean('peak_torque_norm'),
        'avg_cart_residual_norm': mean('mean_cart_residual_norm'),
        'avg_joint_delta_norm': mean('mean_joint_delta_norm'),
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def evaluate_method(method, controller, args, device):
    env = PIHEnv2(
        xml_path=args.xml_path,
        action_type='joint_angle',
        state_type='joint_angle',
    )
    control_hz = 20.0
    rows = []

    for episode_idx in range(args.episodes):
        env.reset(seed=episode_idx)
        env.set_instruction(ROUND_INSTRUCTION)
        controller.reset()

        ft_offset = env.get_force_torque().copy()
        ft_history = [np.zeros(6, dtype=np.float32) for _ in range(10)]
        current_ft = env.get_force_torque() - ft_offset
        ft_history = [current_ft.copy() for _ in range(10)]

        episode_metrics = init_episode_metrics()
        success = False
        step = 0

        while step < args.max_steps:
            env.step_env()

            if not env.env.loop_every(HZ=int(control_hz)):
                continue

            ft_cal = env.get_force_torque() - ft_offset
            ft_history.append(ft_cal.copy())
            ft_history = ft_history[-10:]
            force_norm = float(np.linalg.norm(ft_cal[:3]))
            torque_norm = float(np.linalg.norm(ft_cal[3:]))
            lateral_force = float(np.linalg.norm(ft_cal[:2]))
            axial_force = float(abs(ft_cal[2]))
            bending_torque = float(np.linalg.norm(ft_cal[3:5]))

            episode_metrics['sum_force_norm_all'] += force_norm
            episode_metrics['sum_torque_norm_all'] += torque_norm
            episode_metrics['sum_lateral_force_all'] += lateral_force
            episode_metrics['sum_axial_force_all'] += axial_force
            episode_metrics['sum_bending_torque_all'] += bending_torque
            episode_metrics['peak_force_norm'] = max(episode_metrics['peak_force_norm'], force_norm)
            episode_metrics['peak_torque_norm'] = max(episode_metrics['peak_torque_norm'], torque_norm)
            episode_metrics['all_steps'] += 1

            contact = check_contact(env)
            if contact and not episode_metrics['contact_started']:
                episode_metrics['contact_started'] = True
                episode_metrics['contact_start_step'] = step
            if contact:
                episode_metrics['sum_force_norm_contact'] += force_norm
                episode_metrics['sum_torque_norm_contact'] += torque_norm
                episode_metrics['sum_lateral_force_contact'] += lateral_force
                episode_metrics['sum_axial_force_contact'] += axial_force
                episode_metrics['sum_bending_torque_contact'] += bending_torque
                episode_metrics['contact_steps'] += 1

            residual_state = {
                'contact': contact,
                'state': None,
                'ft_history': None,
            }
            if method.startswith('smolvla+') and contact:
                state_vec, ft_hist = build_residual_obs(env, ft_offset, ft_history)
                residual_state['state'] = state_vec
                residual_state['ft_history'] = ft_hist

            if method in {'act', 'smolvla'}:
                action, cart_norm, dq_norm = controller.act(env)
            else:
                action, cart_norm, dq_norm = controller.act(env, residual_state)
                if contact:
                    episode_metrics['sum_cart_residual_norm'] += cart_norm
                    episode_metrics['sum_joint_delta_norm'] += dq_norm
                    episode_metrics['residual_steps'] += 1

            env.step(action)
            if args.render:
                env.render(idx=episode_idx)

            success = env.check_success()
            step += 1
            if success:
                break

        episode_result = {
            'method': method,
            'episode': episode_idx,
            'success': int(success),
            'steps': step,
            'duration_sec': float(step / control_hz),
            'contact_start_sec': (
                float(episode_metrics['contact_start_step'] / control_hz)
                if episode_metrics['contact_start_step'] >= 0 else -1.0
            ),
        }
        episode_result.update(finalize_episode_metrics(episode_metrics))
        rows.append(episode_result)
        print(
            f"[{method}] ep={episode_idx:>3d} success={int(success)} "
            f"steps={step:>4d} time={episode_result['duration_sec']:.2f}s "
            f"contactF={episode_result['mean_force_norm_contact']:.3f}N "
            f"contactT={episode_result['mean_torque_norm_contact']:.3f}Nm"
        )

    env.env.close_viewer()
    return rows


def main():
    args = parse_args()
    device = torch.device(args.device)
    ensure_dir(args.output_dir)

    all_episode_rows = []
    summary_rows = []

    for method in args.methods:
        controller, error = make_controller(method, args, device)
        if error:
            print(f"Skip {method}: {error}")
            continue

        print(f"\n=== Evaluating {method} ===")
        episode_rows = evaluate_method(method, controller, args, device)
        all_episode_rows.extend(episode_rows)
        summary = compute_summary(method, episode_rows)
        summary_rows.append(summary)

    episode_csv = os.path.join(args.output_dir, 'episode_metrics.csv')
    summary_csv = os.path.join(args.output_dir, 'summary_metrics.csv')
    summary_json = os.path.join(args.output_dir, 'summary_metrics.json')

    write_csv(episode_csv, all_episode_rows)
    write_csv(summary_csv, summary_rows)
    with open(summary_json, 'w', encoding='utf-8') as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)

    if summary_rows:
        print('\n=== Summary ===')
        for row in summary_rows:
            print(
                f"{row['method']:>12s} | success={row['success_rate']:.3f} | "
                f"time={row['avg_episode_time_sec']:.2f}s | "
                f"contactF={row['avg_contact_force_norm_N']:.3f}N | "
                f"contactT={row['avg_contact_torque_norm_Nm']:.3f}Nm"
            )
    print(f"\nSaved: {episode_csv}")
    print(f"Saved: {summary_csv}")
    print(f"Saved: {summary_json}")


if __name__ == '__main__':
    main()
