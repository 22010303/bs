"""
evaluate_insertion_policies.py — Quantitative insertion policy evaluation.

Evaluates any subset of:
  - act
  - smolvla
  - smolvla+sac   (SmolVLA + residual SAC)
  - smolvla+ppo   (SmolVLA + residual PPO)
  - smolvla+td3   (SmolVLA + residual TD3)

For each method, per-episode and over `--episodes` rollouts:
  * success rate
  * success step count + completion time
  * per-step F/T (zero-calibrated at home pose)
  * per-method CSVs:
        perstep_force_torque.csv
        episode_summary.csv
  * per-method plots (one PNG each, xyz separate):
        force_Fx_vs_step.png, force_Fy_vs_step.png, force_Fz_vs_step.png
        torque_Tx_vs_step.png, torque_Ty_vs_step.png, torque_Tz_vs_step.png
  * comparative summary across methods:
        <output_dir>/summary_metrics.csv
        <output_dir>/summary_metrics.json

Usage:
  python evaluate_insertion_policies.py                                # all 5 methods
  python evaluate_insertion_policies.py --methods smolvla smolvla+sac smolvla+ppo smolvla+td3
  python evaluate_insertion_policies.py --episodes 30 --max_steps 800 --no_viewer
"""

import argparse     # 处理命令行参数
import csv
import json
import os
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')  # headless-safe
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

# 策略模型：从lerobot库导入ACT和SmolVLA策略模型及其配置
from lerobot.common.datasets.factory import resolve_delta_timestamps
from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.common.datasets.utils import dataset_to_policy_features
from lerobot.common.policies.act.configuration_act import ACTConfig
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
from lerobot.configs.types import FeatureType

# 环境与RL策略：从本地模块导入仿真环境PIHEnv2，以及SAC、PPO、TD3三种RL算法的残差策略训练器和配置
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
FT_HISTORY_LEN = 10


# ======================================================================
# CLI
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser(description='Evaluate insertion policies quantitatively.')
    # 要评估的策略
    p.add_argument('--methods', nargs='+', default=[
        'smolvla', 'smolvla+sac', 'smolvla+ppo', 'smolvla+td3',
    ], choices=['act', 'smolvla', 'smolvla+sac', 'smolvla+ppo', 'smolvla+td3'])
    # 实验次数、最大步数、随机种子
    p.add_argument('--episodes', type=int, default=20)
    p.add_argument('--max_steps', type=int, default=650)
    p.add_argument('--base_seed', type=int, default=0)
    p.add_argument('--xml_path', type=str, default='./asset/pih.xml')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--dataset_root', type=str, default='./demo_data_pih')
    p.add_argument('--repo_name', type=str, default='ur5e_pih_language')
    p.add_argument('--instruction', type=str, default=ROUND_INSTRUCTION)
    # 各策略对应预训练模型的路径
    p.add_argument('--smolvla_pretrained', type=str,
                   default='./ckpt/smolvla_pih1/checkpoints/020000/pretrained_model')
    p.add_argument('--act_pretrained', type=str, default='./ckpt/act_y')
    p.add_argument('--sac_checkpoint', type=str, default='./ckpt/residual_rl_snapshot/step_190013')
    p.add_argument('--ppo_checkpoint', type=str, default='./ckpt/residual_rl_snapshot_ppo_no_force_history/final')
    p.add_argument('--td3_checkpoint', type=str, default='./ckpt/residual_rl_td3/final')
    # 输出目录、是否启用可视化
    p.add_argument('--output_dir', type=str, default='./eval_results')
    p.add_argument('--no_viewer', action='store_true')
    return p.parse_args()


# ======================================================================
# Helpers
# ======================================================================
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
    features = dataset_to_policy_features(meta.features)    # 将数据集原始特征转换为策略模型所需的特征格式
    output_features = {k: f for k, f in features.items() if f.type is FeatureType.ACTION}   # 所有类型为 FeatureType.ACTION的特征，即策略需要预测的动作
    input_features = {k: f for k, f in features.items() if k not in output_features}        # 除动作特征外的所有其他特征，将作为策略的输入（如状态、图像等）
    return meta, input_features, output_features

# 根据给定的键列表 allowed_keys，过滤输入特征字典. 不同的预训练模型可能只使用了数据集特征的一个子集。在加载模型时，需要根据模型配置文件（config.json）中记录的 input_features键来筛选特征，确保输入与模型期望的结构完全一致
def filter_input_features(input_features, allowed_keys):
    return {k: v for k, v in input_features.items() if k in allowed_keys}


def build_base_policy_obs(env, device, input_feature_keys, step):
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
    obs['timestamp'] = torch.tensor([step / 20.0]).to(device)
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


# ======================================================================
# Controllers
# ======================================================================
class ActController:
    name = 'act'
    base_only = True

    def __init__(self, args, device):
        # 加载数据集特征
        meta, input_features, output_features = load_dataset_features(args.repo_name, args.dataset_root)
        # 读取模型配置
        with open(os.path.join(args.act_pretrained, 'config.json'), 'r', encoding='utf-8') as f:
            cfg_json = json.load(f)
        input_features = filter_input_features(input_features, cfg_json['input_features'].keys())
        # 构建ACT配置对象
        cfg = ACTConfig(
            input_features=input_features,
            output_features=output_features,
            chunk_size=cfg_json['chunk_size'],              # 模型一次处理的时间步数
            n_action_steps=cfg_json['n_action_steps'],      # 每个预测生成的动作步数
            temporal_ensemble_coeff=cfg_json.get('temporal_ensemble_coeff'), # 时间集成系数（可选），用于平滑连续预测
        )
        resolve_delta_timestamps(cfg, meta)                 # 解析时间戳差异，确保时间特征与数据集的时间特性对齐
        # 加载预训练模型
        self.policy = ACTPolicy.from_pretrained(            # 加载预训练模型
            args.act_pretrained, config=cfg, dataset_stats=meta.stats,
        ).to(device).eval()
        # 存储关键属性
        self.device = device                                # 存储设备信息用于后续张量传输
        self.input_feature_keys = set(input_features.keys())# 存储输入特征键集合，用于在act方法中动态构建观测

    def reset(self):
        if hasattr(self.policy, 'reset'):
            self.policy.reset()
    # 动作生成
    def act(self, env, step, residual_state=None):
        data = build_base_policy_obs(env, self.device, self.input_feature_keys, step) # 构建观测字典
        with torch.no_grad():  # torch.no_grad(): 上下文管理器，禁用梯度计算
            action = self.policy.select_action(data)
        action_np = action[0, :7].cpu().numpy().astype(np.float32) # action[0, :7]: 取批次中的第一个样本，前7个维度；.cpu().numpy(): 从GPU张量移动到CPU并转为NumPy数组；.astype(np.float32): 确保数据类型
        action_np[6] = 200.0
        return action_np, 0.0, 0.0


class SmolVLAController:
    name = 'smolvla'
    base_only = True

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
            args.smolvla_pretrained, config=cfg, dataset_stats=meta.stats,
        ).to(device).eval()
        self.device = device
        self.input_feature_keys = set(input_features.keys())

    def reset(self):
        if hasattr(self.policy, 'reset'):
            self.policy.reset()

    def act(self, env, step, residual_state=None):
        data = build_base_policy_obs(env, self.device, self.input_feature_keys, step)
        with torch.no_grad():
            action = self.policy.select_action(data)
        action_np = action[0, :7].cpu().numpy().astype(np.float32)
        action_np[6] = 200.0
        return action_np, 0.0, 0.0


class SmolVLAResidualSACController:
    name = 'smolvla+sac'    # 标识控制器名称
    base_only = False

    def __init__(self, args, device):
        # 创建基座策略
        self.base = SmolVLAController(args, device)
        # 加载SAC检查点
        ckpt = torch.load(
            os.path.join(args.sac_checkpoint, 'sac_checkpoint.pt'),
            map_location=device, weights_only=False,
        )
        # 配置和Actor网络初始化
        self.cfg = ckpt.get('config', ResidualRLConfig())   # 配置获取
        self.actor = ResidualActor(self.cfg).to(device)     # 创建和加载Actor网络：ResidualActor(self.cfg)：根据配置创建SAC的Actor网络；.to(device)：移动到指定设备；
        self.actor.load_state_dict(ckpt['actor'])           # 加载训练好的权重
        self.actor.eval()                                   # 设置为评估模式，关闭Dropout、BatchNorm统计更新等
        self.device = device

    def reset(self):
        self.base.reset()   # 仅调用基座策略的reset方法

    def act(self, env, step, residual_state):
        # 获取基座动作
        base_action, _, _ = self.base.act(env, step)
        # 接触判断，只在接触发生时才激活残差策略
        if not residual_state['contact']:
            return base_action, 0.0, 0.0
        # 构建SAC观测
        state_t = torch.FloatTensor(residual_state['state']).unsqueeze(0).to(self.device)       # 状态向量转换：residual_state['state']：来自build_residual_obs的17维状态向量；.unsqueeze(0)：增加批次维度，从[17]变为[1, 17]
        hist_t = torch.FloatTensor(residual_state['ft_history']).unsqueeze(0).to(self.device)   # 力/力矩历史转换
        # SAC Actor推理
        with torch.no_grad():
            cart = self.actor.deterministic_action(state_t, hist_t).squeeze(0).cpu().numpy()    # 评估时使用确定性动作（均值）；.squeeze(0)：移除批次维度，从[1, 6]变为[6]
        # 逆运动学转换
        dq = cartesian_to_joint_delta(env, cart, self.cfg)
        final_action = base_action.copy()
        final_action[:6] += dq
        return final_action, float(np.linalg.norm(cart)), float(np.linalg.norm(dq))


class SmolVLAResidualPPOController:
    name = 'smolvla+ppo'
    base_only = False

    def __init__(self, args, device):
        self.base = SmolVLAController(args, device)
        ckpt = torch.load(
            os.path.join(args.ppo_checkpoint, 'ppo_checkpoint.pt'),
            map_location=device, weights_only=False,
        )
        self.cfg = ckpt.get('config', ResidualPPOConfig())
        self.trainer = PPOTrainer(config=self.cfg, device=str(device))
        self.trainer.load(args.ppo_checkpoint)

    def reset(self):
        self.base.reset()

    def act(self, env, step, residual_state):
        base_action, _, _ = self.base.act(env, step)
        if not residual_state['contact']:
            return base_action, 0.0, 0.0
        # PPO的观测构建和推理都移动到了PPOTrainer类的 deterministic_action方法内部
        cart = self.trainer.deterministic_action(
            residual_state['state'], residual_state['ft_history']
        )
        dq = cartesian_to_joint_delta(env, cart, self.cfg)
        final_action = base_action.copy()
        final_action[:6] += dq
        return final_action, float(np.linalg.norm(cart)), float(np.linalg.norm(dq))


class SmolVLAResidualTD3Controller:
    name = 'smolvla+td3'
    base_only = False

    def __init__(self, args, device):
        self.base = SmolVLAController(args, device)
        ckpt = torch.load(
            os.path.join(args.td3_checkpoint, 'td3_checkpoint.pt'),
            map_location=device, weights_only=False,
        )
        self.cfg = ckpt.get('config', ResidualTD3Config())
        self.trainer = TD3Trainer(config=self.cfg, device=str(device))
        self.trainer.load(args.td3_checkpoint)

    def reset(self):
        self.base.reset()

    def act(self, env, step, residual_state):
        base_action, _, _ = self.base.act(env, step)
        if not residual_state['contact']:
            return base_action, 0.0, 0.0
        cart = self.trainer.select_action(
            residual_state['state'], residual_state['ft_history'], deterministic=True,
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
        p = os.path.join(args.sac_checkpoint, 'sac_checkpoint.pt')
        if not os.path.isfile(p):
            return None, f'SAC checkpoint not found: {p}'
        return SmolVLAResidualSACController(args, device), None
    if method == 'smolvla+ppo':
        p = os.path.join(args.ppo_checkpoint, 'ppo_checkpoint.pt')
        if not os.path.isfile(p):
            return None, f'PPO checkpoint not found: {p}'
        return SmolVLAResidualPPOController(args, device), None
    if method == 'smolvla+td3':
        p = os.path.join(args.td3_checkpoint, 'td3_checkpoint.pt')
        if not os.path.isfile(p):
            return None, f'TD3 checkpoint not found: {p}'
        return SmolVLAResidualTD3Controller(args, device), None
    return None, f'Unknown method: {method}'


# ======================================================================
# Episode rollout
# ======================================================================
def run_episode(pih, controller, method, device, max_steps, seed, instruction, render):
    """Run one episode; return per-step arrays + summary dict."""
    pih.reset(seed=seed)
    pih.set_instruction(instruction)
    controller.reset()

    # Zero-calibration: after reset's 100-step settle, use current F/T as offset.
    ft_offset = pih.get_force_torque().copy()

    # Initial F/T history for residual controllers
    current_ft = pih.get_force_torque() - ft_offset
    ft_history = [current_ft.copy() for _ in range(FT_HISTORY_LEN)]

    records = {
        'step': [], 'fx': [], 'fy': [], 'fz': [],
        'tx': [], 'ty': [], 'tz': [],
        'cart_norm': [], 'dq_norm': [], 'contact': [],
    }
    step = 0
    success = False
    success_step = -1
    contact_started = False
    contact_start_step = -1

    sum_cart_norm = 0.0
    sum_dq_norm = 0.0
    residual_steps = 0

    t0 = time.time()

    while True:
        pih.step_env()
        if render and not pih.env.is_viewer_alive():
            break
        if not pih.env.loop_every(HZ=20):
            continue

        ft_cal = (pih.get_force_torque() - ft_offset).astype(np.float32)
        records['step'].append(step)
        records['fx'].append(float(ft_cal[0]))
        records['fy'].append(float(ft_cal[1]))
        records['fz'].append(float(ft_cal[2]))
        records['tx'].append(float(ft_cal[3]))
        records['ty'].append(float(ft_cal[4]))
        records['tz'].append(float(ft_cal[5]))

        contact = check_contact(pih)
        records['contact'].append(int(contact))
        if contact and not contact_started:
            contact_started = True
            contact_start_step = step

        # Terminate BEFORE stepping, matching evaluate_act.py
        if pih.check_success():
            success = True
            success_step = step
            records['cart_norm'].append(0.0)
            records['dq_norm'].append(0.0)
            break
        if step >= max_steps:
            records['cart_norm'].append(0.0)
            records['dq_norm'].append(0.0)
            break

        # Update F/T history for residual controllers
        ft_history.append(ft_cal.copy())
        ft_history = ft_history[-FT_HISTORY_LEN:]

        residual_state = {
            'contact': contact,
            'state': None,
            'ft_history': None,
        }
        if not controller.base_only and contact:
            state_vec, ft_hist = build_residual_obs(pih, ft_offset, ft_history)
            residual_state['state'] = state_vec
            residual_state['ft_history'] = ft_hist

        if controller.base_only:
            action, cart_norm, dq_norm = controller.act(pih, step)
        else:
            action, cart_norm, dq_norm = controller.act(pih, step, residual_state)
            if contact:
                sum_cart_norm += cart_norm
                sum_dq_norm += dq_norm
                residual_steps += 1

        records['cart_norm'].append(float(cart_norm))
        records['dq_norm'].append(float(dq_norm))

        pih.step(action)
        if render:
            pih.render(idx=seed)
        step += 1

    elapsed = time.time() - t0

    # Arrays
    for k in records:
        records[k] = np.asarray(records[k], dtype=np.float32)

    # Summary
    fn = np.linalg.norm(np.stack([records['fx'], records['fy'], records['fz']], axis=1), axis=1) \
        if len(records['step']) > 0 else np.zeros(0)
    tn = np.linalg.norm(np.stack([records['tx'], records['ty'], records['tz']], axis=1), axis=1) \
        if len(records['step']) > 0 else np.zeros(0)

    contact_mask = records['contact'].astype(bool)
    def _mean_or_zero(arr):
        return float(arr.mean()) if arr.size else 0.0

    summary = {
        'seed': seed,
        'success': int(success),
        'success_step': success_step,
        'steps_taken': step,
        'elapsed_s': elapsed,
        'duration_sec': float(step / 20.0),
        'contact_start_step': contact_start_step,
        'contact_start_sec': float(contact_start_step / 20.0) if contact_start_step >= 0 else -1.0,
        'contact_steps': int(contact_mask.sum()),
        'peak_force_norm': float(fn.max()) if fn.size else 0.0,
        'peak_torque_norm': float(tn.max()) if tn.size else 0.0,
        'mean_force_norm_all': _mean_or_zero(fn),
        'mean_torque_norm_all': _mean_or_zero(tn),
        'mean_force_norm_contact': _mean_or_zero(fn[contact_mask]) if fn.size else 0.0,
        'mean_torque_norm_contact': _mean_or_zero(tn[contact_mask]) if tn.size else 0.0,
        'mean_cart_residual_norm': float(sum_cart_norm / residual_steps) if residual_steps else 0.0,
        'mean_joint_delta_norm': float(sum_dq_norm / residual_steps) if residual_steps else 0.0,
    }
    return records, summary


# ======================================================================
# CSV writers + plotting
# ======================================================================
def write_perstep_csv(path, all_records):
    """all_records: list of (episode_idx, records_dict)."""
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['episode', 'step', 'fx', 'fy', 'fz',
                    'tx', 'ty', 'tz', 'contact',
                    'cart_residual_norm', 'joint_delta_norm'])
        for ep_idx, rec in all_records:
            for i in range(len(rec['step'])):
                w.writerow([
                    ep_idx, int(rec['step'][i]),
                    f"{rec['fx'][i]:.6f}", f"{rec['fy'][i]:.6f}", f"{rec['fz'][i]:.6f}",
                    f"{rec['tx'][i]:.6f}", f"{rec['ty'][i]:.6f}", f"{rec['tz'][i]:.6f}",
                    int(rec['contact'][i]),
                    f"{rec['cart_norm'][i]:.6f}", f"{rec['dq_norm'][i]:.6f}",
                ])
    print(f"  [csv] per-step -> {path}")


def write_episode_summary_csv(path, summaries, success_rate):
    fields = [
        'episode', 'seed', 'success', 'success_step', 'steps_taken',
        'elapsed_s', 'duration_sec',
        'contact_start_step', 'contact_start_sec', 'contact_steps',
        'peak_force_norm', 'peak_torque_norm',
        'mean_force_norm_all', 'mean_torque_norm_all',
        'mean_force_norm_contact', 'mean_torque_norm_contact',
        'mean_cart_residual_norm', 'mean_joint_delta_norm',
    ]
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(fields)
        for i, s in enumerate(summaries):
            w.writerow([i] + [s[k] if not isinstance(s[k], float) else f"{s[k]:.6f}"
                              for k in fields[1:]])
        w.writerow([])
        n = len(summaries)
        succ = sum(s['success'] for s in summaries)
        w.writerow(['success_rate', f"{success_rate:.4f}", f"{succ}/{n}"])
        if succ > 0:
            avg_steps = np.mean([s['success_step'] for s in summaries if s['success']])
            avg_time  = np.mean([s['elapsed_s']   for s in summaries if s['success']])
            w.writerow(['mean_success_step', f"{avg_steps:.2f}"])
            w.writerow(['mean_success_time_s', f"{avg_time:.2f}"])
    print(f"  [csv] summary  -> {path}")


def plot_per_axis(method, all_records, out_dir):
    """Overlay all episodes per F/T axis; bold line = mean on common step grid."""
    out_dir = Path(out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    channels = [
        ('fx', 'Fx [N]',  'force_Fx_vs_step.png'),
        ('fy', 'Fy [N]',  'force_Fy_vs_step.png'),
        ('fz', 'Fz [N]',  'force_Fz_vs_step.png'),
        ('tx', 'Tx [Nm]', 'torque_Tx_vs_step.png'),
        ('ty', 'Ty [Nm]', 'torque_Ty_vs_step.png'),
        ('tz', 'Tz [Nm]', 'torque_Tz_vs_step.png'),
    ]
    max_len = max(len(rec['step']) for _, rec in all_records) if all_records else 0

    for key, ylabel, fname in channels:
        fig, ax = plt.subplots(figsize=(10, 4.5))
        stacked = np.full((len(all_records), max_len), np.nan, dtype=np.float32)
        for row, (ep, rec) in enumerate(all_records):
            n = len(rec[key])
            stacked[row, :n] = rec[key]
            ax.plot(rec['step'], rec[key], alpha=0.35, linewidth=0.9,
                    label=f'ep{ep}' if len(all_records) <= 10 else None)
        if max_len > 0:
            mean = np.nanmean(stacked, axis=0)
            x = np.arange(max_len)
            ax.plot(x, mean, color='black', linewidth=2.0, label='mean')

        ax.set_xlabel('step (20 Hz)')
        ax.set_ylabel(ylabel)
        ax.set_title(f'{method}: {ylabel} vs step (zero-calibrated)')
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
        print(f"  [plot] {ylabel:8s} -> {out_path}")


# ======================================================================
# Per-method evaluation
# ======================================================================
def evaluate_method(method, controller, args, device, method_out_dir):
    render = not args.no_viewer

    print(f"[env] creating PIHEnv2 from {args.xml_path}")
    pih = PIHEnv2(
        xml_path=args.xml_path,
        action_type='joint_angle',
        state_type='joint_angle',
    )
    pih.set_instruction(args.instruction)

    all_records = []
    summaries = []

    t_all = time.time()
    for ep in range(args.episodes):
        seed = args.base_seed + ep
        print(f"\n--- [{method}] Episode {ep+1}/{args.episodes}  (seed={seed}) ---")
        rec, summ = run_episode(
            pih, controller, method, str(device),
            max_steps=args.max_steps, seed=seed,
            instruction=args.instruction, render=render,
        )
        tag = 'SUCCESS' if summ['success'] else 'FAIL'
        print(f"  [{tag}] success_step={summ['success_step']} "
              f"steps={summ['steps_taken']} time={summ['elapsed_s']:.2f}s "
              f"peakF={summ['peak_force_norm']:.2f}N peakT={summ['peak_torque_norm']:.3f}Nm")
        summaries.append(summ)
        all_records.append((ep, rec))

        if render and not pih.env.is_viewer_alive():
            print("  viewer closed; stopping early")
            break

    # Stats
    n = len(summaries)
    succ = sum(s['success'] for s in summaries)
    success_rate = succ / n if n > 0 else 0.0
    print(f"\n=== {method} summary ===")
    print(f"  episodes evaluated : {n}")
    print(f"  success rate       : {success_rate*100:.1f}%  ({succ}/{n})")
    if succ > 0:
        sst = [s['success_step'] for s in summaries if s['success']]
        sts = [s['elapsed_s']   for s in summaries if s['success']]
        print(f"  mean success step  : {np.mean(sst):.2f}")
        print(f"  mean success time  : {np.mean(sts):.2f}s")
    print(f"  total wall time    : {time.time()-t_all:.1f}s")

    # Write outputs
    ensure_dir(method_out_dir)
    write_perstep_csv(os.path.join(method_out_dir, 'perstep_force_torque.csv'), all_records)
    write_episode_summary_csv(os.path.join(method_out_dir, 'episode_summary.csv'),
                              summaries, success_rate)
    plot_per_axis(method, all_records, method_out_dir)

    pih.env.close_viewer()

    # Comparative summary row
    contact_rows = [s for s in summaries if s['contact_steps'] > 0]
    succ_rows    = [s for s in summaries if s['success']]
    def _mean(key, rows):
        return float(np.mean([r[key] for r in rows])) if rows else 0.0
    return {
        'method': method,
        'episodes': n,
        'success_rate': success_rate,
        'successes': succ,
        'avg_episode_time_sec': _mean('elapsed_s', summaries),
        'avg_success_time_sec': _mean('elapsed_s', succ_rows),
        'avg_steps': _mean('steps_taken', summaries),
        'avg_success_steps': _mean('steps_taken', succ_rows),
        'avg_contact_start_sec': _mean('contact_start_sec', contact_rows),
        'avg_mean_force_norm_contact_N':   _mean('mean_force_norm_contact', contact_rows),
        'avg_mean_torque_norm_contact_Nm': _mean('mean_torque_norm_contact', contact_rows),
        'avg_peak_force_norm_N':   _mean('peak_force_norm', summaries),
        'avg_peak_torque_norm_Nm': _mean('peak_torque_norm', summaries),
        'avg_cart_residual_norm':  _mean('mean_cart_residual_norm', summaries),
        'avg_joint_delta_norm':    _mean('mean_joint_delta_norm', summaries),
    }


# ======================================================================
# Main
# ======================================================================
def main():
    args = parse_args()
    device = torch.device(args.device)
    ensure_dir(args.output_dir)

    summary_rows = []
    for method in args.methods:
        controller, error = make_controller(method, args, device)
        if error:
            print(f"[skip] {method}: {error}")
            continue

        method_dir = os.path.join(args.output_dir, method.replace('+', '_'))
        print(f"\n================ Evaluating: {method} ================")
        print(f"[out] writing to {method_dir}")
        summary = evaluate_method(method, controller, args, device, method_dir)
        summary_rows.append(summary)

        # free policy between methods
        del controller
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Comparative summary
    summary_csv  = os.path.join(args.output_dir, 'summary_metrics.csv')
    summary_json = os.path.join(args.output_dir, 'summary_metrics.json')
    if summary_rows:
        with open(summary_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
            w.writeheader()
            for r in summary_rows:
                w.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                            for k, v in r.items()})
        with open(summary_json, 'w', encoding='utf-8') as f:
            json.dump(summary_rows, f, indent=2, ensure_ascii=False)
        print('\n================ Cross-method summary ================')
        for r in summary_rows:
            print(f"  {r['method']:>12s} | succ={r['success_rate']*100:5.1f}% "
                  f"({r['successes']}/{r['episodes']}) | "
                  f"succ_time={r['avg_success_time_sec']:.2f}s "
                  f"succ_steps={r['avg_success_steps']:.1f} | "
                  f"contactF={r['avg_mean_force_norm_contact_N']:.2f}N "
                  f"contactT={r['avg_mean_torque_norm_contact_Nm']:.3f}Nm | "
                  f"peakF={r['avg_peak_force_norm_N']:.1f}N "
                  f"peakT={r['avg_peak_torque_norm_Nm']:.3f}Nm")
        print(f"\nSaved comparative summary: {summary_csv}")
        print(f"Saved comparative summary: {summary_json}")
    else:
        print("\n[warn] no methods evaluated; nothing to summarize.")


if __name__ == '__main__':
    main()
