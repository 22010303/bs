"""
train_residual_rl_snapshot_all.py — Snapshot-based residual RL trainer that
supports SAC / PPO / TD3 in a single script.

Built on train_residual_rl_snapshot.py (same snapshot reset, SmolVLA base
policy, hierarchical reward, contact-gated residual, excessive-force /
out-of-hole / timeout early stops, wandb + CSV logging, periodic saves,
per-step viewer overlays).

Choose the algorithm with `--algo sac|ppo|td3`.

Usage
-----
  # Step 1: build the snapshot once (if you don't have one yet)
  python create_snapshot.py --snapshot_step 315

  # Step 2: train with any of the three algorithms
  python train_residual_rl_snapshot_all.py --algo sac \\
        --snapshot ./ckpt/residual_rl/snapshot_step315.pkl
  python train_residual_rl_snapshot_all.py --algo ppo \\
        --snapshot ./ckpt/residual_rl/snapshot_step315.pkl
  python train_residual_rl_snapshot_all.py --algo td3 \\
        --snapshot ./ckpt/residual_rl/snapshot_step315.pkl

  # With wandb
  python train_residual_rl_snapshot_all.py --algo ppo --wandb --snapshot ./ckpt/residual_rl/snapshot_step315.pkl
"""

import argparse
import os
from collections import deque

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from mujoco_env.pih_env2 import PIHEnv2
from residual_rl.reward import HierarchicalReward
from residual_rl.snapshot import load_snapshot, restore_snapshot, perturb_ee_pose

# SAC
from residual_rl.config import ResidualRLConfig
from residual_rl.sac_trainer import SACTrainer

# PPO
from residual_rl_ppo.config import ResidualPPOConfig
from residual_rl_ppo.ppo_trainer import PPOTrainer

# TD3
from residual_rl_td3.config import ResidualTD3Config
from residual_rl_td3.td3_trainer import TD3Trainer

#图像转换：用于SmolVLA模型
IMG_TRANSFORM = transforms.Compose([transforms.ToTensor()])

#关键站点：定义任务中的关键位置（插头尖端、孔入口、孔底部）
PEG_TIP_SITE     = 'peg_tip_site'
HOLE_ENTRY_SITE  = 'hole_entry_site'
HOLE_BOTTOM_SITE = 'hole_bottom_site'

ARM_JOINTS = [
    'shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
    'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint',
]
EEF_BODY = 'tool0_link'


# ======================================================================
# SmolVLA helpers (identical to train_residual_rl_snapshot.py)
# ======================================================================
def load_smolvla(cfg):
    # 加载预训练模型
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.common.datasets.utils import dataset_to_policy_features
    from lerobot.common.datasets.factory import resolve_delta_timestamps
    from lerobot.common.policies.smolvla.configuration_smolvla import SmolVLAConfig
    from lerobot.common.policies.smolvla.modeling_smolvla import SmolVLAPolicy
    from lerobot.configs.types import FeatureType

    print(f"[smolvla] loading from {cfg.smolvla_pretrained}")
    meta = LeRobotDatasetMetadata(cfg.smolvla_repo_name, root=cfg.smolvla_dataset_root)
    feats = dataset_to_policy_features(meta.features)
    out_feats = {k: f for k, f in feats.items() if f.type is FeatureType.ACTION}
    in_feats  = {k: f for k, f in feats.items() if k not in out_feats}

    smol_cfg = SmolVLAConfig(
        input_features=in_feats, output_features=out_feats,
        chunk_size=cfg.smolvla_chunk_size, n_action_steps=cfg.smolvla_n_action_steps,
    )
    resolve_delta_timestamps(smol_cfg, meta)

    policy = SmolVLAPolicy.from_pretrained(
        cfg.smolvla_pretrained, config=smol_cfg, dataset_stats=meta.stats,
    )
    policy.to(cfg.smolvla_device).eval()
    for p in policy.parameters():
        p.requires_grad = False
    print(f"[smolvla] {sum(p.numel() for p in policy.parameters())/1e6:.1f}M params, frozen")
    return policy


def get_smolvla_action(policy, pih, device):
    # 获取状态和图像
    state = pih.get_joint_state()[:6]
    a_img, w_img = pih.grab_image()
    # 图像预处理，缩放到256×256并转换为张量
    a = IMG_TRANSFORM(Image.fromarray(a_img).resize((256, 256)))
    w = IMG_TRANSFORM(Image.fromarray(w_img).resize((256, 256)))
    # 构建输入数据
    data = {
        'observation.state': torch.from_numpy(
            np.array([state], dtype=np.float32)).to(device),
        'observation.image': a.unsqueeze(0).to(device),
        'observation.wrist_image': w.unsqueeze(0).to(device),
        'task': [pih.instruction],
    }
    # 推理获取动作
    with torch.no_grad():
        act = policy.select_action(data)
    act_np = act[0, :7].cpu().numpy()
    act_np[6] = 200.0
    return act_np


# ======================================================================
# Geometry / observation helpers
# ======================================================================
def check_contact(pih, ft_offset):
    pt = pih.env.get_p_site(PEG_TIP_SITE)
    he = pih.env.get_p_site(HOLE_ENTRY_SITE)
    xy = np.linalg.norm(pt[:2] - he[:2])
    return (pt[2] < he[2]) and (xy < 0.003)     # z低于孔口且xy距离<3mm


def build_rl_obs(pih, ft_offset, ft_history):
    ft = pih.get_force_torque() - ft_offset     # 校准后的力/力矩
    ee = pih.get_ee_pose()
    pt = pih.env.get_p_site(PEG_TIP_SITE)
    he = pih.env.get_p_site(HOLE_ENTRY_SITE)
    rel = (he - pt).astype(np.float32)          # 相对位置
    # 当前状态：[力/力矩(6), 末端位姿(6), 相对位置(3)]= 15维
    # 历史信息：最近N步的力 / 力矩序列
    return np.concatenate([ft, ee, rel], dtype=np.float32), \
           np.array(list(ft_history), dtype=np.float32)


def get_cart_limits(cfg):
    return np.array([
        cfg.max_residual_pos, cfg.max_residual_pos, cfg.max_residual_pos,
        cfg.max_residual_rot, cfg.max_residual_rot, cfg.max_residual_rot,
    ], dtype=np.float32)

#  笛卡尔到关节转换
def cart_to_joint_delta(pih, cart, cfg):
    # 计算雅可比矩阵
    _, _, J_full = pih.env.get_J_body(EEF_BODY)     # 获取末端执行器（tool0_link）相对于世界坐标的完整几何雅可比矩阵 j_full
    J = J_full[:, pih.env.get_idxs_jac(ARM_JOINTS)] # 获取机械臂6个关节在雅可比矩阵中对应的列索引，从而得到只关乎机械臂运动的雅可比矩阵 j_arm（形状为 [6, 6]）
    # 伪逆解算
    JJT = J @ J.T + cfg.ik_damping_eps * np.eye(6)  # 计算 JJ^T并加上一个很小的阻尼项（ik_damping_eps * I）。这是为了在雅可比矩阵接近奇异（机器人处于奇异位形）时，求逆仍然数值稳定
    J_pinv = J.T @ np.linalg.solve(JJT, np.eye(6))  # 通过求解线性方程组 (JJ^T+λI)x=I来高效计算阻尼伪逆 J^+=J^T(JJ^T+λI)^−1
    # 关节增量
    # 映射到关节空间：joint_delta = j_pinv @ cart_residual。这正是逆运动学的核心公式：Δq=J^+⋅Δx，将笛卡尔空间的变化映射为关节角度的变化。
    # 限幅与返回：np.clip(joint_delta, -cfg.max_joint_delta, cfg.max_joint_delta)对关节增量进行限幅，防止单步变化过大导致不稳定，最后转换为 float32类型返回。
    dq = np.clip(J_pinv @ cart, -cfg.max_joint_delta, cfg.max_joint_delta)
    return dq.astype(np.float32)


# ======================================================================
# Algorithm adapters — a common interface around the three trainers.
# ======================================================================
class AlgoAdapter:
    """Base class — see per-algo subclasses for implementations."""
    name = ''
    checkpoint_filename = ''
    uses_warmup = False
    has_alpha = False

    @property
    def total_steps(self):
        return self.trainer.total_steps

    @total_steps.setter
    def total_steps(self, v):
        self.trainer.total_steps = v

    @property
    def alpha(self):
        return 0.0

    def save(self, path):
        self.trainer.save(path)

    def load(self, path):
        self.trainer.load(path)

    # Subclasses must implement:
    def act(self, state, ft_hist, in_warmup): ...
    def store_step(self, state, ft_hist, action, extras,
                   reward, next_state, next_ft_hist, done): ...
    def step_updates(self): ...
    def on_episode_end(self, last_state, last_ft_hist, last_done): ...
    def log_episode(self, ep_reward, ep_len, ep_info, reward_info): ...


class SACAdapter(AlgoAdapter):
    name = 'sac'
    checkpoint_filename = 'sac_checkpoint.pt'
    uses_warmup = True
    has_alpha = True

    def __init__(self, cfg: ResidualRLConfig, device, use_wandb):
        self.cfg = cfg
        self.trainer = SACTrainer(config=cfg, device=device, use_wandb=use_wandb)
        self.device = self.trainer.device

    @property
    def alpha(self):
        return self.trainer.alpha

    def act(self, state, ft_hist, in_warmup):
        limits = get_cart_limits(self.cfg)
        if in_warmup:   # 预热阶段：随机动作
            cart = np.random.uniform(-limits, limits).astype(np.float32)
        else:   # 正常阶段：策略网络
            s_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            h_t = torch.FloatTensor(ft_hist).unsqueeze(0).to(self.device)
            with torch.no_grad():
                a, _ = self.trainer.actor.sample(s_t, h_t)
                cart = a.squeeze(0).cpu().numpy()
            cart = np.clip(cart, -limits, limits).astype(np.float32)
        return cart, {}

    def store_step(self, state, ft_hist, action, extras, reward, next_state, next_ft_hist, done):
        # 存储经验到回放缓冲区
        self.trainer.store_transition(
            state=state, ft_history=ft_hist,
            action=action, reward=reward,
            next_state=next_state, next_ft_history=next_ft_hist,
            done=done,
        )
        self.trainer.total_steps += 1

    def step_updates(self):
        if self.trainer.total_steps >= self.cfg.warmup_steps and \
                self.trainer.total_steps % self.cfg.update_every == 0:
            for _ in range(self.cfg.updates_per_step):
                self.trainer.update()

    def on_episode_end(self, last_state, last_ft_hist, last_done):
        pass

    def log_episode(self, ep_reward, ep_len, ep_info, reward_info):
        self.trainer.log_episode(ep_reward, ep_len, ep_info,
                                 reward_info=reward_info)


class TD3Adapter(AlgoAdapter):
    name = 'td3'
    checkpoint_filename = 'td3_checkpoint.pt'
    uses_warmup = True
    has_alpha = False

    def __init__(self, cfg: ResidualTD3Config, device, use_wandb):
        self.cfg = cfg
        self.trainer = TD3Trainer(config=cfg, device=device, use_wandb=use_wandb)
        self.device = self.trainer.device

    def act(self, state, ft_hist, in_warmup):
        if in_warmup:   # 随机动作
            limit = self.trainer.action_limit.cpu().numpy()
            cart = np.random.uniform(-limit, limit).astype(np.float32)
        else:   # 策略网络 + 探索噪声
            cart = self.trainer.select_action(state, ft_hist, deterministic=False)
        return cart, {}

    def store_step(self, state, ft_hist, action, extras,
                   reward, next_state, next_ft_hist, done):
        self.trainer.store_transition(
            state=state, ft_history=ft_hist,
            action=action, reward=reward,
            next_state=next_state, next_ft_history=next_ft_hist,
            done=done,
        )
        self.trainer.total_steps += 1

    def step_updates(self):
        if self.trainer.total_steps >= self.cfg.warmup_steps and \
                self.trainer.total_steps % self.cfg.update_every == 0:
            for _ in range(self.cfg.updates_per_step):
                self.trainer.update()

    def on_episode_end(self, last_state, last_ft_hist, last_done):
        pass

    def log_episode(self, ep_reward, ep_len, ep_info, reward_info):
        if not self.trainer.use_wandb:
            return
        import wandb
        log_dict = {
            'episode/reward': ep_reward,
            'episode/length': ep_len,
            'episode/success': float(ep_info.get('success', False)),
            'episode/contact_step': ep_info.get('contact_step', -1),
            'episode/rl_steps': ep_info.get('steps_after_contact', 0),
        }
        if reward_info:
            for k in ['total_force', 'lateral_force', 'axial_force',
                      'bending_torque', 'insertion_progress', 'cumulative_force']:
                if k in reward_info:
                    log_dict[f'episode/{k}'] = reward_info[k]
            for k in ['r_force_mag', 'r_stability', 'r_depth', 'r_progress',
                      'p_fx', 'p_fy', 'p_fz', 'p_tx', 'p_ty']:
                if k in reward_info:
                    log_dict[f'reward/{k}'] = reward_info[k]
        wandb.log(log_dict, step=self.trainer.total_steps)


class PPOAdapter(AlgoAdapter):
    name = 'ppo'
    checkpoint_filename = 'ppo_checkpoint.pt'
    uses_warmup = False
    has_alpha = False

    def __init__(self, cfg: ResidualPPOConfig, device, use_wandb):
        self.cfg = cfg
        self.trainer = PPOTrainer(config=cfg, device=device, use_wandb=use_wandb)
        self.device = self.trainer.device
        # Most recent (next_state, next_ft_hist, done) from the last stored
        # transition — used for GAE bootstrap when the rollout buffer fills up.
        self._last_next = (None, None, False)   # 保存最后一个下一个状态

    def act(self, state, ft_hist, in_warmup):
        # PPO无预热，直接使用策略
        action, log_prob, value = self.trainer.act(state, ft_hist)
        return action, {'log_prob': log_prob, 'value': value}

    def store_step(self, state, ft_hist, action, extras, reward, next_state, next_ft_hist, done):
        # 存储到rollout缓冲区
        self.trainer.store_transition(
            state=state,
            ft_history=ft_hist,
            action=action,
            log_prob=extras['log_prob'],
            reward=reward * self.cfg.reward_scale,  # 奖励缩放
            done=done,
            value=extras['value'],
        )
        self.trainer.total_steps += 1
        self._last_next = (next_state.copy(), next_ft_hist.copy(), bool(done))

    def step_updates(self):
        # 缓冲区满时更新
        if self.trainer.buffer.full:
            ns, nh, nd = self._last_next
            if ns is None:
                ns = np.zeros(self.cfg.state_dim, dtype=np.float32)
                nh = np.zeros((self.cfg.force_history_len, self.cfg.ft_dim),
                              dtype=np.float32)
                nd = False
            self.trainer.finish_rollout(ns, nh, nd) # 计算GAE
            self.trainer.update()

    def on_episode_end(self, last_state, last_ft_hist, last_done):
        # 处理回合结束时的剩余数据
        if self.trainer.buffer.size > 0:
            self.trainer.finish_rollout(last_state, last_ft_hist, last_done)
            self.trainer.update()
            self._last_next = (None, None, False)

    def log_episode(self, ep_reward, ep_len, ep_info, reward_info):
        if not self.trainer.use_wandb:
            return
        import wandb
        log_dict = {
            'episode/reward': ep_reward,
            'episode/length': ep_len,
            'episode/success': float(ep_info.get('success', False)),
            'episode/contact_step': ep_info.get('contact_step', -1),
            'episode/rl_steps': ep_info.get('steps_after_contact', 0),
        }
        if reward_info:
            for k in ['total_force', 'lateral_force', 'axial_force',
                      'bending_torque', 'insertion_progress', 'cumulative_force']:
                if k in reward_info:
                    log_dict[f'episode/{k}'] = reward_info[k]
        wandb.log(log_dict, step=self.trainer.total_steps)

# 算法注册表：算法名 → (适配器类, 配置类, 默认输出目录)
ALGO_REGISTRY = {
    'sac': (SACAdapter, ResidualRLConfig,  './ckpt/residual_rl_snapshot'),
    'ppo': (PPOAdapter, ResidualPPOConfig, './ckpt/residual_rl_snapshot_ppo'),
    'td3': (TD3Adapter, ResidualTD3Config, './ckpt/residual_rl_snapshot_td3'),
}


# ======================================================================
# CLI
# ======================================================================
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--algo', type=str, default='sac',
                   choices=list(ALGO_REGISTRY.keys()),
                   help='RL algorithm: sac | ppo | td3')
    p.add_argument('--snapshot', type=str,
                   default='./ckpt/residual_rl/snapshot_step315.pkl',
                   help='Path to a snapshot produced by create_snapshot.py')
    p.add_argument('--xml_path', type=str, default='./asset/pih.xml')
    p.add_argument('--device', type=str, default='cuda')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--total_steps', type=int, default=200_000)
    p.add_argument('--output_dir', type=str, default=None,
                   help='Override default per-algo output directory')
    p.add_argument('--no_smolvla', action='store_true')
    p.add_argument('--smolvla_pretrained', type=str,
                   default='./ckpt/smolvla_pih1/checkpoints/020000/pretrained_model')
    p.add_argument('--smolvla_dataset_root', type=str, default='./demo_data_pih')
    p.add_argument('--smolvla_repo_name', type=str, default='ur5e_pih_language')
    p.add_argument('--resume', type=str, default=None)
    p.add_argument('--max_steps', type=int, default=650,
                   help='Max ABSOLUTE step index (snapshot base_step counts)')
    p.add_argument('--noise_pos', type=float, default=2e-4,
                   help='EEF position noise half-range (m), per axis')
    p.add_argument('--noise_rot_deg', type=float, default=0.2,
                   help='EEF rotation noise half-range (deg), per axis')
    p.add_argument('--wandb', action='store_true')
    p.add_argument('--wandb_project', type=str, default='residual-rl-pih')
    p.add_argument('--wandb_name', type=str, default=None)
    return p.parse_args()


# ======================================================================
# Main
# ======================================================================
def main():
    args = parse_args()
    device = torch.device(args.device)
    # 1. 获取算法对应的类
    AdapterCls, ConfigCls, default_output_dir = ALGO_REGISTRY[args.algo]
    output_dir = args.output_dir or default_output_dir
    # 2. 创建配置
    cfg = ConfigCls(
        xml_path=args.xml_path, seed=args.seed,
        total_steps=args.total_steps,
        output_dir=output_dir,
        log_dir=os.path.join(output_dir, 'logs'),
        smolvla_pretrained=args.smolvla_pretrained,
        smolvla_dataset_root=args.smolvla_dataset_root,
        smolvla_repo_name=args.smolvla_repo_name,
        smolvla_device=str(device),
    )
    # 3. 设置随机种子
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    # ------------------------------------------------------------------
    # Snapshot
    # ------------------------------------------------------------------
    snap = load_snapshot(args.snapshot)
    base_step = snap['meta']['base_step']
    instruction_ref = snap['pih']['instruction']
    print(f"[snap] base_step={base_step}, peg={snap['pih']['active_peg']}, "
          f"instruction='{instruction_ref}'")

    # ------------------------------------------------------------------
    # Wandb
    # ------------------------------------------------------------------
    use_wandb = args.wandb
    if use_wandb:
        import wandb
        run_config = dict(vars(cfg)) if hasattr(cfg, '__dict__') else {}
        run_config.update({
            'algo':                 args.algo,
            'snapshot_path':        args.snapshot,
            'snapshot_base_step':   base_step,
            'snapshot_peg':         snap['pih']['active_peg'],
            'snapshot_instruction': instruction_ref,
            'noise_pos_m':          args.noise_pos,
            'noise_rot_deg':        args.noise_rot_deg,
            'max_steps':            args.max_steps,
            'no_smolvla':           args.no_smolvla,
        })
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_name or f'residual_rl_snap_{args.algo}_seed{cfg.seed}',
            config=run_config,
        )
        print("Wandb initialized.")

    # ------------------------------------------------------------------
    # SmolVLA (frozen)
    # ------------------------------------------------------------------
    smolvla = None if args.no_smolvla else load_smolvla(cfg)

    # ------------------------------------------------------------------
    # Trainer adapter
    # ------------------------------------------------------------------
    adapter = AdapterCls(cfg, device=str(device), use_wandb=use_wandb)
    if args.resume:
        adapter.load(args.resume)

    # ------------------------------------------------------------------
    # Environment
    # ------------------------------------------------------------------
    print(f"[env] creating PIHEnv2 from {cfg.xml_path}")
    pih = PIHEnv2(
        xml_path=cfg.xml_path,
        action_type='joint_angle',
        state_type='joint_angle',
    )
    # 创建奖励函数和力/力矩历史
    reward_fn = HierarchicalReward(cfg)
    ft_history = deque(maxlen=cfg.force_history_len)

    # ------------------------------------------------------------------
    # Episode state
    # ------------------------------------------------------------------
    step = base_step
    episode = 0
    contact_detected = False
    contact_step = -1
    steps_after_contact = 0
    episode_reward = 0.0
    prev_obs = None          # (state, ft_hist, action, extras)
    last_rinfo = {}
    excessive_force_counter = 0
    ft_offset = snap['ft']['offset'].copy()

    # For PPO episode-end rollout finalization:
    last_next_state = None
    last_next_ft_hist = None

    # ------------------------------------------------------------------
    # Logging
    # ------------------------------------------------------------------
    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.log_dir, exist_ok=True)
    log_file = open(os.path.join(cfg.log_dir, 'train_log.csv'), 'w')
    log_file.write('step,episode,reward,length,success,contact_step,rl_steps\n')

    # ------------------------------------------------------------------
    # Reset helpers
    # ------------------------------------------------------------------
    def reset_from_snapshot():
        nonlocal step, episode, contact_detected, contact_step, steps_after_contact
        nonlocal episode_reward, prev_obs, last_rinfo, excessive_force_counter
        nonlocal ft_offset, last_next_state, last_next_ft_hist

        if smolvla is not None:
            smolvla.reset()

        # 恢复快照
        ft_offset = restore_snapshot(pih, snap)
        # 扰动末端执行器位姿
        info = perturb_ee_pose(pih,
                               pos_noise=args.noise_pos,
                               rot_noise_deg=args.noise_rot_deg,
                               rng=rng)
        # 初始化力/力矩历史
        ft_cal = pih.get_force_torque() - ft_offset
        reward_fn.reset(initial_ft=ft_cal)
        ft_history.clear()
        for _ in range(cfg.force_history_len):
            ft_history.append(ft_cal.copy())
        # 重置计数器
        step = base_step
        episode += 1
        contact_detected = False
        contact_step = -1
        steps_after_contact = 0
        episode_reward = 0.0
        prev_obs = None
        last_rinfo = {}
        excessive_force_counter = 0
        last_next_state = None
        last_next_ft_hist = None

        print(f"\n--- Episode {episode} --- "
              f"dp={info['dp']*1000}mm drpy={np.rad2deg(info['drpy'])}deg  "
              f"ft_cal={ft_cal}")

    def store_terminal(done_reward, is_success):
        """Close out the pending (prev_obs) transition with done=True."""
        nonlocal episode_reward, last_rinfo
        nonlocal last_next_state, last_next_ft_hist
        if prev_obs is None or not contact_detected:
            return
        # 获取最终状态
        ft = pih.get_force_torque() - ft_offset
        ft_history.append(ft.copy())
        state_vec, ft_hist = build_rl_obs(pih, ft_offset, ft_history)
        # 计算终止奖励
        pt = pih.env.get_p_site(PEG_TIP_SITE)
        he = pih.env.get_p_site(HOLE_ENTRY_SITE)
        hb = pih.env.get_p_site(HOLE_BOTTOM_SITE)
        r, rinfo = reward_fn.compute(ft, pt, he, hb, is_success)
        r += done_reward    # 加上终止奖励
        episode_reward += r
        last_rinfo = rinfo
        # 存储最后一步
        adapter.store_step(
            state=prev_obs[0], ft_hist=prev_obs[1],
            action=prev_obs[2], extras=prev_obs[3],
            reward=r,
            next_state=state_vec, next_ft_hist=ft_hist,
            done=True,
        )
        last_next_state = state_vec
        last_next_ft_hist = ft_hist

    # ------------------------------------------------------------------
    # First reset
    # ------------------------------------------------------------------
    reset_from_snapshot()

    print(f"\n=== Snapshot Residual RL ({args.algo.upper()}) ===")
    print(f"base_step        = {base_step}")
    print(f"instruction      = {pih.instruction}")
    print(f"active peg       = {pih.active_peg}")
    print(f"noise pos/rot    = {args.noise_pos*1000:.2f}mm / {args.noise_rot_deg:.2f}deg")
    print(f"max_residual_pos = {cfg.max_residual_pos} m")
    print(f"max_residual_rot = {cfg.max_residual_rot} rad")
    print(f"total RL target  = {cfg.total_steps}\n")

    # ==================================================================
    # MAIN LOOP
    # ==================================================================
    while pih.env.is_viewer_alive() and adapter.total_steps < cfg.total_steps:
        pih.step_env()  # MuJoCo仿真步进

        if not pih.env.loop_every(HZ=20):
            continue

        # ----- termination checks -----
        success = pih.check_success()
        timeout = (step >= args.max_steps)

        excessive_force = False
        if contact_detected:
            ft_cal_check = pih.get_force_torque() - ft_offset
            if abs(ft_cal_check[2]) > cfg.excessive_force_th:
                excessive_force_counter += 1
            else:
                excessive_force_counter = 0
            if excessive_force_counter >= cfg.excessive_force_steps:
                excessive_force = True

        out_of_hole = False
        if contact_detected:
            pt = pih.env.get_p_site(PEG_TIP_SITE)
            he = pih.env.get_p_site(HOLE_ENTRY_SITE)
            xy = np.linalg.norm(pt[:2] - he[:2])
            above = pt[2] > he[2] + 0.005
            if xy > cfg.out_of_hole_xy_th or above:
                out_of_hole = True

        # 处理终止
        if success or timeout or excessive_force or out_of_hole:
            if success:
                reason, terminal_r = 'SUCCESS', 0.0
            elif excessive_force:
                reason, terminal_r = 'EXCESSIVE_FORCE', -cfg.penalty_early_stop
            elif out_of_hole:
                reason, terminal_r = 'OUT_OF_HOLE', -cfg.penalty_early_stop
            else:
                reason, terminal_r = 'TIMEOUT', -cfg.penalty_early_stop * 0.5

            store_terminal(terminal_r, success)
            # PPO: finalize rollout + update if buffer has any data.
            adapter.on_episode_end(
                last_state=last_next_state if last_next_state is not None
                else np.zeros(cfg.state_dim, dtype=np.float32),
                last_ft_hist=last_next_ft_hist if last_next_ft_hist is not None
                else np.zeros((cfg.force_history_len, cfg.ft_dim), dtype=np.float32),
                last_done=True,
            )

            c_tag = f'contact@{contact_step}' if contact_step >= 0 else 'no_contact'
            alpha_tag = f'alpha={adapter.alpha:.3f}' if adapter.has_alpha else ''
            print(f"[{args.algo.upper()} {adapter.total_steps:>7d} | Ep {episode:>4d}] "
                  f"R={episode_reward:>8.2f} L={step-base_step:>4d} {reason} "
                  f"{c_tag} rl_steps={steps_after_contact} {alpha_tag}")
            log_file.write(f"{adapter.total_steps},{episode},{episode_reward:.4f},"
                           f"{step-base_step},{int(success)},{contact_step},"
                           f"{steps_after_contact}\n")
            log_file.flush()

            adapter.log_episode(
                episode_reward, step - base_step,
                {'success': success, 'contact_step': contact_step,
                 'steps_after_contact': steps_after_contact, 'reason': reason},
                reward_info=last_rinfo,
            )

            if adapter.total_steps > 0 and \
                    adapter.total_steps % cfg.save_every < max(steps_after_contact + 1, 1):
                adapter.save(os.path.join(cfg.output_dir,
                                          f'step_{adapter.total_steps}'))

            reset_from_snapshot()
            continue

        # ----- SmolVLA base action -----
        if smolvla is not None:
            base_action = get_smolvla_action(smolvla, pih, str(device))
        else:
            q = pih.get_joint_state()[:6]
            base_action = np.concatenate([q, [200.0]], dtype=np.float32)

        # ----- contact detection -----
        if not contact_detected:
            if check_contact(pih, ft_offset):
                contact_detected = True
                contact_step = step
                pt = pih.env.get_p_site(PEG_TIP_SITE)
                he = pih.env.get_p_site(HOLE_ENTRY_SITE)
                print(f"  >> Insertion started at step {step} "
                      f"(depth={1000*(he[2]-pt[2]):.1f}mm)")

        # ----- RL residual (only after contact) -----
        cart_res = np.zeros(6, dtype=np.float32)
        joint_delta = np.zeros(6, dtype=np.float32)

        if contact_detected and cfg.max_residual_pos > 0:
            # 构建观测
            ft = pih.get_force_torque() - ft_offset
            ft_history.append(ft.copy())
            state_vec, ft_hist = build_rl_obs(pih, ft_offset, ft_history)

            # Close out the pending transition from the previous step存储上一步的转移
            if prev_obs is not None:
                pt = pih.env.get_p_site(PEG_TIP_SITE)
                he = pih.env.get_p_site(HOLE_ENTRY_SITE)
                hb = pih.env.get_p_site(HOLE_BOTTOM_SITE)
                r, rinfo = reward_fn.compute(ft, pt, he, hb, False)
                episode_reward += r
                last_rinfo = rinfo
                adapter.store_step(
                    state=prev_obs[0], ft_hist=prev_obs[1],
                    action=prev_obs[2], extras=prev_obs[3],
                    reward=r,
                    next_state=state_vec, next_ft_hist=ft_hist,
                    done=False,
                )
                last_next_state = state_vec
                last_next_ft_hist = ft_hist
                steps_after_contact += 1
                adapter.step_updates()

            # 获取新的RL动作
            in_warmup = adapter.uses_warmup and adapter.total_steps < cfg.warmup_steps
            cart_res, extras = adapter.act(state_vec, ft_hist, in_warmup=in_warmup)
            joint_delta = cart_to_joint_delta(pih, cart_res, cfg)
            prev_obs = (state_vec.copy(), ft_hist.copy(),
                        cart_res.copy(), dict(extras))
        else:
            ft = pih.get_force_torque() - ft_offset
            ft_history.append(ft.copy())

        final_action = base_action.copy()
        final_action[:6] += joint_delta
        _ = pih.step(final_action)

        # ----- monitoring -----
        ft_cal = pih.get_force_torque() - ft_offset
        lat_f  = np.linalg.norm(ft_cal[:2])
        axial  = ft_cal[2]
        bend_t = np.linalg.norm(ft_cal[3:5])
        tot_f  = np.linalg.norm(ft_cal[:3])
        peg_z  = pih.env.get_p_site(PEG_TIP_SITE)[2]

        pih.render(idx=episode)
        if contact_detected and prev_obs is not None:
            pih.env.viewer_text_overlay(
                text1=f'Residual {args.algo.upper()}',
                text2=f'R={episode_reward:.2f}  '
                      f'|cart|={np.linalg.norm(cart_res):.5f}  '
                      f'|dq|={np.linalg.norm(joint_delta):.5f}  '
                      f'rl_steps={steps_after_contact}',
            )
        pih.env.viewer_text_overlay(
            text1='F/T Calibrated',
            text2=f'Fx={ft_cal[0]:.1f} Fy={ft_cal[1]:.1f} Fz={ft_cal[2]:.1f} '
                  f'|Flat|={lat_f:.1f} |F|={tot_f:.1f}N  '
                  f'Tx={ft_cal[3]:.3f} Ty={ft_cal[4]:.3f} |Tb|={bend_t:.3f}Nm',
        )
        pih.env.viewer_text_overlay(
            text1='Snapshot',
            text2=f'base_step={base_step}  step={step}  '
                  f'contact={contact_detected}  excess={excessive_force_counter}',
        )

        if contact_detected and step % 5 == 0:
            r_str = f'R={episode_reward:>7.2f}' if prev_obs is not None else 'R=  n/a'
            print(f'  step={step:>4d} {r_str} '
                  f'|F|={tot_f:>5.1f}N lat={lat_f:>4.1f} Fz={axial:>5.1f} '
                  f'bend={bend_t:.3f}Nm peg_z={peg_z:.4f} '
                  f'|cart|={np.linalg.norm(cart_res):.5f} '
                  f'|dq|={np.linalg.norm(joint_delta):.5f}')

        step += 1

    # ------------------------------------------------------------------
    # Wrap up
    # ------------------------------------------------------------------
    adapter.save(os.path.join(cfg.output_dir, 'final'))
    log_file.close()
    pih.env.close_viewer()
    if use_wandb:
        import wandb
        wandb.finish()
    print(f"\n[done] {episode} episodes, {adapter.total_steps} RL steps.")


if __name__ == '__main__':
    main()
