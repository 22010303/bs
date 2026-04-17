"""
config.py — All hyperparameters for Residual RL fine-tuning.
"""

from dataclasses import dataclass, field
import numpy as np


@dataclass
class ResidualRLConfig:
    """Centralized configuration for residual RL training."""

    # ======================== Environment ========================
    xml_path: str = './asset/pih.xml'
    control_hz: float = 20.0
    max_episode_steps: int = 400       # max steps in insertion phase
    force_history_len: int = 10        # K: temporal window for F/T history

    # ======================== Residual Policy ========================
    # SAC observes ft + ee pose + relative position and outputs
    # Cartesian residual [dx, dy, dz, drx, dry, drz].
    # The controller converts it to joint angle delta via Jacobian pseudo-inverse.
    state_dim: int = 15                # ft(6) + ee_pose(6) + relative(3)
    ft_dim: int = 6                    # force/torque dimension
    action_dim: int = 6                # 6D Cartesian residual [dx,dy,dz,drx,dry,drz]
    max_residual_pos: float = 0.001    # max position residual (m)
    max_residual_rot: float = 0.005    # max rotation residual (rad)

    # Network architecture
    state_hidden: int = 256
    ft_attention_dim: int = 64
    fusion_hidden: int = 256
    output_hidden: int = 128
    n_attention_heads: int = 1

    # ======================== SAC ========================
    gamma: float = 0.95                # discount factor
    tau: float = 0.002                 # soft target update rate（0.005-0.002）
    lr_actor: float = 1e-4             # 与Critic同步，从2e-4下调
    lr_critic: float = 5e-5            #（1e-4- 5E-5）
    lr_alpha: float = 1e-4             # entropy temperature learning rate
    init_alpha: float = 0.05            # initial entropy temperature
    target_entropy: float = -3.0       # target entropy (relaxed for small residuals)
    batch_size: int = 512              #（256-512）
    replay_capacity: int = 50_000
    warmup_steps: int = 5000           # random actions before training starts
    update_every: int = 1              # SAC update frequency (env steps)
    updates_per_step: int = 1          # gradient steps per env step

    # ======================== Reward Weights ========================
    w_success: float = 100.0

    # --- Force magnitude (primary target) ---
    # Each axis has its own threshold and penalty coefficient
    w_force_baseline: float = 3.0          # per-step bonus for being in hole

    # XY lateral forces (Fx, Fy) — causes jamming
    fx_th: float = 2.0                     # Fx safe threshold (N)
    fy_th: float = 2.0                     # Fy safe threshold (N)
    penalty_fx: float = 0.05               # per-Newton above threshold
    penalty_fy: float = 0.05               # per-Newton above threshold

    # Axial force (Fz) — excessive pushing
    fz_th: float = 15.0                     # Fz safe threshold (N)
    penalty_fz: float = 0.05              # per-Newton above threshold

    # Bending torques (Tx, Ty) — causes skewing
    tx_th: float = 0.3                     # Tx safe threshold (Nm)
    ty_th: float = 0.3                     # Ty safe threshold (Nm)
    penalty_tx: float = 0.1               # per-Nm above threshold
    penalty_ty: float = 0.1               # per-Nm above threshold

    # --- Force stability (secondary target) ---
    force_stability_threshold: float = 1.0 # |dF| below this = "stable" (N)
    w_force_stable: float = 0.5            # bonus when force change is small
    penalty_force_jerk: float = 0.1        # per-Newton penalty on |dF|

    # --- Insertion maintenance ---
    w_depth_hold: float = 1.0              # per-step depth bonus
    w_progress: float = 10.0               # watermark progress bonus
    penalty_retract: float = 5.0           # retract penalty
    penalty_outside: float = 2.0           # per-step outside-hole penalty

    # ======================== Force Thresholds ========================
    # 新增的力/扭矩阈值
    lateral_th: float = 0.5            # 侧向力阈值（N），超过则惩罚
    bending_th: float = 0.5            # 弯曲扭矩阈值（Nm），超过则惩罚
    axial_th: float = 10.0             # 轴向力阈值（N），超过则惩罚

    # ======================== Contact Detection ========================
    contact_xy_threshold: float = 0.01     # XY distance to hole entry (m)
    contact_z_threshold: float = 0.005     # Z distance to hole entry (m)
    contact_force_threshold: float = 15.0  # calibrated |Fz| indicating contact (N)
    max_steps_after_contact: int = 100     # timeout after contact

    # ======================== Early Termination ========================
    excessive_force_th: float = 50.0       # |Fz| threshold for excessive force (N)
    excessive_force_steps: int = 10        # consecutive steps AFTER CONTACT above threshold
    out_of_hole_xy_th: float = 0.04        # XY distance to declare "out of hole" (m)
    penalty_early_stop: float = 200.0      # extra negative reward on failure termination

    # ======================== Jacobian IK ========================
    ik_damping_eps: float = 1e-4           # damped pseudo-inverse regularization
    max_joint_delta: float = 0.01          # safety clamp on joint delta (rad)

    # ======================== Training ========================
    total_steps: int = 100_000
    eval_every: int = 5000
    eval_episodes: int = 10
    save_every: int = 10_000
    log_every: int = 100
    seed: int = 42
    grad_clip_norm: float = 0.5            # gradient clipping max norm

    # ======================== Normalization ========================
    reward_scale: float = 0.1              # reward *= scale before storing
    obs_normalize: bool = True             # running mean/std normalization on state

    # ======================== SmolVLA ========================
    smolvla_pretrained: str = './ckpt/smolvla_pih/checkpoints/020000/pretrained_model'
    smolvla_dataset_root: str = './demo_data_pih'
    smolvla_repo_name: str = 'ur5e_pih_language'
    smolvla_chunk_size: int = 5
    smolvla_n_action_steps: int = 5
    smolvla_device: str = 'cuda'

    # ======================== Paths ========================
    output_dir: str = './ckpt/residual_rl'
    log_dir: str = './ckpt/residual_rl/logs'
