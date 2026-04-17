"""
config.py — Hyperparameters for TD3 residual RL.
"""

from dataclasses import dataclass


@dataclass
class ResidualTD3Config:
    # ======================== Environment ========================
    xml_path: str = './asset/pih.xml'
    control_hz: float = 20.0
    max_episode_steps: int = 400
    force_history_len: int = 10

    # ======================== Residual Policy ========================
    state_dim: int = 15
    ft_dim: int = 6
    action_dim: int = 6
    max_residual_pos: float = 0.001
    max_residual_rot: float = 0.005

    # Network architecture
    state_hidden: int = 256
    ft_attention_dim: int = 64
    fusion_hidden: int = 256
    output_hidden: int = 128
    n_attention_heads: int = 1

    # ======================== TD3 ========================
    gamma: float = 0.95
    tau: float = 0.002
    lr_actor: float = 1e-4
    lr_critic: float = 5e-5
    batch_size: int = 512
    replay_capacity: int = 50_000
    warmup_steps: int = 3000
    update_every: int = 1
    updates_per_step: int = 1
    policy_delay: int = 2
    policy_noise_pos: float = 0.0003
    policy_noise_rot: float = 0.0015
    noise_clip_pos: float = 0.0005
    noise_clip_rot: float = 0.0025
    exploration_noise_pos: float = 0.0004
    exploration_noise_rot: float = 0.002

    # ======================== Reward Weights ========================
    w_success: float = 100.0
    w_force_baseline: float = 3.0
    fx_th: float = 2.0
    fy_th: float = 2.0
    penalty_fx: float = 0.05
    penalty_fy: float = 0.05
    fz_th: float = 15.0
    penalty_fz: float = 0.05
    tx_th: float = 0.3
    ty_th: float = 0.3
    penalty_tx: float = 0.3
    penalty_ty: float = 0.3
    force_stability_threshold: float = 1.0
    w_force_stable: float = 0.5
    penalty_force_jerk: float = 0.1
    w_depth_hold: float = 1.0
    w_progress: float = 10.0
    penalty_retract: float = 5.0
    penalty_outside: float = 2.0

    # ======================== Force Thresholds ========================
    lateral_th: float = 0.5
    bending_th: float = 0.5
    axial_th: float = 10.0

    # ======================== Contact Detection ========================
    contact_xy_threshold: float = 0.01
    contact_z_threshold: float = 0.005
    contact_force_threshold: float = 15.0
    max_steps_after_contact: int = 100

    # ======================== Early Termination ========================
    excessive_force_th: float = 50.0
    excessive_force_steps: int = 15
    out_of_hole_xy_th: float = 0.04
    penalty_early_stop: float = 300.0

    # ======================== Jacobian IK ========================
    ik_damping_eps: float = 1e-4
    max_joint_delta: float = 0.01

    # ======================== Training ========================
    total_steps: int = 50_000
    eval_every: int = 5000
    eval_episodes: int = 10
    save_every: int = 5000
    log_every: int = 100
    seed: int = 42
    grad_clip_norm: float = 0.5

    # ======================== Normalization ========================
    reward_scale: float = 0.1
    obs_normalize: bool = True

    # ======================== SmolVLA ========================
    smolvla_pretrained: str = './ckpt/smolvla_pih/checkpoints/020000/pretrained_model'
    smolvla_dataset_root: str = './demo_data_pih'
    smolvla_repo_name: str = 'ur5e_pih_language'
    smolvla_chunk_size: int = 5
    smolvla_n_action_steps: int = 5
    smolvla_device: str = 'cuda'

    # ======================== Paths ========================
    output_dir: str = './ckpt/residual_rl_td3'
    log_dir: str = './ckpt/residual_rl_td3/logs'

