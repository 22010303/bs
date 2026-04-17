"""
config.py — Hyperparameters for PPO residual RL.
"""

from dataclasses import dataclass


@dataclass
class ResidualPPOConfig:
    # ======================== Environment ========================
    xml_path: str = './asset/pih.xml'
    control_hz: float = 20.0
    max_episode_steps: int = 300
    force_history_len: int = 10

    # ======================== Residual Policy ========================
    state_dim: int = 15
    ft_dim: int = 6
    action_dim: int = 6
    max_residual_pos: float = 0.001
    max_residual_rot: float = 0.001

    # Network architecture
    state_hidden: int = 256
    ft_attention_dim: int = 64
    fusion_hidden: int = 256
    output_hidden: int = 128
    n_attention_heads: int = 1

    # ======================== PPO ========================
    lr_actor: float = 1e-4
    lr_critic: float = 1e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.4
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    max_grad_norm: float = 0.5
    rollout_steps: int = 2048
    ppo_epochs: int = 10
    minibatch_size: int = 256
    normalize_advantage: bool = True
    target_kl: float = 0.03
    update_after_contact_only: bool = True

    # ======================== Reward Weights ========================
    w_success: float = 100.0
    w_force_baseline: float = 4.0
    fx_th: float = 2.0
    fy_th: float = 2.0
    penalty_fx: float = 0.05
    penalty_fy: float = 0.05
    fz_th: float = 15.0
    penalty_fz: float = 0.05
    tx_th: float = 0.3
    ty_th: float = 0.3
    penalty_tx: float = 0.1
    penalty_ty: float = 0.1
    force_stability_threshold: float = 1.0
    w_force_stable: float = 1
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
    total_steps: int = 30_000
    eval_every: int = 5000
    eval_episodes: int = 10
    save_every: int = 10_000
    log_every: int = 100
    seed: int = 42

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
    output_dir: str = './ckpt/residual_rl_ppo'
    log_dir: str = './ckpt/residual_rl_ppo/logs'

