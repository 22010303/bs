"""
aux_config.py — Config for PPO with CLS pooling and auxiliary heads.
"""

from dataclasses import dataclass


@dataclass
class ResidualPPOAuxConfig:
    # ======================== Environment ========================
    xml_path: str = './asset/pih.xml'
    control_hz: float = 20.0
    max_episode_steps: int = 100
    force_history_len: int = 10

    # ======================== Residual Policy ========================
    # state = ft(6) + ee_pose(6) + relative(3) + ee_velocity(6)
    state_dim: int = 21
    ft_dim: int = 6
    action_dim: int = 6
    num_state_classes: int = 3
    max_residual_pos: float = 0.0002
    max_residual_rot: float = 0.005

    # Network architecture
    state_hidden: int = 256
    ft_attention_dim: int = 64
    fusion_hidden: int = 256
    output_hidden: int = 128
    aux_hidden: int = 128
    n_attention_heads: int = 1

    # ======================== PPO ========================
    lr_actor: float = 3e-5
    lr_critic: float = 1e-4
    gamma: float = 0.95
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.02
    max_grad_norm: float = 0.5
    rollout_steps: int = 2048
    ppo_epochs: int = 10
    minibatch_size: int = 256
    normalize_advantage: bool = True
    target_kl: float = 0.03

    # ======================== Auxiliary Loss ========================
    # Ablation switch: CLS pooling inside ForceTemporalAttention.
    use_cls_pooling: bool = True
    # Ablation switch: next-step force prediction head.
    use_force_prediction_head: bool = True
    # Ablation switch: 3-way insertion state classification head.
    use_state_classification_head: bool = False
    force_pred_loss_coef: float = 0.2
    state_cls_loss_coef: float = 0.2

    # ======================== Reward ========================
    w_success: float = 100.0
    w_time_bonus: float = 2.0
    w_force_baseline: float = 2.0
    fx_th: float = 5.0
    fy_th: float = 5.0
    penalty_fx: float = 0.1
    penalty_fy: float = 0.1
    fz_th: float = 15.0
    penalty_fz: float = 0.1
    tx_th: float = 1.0
    ty_th: float = 1.0
    penalty_tx: float = 0.1
    penalty_ty: float = 0.1
    force_stability_threshold: float = 1.0
    w_force_stable: float = 1.0
    penalty_force_jerk: float = 0.3
    w_depth_hold: float = 0.0
    w_progress: float = 100.0
    penalty_retract: float = 100.0
    penalty_outside: float = 2.0

    # ======================== State Label Rules ========================
    state_label_debounce_steps: int = 5
    jam_lateral_force_th: float = 15.0
    jam_torque_th: float = 1.5
    jam_xy_error_th: float = 0.001
    bottom_axial_force_th: float = 30.0
    low_speed_th: float = 0.01
    bottom_progress_th: float = 0.7

    # ======================== Early Termination ========================
    excessive_force_th: float = 20.0
    excessive_force_steps: int = 10
    out_of_hole_xy_th: float = 0.005
    penalty_early_stop: float = 50.0

    # ======================== Jacobian IK ========================
    ik_damping_eps: float = 1e-4
    max_joint_delta: float = 0.01

    # ======================== Training ========================
    total_steps: int = 30_000
    save_every: int = 10_000
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
    output_dir: str = './ckpt/residual_rl_ppo_aux'
    log_dir: str = './ckpt/residual_rl_ppo_aux/logs'
