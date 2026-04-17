"""
residual_policy.py — Attention-augmented MLP residual policy for SAC.

Architecture:
  - ForceTemporalAttention: processes F/T history (K, 6) -> 64D embedding
  - StateMLP: processes state (15D) -> 256D embedding
  - Fusion: concatenates and produces shared features
  - Actor: outputs residual action with tanh squashing
  - Critic (twin Q): outputs state-action value
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .config import ResidualRLConfig


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class ForceTemporalAttention(nn.Module):
    """
    Processes a window of K force/torque readings using self-attention
    to capture temporal dynamics of contact forces.
    """

    def __init__(self, ft_dim=6, embed_dim=64, n_heads=1):
        super().__init__()
        self.embed = nn.Linear(ft_dim, embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=n_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, ft_history):
        """
        Args:
            ft_history: (batch, K, 6) — force/torque history window

        Returns:
            (batch, embed_dim) — temporal embedding
        """
        # Project each timestep: (B, K, 6) -> (B, K, embed_dim)
        x = F.relu(self.embed(ft_history))

        # Self-attention over time dimension
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(attn_out + x)  # residual connection

        # Mean pool over time: (B, K, embed_dim) -> (B, embed_dim)
        return x.mean(dim=1)


class SharedEncoder(nn.Module):
    """
    Shared feature encoder: state MLP + force temporal attention -> fused features.
    """

    def __init__(self, cfg: ResidualRLConfig):
        super().__init__()
        self.cfg = cfg

        # State MLP: 15D -> 256D
        self.state_mlp = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.state_hidden),
            nn.ReLU(),
            nn.Linear(cfg.state_hidden, cfg.state_hidden),
            nn.ReLU(),
        )

        # Force temporal attention: (K, 6) -> 64D
        self.ft_attention = ForceTemporalAttention(
            ft_dim=cfg.ft_dim,
            embed_dim=cfg.ft_attention_dim,
            n_heads=cfg.n_attention_heads,
        )

        # Fusion: (256 + 64) -> 256 -> 128
        fusion_in = cfg.state_hidden + cfg.ft_attention_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, cfg.fusion_hidden),
            nn.ReLU(),
            nn.Linear(cfg.fusion_hidden, cfg.output_hidden),
            nn.ReLU(),
        )

    def forward(self, state, ft_history):
        """
        Args:
            state:      (batch, 15) — concatenated state vector
            ft_history: (batch, K, 6) — force/torque history

        Returns:
            features: (batch, output_hidden) — shared features
        """
        state_feat = self.state_mlp(state)
        ft_feat = self.ft_attention(ft_history)
        fused = torch.cat([state_feat, ft_feat], dim=-1)
        return self.fusion(fused)


class ResidualActor(nn.Module):
    """
    SAC Actor: outputs 6D Cartesian residual [dx,dy,dz,drx,dry,drz].
    Position and rotation have different scaling (max_residual_pos, max_residual_rot).
    """

    def __init__(self, cfg: ResidualRLConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = SharedEncoder(cfg)
        self.mean_head = nn.Linear(cfg.output_hidden, cfg.action_dim)
        self.log_std = nn.Parameter(torch.zeros(cfg.action_dim))

        # Per-dimension scaling: [pos_x, pos_y, pos_z, rot_x, rot_y, rot_z]
        self.register_buffer('action_scale', torch.tensor([
            cfg.max_residual_pos, cfg.max_residual_pos, cfg.max_residual_pos,
            cfg.max_residual_rot, cfg.max_residual_rot, cfg.max_residual_rot,
        ], dtype=torch.float32))

    def forward(self, state, ft_history):
        """
        Returns:
            mean:    (batch, action_dim) — mean of Gaussian before squashing
            log_std: (action_dim,) — fixed log_std (broadcast over batch)
        """
        features = self.encoder(state, ft_history)
        mean = self.mean_head(features)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        return mean, log_std

    def sample(self, state, ft_history):
        """
        Sample action using reparameterization trick + tanh squashing.

        Returns:
            action:   (batch, action_dim) — squashed and scaled residual
            log_prob: (batch, 1) — log probability with squashing correction
        """
        mean, log_std = self.forward(state, ft_history)
        std = log_std.exp().expand_as(mean)  # broadcast fixed std to batch
        dist = Normal(mean, std)

        # Reparameterized sample
        x_t = dist.rsample()
        y_t = torch.tanh(x_t)

        # Scale to per-axis max residual
        action = y_t * self.action_scale

        # Log probability with tanh squashing correction
        log_prob = dist.log_prob(x_t)
        # Enforcing Action Bound (Appendix C of SAC paper)
        log_prob -= torch.log(1 - y_t.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)

        return action, log_prob

    def deterministic_action(self, state, ft_history):
        """For evaluation: use the mean action without sampling."""
        mean, _ = self.forward(state, ft_history)
        return torch.tanh(mean) * self.action_scale


class ResidualCritic(nn.Module):
    """
    SAC Twin Q-Critic: two independent Q-networks for double-Q trick.
    Each takes (state, ft_history, action) and outputs a scalar Q-value.
    """

    def __init__(self, cfg: ResidualRLConfig):
        super().__init__()
        self.cfg = cfg

        # Q1
        self.encoder1 = SharedEncoder(cfg)
        self.q1_head = nn.Sequential(
            nn.Linear(cfg.output_hidden + cfg.action_dim, cfg.output_hidden),
            nn.ReLU(),
            nn.Linear(cfg.output_hidden, 1),
        )

        # Q2
        self.encoder2 = SharedEncoder(cfg)
        self.q2_head = nn.Sequential(
            nn.Linear(cfg.output_hidden + cfg.action_dim, cfg.output_hidden),
            nn.ReLU(),
            nn.Linear(cfg.output_hidden, 1),
        )

    def forward(self, state, ft_history, action):
        """
        Args:
            state:      (batch, 15)
            ft_history: (batch, K, 6)
            action:     (batch, 6) — residual action

        Returns:
            q1, q2: (batch, 1) each
        """
        feat1 = self.encoder1(state, ft_history)
        feat2 = self.encoder2(state, ft_history)

        q1 = self.q1_head(torch.cat([feat1, action], dim=-1))
        q2 = self.q2_head(torch.cat([feat2, action], dim=-1))

        return q1, q2

    def q1(self, state, ft_history, action):
        """Single Q-value for policy gradient (uses Q1 only)."""
        feat1 = self.encoder1(state, ft_history)
        return self.q1_head(torch.cat([feat1, action], dim=-1))
