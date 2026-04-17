"""
ppo_policy.py — PPO actor-critic for Cartesian residual RL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .config import ResidualPPOConfig


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class ForceTemporalAttention(nn.Module):
    def __init__(self, ft_dim=6, embed_dim=64, n_heads=1):
        super().__init__()
        self.embed = nn.Linear(ft_dim, embed_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=n_heads, batch_first=True
        )
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, ft_history):
        x = F.relu(self.embed(ft_history))
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(attn_out + x)
        return x.mean(dim=1)


class SharedEncoder(nn.Module):
    def __init__(self, cfg: ResidualPPOConfig):
        super().__init__()
        self.state_mlp = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.state_hidden),
            nn.ReLU(),
            nn.Linear(cfg.state_hidden, cfg.state_hidden),
            nn.ReLU(),
        )
        self.ft_attention = ForceTemporalAttention(
            ft_dim=cfg.ft_dim,
            embed_dim=cfg.ft_attention_dim,
            n_heads=cfg.n_attention_heads,
        )
        self.fusion = nn.Sequential(
            nn.Linear(cfg.state_hidden + cfg.ft_attention_dim, cfg.fusion_hidden),
            nn.ReLU(),
            nn.Linear(cfg.fusion_hidden, cfg.output_hidden),
            nn.ReLU(),
        )

    def forward(self, state, ft_history):
        state_feat = self.state_mlp(state)
        ft_feat = self.ft_attention(ft_history)
        return self.fusion(torch.cat([state_feat, ft_feat], dim=-1))


class PPOActorCritic(nn.Module):
    def __init__(self, cfg: ResidualPPOConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = SharedEncoder(cfg)
        self.actor_mean = nn.Linear(cfg.output_hidden, cfg.action_dim)
        self.critic = nn.Linear(cfg.output_hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(cfg.action_dim))

        self.register_buffer('action_scale', torch.tensor([
            cfg.max_residual_pos, cfg.max_residual_pos, cfg.max_residual_pos,
            cfg.max_residual_rot, cfg.max_residual_rot, cfg.max_residual_rot,
        ], dtype=torch.float32))

    def _dist_and_value(self, state, ft_history):
        feat = self.encoder(state, ft_history)
        mean = self.actor_mean(feat)
        value = self.critic(feat)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp().expand_as(mean)
        return Normal(mean, std), value

    def act(self, state, ft_history):
        dist, value = self._dist_and_value(state, ft_history)
        pre_tanh = dist.rsample()
        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale

        log_prob = dist.log_prob(pre_tanh)
        log_prob -= torch.log(1 - squashed.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, value

    def evaluate_actions(self, state, ft_history, action):
        dist, value = self._dist_and_value(state, ft_history)
        scaled_action = torch.clamp(action / self.action_scale, -0.999999, 0.999999)
        pre_tanh = torch.atanh(scaled_action)
        log_prob = dist.log_prob(pre_tanh)
        log_prob -= torch.log(1 - scaled_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy, value

    def deterministic_action(self, state, ft_history):
        dist, _ = self._dist_and_value(state, ft_history)
        return torch.tanh(dist.mean) * self.action_scale

