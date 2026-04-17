"""
td3_policy.py — TD3 actor and twin critic for Cartesian residual RL.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ResidualTD3Config


class ForceTemporalAttention(nn.Module):
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
        x = F.relu(self.embed(ft_history))
        attn_out, _ = self.attention(x, x, x)
        x = self.norm(attn_out + x)
        return x.mean(dim=1)


class SharedEncoder(nn.Module):
    def __init__(self, cfg: ResidualTD3Config):
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


class TD3Actor(nn.Module):
    def __init__(self, cfg: ResidualTD3Config):
        super().__init__()
        self.encoder = SharedEncoder(cfg)
        self.mean_head = nn.Linear(cfg.output_hidden, cfg.action_dim)
        self.register_buffer('action_scale', torch.tensor([
            cfg.max_residual_pos, cfg.max_residual_pos, cfg.max_residual_pos,
            cfg.max_residual_rot, cfg.max_residual_rot, cfg.max_residual_rot,
        ], dtype=torch.float32))

    def forward(self, state, ft_history):
        features = self.encoder(state, ft_history)
        return torch.tanh(self.mean_head(features)) * self.action_scale


class TD3Critic(nn.Module):
    def __init__(self, cfg: ResidualTD3Config):
        super().__init__()
        self.encoder1 = SharedEncoder(cfg)
        self.encoder2 = SharedEncoder(cfg)
        self.q1_head = nn.Sequential(
            nn.Linear(cfg.output_hidden + cfg.action_dim, cfg.output_hidden),
            nn.ReLU(),
            nn.Linear(cfg.output_hidden, 1),
        )
        self.q2_head = nn.Sequential(
            nn.Linear(cfg.output_hidden + cfg.action_dim, cfg.output_hidden),
            nn.ReLU(),
            nn.Linear(cfg.output_hidden, 1),
        )

    def forward(self, state, ft_history, action):
        feat1 = self.encoder1(state, ft_history)
        feat2 = self.encoder2(state, ft_history)
        q1 = self.q1_head(torch.cat([feat1, action], dim=-1))
        q2 = self.q2_head(torch.cat([feat2, action], dim=-1))
        return q1, q2

    def q1(self, state, ft_history, action):
        feat1 = self.encoder1(state, ft_history)
        return self.q1_head(torch.cat([feat1, action], dim=-1))

