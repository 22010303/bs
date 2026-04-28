"""
aux_policy.py — PPO actor-critic with CLS pooling and auxiliary heads.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from .aux_config import ResidualPPOAuxConfig


LOG_STD_MIN = -20
LOG_STD_MAX = 2


class ForceTemporalAttention(nn.Module):
    def __init__(self, cfg: ResidualPPOAuxConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Linear(cfg.ft_dim, cfg.ft_attention_dim)
        self.attention = nn.MultiheadAttention(
            embed_dim=cfg.ft_attention_dim,
            num_heads=cfg.n_attention_heads,
            batch_first=True,
        )
        self.norm = nn.LayerNorm(cfg.ft_attention_dim)
        # Ablation point 1: use a learnable CLS token instead of mean pooling.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.ft_attention_dim))

    def forward(self, ft_history, return_attention=False):
        x = F.relu(self.embed(ft_history))
        if self.cfg.use_cls_pooling:
            bsz = x.shape[0]
            cls = self.cls_token.expand(bsz, -1, -1)
            x = torch.cat([cls, x], dim=1)
            attn_out, attn_weights = self.attention(x, x, x)
            x = self.norm(attn_out + x)
            cls_feat = x[:, 0]
            if not return_attention:
                return cls_feat
            # Return only the CLS-token attention over the K history frames.
            cls_attn = attn_weights[:, 0, 1:]
            return cls_feat, cls_attn
        attn_out, attn_weights = self.attention(x, x, x)
        x = self.norm(attn_out + x)
        pooled = x.mean(dim=1)
        if not return_attention:
            return pooled
        # For mean-pooling analysis we still expose a comparable K-step score by
        # averaging query attention over all history tokens.
        mean_attn = attn_weights.mean(dim=1)
        return pooled, mean_attn


class SharedEncoder(nn.Module):
    def __init__(self, cfg: ResidualPPOAuxConfig):
        super().__init__()
        self.state_mlp = nn.Sequential(
            nn.Linear(cfg.state_dim, cfg.state_hidden),
            nn.ReLU(),
            nn.Linear(cfg.state_hidden, cfg.state_hidden),
            nn.ReLU(),
        )
        self.ft_attention = ForceTemporalAttention(cfg)
        self.fusion = nn.Sequential(
            nn.Linear(cfg.state_hidden + cfg.ft_attention_dim, cfg.fusion_hidden),
            nn.ReLU(),
            nn.Linear(cfg.fusion_hidden, cfg.output_hidden),
            nn.ReLU(),
        )

    def forward(self, state, ft_history, return_attention=False):
        state_feat = self.state_mlp(state)
        if return_attention:
            ft_feat, attn_scores = self.ft_attention(ft_history, return_attention=True)
        else:
            ft_feat = self.ft_attention(ft_history)
            attn_scores = None
        fused = self.fusion(torch.cat([state_feat, ft_feat], dim=-1))
        if return_attention:
            return fused, {
                'attention_scores': attn_scores,
                'state_feat': state_feat,
                'ft_feat': ft_feat,
            }
        return fused


class PPOAuxActorCritic(nn.Module):
    def __init__(self, cfg: ResidualPPOAuxConfig):
        super().__init__()
        self.cfg = cfg
        # Shared backbone for policy/value/auxiliary heads.
        self.encoder = SharedEncoder(cfg)
        self.actor_mean = nn.Linear(cfg.output_hidden, cfg.action_dim)
        self.critic = nn.Linear(cfg.output_hidden, 1)
        self.log_std = nn.Parameter(torch.zeros(cfg.action_dim))

        # Ablation point 2: next-step force prediction head.
        self.force_predictor = nn.Sequential(
            nn.Linear(cfg.output_hidden, cfg.aux_hidden),
            nn.ReLU(),
            nn.Linear(cfg.aux_hidden, cfg.ft_dim),
        )
        # Ablation point 3: explicit 3-way state classification head.
        self.state_classifier = nn.Sequential(
            nn.Linear(cfg.output_hidden, cfg.aux_hidden),
            nn.ReLU(),
            nn.Linear(cfg.aux_hidden, cfg.num_state_classes),
        )

        self.register_buffer('action_scale', torch.tensor([
            cfg.max_residual_pos, cfg.max_residual_pos, cfg.max_residual_pos,
            cfg.max_residual_rot, cfg.max_residual_rot, cfg.max_residual_rot,
        ], dtype=torch.float32))

    def _forward_all(self, state, ft_history, return_attention=False):
        if return_attention:
            feat, extra = self.encoder(state, ft_history, return_attention=True)
        else:
            feat = self.encoder(state, ft_history)
            extra = {}
        mean = self.actor_mean(feat)
        value = self.critic(feat)
        next_force = self.force_predictor(feat)
        state_logits = self.state_classifier(feat)
        log_std = torch.clamp(self.log_std, LOG_STD_MIN, LOG_STD_MAX)
        std = log_std.exp().expand_as(mean)
        if return_attention:
            extra['feat'] = feat
            return Normal(mean, std), value, next_force, state_logits, extra
        return Normal(mean, std), value, next_force, state_logits

    def act(self, state, ft_history):
        dist, value, _, _ = self._forward_all(state, ft_history)
        pre_tanh = dist.rsample()
        squashed = torch.tanh(pre_tanh)
        action = squashed * self.action_scale
        log_prob = dist.log_prob(pre_tanh)
        log_prob -= torch.log(1 - squashed.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, value

    def evaluate_actions(self, state, ft_history, action):
        dist, value, next_force, state_logits = self._forward_all(state, ft_history)
        scaled_action = torch.clamp(action / self.action_scale, -0.999999, 0.999999)
        pre_tanh = torch.atanh(scaled_action)
        log_prob = dist.log_prob(pre_tanh)
        log_prob -= torch.log(1 - scaled_action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        entropy = dist.entropy().sum(dim=-1, keepdim=True)
        return log_prob, entropy, value, next_force, state_logits

    def deterministic_action(self, state, ft_history):
        dist, _, _, _ = self._forward_all(state, ft_history)
        return torch.tanh(dist.mean) * self.action_scale

    def analyze(self, state, ft_history):
        """
        Analysis-only path.

        Returns:
          - deterministic action
          - value
          - predicted next force
          - predicted state logits
          - shared encoder feature
          - attention scores over the K history steps
          - intermediate encoder features for ablation analysis
        """
        dist, value, next_force, state_logits, extra = self._forward_all(
            state, ft_history, return_attention=True
        )
        action = torch.tanh(dist.mean) * self.action_scale
        return {
            'action': action,
            'value': value,
            'next_force': next_force,
            'state_logits': state_logits,
            'feat': extra['feat'],
            'attention_scores': extra['attention_scores'],
            'state_feat': extra['state_feat'],
            'ft_feat': extra['ft_feat'],
        }
