"""
sac_trainer.py — SAC trainer for residual RL.

Features:
  - Gradient clipping (max_norm=0.5) on all three losses
  - Wandb logging: Q values, losses, alpha, entropy, reward, force metrics
  - Running observation normalization
  - Reward scaling
"""

import os
import copy
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .config import ResidualRLConfig
from .residual_policy import ResidualActor, ResidualCritic
from .replay_buffer import ReplayBuffer


class RunningNormalizer:
    """Running mean/std for observation normalization."""

    def __init__(self, dim, device='cpu'):
        self.mean = torch.zeros(dim, device=device)
        self.var = torch.ones(dim, device=device)
        self.count = 1e-4

    def update(self, x):
        """Update with a single observation (numpy array)."""
        x_t = torch.FloatTensor(x).to(self.mean.device)
        self.count += 1
        delta = x_t - self.mean
        self.mean += delta / self.count
        self.var += delta * (x_t - self.mean)

    def normalize(self, x_tensor):
        """Normalize a batch tensor."""
        std = torch.sqrt(self.var / max(self.count, 1) + 1e-8)
        return (x_tensor - self.mean) / std

    def state_dict(self):
        return {'mean': self.mean, 'var': self.var, 'count': self.count}

    def load_state_dict(self, d):
        self.mean = d['mean']
        self.var = d['var']
        self.count = d['count']


class SACTrainer:

    def __init__(self, config: ResidualRLConfig = None, device='cuda', use_wandb=False):
        self.cfg = config or ResidualRLConfig()
        self.device = torch.device(device)
        self.use_wandb = use_wandb

        # Networks
        self.actor = ResidualActor(self.cfg).to(self.device)
        self.critic = ResidualCritic(self.cfg).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Optimizers
        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.cfg.lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.cfg.lr_critic)

        # Auto entropy tuning (DISABLED — using fixed alpha)
        # self.log_alpha = torch.tensor(
        #     np.log(self.cfg.init_alpha), dtype=torch.float32,
        #     device=self.device, requires_grad=True
        # )
        # self.alpha_optimizer = Adam([self.log_alpha], lr=self.cfg.lr_alpha)
        # self.target_entropy = self.cfg.target_entropy

        # Fixed alpha (constant, no gradient)
        self._fixed_alpha = self.cfg.init_alpha

        # Replay buffer
        self.buffer = ReplayBuffer(
            capacity=self.cfg.replay_capacity,
            state_dim=self.cfg.state_dim,
            action_dim=self.cfg.action_dim,
            ft_dim=self.cfg.ft_dim,
            ft_history_len=self.cfg.force_history_len,
        )

        # Observation normalizer
        self.obs_normalizer = RunningNormalizer(self.cfg.state_dim, device=self.device)

        # Counters
        self.total_steps = 0
        self.total_episodes = 0
        self.best_eval_reward = -float('inf')
        self._last_update_info = {}

    @property
    def alpha(self):
        # return self.log_alpha.exp()  # auto-tuned version
        return self._fixed_alpha       # fixed constant

    # ------------------------------------------------------------------
    # SAC update with gradient clipping + detailed metrics
    # ------------------------------------------------------------------
    def update(self):
        if len(self.buffer) < self.cfg.batch_size:
            return {}

        batch = self.buffer.sample(self.cfg.batch_size, device=self.device)
        state = batch['state']
        ft_history = batch['ft_history']
        action = batch['action']
        reward = batch['reward']
        next_state = batch['next_state']
        next_ft_history = batch['next_ft_history']
        done = batch['done']

        # Normalize observations
        if self.cfg.obs_normalize:
            state = self.obs_normalizer.normalize(state)
            next_state = self.obs_normalizer.normalize(next_state)

        # Scale reward
        reward = reward * self.cfg.reward_scale

        # --- Critic update ---
        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state, next_ft_history)
            q1_next, q2_next = self.critic_target(next_state, next_ft_history, next_action)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_prob
            q_target = reward + (1 - done) * self.cfg.gamma * q_next

        q1, q2 = self.critic(state, ft_history, action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.cfg.grad_clip_norm
        )
        self.critic_optimizer.step()

        # --- Actor update ---
        new_action, log_prob = self.actor.sample(state, ft_history)
        q1_new = self.critic.q1(state, ft_history, new_action)
        actor_loss = (self.alpha * log_prob - q1_new).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        actor_grad_norm = nn.utils.clip_grad_norm_(
            self.actor.parameters(), self.cfg.grad_clip_norm
        )
        self.actor_optimizer.step()

        # --- Alpha update (DISABLED — fixed alpha) ---
        # alpha_loss = -(self.log_alpha * (log_prob.detach() + self.target_entropy)).mean()
        # self.alpha_optimizer.zero_grad()
        # alpha_loss.backward()
        # self.alpha_optimizer.step()
        alpha_loss_val = 0.0

        # --- Soft target update ---
        with torch.no_grad():
            for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                pt.data.mul_(1 - self.cfg.tau)
                pt.data.add_(self.cfg.tau * p.data)

        # --- Metrics ---
        entropy = -log_prob.mean().item()
        info = {
            'critic_loss': critic_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha_loss': alpha_loss_val,
            'alpha': self._fixed_alpha,
            'entropy': entropy,
            'q1_mean': q1.mean().item(),
            'q2_mean': q2.mean().item(),
            'q1_std': q1.std().item(),
            'q_target_mean': q_target.mean().item(),
            'critic_grad_norm': critic_grad_norm.item(),
            'actor_grad_norm': actor_grad_norm.item(),
            'reward_batch_mean': reward.mean().item(),
            'reward_batch_std': reward.std().item(),
            'log_std_mean': self.actor.log_std.data.mean().item(),
        }
        self._last_update_info = info

        # Wandb logging (per-update)
        if self.use_wandb:
            import wandb
            wandb.log({f'sac/{k}': v for k, v in info.items()}, step=self.total_steps)

        return info

    # ------------------------------------------------------------------
    # Log episode metrics to wandb
    # ------------------------------------------------------------------
    def log_episode(self, ep_reward, ep_len, ep_info, reward_info=None):
        """Log episode-level metrics."""
        if not self.use_wandb:
            return
        import wandb

        log_dict = {
            'episode/reward': ep_reward,
            'episode/length': ep_len,
            'episode/success': float(ep_info.get('success', False)),
            'episode/contact_step': ep_info.get('contact_step', -1),
            'episode/rl_steps': ep_info.get('steps_after_contact', 0),
        }

        # Force metrics from last reward_info
        if reward_info:
            for k in ['total_force', 'lateral_force', 'axial_force',
                       'bending_torque', 'insertion_progress', 'cumulative_force']:
                if k in reward_info:
                    log_dict[f'episode/{k}'] = reward_info[k]
            for k in ['r_force_mag', 'r_stability', 'r_depth', 'r_progress',
                       'p_fx', 'p_fy', 'p_fz', 'p_tx', 'p_ty']:
                if k in reward_info:
                    log_dict[f'reward/{k}'] = reward_info[k]

        wandb.log(log_dict, step=self.total_steps)

    # ------------------------------------------------------------------
    # Normalize + store transition
    # ------------------------------------------------------------------
    def store_transition(self, state, ft_history, action, reward, next_state, next_ft_history, done):
        """Update normalizer + store in buffer."""
        if self.cfg.obs_normalize:
            self.obs_normalizer.update(state)

        self.buffer.add(
            state=state,
            ft_history=ft_history,
            action=action,
            reward=reward,
            next_state=next_state,
            next_ft_history=next_ft_history,
            done=done,
        )

    # ------------------------------------------------------------------
    # Save / Load
    # ------------------------------------------------------------------
    def save(self, path):
        os.makedirs(path, exist_ok=True)
        save_dict = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            # 'log_alpha': self.log_alpha.data,  # disabled: fixed alpha
            # 'alpha_optimizer': self.alpha_optimizer.state_dict(),
            'fixed_alpha': self._fixed_alpha,
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'best_eval_reward': self.best_eval_reward,
            'config': self.cfg,
        }
        if self.cfg.obs_normalize:
            save_dict['obs_normalizer'] = self.obs_normalizer.state_dict()
        torch.save(save_dict, os.path.join(path, 'sac_checkpoint.pt'))
        print(f"  Saved -> {path}")

    def load(self, path):
        ckpt = torch.load(os.path.join(path, 'sac_checkpoint.pt'),
                          weights_only=False, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.critic.load_state_dict(ckpt['critic'])
        self.critic_target.load_state_dict(ckpt['critic_target'])
        self.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
        self.critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
        # self.log_alpha.data = ckpt['log_alpha']  # disabled: fixed alpha
        # self.alpha_optimizer.load_state_dict(ckpt['alpha_optimizer'])
        if 'fixed_alpha' in ckpt:
            self._fixed_alpha = ckpt['fixed_alpha']
        self.total_steps = ckpt['total_steps']
        self.total_episodes = ckpt['total_episodes']
        self.best_eval_reward = ckpt.get('best_eval_reward', -float('inf'))
        if 'obs_normalizer' in ckpt:
            self.obs_normalizer.load_state_dict(ckpt['obs_normalizer'])
        print(f"  Loaded <- {path} (step {self.total_steps})")
