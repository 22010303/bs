"""
td3_trainer.py — TD3 trainer for residual RL.
"""

import os
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from residual_rl.replay_buffer import ReplayBuffer
from .config import ResidualTD3Config
from .td3_policy import TD3Actor, TD3Critic


class RunningNormalizer:
    def __init__(self, dim, device='cpu'):
        self.mean = torch.zeros(dim, device=device)
        self.var = torch.ones(dim, device=device)
        self.count = 1e-4

    def update(self, x):
        x_t = torch.FloatTensor(x).to(self.mean.device)
        self.count += 1
        delta = x_t - self.mean
        self.mean += delta / self.count
        self.var += delta * (x_t - self.mean)

    def normalize(self, x_tensor):
        std = torch.sqrt(self.var / max(self.count, 1) + 1e-8)
        return (x_tensor - self.mean) / std

    def state_dict(self):
        return {'mean': self.mean, 'var': self.var, 'count': self.count}

    def load_state_dict(self, d):
        self.mean = d['mean']
        self.var = d['var']
        self.count = d['count']


class TD3Trainer:
    def __init__(self, config: ResidualTD3Config = None, device='cuda', use_wandb=False):
        self.cfg = config or ResidualTD3Config()
        self.device = torch.device(device)
        self.use_wandb = use_wandb

        self.actor = TD3Actor(self.cfg).to(self.device)
        self.actor_target = copy.deepcopy(self.actor).to(self.device)
        self.critic = TD3Critic(self.cfg).to(self.device)
        self.critic_target = copy.deepcopy(self.critic).to(self.device)
        for p in self.actor_target.parameters():
            p.requires_grad = False
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor_optimizer = Adam(self.actor.parameters(), lr=self.cfg.lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=self.cfg.lr_critic)

        self.buffer = ReplayBuffer(
            capacity=self.cfg.replay_capacity,
            state_dim=self.cfg.state_dim,
            action_dim=self.cfg.action_dim,
            ft_dim=self.cfg.ft_dim,
            ft_history_len=self.cfg.force_history_len,
        )
        self.obs_normalizer = RunningNormalizer(self.cfg.state_dim, device=self.device)

        self.total_steps = 0
        self.total_episodes = 0
        self.best_eval_reward = -float('inf')
        self._update_step = 0

        self.action_limit = torch.tensor([
            self.cfg.max_residual_pos, self.cfg.max_residual_pos, self.cfg.max_residual_pos,
            self.cfg.max_residual_rot, self.cfg.max_residual_rot, self.cfg.max_residual_rot,
        ], dtype=torch.float32, device=self.device)
        self.policy_noise = torch.tensor([
            self.cfg.policy_noise_pos, self.cfg.policy_noise_pos, self.cfg.policy_noise_pos,
            self.cfg.policy_noise_rot, self.cfg.policy_noise_rot, self.cfg.policy_noise_rot,
        ], dtype=torch.float32, device=self.device)
        self.noise_clip = torch.tensor([
            self.cfg.noise_clip_pos, self.cfg.noise_clip_pos, self.cfg.noise_clip_pos,
            self.cfg.noise_clip_rot, self.cfg.noise_clip_rot, self.cfg.noise_clip_rot,
        ], dtype=torch.float32, device=self.device)
        self.exploration_noise = torch.tensor([
            self.cfg.exploration_noise_pos, self.cfg.exploration_noise_pos, self.cfg.exploration_noise_pos,
            self.cfg.exploration_noise_rot, self.cfg.exploration_noise_rot, self.cfg.exploration_noise_rot,
        ], dtype=torch.float32, device=self.device)

    def select_action(self, state, ft_history, deterministic=False):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        hist_t = torch.FloatTensor(ft_history).unsqueeze(0).to(self.device)
        if self.cfg.obs_normalize:
            state_t = self.obs_normalizer.normalize(state_t)
        with torch.no_grad():
            action = self.actor(state_t, hist_t).squeeze(0)
        if not deterministic:
            action = action + torch.randn_like(action) * self.exploration_noise
        action = torch.max(torch.min(action, self.action_limit), -self.action_limit)
        return action.cpu().numpy().astype(np.float32)

    def store_transition(self, state, ft_history, action, reward, next_state, next_ft_history, done):
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

    def update(self):
        if len(self.buffer) < self.cfg.batch_size:
            return {}

        self._update_step += 1
        batch = self.buffer.sample(self.cfg.batch_size, device=self.device)
        state = batch['state']
        ft_history = batch['ft_history']
        action = batch['action']
        reward = batch['reward'] * self.cfg.reward_scale
        next_state = batch['next_state']
        next_ft_history = batch['next_ft_history']
        done = batch['done']

        if self.cfg.obs_normalize:
            state = self.obs_normalizer.normalize(state)
            next_state = self.obs_normalizer.normalize(next_state)

        with torch.no_grad():
            noise = torch.randn_like(action) * self.policy_noise
            noise = torch.max(torch.min(noise, self.noise_clip), -self.noise_clip)
            next_action = self.actor_target(next_state, next_ft_history) + noise
            next_action = torch.max(torch.min(next_action, self.action_limit), -self.action_limit)
            q1_next, q2_next = self.critic_target(next_state, next_ft_history, next_action)
            q_target = reward + (1 - done) * self.cfg.gamma * torch.min(q1_next, q2_next)

        q1, q2 = self.critic(state, ft_history, action)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        critic_grad_norm = nn.utils.clip_grad_norm_(
            self.critic.parameters(), self.cfg.grad_clip_norm
        )
        self.critic_optimizer.step()

        info = {
            'critic_loss': critic_loss.item(),
            'critic_grad_norm': critic_grad_norm.item(),
            'q1_mean': q1.mean().item(),
            'q2_mean': q2.mean().item(),
            'q_target_mean': q_target.mean().item(),
            'reward_batch_mean': reward.mean().item(),
            'reward_batch_std': reward.std().item(),
        }

        if self._update_step % self.cfg.policy_delay == 0:
            actor_action = self.actor(state, ft_history)
            actor_loss = -self.critic.q1(state, ft_history, actor_action).mean()

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            actor_grad_norm = nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.cfg.grad_clip_norm
            )
            self.actor_optimizer.step()

            with torch.no_grad():
                for p, pt in zip(self.critic.parameters(), self.critic_target.parameters()):
                    pt.data.mul_(1 - self.cfg.tau)
                    pt.data.add_(self.cfg.tau * p.data)
                for p, pt in zip(self.actor.parameters(), self.actor_target.parameters()):
                    pt.data.mul_(1 - self.cfg.tau)
                    pt.data.add_(self.cfg.tau * p.data)

            info.update({
                'actor_loss': actor_loss.item(),
                'actor_grad_norm': actor_grad_norm.item(),
            })

        if self.use_wandb:
            import wandb
            wandb.log({f'td3/{k}': v for k, v in info.items()}, step=self.total_steps)

        return info

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        save_dict = {
            'actor': self.actor.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'best_eval_reward': self.best_eval_reward,
            'config': self.cfg,
        }
        if self.cfg.obs_normalize:
            save_dict['obs_normalizer'] = self.obs_normalizer.state_dict()
        torch.save(save_dict, os.path.join(path, 'td3_checkpoint.pt'))
        print(f"  Saved -> {path}")

    def load(self, path):
        ckpt = torch.load(os.path.join(path, 'td3_checkpoint.pt'),
                          weights_only=False, map_location=self.device)
        self.actor.load_state_dict(ckpt['actor'])
        self.actor_target.load_state_dict(ckpt['actor_target'])
        self.critic.load_state_dict(ckpt['critic'])
        self.critic_target.load_state_dict(ckpt['critic_target'])
        self.actor_optimizer.load_state_dict(ckpt['actor_optimizer'])
        self.critic_optimizer.load_state_dict(ckpt['critic_optimizer'])
        self.total_steps = ckpt['total_steps']
        self.total_episodes = ckpt['total_episodes']
        self.best_eval_reward = ckpt.get('best_eval_reward', -float('inf'))
        if 'obs_normalizer' in ckpt:
            self.obs_normalizer.load_state_dict(ckpt['obs_normalizer'])
        print(f"  Loaded <- {path} (step {self.total_steps})")

