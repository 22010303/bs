"""
aux_trainer.py — PPO trainer with force prediction and state classification.
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from .aux_config import ResidualPPOAuxConfig
from .aux_policy import PPOAuxActorCritic
from .aux_rollout_buffer import PPOAuxRolloutBuffer


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


class PPOAuxTrainer:
    def __init__(self, config: ResidualPPOAuxConfig = None, device='cuda', use_wandb=False):
        self.cfg = config or ResidualPPOAuxConfig()
        self.device = torch.device(device)
        self.use_wandb = use_wandb

        self.policy = PPOAuxActorCritic(self.cfg).to(self.device)
        self.optimizer = Adam(self.policy.parameters(), lr=self.cfg.lr_actor)
        self.buffer = PPOAuxRolloutBuffer(
            capacity=self.cfg.rollout_steps,
            state_dim=self.cfg.state_dim,
            action_dim=self.cfg.action_dim,
            ft_dim=self.cfg.ft_dim,
            ft_history_len=self.cfg.force_history_len,
        )
        self.obs_normalizer = RunningNormalizer(self.cfg.state_dim, device=self.device)
        self.total_steps = 0
        self.total_episodes = 0
        self.best_eval_reward = -float('inf')

    def log_episode(self, ep_reward, ep_len, ep_info, reward_info=None):
        if not self.use_wandb:
            return
        import wandb

        log_dict = {
            'episode/reward': ep_reward,
            'episode/length': ep_len,
            'episode/success': float(ep_info.get('success', False)),
            'episode/contact_step': ep_info.get('contact_step', -1),
            'episode/rl_steps': ep_info.get('steps_after_contact', 0),
            'episode/state_ratio_smooth': ep_info.get('state_ratio_smooth', 0.0),
            'episode/state_ratio_jamming': ep_info.get('state_ratio_jamming', 0.0),
            'episode/state_ratio_bottoming': ep_info.get('state_ratio_bottoming', 0.0),
            'episode/state_count_smooth': ep_info.get('state_count_smooth', 0),
            'episode/state_count_jamming': ep_info.get('state_count_jamming', 0),
            'episode/state_count_bottoming': ep_info.get('state_count_bottoming', 0),
        }
        if reward_info:
            for k in [
                'total_force', 'lateral_force', 'axial_force',
                'bending_torque', 'insertion_progress', 'cumulative_force',
            ]:
                if k in reward_info:
                    log_dict[f'episode/{k}'] = reward_info[k]
        wandb.log(log_dict, step=self.total_steps)

    def store_transition(
        self, state, ft_history, action, log_prob, reward, done, value,
        next_force_target, state_label
    ):
        if self.cfg.obs_normalize:
            self.obs_normalizer.update(state)
        self.buffer.add(
            state, ft_history, action, log_prob, reward, done, value,
            next_force_target, state_label,
        )

    def _normalize_state(self, state):
        return self.obs_normalizer.normalize(state) if self.cfg.obs_normalize else state

    def act(self, state, ft_history):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        hist_t = torch.FloatTensor(ft_history).unsqueeze(0).to(self.device)
        state_t = self._normalize_state(state_t)
        with torch.no_grad():
            action, log_prob, value = self.policy.act(state_t, hist_t)
        return action.squeeze(0).cpu().numpy(), float(log_prob.item()), float(value.item())

    def finish_rollout(self, last_state, last_ft_history, last_done):
        last_value = np.array([[0.0]], dtype=np.float32)
        if not last_done and self.buffer.size > 0:
            s_t = torch.FloatTensor(last_state).unsqueeze(0).to(self.device)
            h_t = torch.FloatTensor(last_ft_history).unsqueeze(0).to(self.device)
            s_t = self._normalize_state(s_t)
            with torch.no_grad():
                _, value, _, _ = self.policy._forward_all(s_t, h_t)
            last_value = value.cpu().numpy()
        self.buffer.compute_returns_and_advantages(last_value, self.cfg.gamma, self.cfg.gae_lambda)

    def update(self):
        data = self.buffer.get(device=self.device)
        if data['state'].shape[0] == 0:
            return {}

        states = self._normalize_state(data['state'])
        ft_histories = data['ft_history']
        actions = data['action']
        old_log_probs = data['log_prob']
        advantages = data['advantage']
        returns = data['return']
        next_force_targets = data['next_force_target']
        state_labels = data['state_label']

        if self.cfg.normalize_advantage:
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = states.shape[0]
        idxs = np.arange(batch_size)
        metrics = {}

        for _ in range(self.cfg.ppo_epochs):
            np.random.shuffle(idxs)
            for start in range(0, batch_size, self.cfg.minibatch_size):
                mb = idxs[start:start + self.cfg.minibatch_size]
                mb_states = states[mb]
                mb_histories = ft_histories[mb]
                mb_actions = actions[mb]
                mb_old_log_probs = old_log_probs[mb]
                mb_advantages = advantages[mb]
                mb_returns = returns[mb]
                mb_next_force = next_force_targets[mb]
                mb_state_labels = state_labels[mb]

                new_log_probs, entropy, values, pred_next_force, state_logits = \
                    self.policy.evaluate_actions(mb_states, mb_histories, mb_actions)

                ratio = (new_log_probs - mb_old_log_probs).exp()
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(
                    ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef
                )
                actor_loss = torch.max(pg_loss1, pg_loss2).mean()
                value_loss = 0.5 * (mb_returns - values).pow(2).mean()
                entropy_loss = entropy.mean()

                if self.cfg.use_force_prediction_head:
                    force_pred_loss = F.mse_loss(pred_next_force, mb_next_force)
                else:
                    force_pred_loss = torch.zeros((), device=self.device)

                if self.cfg.use_state_classification_head:
                    state_cls_loss = F.cross_entropy(state_logits, mb_state_labels)
                    pred_labels = state_logits.argmax(dim=-1)
                    state_cls_acc = (pred_labels == mb_state_labels).float().mean().item()
                else:
                    state_cls_loss = torch.zeros((), device=self.device)
                    state_cls_acc = 0.0

                total_loss = (
                    actor_loss +
                    self.cfg.value_coef * value_loss -
                    self.cfg.entropy_coef * entropy_loss +
                    self.cfg.force_pred_loss_coef * force_pred_loss +
                    self.cfg.state_cls_loss_coef * state_cls_loss
                )

                self.optimizer.zero_grad()
                total_loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(), self.cfg.max_grad_norm)
                self.optimizer.step()

                approx_kl = (mb_old_log_probs - new_log_probs).mean().item()
                pred_labels = state_logits.argmax(dim=-1)
                state_cls_acc = (pred_labels == mb_state_labels).float().mean().item()
                metrics = {
                    'actor_loss': actor_loss.item(),
                    'value_loss': value_loss.item(),
                    'entropy': entropy_loss.item(),
                    'approx_kl': approx_kl,
                    'grad_norm': grad_norm.item(),
                    'force_pred_loss': force_pred_loss.item(),
                    'state_cls_loss': state_cls_loss.item(),
                    'state_cls_acc': state_cls_acc,
                    'return_mean': returns.mean().item(),
                }
                if approx_kl > self.cfg.target_kl:
                    break
            if metrics.get('approx_kl', 0.0) > self.cfg.target_kl:
                break

        if self.use_wandb and metrics:
            import wandb
            wandb.log({f'ppo_aux/{k}': v for k, v in metrics.items()}, step=self.total_steps)

        self.buffer.reset()
        return metrics

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        save_dict = {
            'policy': self.policy.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_steps': self.total_steps,
            'total_episodes': self.total_episodes,
            'best_eval_reward': self.best_eval_reward,
            'config': self.cfg,
        }
        if self.cfg.obs_normalize:
            save_dict['obs_normalizer'] = self.obs_normalizer.state_dict()
        torch.save(save_dict, os.path.join(path, 'ppo_aux_checkpoint.pt'))
        print(f"  Saved -> {path}")

    def load(self, path):
        ckpt = torch.load(os.path.join(path, 'ppo_aux_checkpoint.pt'),
                          weights_only=False, map_location=self.device)
        self.policy.load_state_dict(ckpt['policy'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.total_steps = ckpt['total_steps']
        self.total_episodes = ckpt['total_episodes']
        self.best_eval_reward = ckpt.get('best_eval_reward', -float('inf'))
        if 'obs_normalizer' in ckpt:
            self.obs_normalizer.load_state_dict(ckpt['obs_normalizer'])
        print(f"  Loaded <- {path} (step {self.total_steps})")
