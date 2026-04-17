"""
rollout_buffer.py — On-policy rollout storage for PPO.
"""

import numpy as np
import torch


class PPORolloutBuffer:
    def __init__(self, capacity, state_dim, action_dim, ft_dim, ft_history_len):
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ft_dim = ft_dim
        self.ft_history_len = ft_history_len
        self.reset()

    def reset(self):
        self.ptr = 0
        self.full = False
        c = self.capacity
        self.states = np.zeros((c, self.state_dim), dtype=np.float32)
        self.ft_histories = np.zeros((c, self.ft_history_len, self.ft_dim), dtype=np.float32)
        self.actions = np.zeros((c, self.action_dim), dtype=np.float32)
        self.log_probs = np.zeros((c, 1), dtype=np.float32)
        self.rewards = np.zeros((c, 1), dtype=np.float32)
        self.dones = np.zeros((c, 1), dtype=np.float32)
        self.values = np.zeros((c, 1), dtype=np.float32)
        self.advantages = np.zeros((c, 1), dtype=np.float32)
        self.returns = np.zeros((c, 1), dtype=np.float32)

    def add(self, state, ft_history, action, log_prob, reward, done, value):
        self.states[self.ptr] = state
        self.ft_histories[self.ptr] = ft_history
        self.actions[self.ptr] = action
        self.log_probs[self.ptr] = log_prob
        self.rewards[self.ptr] = reward
        self.dones[self.ptr] = float(done)
        self.values[self.ptr] = value
        self.ptr += 1
        if self.ptr >= self.capacity:
            self.full = True
            self.ptr = self.capacity

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        last_adv = 0.0
        size = self.size
        for t in reversed(range(size)):
            if t == size - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value = last_value
            else:
                next_non_terminal = 1.0 - self.dones[t + 1]
                next_value = self.values[t + 1]

            delta = self.rewards[t] + gamma * next_value * next_non_terminal - self.values[t]
            last_adv = delta + gamma * gae_lambda * next_non_terminal * last_adv
            self.advantages[t] = last_adv
        self.returns[:size] = self.advantages[:size] + self.values[:size]

    @property
    def size(self):
        return self.capacity if self.full else self.ptr

    def get(self, device='cpu'):
        size = self.size
        return {
            'state': torch.FloatTensor(self.states[:size]).to(device),
            'ft_history': torch.FloatTensor(self.ft_histories[:size]).to(device),
            'action': torch.FloatTensor(self.actions[:size]).to(device),
            'log_prob': torch.FloatTensor(self.log_probs[:size]).to(device),
            'reward': torch.FloatTensor(self.rewards[:size]).to(device),
            'done': torch.FloatTensor(self.dones[:size]).to(device),
            'value': torch.FloatTensor(self.values[:size]).to(device),
            'advantage': torch.FloatTensor(self.advantages[:size]).to(device),
            'return': torch.FloatTensor(self.returns[:size]).to(device),
        }

