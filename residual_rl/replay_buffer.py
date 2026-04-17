"""
replay_buffer.py — Off-policy replay buffer for SAC training.

Stores transitions with force/torque history for the temporal attention module.
"""

import numpy as np
import torch


class ReplayBuffer:
    """
    Fixed-size replay buffer with force history support.

    Each transition stores:
      - state: (state_dim,) — concatenated state vector
      - ft_history: (K, ft_dim) — force/torque history window
      - action: (action_dim,) — residual action taken
      - reward: scalar
      - next_state: (state_dim,)
      - next_ft_history: (K, ft_dim)
      - done: bool
    """

    def __init__(self, capacity, state_dim, action_dim, ft_dim=6, ft_history_len=10):
        self.capacity = capacity
        self.ptr = 0
        self.size = 0

        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.ft_histories = np.zeros((capacity, ft_history_len, ft_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.next_ft_histories = np.zeros((capacity, ft_history_len, ft_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

    def add(self, state, ft_history, action, reward, next_state, next_ft_history, done):
        """Add a single transition."""
        self.states[self.ptr] = state
        self.ft_histories[self.ptr] = ft_history
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.next_ft_histories[self.ptr] = next_ft_history
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device='cpu'):
        """
        Sample a random mini-batch.

        Returns:
            dict of tensors on the specified device
        """
        idxs = np.random.randint(0, self.size, size=batch_size)

        return {
            'state': torch.FloatTensor(self.states[idxs]).to(device),
            'ft_history': torch.FloatTensor(self.ft_histories[idxs]).to(device),
            'action': torch.FloatTensor(self.actions[idxs]).to(device),
            'reward': torch.FloatTensor(self.rewards[idxs]).to(device),
            'next_state': torch.FloatTensor(self.next_states[idxs]).to(device),
            'next_ft_history': torch.FloatTensor(self.next_ft_histories[idxs]).to(device),
            'done': torch.FloatTensor(self.dones[idxs]).to(device),
        }

    def __len__(self):
        return self.size
