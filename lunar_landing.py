import pandas as pd
import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.autograd as autograd

from torch.autograd import Variable
from collections import deque, namedtuple

class Network(nn.Module):
  def __init__(self, state_size, action_size, seed=42) -> None:
    """

    Args:
      state_size: Number of states in the environment
      action_size: Number of actions that can be taken
      seed: Random seed number
    """
    super(Network, self).__init__()
    self.seed = torch.manual_seed(seed)
    self.fc1 = nn.Linear(state_size, 64)
    self.fc2 = nn.Linear(64, 64)
    self.fc3 = nn.Linear(64, action_size)

  def forward(self, state):
    x = self.fc1(state)
    x = F.relu(x)
    x = self.fc2(x)
    x = F.relu(x)
    return self.fc3(x)

import gymnasium as gym
env = gym.make('LunarLander-v2')
state_size = env.observation_space.shape
number_actions = env.action_space.n

# Initializing Hyperparameters (Tweak these variables to optimize the model)
learning_rate = 5e-4
mini_batch_size = 100

# Discount Factor
gamma = 0.99
replay_buffer_size = int(1e5)

# Interpolation Parameter
tau = 1e-3

class ReplayMemory(object):
  def __init__(self,capacity) -> None:
    self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    self.capacity = capacity
    self.memory = []

  def push(self, event):
    self.memory.append(event)
    if len(self.memory) > self.capacity:
      del self.memory[0]

  def sample(self, batch_size):
    experiences = random.sample(self.memory, k=batch_size)
    states = torch.from_numpy(
        np.vstack(
            [e [0] for e in experiences if e is not None])).float().to(self.device)

    actions = torch.from_numpy(
        np.vstack(
            [e [1] for e in experiences if e is not None])).long().to(self.device)

    rewards = torch.from_numpy(
        np.vstack(
            [e [2] for e in experiences if e is not None])).float().to(self.device)

    next_states = torch.from_numpy(
        np.vstack(
            [e [3] for e in experiences if e is not None])).float().to(self.device)

    dones = torch.from_numpy(
        np.vstack(
            [e [4] for e in experiences if e is not None])).astype(np.uint8).to(self.device)