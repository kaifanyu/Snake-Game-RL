import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter


class NN(nn.Module):
    def __init__(self, num_states, num_actions, dim):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(num_states, dim),
            nn.ReLU(),
            nn.Linear(dim, num_actions),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        return self.nn(x)

class Policy():
    def __init__(self):
        self.hidden_dim = 64
        self.discount = 0.9
        self.learning_rate = 0.001
        self.num_states = 11
        self.num_actions = 3
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.policy = NN(num_states=self.num_states, num_actions=self.num_actions, dim=self.hidden_dim)
        self.policy.to(self.device)

        
        # Value network (outputs baseline value function V(s))
        self.value_net = NN(num_states=self.num_states, num_actions=1, dim=self.hidden_dim)  # Outputs a scalar V(s)
        self.value_net.to(self.device)


        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.learning_rate)

    def action(self, state):
        # convert state to tensor
        state_tensor = torch.tensor(state).float().unsqueeze(0)
        with torch.no_grad():  # No gradient needed during inference
            action_probs = self.policy(state_tensor)  # Get action probabilities

        # sample action from action distribution
        action = torch.multinomial(action_probs, num_samples=1).item()

        return action

    def update(self, states, actions, reward):
        #sequence contains an entire time-series of how we reached the terminal goal
        # convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)


        for i in len(states):

