import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter

class Value_NN(nn.Module):
    def __init__(self, num_states, output, dim):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(num_states, dim),
            nn.ReLU(),
            nn.Linear(dim, output)
        )

    def forward(self, x):
        return self.nn(x)

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
        self.value_learning_rate = 0.01
        self.theta_learning_rate = 0.001
        self.num_states = 11
        self.num_actions = 3
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.policy_net = NN(num_states=self.num_states, num_actions=self.num_actions, dim=self.hidden_dim)
        self.policy_net.to(self.device)

        
        # Value network (outputs baseline value function V(s))
        self.value_net = Value_NN(num_states=self.num_states, output=1, dim=self.hidden_dim)  # Outputs a scalar V(s)
        self.value_net.to(self.device)


        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.theta_learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.value_learning_rate)

    def action(self, state):
        # convert state to tensor
        state_tensor = torch.tensor(state).float().unsqueeze(0)
        with torch.no_grad():  # No gradient needed during inference
            action_probs = self.policy_net(state_tensor)  # Get action probabilities

        # sample action from action distribution
        action = torch.multinomial(action_probs, num_samples=1).item()

        return action

    def update(self, states, actions, reward):
        #sequence contains an entire time-series of how we reached the terminal goal
        # convert to tensors
        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)

        # Compute Returns G_t with Discounting
        G = torch.zeros(len(states), dtype=torch.float32).to(self.device)
        G[-1] = reward  # Assign last reward

        # GT = reward, GT-1 = discount * reward...
        for k in reversed(range(len(states) - 1)):  
            G[k] = self.discount * G[k + 1]  # Discounted return

        # Compute Advantage A = G - V(s, w) for all states
        V_pred = self.value_net(states).squeeze()  # Get predicted values V(s)
        advantage = G - V_pred  # Compute advantage function

        # Compute Policy Loss with Discount Factor
        action_probs = self.policy_net(states)  # Get policy output
        action_log_probs = torch.log(action_probs.gather(1, actions.unsqueeze(1)).squeeze())

        # Create discount factors: [γ^0, γ^1, ..., γ^(T-1)]
        discount_factors = torch.pow(self.discount, torch.arange(len(states), dtype=torch.float32).to(self.device))

        # Apply discounting to the advantage function
        policy_loss = -torch.sum(discount_factors * action_log_probs * advantage.detach())  # Negative for gradient ascent

        # Compute Value Function Loss (MSE Loss)
        value_loss = torch.nn.functional.mse_loss(V_pred, G)

        # Update Policy Network (Gradient Ascent)
        self.policy_optimizer.zero_grad()   # reset gradients
        # pass in discount * log(action prob) * advantage to backpropagate together with loss function of the hidden neural network
        policy_loss.backward()              # computes gradient (loss function)
        self.policy_optimizer.step()        # updates theta values

        # Update Value Network (Gradient Descent)
        self.value_optimizer.zero_grad()
        value_loss.backward()
        self.value_optimizer.step()