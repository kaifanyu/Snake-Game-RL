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

class PPO():
    def __init__(self):
        self.hidden_dim = 64
        self.num_states = 11
        self.num_actions = 3
        self.discount = 0.9
        self.value_learning_rate = 0.01
        self.theta_learning_rate = 0.001
        self.epsilon = 0.2
        self.max_change = 1 + self.epsilon
        self.min_change = 1 - self.epsilon
        self.GAE = 0.95 # lambda value between 0.9 and 1
        self.batch_size = 2048
        self.mini_batch = 256
        self.num_mini_batch = self.batch_size // self.mini_batch
        self.ppo_epochs = 6
        self.weighting_factor = 0.5
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # policy network
        self.policy_net = NN(num_states=self.num_states, num_actions=self.num_actions, dim=self.hidden_dim)
        self.policy_net.to(self.device)
        
        # Value network (outputs baseline value function V(s))
        self.value_net = Value_NN(num_states=self.num_states, output=1, dim=self.hidden_dim)  # Outputs a scalar V(s)
        self.value_net.to(self.device)


        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.theta_learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.value_learning_rate)

        self.writer = SummaryWriter('runs/snake_dqn')  # specify log dir

        # load model if exist
        self._load_model()

    def _load_model(self):
        value_checkpoint_path = 'models/ppo_value_net.pth'
        policy_checkpoint_path = 'models/ppo_policy_net.pth'
        if os.path.exists(value_checkpoint_path) and os.path.exists(policy_checkpoint_path):
            self.policy_net.load_state_dict(torch.load(policy_checkpoint_path, map_location=self.device, weights_only=True))
            self.value_net.load_state_dict(torch.load(value_checkpoint_path, map_location=self.device, weights_only=True))
            print(f"Loaded existing weights from {policy_checkpoint_path} and {value_checkpoint_path}.")
        else:
            print("No existing checkpoint found. Starting fresh.")

    def action(self, state):
        # convert state to tensor
        state_tensor = torch.tensor(state).float().unsqueeze(0)
        with torch.no_grad():  # No gradient needed during inference
            action_probs = self.policy_net(state_tensor)  # Get action probabilities

        # sample action from action distribution
        action = torch.multinomial(action_probs, num_samples=1).item()

        return action

    def update(self, batch, global_step):

        # Unpack batch into separate lists
        old_states, action_indices, new_states, rewards, game_overs = zip(*batch)

        # Convert to tensors
        old_states = torch.tensor(old_states, dtype=torch.float32).to(self.device)
        action_indices = torch.tensor(action_indices, dtype=torch.long).to(self.device)
        new_states = torch.tensor(new_states, dtype=torch.float32).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        game_overs = torch.tensor(game_overs, dtype=torch.bool).to(self.device)

        with torch.no_grad():
            # Get old policy probability distribution based on state
            policy_old = self.policy_net(old_states)
            old_probs = policy_old.gather(1, action_indices.unsqueeze(1)).squeeze(1)
            old_log_probs = torch.log(old_probs + 1e-8)

        # Compute V(s) - Value of old states
        V_s = self.value_net(old_states).squeeze(1)  # shape [N]
        # Compute V(st+1) - Value of new states
        V_snew = self.value_net(new_states).squeeze(1)  # shape [N]

        advantages = torch.zeros_like(rewards).to(self.device)
        returns = torch.zeros_like(rewards).to(self.device)

        # Compute delta = r + discount * V(s') - V(s) (only if not terminal)
        delta = rewards + V_snew * self.discount * (~game_overs) - V_s

        # Compute advantages using Generalized Advantage Estimation (GAE)
        advantages[-1] = delta[-1]
        for t in reversed(range(len(rewards) - 1)):
            if game_overs[t]:
                advantages[t] = delta[t]
            else:
                advantages[t] = delta[t] + (self.discount * self.GAE * advantages[t+1])

        returns = advantages + V_s  # Returns(t) = Advantage(t) + V(s)

        # Mini-batch training: split batch into smaller batches
        for epoch in range(self.ppo_epochs):
            indices = torch.randperm(self.batch_size)

            for t in range(self.num_mini_batch):
                idx = indices[t * self.mini_batch: (t + 1) * self.mini_batch]

                # Select mini-batch samples
                mini_states = old_states[idx]
                mini_actions = action_indices[idx]
                mini_advantages = advantages[idx].detach()
                mini_returns = returns[idx].detach()
                mini_old_log_probs = old_log_probs[idx].detach()

                # Get new policy π(new)
                new_policy_probs = self.policy_net(mini_states)
                new_probs = new_policy_probs.gather(1, mini_actions.unsqueeze(1)).squeeze(1)
                new_log_probs = torch.log(new_probs + 1e-8)


                # Compute policy ratio r(θ)
                ratio = torch.exp(new_log_probs - mini_old_log_probs)

                # Compute clipped policy loss
                unclipped = ratio * mini_advantages
                clipped = torch.clamp(ratio, self.min_change, self.max_change) * mini_advantages
                policy_loss = -torch.mean(torch.min(unclipped, clipped))

                # Compute value loss
                value_loss = nn.functional.mse_loss(self.value_net(mini_states).squeeze(1), mini_returns)

                total_loss = policy_loss + value_loss

                # Backpropagation
                self.policy_optimizer.zero_grad()
                self.value_optimizer.zero_grad()
                total_loss.backward()
                self.policy_optimizer.step()
                self.value_optimizer.step()

        self.writer.add_scalar("Loss/total_loss", total_loss.item(), global_step)
        self.writer.add_scalar("Loss/policy_loss", policy_loss.item(), global_step)
        self.writer.add_scalar("Loss/value_loss", value_loss.item(), global_step)

    def save_model(self):
        print("PPO Model saved")
        torch.save(self.policy_net.state_dict(), 'models/ppo_policy_net.pth')
        torch.save(self.value_net.state_dict(), 'models/ppo_value_net.pth')