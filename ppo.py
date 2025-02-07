import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from collections import deque


class ValueNet(nn.Module):
    def __init__(self, num_states, alpha, fc1, chkpt_dir='models/ppo_value.pth'):
        super(ValueNet, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(num_states, fc1),
            nn.ReLU(),
            nn.Linear(fc1, 1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.checkpoint_file = chkpt_dir
    
    def forward(self, x):
        return self.critic(x)
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, weights_only=True))    

class PolicyNet(nn.Module):
    def __init__(self, num_states, num_actions, alpha, fc1, chkpt_dir='models/ppo_policy.pth'):
        super(PolicyNet, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(num_states, fc1),
            nn.ReLU(),
            nn.Linear(fc1, num_actions),
            nn.Softmax(dim=-1)
        )

        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.checkpoint_file = chkpt_dir

    def forward(self, x):
        return self.actor(x)
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.checkpoint_file, weights_only=True))    

    
class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.actions = []
        self.action_probs = []
        self.values = []
        self.rewards = []
        self.done = []

        self.batch_size = batch_size
    
    
    def generate_batches(self):
        n_states = len(self.states)

        # Convert lists to tensors
        states = torch.tensor(self.states, dtype=torch.float32)
        actions = torch.tensor(self.actions, dtype=torch.int64)
        action_probs = torch.tensor(self.action_probs, dtype=torch.float32)
        values = torch.tensor(self.values, dtype=torch.float32)
        rewards = torch.tensor(self.rewards, dtype=torch.float32)
        dones = torch.tensor(self.dones, dtype=torch.float32)

        # Shuffle indices
        indices = torch.randperm(n_states)  # Shuffles indices
        batches = [indices[i : i + self.batch_size] for i in range(0, n_states, self.batch_size)]

        return states[batches], actions[batches], action_probs[batches], values[batches], rewards[batches], dones[batches]
    
    def store_memory(self, state, action, action_prob, value, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.action_probs.append(action_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.actions = []
        self.action_probs = []
        self.values = []
        self.rewards = []
        self.dones = []

class PPO:
    def __init__(self, agent, game):

        self.agent = agent
        self.game = game

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

        self.memory = PPOMemory(self.batch_size)

        # policy network
        self.policy_net = PolicyNet(self.num_states, self.num_actions, self.hidden_dim)
        self.policy_net.to(self.device)
        
        # Value network (outputs baseline value function V(s))
        self.value_net = ValueNet(self.num_states, self.num_actions, self.value_learning_rate)  # Outputs a scalar V(s)
        self.value_net.to(self.device)

        # Optimizers
        self.policy_optimizer = optim.Adam(self.policy_net.parameters(), lr=self.theta_learning_rate)
        self.value_optimizer = optim.Adam(self.value_net.parameters(), lr=self.value_learning_rate)

        self.writer = SummaryWriter('runs/ppo')  # specify log dir
        self.loss_history = []
        # load model if exist
        self._load_model()

    def train(self):

        for _ in range(self.ppo_epochs):
            states, actions, old_probs, values, rewards, dones, = self.memory.generate_batches()

            advantage = torch.zeros_like(rewards)

            # Compute Advantage Value
            for t in reversed(range(len(rewards))):
                # delta = r + discount * value[t+1] - value[t] 
                delta = rewards[t] + self.discount * values[t + 1] * (1-dones[t]) - values[t]
                # a = delta + self.discount * gae_lambda * a[t+1]
                a_t = delta + self.discount * self.GAE * (1 - dones[t]) * (advantage[t+1] if t + 1 < len(rewards) else 0)
                advantage[t] = a_t

            for i in range(len(states)):
                # policy of new states
                new_probs = self.policy_net(states[i]).log_prob(actions[i])
                new_value = self.value_net(states[i]).squeeze(0)

                ratio = new_probs.exp() / old_probs[i].exp()

                unclipped = ratio * advantage[i]

                clipped = torch.clamp(ratio, self.min_change, self.max_change) * advantage[i]

                policy_loss = -torch.min(unclipped, clipped).mean()

                returns = advantage[i] + values[i]
                value_loss = (returns - new_value) ** 2
                value_loss = value_loss.mean()

                total_loss = policy_loss + self.weighting_factor * value_loss

                self.policy_net.optimizer.zero_grad()
                self.value_net.optimizer.zero_grad()

                total_loss.backward()

                self.policy_net.optimizer.step()
                self.value_net.optimizer.step()

        self.memory.clear_memory()

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)


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

    def _save_model(self):
        print("PPO Model saved")
        torch.save(self.policy_net.state_dict(), 'models/ppo_policy_net.pth')
        torch.save(self.value_net.state_dict(), 'models/ppo_value_net.pth')