import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np



class DQN(nn.Module):
    def __init__(self, num_states, num_actions, dim):
        super().__init__()
    
        # 2 layer Neural Network
        # 11 -> 256 -> 256 -> 3
        self.dqn = nn.Sequential(
            # num_states -> dim
            nn.Linear(num_states, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU(),
            nn.Linear(dim, num_actions)
        )

        
    def forward(self, x):
        x = self.dqn(x)
        return x

class DeepQLearn():
    def __init__(self): 
        self.hidden_dim = 256
        self.epsilon = 0.9
        self.epsilon_min = 0.1
        self.discount = 0.9
        self.learning_rate = 0.1
        self.num_states = 11
        self.num_actions = 3
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # initialize Qnet
        self.policy_nn = DQN(num_states=self.num_states, num_actions=self.num_actions, dim=self.hidden_dim)
        self.policy_nn.to(self.device)

        self.target_nn = DQN(num_states=self.num_states, num_actions=self.num_actions, dim=self.hidden_dim)
        self.target_nn.to(self.device)

        # loss function
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.policy_nn.parameters(), lr=self.learning_rate)

    def action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, 2)
        else:
            return 1

    def update(self, batch):

        current_q_list = []
        target_q_list = []
        for old_state, action_index, new_state, reward, game_over in batch:
            new_state = torch.tensor(new_state).float().unsqueeze(0)
            old_state = torch.tensor(old_state).float().unsqueeze(0)

            if game_over:
                target = torch.FloatTensor([reward])
            else:
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount * self.target_nn(new_state, self.num_states).argmax().item()
                    )

            # append the output of each nn to the q list
            current_q_list.append(self.policy_nn(old_state, self.num_states))

            target_q = self.target_nn(old_state, self.num_states)
            target_q[action_index] = target
            target_q_list.append(target_q)

        loss = self.loss(torch.stack(current_q_list), torch.stack(target_q_list))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

