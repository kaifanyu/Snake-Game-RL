import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
import os
from torch.utils.tensorboard import SummaryWriter
from collections import deque

class NN(nn.Module):
    def __init__(self, num_states, num_actions, dim):
        super().__init__()
        # 2 layer Neural Network
        # 11 -> 256 -> 256 -> 3 3
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

class Vanilla_DQN():
    def __init__(self, agent, game): 

        self.agent = agent
        self.game = game

        self.hidden_dim = 256
        self.epsilon = 0.9
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.99
        self.discount = 0.9
        self.learning_rate = 0.001
        self.num_states = 11
        self.num_actions = 3
        self.batch_size = 1000

        self.loss_history = []
        
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        # initialize policy network
        self.policy_netowrk = NN(num_states=self.num_states, num_actions=self.num_actions, dim=self.hidden_dim)
        self.policy_netowrk.to(self.device)

        # loss function
        self.loss = nn.MSELoss()
        self.optimizer = optim.Adam(self.policy_netowrk.parameters(), lr=self.learning_rate)

        self.writer = SummaryWriter('runs/vanilla_dqn')  # specify log dir

        self._load_model()

    def action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            state_tensor = torch.tensor(state).float().unsqueeze(0)
            q_values = self.policy_netowrk(state_tensor)
            action = torch.argmax(q_values, dim=1).item()
            return action
        
    def update(self, batch, global_step):
        # Unpack batch into separate lists
        old_states, action_indices, new_states, rewards, game_overs = zip(*batch)

        # (1000, 11), ex: [[0,1,2], [3,4,5]]
        old_states = torch.tensor(old_states, dtype=torch.float32)
        # (1000, 1), ex: [[1], [2]]
        action_indices = torch.tensor(action_indices, dtype=torch.long).unsqueeze(1)
        # (1000, 11), ex: [[1,2,3], [4,5,6]]
        new_states = torch.tensor(new_states, dtype=torch.float32)
        # (1000, 1), ex: [[0], [-10]]
        rewards = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        # (1000, 1), ex: [[1], [0]]
        game_overs = torch.tensor(game_overs, dtype=torch.float32).unsqueeze(1)
        
        # list of q values that corresponds to old_state and action a taken at that state
        Q_pred = self.policy_netowrk(old_states).gather(1, action_indices)

        # disable gradient calculation because we won't use this value for back propagation, we use the value after bellman equation
        with torch.no_grad():
            # max(Q(next_state)), we don't care which action we just want the highest Q value possible
            next_q_value = self.policy_netowrk(new_states) #-> [[0.1,0.5,0.3], [-0.1, 0.3, -0.2]]
            max_q_value = next_q_value.max(1, keepdim=True)[0] # ->[[0.5], [0.3]]. the [0] if for selecting the value and not the index at [1]

        # if game_over = 1, then 1-1=0 and Q_target = rewards
        # (1000,1) + (0/1) * (scalar) * (1000, 1) -> (1000, 1)
        Q_target = rewards + (1 - game_overs) * self.discount * max_q_value

        # 1/N * (Q_pred - Q_target) ^ 2
        loss = self.loss(Q_pred, Q_target)

        # Clears Gradients from Previous Iterations
        self.optimizer.zero_grad()
        # Backpropagation, how much each weight contributed to the error.
        # After loss.backward(), every weight in the network knows how to change to reduce loss.
        loss.backward()
        # w = w - (learning rate) * dLoss / dWeight. Actually updating the weight
        self.optimizer.step()

        self.loss_history.append(loss.item())
        # Log training loss
        self.writer.add_scalar("Loss/train", loss.item(), global_step)
    
    def train(self):

        MAX_MEMORY = 100000
        memory = deque([], maxlen=MAX_MEMORY)
        model_saved = False

        global_step = 0

        try:
            for episode in range(self.agent.max_games):
                while True:
                    #counter
                    global_step += 1

                    # get current state of self.game
                    old_state = self.agent.get_state(self.game)

                    # get action from self
                    action_index = self.action(old_state)
                    action_vector = self.agent.move[action_index]

                    # play the action in self.game
                    reward, game_over, score = self.game.play_step(action_vector)

                    # get the new state after action
                    new_state = self.agent.get_state(self.game)
                    
                    sequence = (old_state, action_index, new_state, reward, game_over)
                    # append this sequence in memory
                    memory.append(sequence)

                    # Sample batch and update
                    if len(memory) >= self.batch_size:
                        batch = random.sample(memory, self.batch_size)
                        self.update(batch, global_step)

                    if game_over:
                        self.game.reset()
                        self.agent.score_history.append(score)
                        break

                print("Game: ", episode, "score: ", score)
                
                self.update_epsilon()

                self.writer.add_scalar("Score/episodes", score, episode)
                self.writer.add_scalar("Params/epsilon", self.epsilon, episode)

        except KeyboardInterrupt:
            self._save_model()
            self.agent.save("vanilla_dqn", self.loss_history)
            model_saved = True
        finally:
            if not model_saved:
                self._save_model()
                self.agent.save("vanilla_dqn", self.loss_history)
            
    def _save_model(self):
        print("Model saved")
        torch.save(self.policy_netowrk.state_dict(), 'models/dqn_vanilla.pth')

    def _load_model(self):
        checkpoint_path = 'models/dqn_vanilla.pth'
        if os.path.exists(checkpoint_path):
            self.policy_netowrk.load_state_dict(torch.load(checkpoint_path, map_location=self.device, weights_only=True))
            self.epsilon = 0.1
            print(f"Loaded existing weights from {checkpoint_path}, setting epsilon to 0.1")
        else:
            print("No existing checkpoint found. Starting fresh.")
        
    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)