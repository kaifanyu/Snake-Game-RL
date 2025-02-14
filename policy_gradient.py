import torch
import torch.nn as nn
import torch.optim as optim
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
    def __init__(self, agent, game):

        self.agent = agent
        self.game = game

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

        self.writer = SummaryWriter('runs/policy_gradient')  # specify log dir
        self.loss_history = []

        # load model if exist
        self._load_model()

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
        states = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
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
        discount_factors = torch.pow(self.discount, torch.arange(len(states), dtype=torch.float32).to(self.device)).unsqueeze(1)

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

        self.loss_history.append(policy_loss.item())


    def train(self):
        self.game.reset()

        global_step = 0
        model_saved = False

        try:
            for episode in range(self.agent.max_games):
                states = []
                actions = []
                while True:
                    #counter
                    global_step += 1

                    # get current state of self.game
                    old_state = self.agent.get_state(self.game)

                    states.append(old_state)

                    # get action from self
                    action_index = self.action(old_state)
                    action_vector = self.agent.move[action_index]
                    
                    actions.append(action_index)

                    # play the action in self.game
                    reward, game_over, score = self.game.play_step(action_vector)

                    if game_over:
                        self.game.reset()
                        self.agent.score_history.append(score)
                        self.update(states, actions, reward)    #update full episodes 
                        break

                print("Game: ", episode, "score: ", score)

        except KeyboardInterrupt:
            self._save_model()
            self.agent.save("policy_gradient", self.loss_history)
            model_saved = True
        finally:
            if not model_saved:
                self._save_model()
                self.agent.save("policy_gradient", self.loss_history)
            
    def _save_model(self):
        print("Policy Gradient Model saved")
        torch.save(self.policy_net.state_dict(), 'models/pg_policy_net.pth')
        torch.save(self.value_net.state_dict(), 'models/pg_value_net.pth')

    
    def _load_model(self):
        value_checkpoint_path = 'models/pg_value_net.pth'
        policy_checkpoint_path = 'models/pg_policy_net.pth'
        if os.path.exists(value_checkpoint_path) and os.path.exists(policy_checkpoint_path):
            self.policy_net.load_state_dict(torch.load(policy_checkpoint_path, map_location=self.device, weights_only=True))
            self.value_net.load_state_dict(torch.load(value_checkpoint_path, map_location=self.device, weights_only=True))
            print(f"Loaded existing weights from {policy_checkpoint_path} and {value_checkpoint_path}.")
        else:
            print("No existing checkpoint found. Starting fresh.")
            