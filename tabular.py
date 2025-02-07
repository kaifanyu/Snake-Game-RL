import numpy as np
import random
import pickle 
import os 

# Algorithm:
# After each step, we update Q_table:
# Q(s, a) = Q(s, a) + a[r + y * max(Q(s',a') - Q(s, a))]

class Tabular:
    
    def __init__(self, agent, game):

        self.agent = agent
        self.game = game

        self.num_states = 11 # danger_up, danger_left, danger_right, 
        self.num_actions = 3 # left right up
        self.epsilon = 0.9  # starting epsilon, decrement over time
        self.discount = 0.9 # discount variable
        self.alpha = 0.1 # Learning Rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        self.Q = dict()

        self._load_model()

    # update the tabular based on latest state
    def action(self, state):
        
        if state not in self.Q:
            self.Q[state] = np.zeros(self.num_actions)

        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, 2)
        else:
            action = np.argmax(self.Q[state])

        # return action either 0, 1, 2
        return action

    # pass in action either 0, 1, 2
    def update(self, old_state, new_state, action, reward, game_over):
        # creat dictionary entry if it doesn't already exist
        if new_state not in self.Q:
            self.Q[new_state] = np.zeros(self.num_actions)
        
        # store old_value given old_state and action took
        old_value = self.Q[old_state][action]

        # find Q value of new_state after we took the action
        future = max(self.Q[new_state]) if not game_over else 0

        # new Q value 
        new_value = old_value + self.alpha * (reward + self.discount * future - old_value) 

        # store new Q value in old_state with action
        self.Q[old_state][action] = new_value

    
    def train(self):
        self.game.reset()
        
        for i in range(self.agent.max_games):
            while True: 
                # get the current state of the game
                old_state = self.agent.get_state(self.game)

                # figure out what move it will play next
                action_index = self.action(old_state)
                
                # convert the index to move
                action_vector = self.agent.move[action_index]

                # play the action in the game
                reward, game_over, score = self.game.play_step(action_vector)

                # get the new state of the game
                new_state = self.agent.get_state(self.game)

                # update the model, in this case, update the Q table
                self.update(old_state, new_state, action_index, reward, game_over)

                if game_over:
                    self.game.reset()
                    self.agent.score_history.append(score)
                    break

            print("game: ", i , "score: ", score)

            # decrement epsilon so over time, we explore less and follow our model
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self._save_model()
        self.agent.save("tabular")

    def _save_model(self):
        with open('models/tabular.pkl', 'wb') as f:
            pickle.dump(self.Q, f)


    def _load_model(self):
        model_checkpoint = "models/tabular.pkl"

        if os.path.exists(model_checkpoint):
            with open(model_checkpoint, 'rb') as f:
                self.Q = pickle.load(f)
                print(f"Loaded Q table from {model_checkpoint}")
        else:
            print("No existing Q table found.")