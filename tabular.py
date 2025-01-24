import numpy as np
import random
import pickle 

# Algorithm:
# After each step, we update Q_table:
# Q(s, a) = Q(s, a) + a[r + y * max(Q(s',a') - Q(s, a))]

class tabular:
    
    def __init__(self):
        self.num_states = 11 # danger_up, danger_left, danger_right, 
        self.num_actions = 3 # left right up
        self.epsilon = 0.1  # starting epsilon, decrement over time
        self.discount = 0.9 # discount variable
        self.alpha = 0.1 # Learning Rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.9995
        self.Q = dict()
            
        with open('tabular.pkl', 'rb') as f:
            data = pickle.load(f)
        self.Q = data['Q_table']

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
        if new_state not in self.Q:
            self.Q[new_state] = np.zeros(self.num_actions)
        
        old_value = self.Q[old_state][action]

        future = max(self.Q[new_state]) if not game_over else 0

        new_value = old_value + self.alpha * (reward + self.discount * future - old_value) 
        self.Q[old_state][action] = new_value