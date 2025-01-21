import numpy as np
import random

# Algorithm:
# After each step, we update Q_table:
# Q(s, a) = Q(s, a) + a[r + y * max(Q(s',a') - Q(s, a))]


class tabular:
    
    def __init__(self):
        self.num_states = 11 # danger_up, danger_left, danger_right, 
        self.num_actions = 3 # left right up
        self.epsilon = 0.9  # starting epsilon, decrement over time
        self.discount = 0.9 # discount variable
        self.alpha = 0.1 # Learning Rate
        self.Q = dict()
        self.Q_table = np.zeros((self.num_states, self.num_actions))

    # update the tabular based on latest state
    def action(self, state):

        # first we find out what to do. If it is less than epislon, we pick a random move
        if random.uniform(0, 1) < self.epsilon:
            action = random.randint(0, 2)
        else:
            action = np.argmax(self.Q_table[state][0], self.Q_table[state][1], self.Q_table[state][2])
        
        return action


    def update(self, old_state, new_state, action, reward, game_over):
        old_value = self.Q[old_state, action]

        future = max(self.Q[new_state, action] for a in [0, 1, 2]) if not game_over else 0

        self.Q[old_state, action] = old_value + self.alpha * (reward + self.discount * future - old_value) 

        # decay epsilon
        # self.epsilon = max(self.epsilon_min, self.epsilon_decay * self.epsilon)

        