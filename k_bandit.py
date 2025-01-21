import random
import matplotlib.pyplot as plt

# Number of arms (k) and their rewards
targets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

class EGreedy:
    def __init__(self, e=0.1, rounds=100):
        self.epsilon = e  # Exploration probability
        self.rounds = rounds  # Number of rounds to simulate
        self.history = []  # History of selected actions
        self.explore_count = 0  # Number of exploratory actions
        self.total_reward = 0  # Total accumulated reward
        self.estimated_values = [0] * len(targets)  # Action value estimates
        self.action_counts = [0] * len(targets)  # Counts of actions taken

    def simulate(self):
        for _ in range(self.rounds):
            # Decide whether to explore or exploit
            if random.uniform(0, 1) < self.epsilon:
                # Explore: Choose a random action
                action = random.randint(0, len(targets) - 1)
                self.explore_count += 1
            else:
                # Exploit: Choose the action with the highest estimated value
                action = self.estimated_values.index(max(self.estimated_values))

            # Simulate reward for the chosen action (using targets as true values)
            reward = targets[action] + random.gauss(0, 1)  # Adding some noise

            # Update action value estimate using incremental formula
            self.action_counts[action] += 1
            self.estimated_values[action] += (reward - self.estimated_values[action]) / self.action_counts[action]

            # Track total reward and history
            self.total_reward += reward
            self.history.append(reward)

        # Calculate average reward
        self.total_reward /= self.rounds


# # Create an instance of the EGreedy class and simulate
# eg = EGreedy(e=0.1, rounds=1000)
# eg.simulate()

# # Print results
# print("Total Rounds:", eg.rounds)
# print("Total Average Reward:", eg.total_reward)
# print("Exploration Count:", eg.explore_count)

# # Plot the reward history
# plt.plot(eg.history, label='Reward per Step')
# plt.axhline(y=eg.total_reward, color='r', linestyle='--', label='Average Reward')
# plt.xlabel('Steps')
# plt.ylabel('Reward')
# plt.title(f'e-Greedy Simulation (e={eg.epsilon})')
# plt.legend()
# plt.show()

import random
import math
import matplotlib.pyplot as plt

# Number of arms (k) and their true rewards
targets = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

class GradientBandit:
    def __init__(self, alpha=0.1, rounds=1000):
        self.alpha = alpha  # Step-size parameter
        self.rounds = rounds  # Number of rounds to simulate
        self.preferences = [0] * len(targets)  # Preferences for each action
        self.probabilities = [1 / len(targets)] * len(targets)  # Action probabilities
        self.total_reward = 0  # Total accumulated reward
        self.average_reward = 0  # Average reward over time
        self.history = []  # History of rewards

    def softmax(self):
        """Compute action probabilities using softmax over preferences."""
        exp_preferences = [math.exp(p) for p in self.preferences]
        total = sum(exp_preferences)
        self.probabilities = [ep / total for ep in exp_preferences]

    def simulate(self):
        for t in range(1, self.rounds + 1):
            # Compute action probabilities
            self.softmax()

            # Choose an action based on probabilities
            action = random.choices(range(len(targets)), weights=self.probabilities, k=1)[0]

            # Simulate reward for the chosen action (true value + noise)
            reward = targets[action] + random.gauss(0, 1)

            # Update average reward
            self.average_reward += (reward - self.average_reward) / t

            # Update preferences
            for a in range(len(self.preferences)):
                if a == action:
                    self.preferences[a] += self.alpha * (reward - self.average_reward) * (1 - self.probabilities[a])
                else:
                    self.preferences[a] -= self.alpha * (reward - self.average_reward) * self.probabilities[a]

            # Track total reward and reward history
            self.total_reward += reward
            self.history.append(reward)

        # Calculate average reward over all rounds
        self.total_reward /= self.rounds

# Create an instance of the GradientBandit class and simulate
gb = GradientBandit(alpha=0.1, rounds=1000)
gb.simulate()

# Print results
print("Total Rounds:", gb.rounds)
print("Total Average Reward:", gb.total_reward)

# Plot the reward history
plt.plot(gb.history, label='Reward per Step')
plt.axhline(y=gb.total_reward, color='r', linestyle='--', label='Average Reward')
plt.xlabel('Steps')
plt.ylabel('Reward')
plt.title(f'Gradient Bandit Algorithm (alpha={gb.alpha})')
plt.legend()
plt.show()
