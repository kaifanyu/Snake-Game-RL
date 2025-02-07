from agent import *


agent = Agent()
game = SnakeGame()

# model = Tabular(agent, game)
# model = Vanilla_DQN(agent, game)
# model = Double_DQN(agent, game)
# model = Policy(agent, game)
model = PPO(agent, game)
model.train()