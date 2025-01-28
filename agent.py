from snake_game import *
import pickle
from collections import deque
from qlearning import *
from dqn import *

class Agent:
    #record the 'states' of the game
    def __init__(self, model):
        self.n_games = 0
        self.model = model
        self.score_history = []
        self.move = [[1,0,0], [0,1,0],[0,0,1]] #straight right left
        self.max_games = 2000

    def get_state(self, game):
        head = game.snake[0]
        
        block_left = Point(head.x - 20, head.y)
        block_right = Point(head.x + 20, head.y)
        block_up = Point(head.x, head.y - 20)
        block_down = Point(head.x, head.y + 20)

        direction_left = (game.direction == Direction.LEFT)
        direction_right = (game.direction == Direction.RIGHT)
        direction_up = (game.direction == Direction.UP)
        direction_down = (game.direction == Direction.DOWN)

        danger_straight = ((game.is_collision(block_right) and direction_right) or
            (game.is_collision(block_left) and direction_left) or
            (game.is_collision(block_up) and direction_up) or
            (game.is_collision(block_down) and direction_down))
        
        danger_right = ((game.is_collision(block_right) and direction_up) or
            (game.is_collision(block_left) and direction_down) or
            (game.is_collision(block_up) and direction_left) or
            (game.is_collision(block_down) and direction_right))
        
        danger_left = ((game.is_collision(block_right) and direction_down) or
            (game.is_collision(block_left) and direction_up) or
            (game.is_collision(block_up) and direction_right) or
            (game.is_collision(block_down) and direction_left))
        
        state = [
            # danger ahead 
            danger_straight,
            danger_right,
            danger_left,

            # current direction 
            direction_left,
            direction_right,
            direction_up,
            direction_down,

            # food location in response to current location
            game.food.x < game.head.x,  #food is to the left
            game.food.x > game.head.x,  #food is to the right
            
            game.food.y < game.head.y,  #food is above us
            game.food.y > game.head.y   #food is below us
        ]

        # keep result as a state tuple
        result = tuple(map(int, state))
        return result


def train_tabular():
    model = tabular()
    agent = Agent(model)
    game = SnakeGame()
    game.reset()
    

    for i in range(agent.max_games):

        while True: 
            # get the current state of the game
            old_state = agent.get_state(game)

            # figure out what move it will play next
            action_index = agent.model.action(old_state)
            
            # convert the index to move
            action_vector = agent.move[action_index]

            # play the action in the game
            reward, game_over, score = game.play_step(action_vector)

            # get the new state of the game
            new_state = agent.get_state(game)

            # update the model, in this case, update the Q table
            agent.model.update(old_state, new_state, action_index, reward, game_over)

            if game_over:
                game.reset()
                agent.n_games += 1
                break

        agent.score_history.append(score)
        # decrement epsilon so over time, we explore less and follow our model
        # agent.model.epsilon = max(agent.model.epsilon_min,
        #                         agent.model.epsilon * agent.model.epsilon_decay)

    data = {
        'Q_table': agent.model.Q,
        'score': agent.score_history,
    }

    # 1) Open a file in 'write binary' mode
    with open('qlearn.pkl', 'wb') as f:
        # 2) Use pickle.dump to write the data to the file
        pickle.dump(data, f)


def train_dqn():
    model = DeepQLearn()
    agent = Agent(model=model)
    game = SnakeGame()
    game.reset()

    episodes = 100
    memory = deque([], maxlen=episodes)

    print("Starting Game")
    for i in range(agent.max_games):
        recieve_reward = False

        while True:
            print("Game: ", i)
            # get current state of game
            old_state = agent.get_state(game)

            # get action from model
            action_index = agent.model.action(old_state)
            action_vector = agent.move[action_index]

            # play the action in game
            reward, game_over, score = game.play_step(action_vector)

            # get the new state after action
            new_state = agent.get_state(game)

            # if we recieved a reward and we have not yet recieved a reward
            if reward > 0 and not recieve_reward:
                recieve_reward = True   #set it to true

            memory.append((old_state, action_index, new_state, reward, game_over))
            # update the model
            # if our memory is full and we have recieved reward
            if len(memory) > episodes and recieve_reward:
                mini_batch = random.sample(memory, 32)
                agent.model.update(mini_batch)

            # agent.model.update(old_state, new_state, action_index, reward, game_over)
            if game_over:
                game.reset()
                agent.n_games += 1
                break

        agent.score_history.append(score)
        agent.model.epsilon = max(agent.model.epsilon_min,
                                agent.model.epsilon * agent.model.epsilon_decay)

if __name__ == "__main__":
    train_dqn()