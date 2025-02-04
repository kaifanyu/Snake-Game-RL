import pickle
from collections import deque
import signal
import sys
from snake_game import *
from qlearning import *
from double_dqn import *
from vanilla_dqn import *

class Agent:
    #record the 'states' of the game
    def __init__(self):
        self.n_games = 0
        self.score_history = []
        self.move = [[1,0,0], [0,1,0],[0,0,1]] #straight right left
        self.max_games = 1000


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

def train_dqn_vanilla():
    model = Vanilla_DQN()
    agent = Agent()
    game = SnakeGame()
    game.reset()

    MAX_MEMORY = 100000
    memory = deque([], maxlen=MAX_MEMORY)

    global_step = 0

    try:
        for episode in range(agent.max_games):
            while True:
                #counter
                global_step += 1

                # get current state of game
                old_state = agent.get_state(game)

                # get action from model
                action_index = model.action(old_state)
                action_vector = agent.move[action_index]

                # play the action in game
                reward, game_over, score = game.play_step(action_vector)

                # get the new state after action
                new_state = agent.get_state(game)
                
                sequence = (old_state, action_index, new_state, reward, game_over)
                # append this sequence in memory
                memory.append(sequence)

                # Sample batch and update
                if len(memory) >= model.batch_size:
                    batch = random.sample(memory, model.batch_size)
                    model.update(batch, global_step)

                if game_over:
                    game.reset()
                    agent.n_games += 1
                    break

            print("Game: ", episode, "score: ", score)
            
            model.update_epsilon()
            model.writer.add_scalar("Score/episodes", score, episode)
            model.writer.add_scalar("Params/epsilon", model.epsilon, episode)

    except KeyboardInterrupt:
        model.save_model()
    finally:
        model.save_model()

def train_dqn_target_policy():
    model = Double_DQN()
    agent = Agent()
    game = SnakeGame()
    game.reset()

    MAX_MEMORY = 100000
    memory = deque([], maxlen=MAX_MEMORY)

    global_step = 0

    try:
        for episode in range(agent.max_games):
            while True:
                #counter
                global_step += 1

                # get current state of game
                old_state = agent.get_state(game)

                # get action from model
                action_index = model.action(old_state)
                action_vector = agent.move[action_index]

                # play the action in game
                reward, game_over, score = game.play_step(action_vector)

                # get the new state after action
                new_state = agent.get_state(game)
                
                sequence = (old_state, action_index, new_state, reward, game_over)
                # append this sequence in memory
                memory.append(sequence)

                # Sample batch and update
                if len(memory) >= model.batch_size:
                    batch = random.sample(memory, model.batch_size)
                    model.update(batch, global_step)

                if game_over:
                    game.reset()
                    agent.n_games += 1
                    break

            print("Game: ", episode, "score: ", score)
            
            model.update_epsilon()
            model.writer.add_scalar("Score/episodes", score, episode)
            model.writer.add_scalar("Params/epsilon", model.epsilon, episode)

    except KeyboardInterrupt:
        model.save_model()
    finally:
        model.save_model()

if __name__ == "__main__":
    train_dqn_target_policy()