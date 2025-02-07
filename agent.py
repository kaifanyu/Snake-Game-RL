import pickle
from snake_game import *
from tabular import *
from double_dqn import *
from vanilla_dqn import *
from policy_gradient import *
from ppo import *

class Agent:
    #record the 'states' of the game
    def __init__(self):
        self.max_games = 10
        self.score_history = []
        self.move = [[1,0,0], [0,1,0],[0,0,1]] #straight right left


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

    def save(self, type, loss=[]):
        result_file = "result.pkl"
        output = {}

        # Load existing results if the file exists
        if os.path.exists(result_file):
            with open(result_file, 'rb') as f:
                output = pickle.load(f)
                print(f"Loaded results from {result_file}")

        # Ensure the type exists in output
        if type not in output:
            output[type] = {"score": [], "loss": []}  # Initialize if not present
            
        # Append the latest score history
        output[type]["score"].extend(self.score_history)
        output[type]["loss"].extend(loss)
        
        # Save the updated results back to file
        with open(result_file, 'wb') as f:
            pickle.dump(output, f)

        print(f"Updated result file with {len(self.score_history)} new scores.")
