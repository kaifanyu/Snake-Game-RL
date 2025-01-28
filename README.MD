### Reinforcement Learning with Snake Game

---

### Instructions 

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run `agent.py` with a specified model or load a pre-trained model.

3. The file `tabular.pkl` contains a pre-trained snake game model using the Q-table method over 10000 games.

---

### States Representation
The game state consists of 11 boolean elements:

1. **Danger Straight**: `1` if moving straight results in a collision, otherwise `0`.
2. **Danger Right**: `1` if moving right results in a collision, otherwise `0`.
3. **Danger Left**: `1` if moving left results in a collision, otherwise `0`.
4. **Direction Left**: `1` if the snake is moving left, otherwise `0`.
5. **Direction Right**: `1` if the snake is moving right, otherwise `0`.
6. **Direction Up**: `1` if the snake is moving up, otherwise `0`.
7. **Direction Down**: `1` if the snake is moving down, otherwise `0`.
8. **Food Left**: `1` if the food is to the left of the snake's head, otherwise `0`.
9. **Food Right**: `1` if the food is to the right of the snake's head, otherwise `0`.
10. **Food Up**: `1` if the food is above the snake's head, otherwise `0`.
11. **Food Down**: `1` if the food is below the snake's head, otherwise `0`.


## Model 1: Q-Table

### Tabular Q-Learning Overview
Quality-Learning Q(s,a) represents the cumulative reward the agent can obtain by taking action (a) at state (s). It measures the 'Quality' of the action in the state.

Q(s,a) = argmax(a)Q(s,a): find the best action a that maximizes Q(s,a)
## Type: Model Free, Off-Polify RL
---

## Model 2: Value Iteration

### Overview
It pre-computes a matrix using DP to find the best action per state before playing the game. 
