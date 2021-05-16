import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

from core.utils import epsilon_function
from core.dqn import DQNAgent
from core.games import BooleanToy


# Board parameters
board_size = 5
game = BooleanToy(board_size, board_size)

# DQN Parameters
memory_size = 2048
minimum_memory_size = 32
batch_size = 32
state_shape = game.state_shape
action_shape = game.action_shape
update_target = 64
discount = 0.99

# Network parameters
hidden_units = 16

# Learning parameters
num_episodes = 75
explore_episodes = 50
e_max = 1
e_min = 0.05
epsilon_array = np.concatenate((epsilon_function(e_max, e_min, 50, explore_episodes),
                                [e_min] * 25))
max_score = 400
max_moves = 300

# Build base model
base_model = keras.Sequential(
    [
        layers.InputLayer(input_shape=state_shape),
        layers.Dense(hidden_units, activation="relu", name="h1", bias_initializer="normal"),
        layers.Dense(action_shape[0], activation="linear", name="output")
    ]
)

# Get base model weights and initialize parameters
base_weights = base_model.get_weights()
dqn_params = {'memory_size': memory_size,
              'minimum_memory_size': minimum_memory_size,
              'batch_size': batch_size,
              'state_shape': state_shape,
              'action_shape': action_shape,
              'update_target': update_target,
              'discount': discount,
              'optimizer': keras.optimizers.SGD(),
              'loss': keras.losses.MeanSquaredError()}

# Initialize agent
agent = DQNAgent(base_model, base_weights, dqn_params)

# Run episodes
for episode in range(num_episodes):

    game.reset()

    # Perform epsilon-greedy action
    epsilon = epsilon_array[episode]

    finished = False
    while not finished:

        # Main agent's move
        start_state = game.get_state_from_board()
        if np.random.uniform() < epsilon:
            act = np.random.randint(action_shape[0])

        else:
            q = agent.get_q(start_state)
            act = np.argmax(q)

        # Play
        reward = game.step(act)
        if game.game_over:
            finished = True
            end_state = game.get_state_from_board()
            transition = [start_state, act, reward, end_state, True]
            agent.update_memory(transition)

        else:
            end_state = game.get_state_from_board()
            transition = [start_state, act, reward, end_state, False]
            agent.update_memory(transition)

        if game.score >= max_score or game.frames_counter >= max_moves:
            finished = True

        agent.train()

    print(f"Episode: {episode}, Moves: {game.frames_counter}, Score: {game.score},"
          f"Loss: {agent.callback.loss[-1]}")

# Save model
keras.models.save_model(agent.online, "./hmodel0")
