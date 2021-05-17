from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from scipy.io import savemat

from core.hex import HexBoard
from core.dqn import DQNAgent
from core.utils import epsilon_function


# Board parameters
board_size = 3
grid_size = board_size ** 2

# DQN Parameters
memory_size = 2048
minimum_memory_size = 32
batch_size = 32
state_shape = (board_size, board_size, 1)
action_shape = (grid_size, )
update_target = 64
discount = 0.99

# Network parameters
c1_filters = 9
c1_kernel = 2

# Learning parameters
num_episodes = 30000
explore_episodes = 10000
e_max = 1
e_min = 0.05
epsilon_array = np.concatenate((epsilon_function(e_max, e_min, explore_episodes, explore_episodes),
                                [e_min] * 20000))
reward = 1
punishment = -1

# Build base model
base_model = keras.Sequential(
    [
        layers.InputLayer(input_shape=state_shape),
        layers.Conv2D(c1_filters, c1_kernel, activation="relu", name="c1", bias_initializer="normal"),
        layers.Conv2D(c1_filters, c1_kernel, activation="relu", name="c2", bias_initializer="normal"),
        layers.Flatten(),
        layers.Dense(grid_size, activation="linear", name="output")
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

# Initialize agents
agent = DQNAgent(base_model, base_weights, dqn_params)

# Run episodes
# In this script, we reward for completing the game both in winning and losing scenarios
num_crash = 0
for episode in range(num_episodes):

    board = HexBoard(board_size, board_size)

    # Perform epsilon-greedy action
    epsilon = epsilon_array[episode]

    # Opponent beginns randomly
    if np.random.randint(0, 2):
        board.play(2, *board.get_xy(np.random.choice(board.get_legal())))

    finished = False
    number_moves = 0
    while not finished:

        # Main agent's move
        start_state = board.get_state()
        if np.random.uniform() < epsilon:
            move = np.random.randint(grid_size)

        else:
            q = agent.get_q(start_state)
            move = np.argmax(q)

        # Try to play
        board.play(1, *board.get_xy(move))
        if board.crash:
            finished = True
            # Just fill the end state with something with the same size
            transition = [start_state, move, punishment, start_state, True]
            agent.update_memory(transition)
            num_crash += 1

        elif not np.any(board.get_legal()):
            finished = True
            end_state = board.get_state()
            transition = [start_state, move, reward, end_state, True]
            agent.update_memory(transition)

        # Continue playing (no matter if anyone has already won)
        else:
            # Choose random legal move
            board.play(2, *board.get_xy(np.random.choice(board.get_legal())))

            if not np.any(board.get_legal()):
                finished = True

            end_state = board.get_state()
            transition = [start_state, move, reward, end_state, finished]
            agent.update_memory(transition)

        agent.train()

        number_moves += 1

    print(f"Episode: {episode}, Moves: {number_moves}, Crash: {num_crash}, Loss: {agent.callback.loss[-1]}")

# Save model
keras.models.save_model(agent.online, "./hmodel0")
savemat('./hmodel0/loss.mat', {'loss': loss})
