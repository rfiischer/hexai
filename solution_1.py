from tensorflow import keras
from tensorflow.keras import layers
import numpy as np

from core.hex import HexBoard
from core.dqn import DQNAgent


# Board parameters
board_size = 11
grid_size = board_size ** 2

# DQN Parameters
memory_size = 3000
minimum_memory_size = 1000
batch_size = 32
input_shape = (grid_size, )
action_shape = (grid_size, )
update_target = 500
discount = 0.9

# Network parameters
factor = 2

# Learning parameters
num_generations = 1
num_episodes = 10000
explore_episodes = 1000
min_epsilon = 0.01
epsilon_array = np.linspace(1, min_epsilon, explore_episodes)
crash_reward = -100
win_reward = 10
lose_reward = -10

# Build base model
base_model = keras.Sequential(
    [
        layers.InputLayer(input_shape=(grid_size,)),
        layers.Dense(factor * grid_size, activation="relu", name="h1", bias_initializer="normal"),
        layers.Dense(grid_size, name="output")
    ]
)

# Get base model weights and initialize parameters
base_weights = base_model.get_weights()
dqn_params = {'memory_size': memory_size,
              'minimum_memory_size': minimum_memory_size,
              'batch_size': batch_size,
              'input_shape': input_shape,
              'action_shape': action_shape,
              'update_target': update_target,
              'discount': discount}

# Initialize agents
agents = [DQNAgent(base_model, base_weights, dqn_params)]

# Each generation corresponds to duplicating the current agent and training it on its past selves
for generation in range(num_generations):

    # Main agent to train
    main_agent = DQNAgent(base_model, agents[-1].online.get_weights(), dqn_params)

    # For each past agent, including itself
    for past in range(generation + 1):

        opponent_agent = agents[past]

        # Run episodes
        for episode in range(num_episodes):

            board = HexBoard(board_size, board_size)

            # Perform epsilon-greedy action
            if episode < explore_episodes:
                epsilon = epsilon_array[episode]

            else:
                epsilon = min_epsilon

            finished = False
            number_plays = 0
            while not finished:

                # Main agent's move
                start_state = board.get_state()
                if np.random.uniform() < epsilon:
                    move = np.random.randint(grid_size)

                else:
                    q = main_agent.get_q(start_state)
                    move = np.argmax(q)

                # Try to play
                board.play(1, *board.get_xy(move))
                if board.crash:
                    finished = True
                    # Just fill the end state with something with the same size
                    transition = [start_state, move, crash_reward, start_state, True]
                    main_agent.update_memory(transition)

                # Main agent wins
                elif board.winner is not None:
                    finished = True
                    transition = [start_state, move, win_reward, start_state, True]
                    main_agent.update_memory(transition)

                # No winner yet (second agent plays)
                else:
                    move_order = np.argsort(opponent_agent.get_q(board.get_state())).flatten()

                    # See if best move is illegal move
                    board.play(2, *board.get_xy(move_order[-1]))
                    attempt_counter = 1
                    while board.crash:
                        board.crash = False
                        board.play(2, *board.get_xy(move_order[-attempt_counter - 1]))
                        attempt_counter += 1

                    # See if opponent wins
                    if board.winner is not None:
                        finished = True
                        transition = [start_state, move, lose_reward, start_state, True]
                        main_agent.update_memory(transition)

                    else:
                        end_state = board.get_state()
                        transition = [start_state, move, 0, end_state, False]
                        main_agent.update_memory(transition)

                main_agent.train()

                number_plays += 1

            print(f"Generation: {generation}, Agent: {past}, Episode: {episode}, Plays: {number_plays}")

    agents.append(main_agent)
