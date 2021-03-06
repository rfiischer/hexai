from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
from scipy.io import savemat
import gym

from core.dqn import DQNAgent
from core.utils import epsilon_function, gym_benchmark


# Gym
env = gym.make("CartPole-v0")
benchmark = gym.make("CartPole-v0")

# DQN Parameters
memory_size = 4096
minimum_memory_size = 128
batch_size = 128
state_shape = env.observation_space.shape
action_shape = env.action_space.shape
update_target = 512
discount = 0.99

# Network parameters
d1_size = 32

# Learning parameters
num_episodes = 1000
explore_episodes = 250
epoch_size = 10
num_benchmark = 10
episode_length = 200
e_max = 1
e_min = 0.001
epsilon_array = np.concatenate((epsilon_function(e_max, e_min, explore_episodes, explore_episodes),
                                [e_min] * 750))
reward = 1
punishment = -1

# Build base model
base_model = keras.Sequential(
    [
        layers.InputLayer(input_shape=state_shape),
        layers.Dense(d1_size, activation="relu", name="d1", bias_initializer="normal"),
        layers.Dense(env.action_space.n, activation="linear", name="output")
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
              'optimizer': keras.optimizers.SGD(lr=1e-2),
              'loss': keras.losses.MeanSquaredError()}

# Initialize agents
agent = DQNAgent(base_model, base_weights, dqn_params)

# Run episodes
num_fail = 0
average_mistakes = [-1]
for episode in range(num_episodes):

    # Perform epsilon-greedy action
    epsilon = epsilon_array[episode]

    # Make initial observation
    observation = env.reset()

    for i_episode in range(episode_length):

        if np.random.uniform() < epsilon:
            action = env.action_space.sample()

        else:
            q = agent.get_q(observation)
            action = np.argmax(q)

        new_observation, reward, stop, _ = env.step(action)
        transition = [observation.reshape((1, -1)), action, reward, new_observation.reshape((1, -1)), stop]
        agent.update_memory(transition)
        observation = new_observation

        agent.train()

        if stop:
            print(f"Episode: {episode}, Crash: {num_fail}, Moves: {i_episode}, Loss: {agent.callback.loss[-1]}")

            if i_episode < episode_length - 1:
                num_fail += 1

            break

    if not episode % epoch_size:
        average = gym_benchmark(num_benchmark, agent.online,
                                benchmark, episode_length) / num_benchmark

        average_mistakes.append(average)

        print(f"Average: {average_mistakes[-1]}")

        if average == 0:
            keras.models.save_model(agent.online, "./gmodel0_best")

            savemat('./gmodel0_best/loss.mat',
                    {'loss': agent.callback.loss[1:], 'average_mistakes': average_mistakes})

# Save model
keras.models.save_model(agent.online, "./gmodel0")
savemat('./gmodel0/loss.mat', {'loss': agent.callback.loss[1:], 'average_mistakes': average_mistakes})
