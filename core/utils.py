import numpy as np

from core.hex import HexBoard


def epsilon_function(e_max, e_min, episodes, period):
    number = episodes // period + 1

    epsilon = np.linspace(e_max, e_min, period)

    return np.tile(epsilon, number)[:episodes]


def hex_benchmark(num_games, model):
    mistake = 0
    for i in range(num_games):
        board = HexBoard(3, 3)

        if np.random.randint(0, 2):
            board.play(2, *board.get_xy(np.random.choice(board.get_legal())))

        while not board.crash and np.any(board.get_legal()):
            move = np.argmax(model(board.get_state()))
            board.play(1, *board.get_xy(move))

            if np.any(board.get_legal()):
                move = np.random.choice(board.get_legal())
                board.play(2, *board.get_xy(move))

        if board.crash:
            mistake += 1

    return mistake


def gym_benchmark(num_attempts, model, env, episode_length=200, render=False):
    fail = 0
    for i in range(num_attempts):
        observation = env.reset()

        for j in range(episode_length):
            if render:
                env.render()
            action = np.argmax(model(observation.reshape((1, -1))))
            observation, reward, done, _ = env.step(action)

            if done and j < episode_length - 1:
                fail += 1
                break

    if render:
        env.close()

    return fail
