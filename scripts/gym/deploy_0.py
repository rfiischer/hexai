from tensorflow import keras
import gym

from core.utils import gym_benchmark


model = keras.models.load_model("./gmodel0")
env = gym.make('CartPole-v0')
gym_benchmark(10, model, env, render=True)
