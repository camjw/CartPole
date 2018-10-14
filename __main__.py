import gym
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lib import controller as control


def main(env_name, num_episodes, batch_size, total_memory, max_epsilon,
    min_epsilon, lamb, gamma, session, location):

    cartpole_control = control.Controller(env_name, num_episodes, batch_size,
        total_memory, max_epsilon, min_epsilon, lamb, gamma, session, location)

    cartpole_control.learn_game(num_episodes)
    cartpole_control.plot_rewards()

if __name__ == "__main__":
    ENV_NAME = "CartPole-v0"
    NUM_EPISODES = 4096
    BATCH_SIZE = 16
    TOTAL_MEMORY = 100000
    MAX_EPSILON = 1
    MIN_EPSILON = 0.01
    LAMBDA = 0.00001
    GAMMA = 0.99
    SESSION = tf.Session()
    LOCATION = "data/CartPole_test/"
    main(ENV_NAME, NUM_EPISODES, BATCH_SIZE, TOTAL_MEMORY, MAX_EPSILON,
         MIN_EPSILON, LAMBDA, GAMMA, SESSION, LOCATION)

    hpDict = { "env_name": ENV_NAME,
               "num_episodes": NUM_EPISODES,
               "batch_size": BATCH_SIZE,
               "total_memory": TOTAL_MEMORY,
               "max_epsilon": MAX_EPSILON,
               "min_epsilon": MIN_EPSILON,
               "lamb": LAMBDA,
               "gamma": GAMMA,
               "location": "data/CartPole_test",
               }

    with open(LOCATION + "hyperparameter_dict.txt", "wb") as parameterDict:
        pickle.dump(hpDict, parameterDict)
