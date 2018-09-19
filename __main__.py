import gym
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lib import experience_buffer
from lib import game_handler
from lib import model_holder


def main(env_name, num_episodes, batch_size, total_memory, max_epsilon,
         min_epsilon, lamb, gamma, session, location):
    env = gym.make(env_name)

    num_states = env.env.observation_space.shape[0]
    num_actions = env.env.action_space.n

    model = model_holder.ModelHolder(num_actions, num_states, batch_size)
    memory = experience_buffer.ExperienceBuffer(total_memory)

    with session as sess:
        sess.run(model._var_init)
        gamehandler = game_handler.GameHandler(env, sess, model, memory,
                                               max_epsilon, min_epsilon,
                                               lamb, gamma)
        count = 0
        while count < num_episodes:
            if count % 10 == 0:
                print('Episode {} of {}'.format(count+1, num_episodes))
            gamehandler.run()
            count += 1

        model._saver.save(sess, LOCATION, global_step=1000)
        sns.set(style='darkgrid', context='talk', palette='Dark2')
        data = pd.Series(gamehandler._reward_store)
        data.to_pickle(location + "_reward_store.pickle")
        rolling_mean = data.rolling(window=100).mean()

        plt.plot(rolling_mean)
        plt.show()
        plt.close("all")

        while True:
            command = input("\nDo you want to see the AI play?\n")
            if command in ["n", "N"]:
                break
            else:
                gamehandler.run(render=True)

def load_saved_network(env_name, filename):

    env = gym.make(env_name)

    with tf.Session() as sess:
         pass

if __name__ == "__main__":
    ENV_NAME = "CartPole-v0"
    NUM_EPISODES = 1000
    BATCH_SIZE = 300
    TOTAL_MEMORY = 10000
    MAX_EPSILON = 0.5
    MIN_EPSILON = 0.01
    LAMBDA = 0.001
    GAMMA = 0.999
    SESSION = tf.Session()
    LOCATION = "data/CartPole_24_48_1000/"
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
               "location": "data/CartPole_24_48_1000/",
               }

    with open(LOCATION + "hyperparameter_dict.txt", "wb") as parameterDict:
        pickle.dump(hpDict, parameterDict)
