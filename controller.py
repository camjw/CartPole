import gym
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from lib import experience_buffer as eb
from lib import game_handler as gh
from lib import model_holder as mh


class Controller:

    def __init__(self, env_name, num_episodes, batch_size, total_memory, max_epsilon,
             min_epsilon, lamb, gamma, location, session, location):

        self.game = gym.make(env_name)
        self.num_states = self.game.env.observation_space.shape[0]
        self.num_actions = self.game.env.action_space.n

        self.num_episodes = num_episodes

        self.model = mh.ModelHolder(self.num_actions, self.num_states,
                                              batch_size)
        self.memory = eb.ExperienceBuffer(total_memory)

        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.lamb = lamb

        self.gamma = gamma
        self.location = location

        self.session = session

        self.handler = gh.GameHandler(self.game, self.session, self.model,
                           self.memory, self.max_epsilon, self.min_epsilon,
                           self.lamb, self.gamma)
        self.parameters = { "env_name": env_name,
                            "total_memory": total_memory,
                            "batch_size": batch_size
                            "max_epsilon": max_epsilon,
                            "min_epsilon": min_epsilon,
                            "lambda": lamb,
                            "gamma": gamma,
                            "location": location
                          }

    def learn_game(self, num_episodes, feedback=True, location=None):
        self.parameters["num_episodes"] = num_episodes
        with self.session as sess:
            sess.run(self.model._var_init)

            count = 0
            while count < num_episodes:
                if feedback:
                    if count % 100 == 0:
                        print('Episode {} of {}'.format(count+1, num_episodes))
                self.handler.run()
                count += 1
            if location is not None:
                self.model._saver.save(sess, location, global_step=1000)
                with open(LOCATION + "hyperparameter_dict.txt", "wb") as params:
                    pickle.dump(self.parameters, params)

    def test_learning(self, num_episodes):
        score = self.handler.test_learning(num_episodes)
        print "The mean over {} steps was {}. The minimum score was {}.".format(
                score[0], score[1])


    def load_network():

    def plot_rewards():

    def render_play():
