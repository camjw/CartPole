import gym
import tensorflow as tf
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import lib.experience_buffer as eb
import lib.game_handler as gh
import lib.model_holder as mh


class Controller:

    def __init__(self, env_name, num_episodes, batch_size, total_memory,
        max_epsilon, min_epsilon, lamb, gamma, session, location):

        self.game = gym.make(env_name)
        self.num_states = self.game.env.observation_space.shape[0]
        try:
            self.num_actions = self.game.env.action_space.n
        except:
            self.num_actions = self.game.env.action_space.shape[0]


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
                            "batch_size": batch_size,
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
            while True:
                command = input("\nDo you want to see the AI play?\n")
                if command in ["n", "N"]:
                    break
                else:
                    self.handler.run(render=True)

    def test_learning(self, num_episodes):
        score = self.handler.test_learning(num_episodes)
        print('The mean over {} steps was {}. The minimum score was {}.'.format(
            num_episodes, score[0], score[1]))


    def load_network(self):
        pass

    def plot_rewards(self):
        data = pd.Series(self.handler._reward_store)
        data.to_pickle(self.location + "_reward_store.pickle")
        rolling_mean = data.rolling(window=100).mean()

        plt.plot(rolling_mean)
        plt.show()
        plt.close("all")

    def render_play(self, num_episodes):
        for episode in range(num_episodes):
            observation = self.game.reset()
            finished = True
            while True:
                self.game.render()
                action = self.handler._choose_action()
                observation, reward, done, info = self.game.step(action)

                if done and finished:
                    finished = False
                    print("Episode finished after {} timesteps".format(t+1))
                    break
