import random
import math
import numpy as np
import pandas as pd


class GameHandler:
    ''' This class deals with all interactions with the game/gym challenge. The
        class takes env, an OpenAI gym environment, sess: a tensorflow session,
        model which will be a tensorflow network, memory which will be an
        ExperienceBuffer object, some epsilons which dictate when the
        GameHandler will take random action, a lamb (short for lambda) which
        also dictates the rate of decrease of epsilon, and a gamma which is the
        amount we discount future reward by.'''

    def __init__(self, env, sess, model, memory, max_epsilon, min_epsilon, lamb,
                 gamma):
        self.env = env
        self.sess = sess
        self.model = model
        self.memory = memory
        self.max_epsilon = max_epsilon
        self.min_epsilon = min_epsilon
        self.epsilon = max_epsilon
        self.lamb = lamb  # can't use lambda because of lambda functions
        self.gamma = gamma
        self.reward_store = []
        self.steps = 0
        self.total_steps = 0

    def run(self, render=False):
        state = self.env.reset()
        total_reward = 0

        while True:
            if render:
                self.env.render()

            action = self.choose_action(state)
            new_state, reward, done, _ = self.env.step(action)

            if done:
                new_state = None

            self.memory.add_sample((state, action, reward, new_state))
            self.replay()

            self.steps += 1
            self.epsilon = self.min_epsilon + ((self.max_epsilon
                                                - self.min_epsilon) * math.exp(- self.lamb
                                                                               * self.total_steps))
            state = new_state
            total_reward += reward

            if done:
                self.reward_store += [total_reward]
                break

        print("Rewarded {} in total.".format(total_reward))
        self.total_steps, self.steps = self.total_steps + self.steps, 0

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.model.action_size - 1)
        else:
            return np.argmax(self.model.predict_one(state, self.sess))

    def replay(self):
        batch = self.memory.take_sample(self.model.batch_size)
        states = np.array([entry[0] for entry in batch])
        new_states = np.array([(np.zeros(self.model.observation_size) if
                                entry[3] is None else entry[3])
                               for entry in batch])

        # predict Q(s,a) given the batch of states
        q_s_a = self.model.predict_batch(states, self.sess)

        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self.model.predict_batch(new_states, self.sess)

        # setup training arrays
        x = np.zeros((len(batch), self.model.observation_size))
        y = np.zeros((len(batch), self.model.action_size))

        for i, entry in enumerate(batch):
            state, action, reward, next_state = entry[0], entry[1], entry[2], entry[3]

            # get the current q values for all actions in state
            current_q = q_s_a[i]

            # update the q value for action
            if next_state is None:
                # in this case, the game completed after action, so there is
                # no max Q(s',a') prediction possible
                current_q[action] = reward
            else:
                current_q[action] = reward + self.gamma * np.amax(q_s_a_d[i])

            x[i] = state
            y[i] = current_q

        self.model.train_batch(self.sess, x, y)

    def test_learning(self, num_episodes):
        rewards = []
        for episode in range(num_episodes):
            state = self.env.reset()
            total_reward = 0
            while True:
                action = self.choose_action(state)
                new_state, reward, done, _ = self.env.step(action)
                if done:
                    new_state = None
                state = new_state
                total_reward += reward
                if done:
                    rewards += [total_reward]
                    break

        return [np.array(rewards).mean(), np.array(rewards).min()]
