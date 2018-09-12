class GameHandler
    ''' This class deals with all interactions with the game/gym challenge. The class takes env, an OpenAI gym
        environment, sess: a tensorflow session, model which will be a tensorflow network, memory which will be an
        ExperienceBuffer object, some epsilons which dictate when the GameHandler will take random action, a lamb
        (short for lambda) which also dictates the rate of decrease of epsilon, and a gamma which is the amount we discount
        future reward by.'''

    def __init__(env, sess, model, memory, max_epsilon, min_epsilon, lamb, gamma):
        self.env = env
        self._sess = sess
        self._model = model
        self._memory = memory
        self._max_epsilon = max_epsilon
        self._min_epsilon = min_epsilon
        self._epsilon = max_epsilon
        self._lamb = lamb # can't use lambda because python wants to create a lambda function
        self._gamma = gamma
        self._reward_store = []

    def run(self, render=False):
        state = self.env.reset()
        total_reward = 0

        while True:
            if render:
                self.env.render()

            action = self._choose_action(state)
            new_state, reward, done, _ = self._env.step(action)

            if done:
                new_state = None

            self._memory.add_sample((state, action, reward, new_state))
            self._replay()

            self._steps += 1
            self._epsilon = self._min_epsilon + ((self._max_epsilon - self.min_epsilon)
                                                * math.exp(- self._lamb * self._steps))
            state = next_state
            total_reward += reward

            if done:
                self._reward_store += [total_reward]
                break

        print("We made it through {} steps and were rewarded {} in total.".format(self._steps, total_reward))

    def _choose_action(self, state):
        if random.random() < self._epsilon:
            return random.randint(0, self._model.num_actions -1)
        else:
            return np.argmax(self._model.predict_batch(state.reshape(1, self._model.num_states), self._sess))

    def _replay(self):
        batch = self._memory.take_sample(self._model.batch_size)
        states = np.array([entry[0] for entry in batch])
        new_states = np.array([(np.zeros(self._model.num_states) if entry[3]
                                is None else entry[3]) for entry in batch])

        # predict Q(s,a) given the batch of states
        q_s_a = self._model.predict_batch(states, self._sess)

        # predict Q(s',a') - so that we can do gamma * max(Q(s'a')) below
        q_s_a_d = self._model.predict_batch(next_states, self._sess)

        # setup training arrays
        x = np.zeros((len(batch), self._model.num_states))
        y = np.zeros((len(batch), self._model.num_actions))

        for i, entry in enumerate(batch):
            state, action, reward, next_state = entry[0], entry[1], entry[2], entry[3]

            # get the current q values for all actions in state
            current_q = q_s_a[i]

            # update the q value for action
            if next_state is None:
                # in this case, the game completed after action, so there is no max Q(s',a')
                # prediction possible
                current_q[action] = reward
            else:
                current_q[action] = reward + self._gamma * np.amax(q_s_a_d[i])

            x[i] = state
            y[i] = current_q

        self._model.train_batch(self._sess, x, y)