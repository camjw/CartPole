import gym


class RandomPlayer:
    def __init__(self, env_name):
        self.env_name = env_name
        self.game = gym.make(env_name)

    def play(self, num_episodes):

        for episode in range(num_episodes):
            observation = self.game.reset()
            finished = True
            for t in range(200):
                self.game.render()
                action = self.game.action_space.sample()
                observation, reward, done, info = self.game.step(action)

                if done and finished:
                    finished = False
                    print("Episode finished after {} timesteps".format(t + 1))


if __name__ == "__main__":
    rando = RandomPlayer("CartPole-v0")
    rando.play(10)
