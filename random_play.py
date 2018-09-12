import gym

env = gym.make('CartPole-v0')

for i_episode in range(5):
    observation = env.reset()
    done_test = True
    for t in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if done and done_test:
            done_test = False
            print("Episode finished after {} timesteps".format(t+1))
