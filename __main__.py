import gym
import tensorflow as tf
import matplotlib.pyplot as plt
from lib import experience_buffer
from lib import game_handler
from lib import model_holder


def main(env_name, num_episodes, batch_size, total_memory, max_epsilon, min_epsilon, lamb, gamma):
    env = gym.make(env_name)

    num_states = env.env.observation_space.shape[0]
    num_actions = env.env.action_space.n

    model = model_holder.ModelHolder(num_actions, num_states, batch_size)
    memory = experience_buffer.ExperienceBuffer(total_memory)

    with tf.Session() as sess:
        sess.run(model._var_init)
        gamehandler = game_handler.GameHandler(env, sess, model, memory, max_epsilon, min_epsilon, lamb, gamma)
        count = 0
        while count < num_episodes:
            if count % 10 == 0:
                print('Episode {} of {}'.format(count+1, num_episodes))
            gamehandler.run()
            count += 1

        #model._saver.save(sess, data)
        plt.plot(gamehandler._reward_store)
        plt.show()
        plt.close("all")

        while True:
            command = input("\nDo you want to see the AI play?\n")
            if command in ["n", "N"]:
                break
            else:
                gamehandler.run(render=True)

if __name__ == "__main__":
    ENV_NAME = "CartPole-v0"
    NUM_EPISODES = 500
    BATCH_SIZE = 5
    TOTAL_MEMORY = 50000
    MAX_EPSILON = 999
    MIN_EPSILON = 0.00001
    LAMBDA = 0.01
    GAMMA = 0.95
    main(ENV_NAME, NUM_EPISODES, BATCH_SIZE, TOTAL_MEMORY, MAX_EPSILON, MIN_EPSILON, LAMBDA, GAMMA)
