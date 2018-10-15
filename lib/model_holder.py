import tensorflow as tf


class ModelHolder:
    ''' This class holds all the parameters about the neural model but not the
        training functions. '''

    def __init__(self, action_size, observation_size, batch_size,
                 hidden_size=256, keep_prob=0.9):
        self.observation_size = observation_size
        self.action_size = action_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.dropout = None
        self.keep_prob = keep_prob

        # Define the placeholders. These will become Tensorflow Placeholders.
        self.states = None
        self.actions = None

        # The output operations. The _logits is going to be a Tensorflow layer,
        # the _optimizer is going to be AdamOptimizer and _var_init is just a
        # variable which holds the Tensorflow global variable initalizer.
        self.logits = None
        self.optimizer = None
        self.var_init = None

        # This lets us save the model after we've finished
        self.saver = None

        # Now we setup the model.
        self.define_model()

    def define_model(self):
        ''' This function is just to define all the variables in the neural
            network and itinialize them as tensorflow variables. '''

        self.states = tf.placeholder(shape=[None, self.observation_size],
                                     dtype=tf.float32)
        self.q_s_a = tf.placeholder(shape=[None, self.action_size],
                                    dtype=tf.float32)
        self.dropout = tf.placeholder(dtype=tf.float32)

        # Create two fully connected hidden layers using just Tensorflow
        # variables and tanh functions. This could be done with a Keras
        # sequential or using tf.layers but this is the most basic way to do
        # it.

        l_1_weights = tf.Variable(tf.truncated_normal([self.observation_size,
                                                       self.hidden_size]))
        l_2_weights = tf.Variable(tf.truncated_normal([self.hidden_size,
                                                       self.action_size]))

        l_1_biases = tf.Variable(tf.zeros([self.hidden_size]))
        l_2_biases = tf.Variable(tf.zeros([self.action_size]))

        self.saver = tf.train.Saver([l_1_weights, l_2_weights,
                                     l_1_biases, l_2_biases, ])

        fully_connected_1 = tf.nn.tanh(tf.matmul(self.states, l_1_weights)
                                       + l_1_biases)
        dropped_layer_1 = tf.nn.dropout(fully_connected_1, self.dropout)

        self.logits = tf.matmul(dropped_layer_1, l_2_weights) + l_2_biases

        self.loss = tf.losses.mean_squared_error(self.q_s_a, self.logits)

        self.optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(
            self.loss)
        self.var_init = tf.global_variables_initializer()

    def predict_one(self, state, sess):
        return sess.run(self.logits, feed_dict={
            self.states: state.reshape(1, self.observation_size),
            self.dropout: self.keep_prob})

    def predict_batch(self, states, sess):
        return sess.run(self.logits, feed_dict={self.states: states,
                                                self.dropout: self.keep_prob})

    def train_batch(self, sess, state_batch, reward_batch):
        sess.run(self.optimizer, feed_dict={self.states: state_batch,
                                            self.q_s_a: reward_batch,
                                            self.dropout: self.keep_prob})

    def load_network(self, sess, filename):
        load_save = tf.train.import_meta_graph(filename)
        load_save.restore(sess, tf.train.latest_checkpoint('./'))
