import tensorflow as tf

class ModelHolder:
    ''' This class holds all the parameters about the neural model but not the
        training functions. '''

    def __init__(self, action_size, observation_size, batch_size,
                 hidden_size_1=50, hidden_size_2=50, reg_factor=0.0001):
        ''' This is the first time I've written a class where all attributes are
            declared in __init__, even if they are initialized to None. '''

        self._observation_size = observation_size
        self._action_size = action_size
        self._batch_size = batch_size
        self._hidden_size_1 = hidden_size_1
        self._hidden_size_2 = hidden_size_2

        # Define the placeholders. These will become Tensorflow Placeholders.
        self._states = None
        self._actions = None

        # The output operations. The _logits is going to be a Tensorflow layer,
        # the _optimizer is going to be AdamOptimizer and _var_init is just a
        # variable which holds the Tensorflow global variable initalizer.
        self._logits = None
        self._optimizer = None
        self._var_init = None

        # This lets us save the model after we've finished
        self._saver = None

        # This lets us do regularisation
        self._reg_factor = reg_factor


        # Now we setup the model.
        self.define_model()

    def define_model(self):
        ''' This function is just to define all the variables in the neural
            network and itinialize them as tensorflow variables. '''

        self._states = tf.placeholder(shape=[None, self._observation_size],
                                      dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._action_size],
                                     dtype=tf.float32)

        # Create two fully connected hidden layers using just Tensorflow
        # variables and ReLU functions. This could be done with a Keras
        # sequential or using tf.layers but this is the most basic way to do it.

        l_1_weights = tf.Variable(tf.truncated_normal([self._observation_size,
                                                       self._hidden_size_1]))
        l_2_weights = tf.Variable(tf.truncated_normal([self._hidden_size_1,
                                                       self._hidden_size_2]))
        l_3_weights = tf.Variable(tf.truncated_normal([self._hidden_size_2,
                                                        self._action_size]))
        l_1_biases = tf.Variable(tf.zeros([self._hidden_size_1]))
        l_2_biases = tf.Variable(tf.zeros([self._hidden_size_2]))
        l_3_biases = tf.Variable(tf.zeros([self._action_size]))
        self._saver = tf.train.Saver([l_1_weights, l_2_weights,
                                      l_1_biases, l_2_biases])
        fully_connected_1 = tf.nn.tanh(tf.matmul(self._states, l_1_weights)
                                       + l_1_biases)
        fully_connected_2 = tf.nn.tanh(tf.matmul(fully_connected_1, l_2_weights)
                                        + l_2_biases)
        self._logits = tf.matmul(fully_connected_2, l_3_weights) + l_3_biases

        self.loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)

        # This is the regularisation bit
        for w in [l_1_weights, l_2_weights, l_3_weights]:
            self.loss += self._reg_factor * tf.reduce_sum(tf.square(w))

        self._optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(self.loss)
        self._var_init = tf.global_variables_initializer()

    def predict_one(self, state, sess):
        return sess.run(self._logits, feed_dict={self._states:
                                                 state.reshape(1, self._observation_size)})

    def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states})

    def train_batch(self, sess, state_batch, reward_batch):
        sess.run(self._optimizer, feed_dict={self._states: state_batch,
                                             self._q_s_a: reward_batch})

    def load_network(self, sess, filename):
        load_save = tf.train.import_meta_graph(filename)
        load_save.restore(sess, tf.train.latest_checkpoint('./'))
