import tensorflow as tf

class ModelHolder:
    ''' This class holds all the parameters about the neural model but not the
        training functions. '''

    def __init__(self, action_size, observation_size, batch_size,
                 hidden_size=256, keep_prob=0.9):
        ''' This is the first time I've written a class where all attributes are
            declared in __init__, even if they are initialized to None. This is
            so not true... I barely declare half of the attrs...'''

        self._observation_size = observation_size
        self._action_size = action_size
        self._batch_size = batch_size
        self._hidden_size = hidden_size
        self._dropout = None
        self._keep_prob = keep_prob


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


        # Now we setup the model.
        self.define_model()

    def define_model(self):
        ''' This function is just to define all the variables in the neural
            network and itinialize them as tensorflow variables. '''

        self._states = tf.placeholder(shape=[None, self._observation_size],
                                      dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._action_size],
                                     dtype=tf.float32)
        self._dropout = tf.placeholder(dtype=tf.float32)


        # Create two fully connected hidden layers using just Tensorflow
        # variables and tanh functions. This could be done with a Keras
        # sequential or using tf.layers but this is the most basic way to do it.

        l_1_weights = tf.Variable(tf.truncated_normal([self._observation_size,
                                                       self._hidden_size]))
        l_2_weights = tf.Variable(tf.truncated_normal([self._hidden_size,
                                                       self._action_size]))

        l_1_biases = tf.Variable(tf.zeros([self._hidden_size]))
        l_2_biases = tf.Variable(tf.zeros([self._action_size]))


        self._saver = tf.train.Saver([l_1_weights, l_2_weights,
                                      l_1_biases, l_2_biases,])

        fully_connected_1 = tf.nn.tanh(tf.matmul(self._states, l_1_weights)
                                       + l_1_biases)
        dropped_layer_1 = tf.nn.dropout(fully_connected_1, self._dropout)

        self._logits = tf.matmul(dropped_layer_1, l_2_weights) + l_2_biases

        self.loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)

        self._optimizer = tf.train.AdamOptimizer(learning_rate=0.0001).minimize(self.loss)
        self._var_init = tf.global_variables_initializer()

    def predict_one(self, state, sess):
        return sess.run(self._logits, feed_dict={
            self._states: state.reshape(1, self._observation_size),
            self._dropout: self._keep_prob})

    def predict_batch(self, states, sess):
        return sess.run(self._logits, feed_dict={self._states: states,
            self._dropout: self._keep_prob})

    def train_batch(self, sess, state_batch, reward_batch):
        sess.run(self._optimizer, feed_dict={self._states: state_batch,
                                             self._q_s_a: reward_batch,
                                             self._dropout: self._keep_prob})

    def load_network(self, sess, filename):
        load_save = tf.train.import_meta_graph(filename)
        load_save.restore(sess, tf.train.latest_checkpoint('./'))
