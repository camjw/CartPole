class ModelHolder:
    ''' This class holds all the parameters about the neural model but not the
        training functions. '''

    def __init__(self, action_size, observation_size, batch_size, hidden_size=24):
        ''' This is the first time I've written a class where all attributes are
            declared in __init__, even if they are initialized to None. '''

        self._observation_size = observation_size
        self._action_size = action_size
        self._batch_size = batch_size
        self._hidden_size = hidden_size

        # Define the placeholders. These will become Tensorflow Placeholders.
        self._states = None
        self._actions = None

        # The output operations. The _logits is going to be a Tensorflow layer,
        # the _optimizer is going to be AdamOptimizer and _var_init is just a
        # variable which holds the Tensorflow global variable initalizer.
        self._logits = None
        self._optimizer = None
        self._var_init = None

        # Now we setup the model.
        self._define_model()

    def _define_model():
        ''' This function is just to define all the variables in the neural
            network and itinialize them as tensorflow variables. '''
        
        self._states = tf.placeholder(shape=[None, self._observation_size],
                                      dtype=tf.float32)
        self._q_s_a = tf.placeholder(shape=[None, self._action_size],
                                     dtype=tf.float32)

        # Create two fully connected hidden layers using just Tensorflow
        # variables and ReLU functions. This could be done with a Keras
        # sequential or using tf.layers but this is the most basic way to do it.

        l_1_weights = tf.Variable(tf.truncated_normal([self._states,
                                                           self._hidden_size]))
        l_2_weights = tf.Variable(tf.truncated_normal([self._hidden_size,
                                                       self._hidden_size]))
        l_3_weights = tf.Variable(tf.truncated_normal([self._hidden_size,
                                                       self._action_size]))
        l_1_biases = tf.Variable(tf.zeros([hidden_nodes]))
        l_2_biases = tf.Variable(tf.zeros([hidden_nodes]))
        l_3_biases = tf.variable(tf.zeros([self._action_size]))

        fully_connected_1 = tf.nn.relu(tf.matmul(self._states, l_1_weights)
                                       + l_1_biases)
        fully_connected_2 = tf.nn.relu(tf.matmul(fully_connected_1, l_2_weights)
                                       + l_2_biases)
        self._logits = tf.matmul(fully_connected_2, l_3_weights) + l_3_biases

        loss = tf.losses.mean_squared_error(self._q_s_a, self._logits)
        self._optimizer = tf.train.AdamOptimizer().minimize(loss)
        self._var_init = tf.global_variables_initializer()
