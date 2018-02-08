import tensorflow as tf
from tensorflow.contrib.rnn import LSTMStateTuple
import TFCommon.Initializer as Initializer
from tensorflow.contrib.keras import activations
from tensorflow.python.ops import array_ops


# Copied from "https://github.com/teganmaharaj/zoneout/blob/master/zoneout_seq2seq.py"
# Wrapper for the TF RNN cell
# For an LSTM, the 'cell' is a tuple containing state and cell
# We use TF's dropout to implement zoneout
class ZoneoutWrapper(tf.nn.rnn_cell.RNNCell):
    """Operator adding zoneout to all states (states+cells) of the given cell."""

    def __init__(self, cell, state_zoneout_prob, is_training=True, seed=None):
        if not isinstance(cell, tf.nn.rnn_cell.RNNCell):
            raise TypeError("The parameter cell is not an RNNCell.")
        if (isinstance(state_zoneout_prob, float) and
                not (state_zoneout_prob >= 0.0 and state_zoneout_prob <= 1.0)):
            raise ValueError("Parameter zoneout_prob must be between 0 and 1: %d"
                             % state_zoneout_prob)
        self._cell = cell
        self._zoneout_prob = state_zoneout_prob
        self._seed = seed
        self.is_training = is_training

    @property
    def state_size(self):
        return self._cell.state_size

    @property
    def output_size(self):
        return self._cell.output_size

    def __call__(self, inputs, state, scope=None):
        if isinstance(self.state_size, tuple) != isinstance(self._zoneout_prob, tuple):
            raise TypeError("Subdivided states need subdivided zoneouts.")
        if isinstance(self.state_size, tuple) and len(tuple(self.state_size)) != len(tuple(self._zoneout_prob)):
            raise ValueError("State and zoneout need equally many parts.")
        output, new_state = self._cell(inputs, state, scope)
        if isinstance(self.state_size, tuple):
            if self.is_training:
                new_state = tuple((1 - state_part_zoneout_prob) * tf.nn.dropout(
                    new_state_part - state_part, (1 - state_part_zoneout_prob), seed=self._seed) + state_part
                                  for new_state_part, state_part, state_part_zoneout_prob in
                                  zip(new_state, state, self._zoneout_prob))
            else:
                new_state = tuple(state_part_zoneout_prob * state_part + (1 - state_part_zoneout_prob) * new_state_part
                                  for new_state_part, state_part, state_part_zoneout_prob in
                                  zip(new_state, state, self._zoneout_prob))
        else:
            if self.is_training:
                new_state = (1 - self._zoneout_prob) * tf.nn.dropout(
                    new_state - state, (1 - self._zoneout_prob), seed=self._seed) + state
            else:
                new_state = self._zoneout_prob * state + (1 - self._zoneout_prob) * new_state
        return output, new_state


class GRUCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, num_units, init_state=None, gate_activation="sigmoid", reuse=None):
        self.__num_units = num_units
        if gate_activation == "sigmoid":
            self.__gate_activation = tf.sigmoid
        elif gate_activation == "hard_sigmoid":
            self.__gate_activation = activations.hard_sigmoid
        else:
            raise ValueError
        self.__init_state = init_state
        self.__reuse = reuse

    @property
    def state_size(self):
        return self.__num_units

    @property
    def output_size(self):
        return self.__num_units

    def zero_state(self, batch_size, dtype):
        return super(GRUCell, self).zero_state(batch_size, dtype)

    def init_state(self, batch_size, dtype):
        if self.__init_state is not None:
            return self.__init_state
        else:
            return self.zero_state(batch_size, dtype)

    def __call__(self, x, h_prev, scope=None):
        with tf.variable_scope(scope or type(self).__name__):

            # Check if the input size exist.
            input_size = x.shape.with_rank(2)[1].value
            if input_size is None:
                raise ValueError("Expecting input_size to be set.")

            ### get weights.
            W_shape = (input_size, self.output_size)
            U_shape = (self.output_size, self.output_size)
            b_shape = (self.output_size,)
            Wrz = tf.get_variable(name="Wrz", shape=(input_size, 2 * self.output_size))
            Wh = tf.get_variable(name='Wh', shape=W_shape)
            Urz = tf.get_variable(name="Urz", shape=(self.output_size, 2 * self.output_size),
                                  initializer=Initializer.random_orthogonal_initializer())
            Uh = tf.get_variable(name='Uh', shape=U_shape,
                                 initializer=Initializer.random_orthogonal_initializer())
            brz = tf.get_variable(name="brz", shape=(2 * self.output_size),
                                  initializer=tf.constant_initializer(0.0))
            bh = tf.get_variable(name='bh', shape=b_shape,
                                 initializer=tf.constant_initializer(0.0))

            ### calculate r and z
            rz = self.__gate_activation(tf.matmul(x, Wrz) + tf.matmul(h_prev, Urz) + brz)
            r, z = array_ops.split(rz, num_or_size_splits=2, axis=1)

            ### calculate candidate
            h_slash = tf.tanh(tf.matmul(x, Wh) + tf.matmul(r * h_prev, Uh) + bh)

            ### final cal
            new_h = (1-z) * h_prev + z * h_slash

            return new_h, new_h
