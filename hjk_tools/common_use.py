import tensorflow as tf
from tensorflow.python.ops import array_ops
import math

#tile
# if use loss as mean, then gradient is same as not tile.
a = tf.get_variable((1, 2, 2), dtype=tf.float32, initializer=tf.constant_initializer(1))
b = tf.tile(a, (4, 1, 1))
c = tf.get_variable((4, 2, 2), dtype=tf.float32, initializer=tf.constant_initializer(2))
loss = tf.reduce_mean(tf.abs(c - a))

#dense and dropout
a = tf.get_variable((2, 2, 3), dtype=tf.float32, initializer=tf.constant_initializer(1))
b = tf.layers.dense(a, 3, activation=tf.nn.relu, kernel_initializer=tf.constant_initializer(3), bias_initializer=tf.constant_initializer(0))
c = tf.layers.dropout(b, rate=0.5, training=True)

#use sequence_mask and expand_dims and * to mask value
a = tf.get_variable('a', (2, 3, 3), dtype=tf.float32, initializer=tf.constant_initializer(1))
a_mask = tf.get_variable('x', shape=(2,), dtype=tf.int32, initializer=tf.constant_initializer([1, 2]))
mask = array_ops.sequence_mask(a_mask, 3, tf.float32)
mask = tf.expand_dims(mask, axis=-1)
c = a * mask

# max_pooling1d, [batch, time, channel], max(time1, time2), use channel asif size 1.  if need mask, max_pooling also need.
b = tf.layers.max_pooling1d(a, pool_size=2, strides=1, padding='same')

#math.inf, tf.where and tf.cast
a = tf.get_variable(name='a', shape=(2, 3), dtype=tf.float32, initializer=tf.constant_initializer(1))
l = tf.get_variable(name='l', shape=(2,), dtype=tf.int32, initializer=tf.constant_initializer([1, 2]))
out = -math.inf * tf.ones(shape=(2, 3), dtype=tf.float32)
mask = array_ops.sequence_mask(l, a.shape[1])
e = tf.where(mask, x=a, y=out)
mask = tf.cast(mask, tf.float32)
f = mask * a

#batch_normlization
