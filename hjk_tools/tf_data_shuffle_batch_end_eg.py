import tensorflow as tf
import os
import numpy as np

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

x = np.arange(0, 6000)
# print(x)

dataset = tf.data.Dataset.from_tensor_slices(x)
dataset = dataset.batch(32)
dataset = dataset.shuffle(buffer_size=10000)
dataset = dataset.repeat(5)



iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    value = sess.run(next_element)
    print(value)
    # while True:
    #     try:
    #         value = sess.run(next_element)
    #         # print(value)
    #     except:
    #         print('finish')
    #         break