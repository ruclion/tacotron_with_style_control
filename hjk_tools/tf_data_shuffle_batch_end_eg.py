import tensorflow as tf
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

dataset = tf.data.Dataset.from_tensor_slices([1, 2, 3, 4, 5])
dataset = dataset.batch(2)
dataset = dataset.shuffle(buffer_size=10)
dataset = dataset.repeat(5)



iterator = dataset.make_one_shot_iterator()
next_element = iterator.get_next()

with tf.Session() as sess:
    while True:
        try:
            value = sess.run(next_element)
            print(value)
        except:
            print('finish')
            break