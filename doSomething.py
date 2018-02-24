import os
import numpy as np

txt = "abc def"
txt = np.asarray(txt)
txt = txt.tostring()
print(type(txt))
print(txt)
txt = txt.decode()
print(txt)

a = np.asarray([1])
print(a.shape)

i = 1
j = i
j = j - 2
print(i, j)


'''
import tensorflow as tf
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
'''