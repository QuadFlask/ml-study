# coding=utf-8
# 선형회귀

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# create points

num_points = 1000
vectors_set = []

for i in xrange(num_points):
    x1 = np.random.normal(0., 0.55)
    y1 = x1 * 0.1 + 0.3 + np.random.normal(0.0, 0.03)

    vectors_set.append([x1, y1])

x_data = [v[0] for v in vectors_set]
y_data = [v[1] for v in vectors_set]

# plt.plot(x_data, y_data, 'ro')
# plt.legend()
# plt.show()

# training

W = tf.Variable(tf.random_uniform([1], -1., 1.))
b = tf.Variable(tf.zeros([1]))
y = W * x_data + b

loss = tf.reduce_mean(tf.square(y - y_data))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.5)
train = optimizer.minimize(loss)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for step in xrange(8):
        sess.run(train)
        print step, sess.run(W), sess.run(b), sess.run(loss)

    plt.plot(x_data, y_data, 'ro')
    plt.plot(x_data, sess.run(W) * x_data + sess.run(b))
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
