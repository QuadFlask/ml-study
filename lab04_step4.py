import tensorflow as tf
import numpy as np

xy = np.loadtxt('train_lab04.txt', unpack=True, dtype='float32')

x_data = xy[0:-1]
y_data = xy[-1]

print x_data
print y_data

W = tf.Variable(tf.random_uniform([1, 3], -5.0, 5.0))

hypothesis = tf.matmul(W, x_data)

cost = tf.reduce_mean(tf.square(hypothesis - y_data))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for step in xrange(601):
    sess.run(train)
    if step % 10 == 0:
        print step, sess.run(cost), sess.run(W)
