# coding=utf-8
# logistic regression -> 0, 1 구분을 sigmoid 함수로?


import tensorflow as tf
import numpy as np

xy = np.loadtxt('train_lab05.txt', unpack=True, dtype='float32')

x_data = xy[0:-1]
y_data = xy[-1]

print x_data
print y_data

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)

W = tf.Variable(tf.random_uniform([1, len(x_data)], -1.0, 1.0))

h = tf.matmul(W, X)
hypothesis = tf.div(1., 1. + tf.exp(-h))  # sigmoid

cost = -tf.reduce_mean(Y * tf.log(hypothesis) + (1 - Y) * tf.log(1 - hypothesis))

a = tf.Variable(0.1)
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for step in xrange(2001):
    sess.run(train, feed_dict={X: x_data, Y: y_data})
    if step % 10 == 0:
        print step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W)

print '-----------------------------'

print sess.run(hypothesis, feed_dict={X: [[1], [2], [2]]}) > 0.5  # 2시간 공부하고 2번 수업 -> fail
print sess.run(hypothesis, feed_dict={X: [[1], [5], [5]]}) > 0.5  # 5시간 공부하고 5번 수업 -> pass

print sess.run(hypothesis, feed_dict={X: [[1, 1], [4, 3], [3, 5]]}) > 0.5  # a: 4시간 공부 3번 수업 b: 3시간 공부 5번 수업
