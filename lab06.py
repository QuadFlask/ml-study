# coding=utf-8
# Multinomial classification
# 여러 카테고리를 여러개의 선으로 구분
# 각 카테고리에 대한 확률을 계산 -> softmax

import tensorflow as tf
import numpy as np

xy = np.loadtxt('train_lab06.txt', unpack=True, dtype='float32')
x_data = np.transpose(xy[0:3])  # 행렬의 x,y 를 바꿔줌
y_data = np.transpose(xy[3:])

X = tf.placeholder("float", [None, 3])  # x1, x2, 1(bias)
Y = tf.placeholder("float", [None, 3])  # A, B, C => 3 classes
W = tf.Variable(tf.zeros([3, 3]))

hypothesis = tf.nn.softmax(tf.matmul(X, W))

learning_rate = 0.001

cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(hypothesis), reduction_indices=1))

optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for step in xrange(2001):
        sess.run(optimizer, feed_dict={X: x_data, Y: y_data})
        if step % 200 == 0:
            print step, sess.run(cost, feed_dict={X: x_data, Y: y_data}), sess.run(W)

    # test

    print '--------------'

    a = sess.run(hypothesis, feed_dict={X: [[1, 11, 7]]})
    print a, sess.run(tf.arg_max(a, 1))  # arg_max 제일 큰 값이 있는 인덱스를 리턴

    b = sess.run(hypothesis, feed_dict={X: [[1, 3, 4]]})
    print b, sess.run(tf.arg_max(b, 1))

    c = sess.run(hypothesis, feed_dict={X: [[1, 1, 0]]})
    print c, sess.run(tf.arg_max(c, 1))

    all = sess.run(hypothesis, feed_dict={X: [[1, 11, 7], [1, 3, 4], [1, 1, 0]]})
    print all, sess.run(tf.arg_max(all, 1))
