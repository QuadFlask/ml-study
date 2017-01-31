# coding=utf-8
# 정확도 91% 정도가 나온다!

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(threshold=np.nan)
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1

x = tf.placeholder('float', [None, 784], name='x-input')
y = tf.placeholder('float', [None, 10], name='y-input')

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

activation = tf.nn.softmax(tf.matmul(x, W) + b)
cost = tf.reduce_sum(-tf.reduce_sum(y * tf.log(activation), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys}) / total_batch

            if epoch % display_step == 0:
                print("Epoch:", "%04d" % (epoch + 1), "cost:", "{:.9f}".format(avg_cost))

        print("optimization finished")

        correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
