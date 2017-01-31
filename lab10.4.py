# coding=utf-8
# More deep & dropout
# 더 깊게하고(5개의 레이어) 드랍아웃을 해보자!
# 깊게 할경우 오버피팅이 될 수 있으니 드랍아웃해서 오버피팅을 막아보자
# 정확도가 97.9~98.1% 정도가 나온다.
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(threshold=np.nan)

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100
display_step = 1
dropout_rate = tf.placeholder("float")


def xavier_init(n_inputs, n_outputs, uniform=True):
    if uniform:
        init_range = tf.sqrt(6.0 / (n_inputs + n_outputs))
        return tf.random_uniform_initializer(-init_range, init_range)
    else:
        stddev = tf.sqrt(3.0 / (n_inputs + n_outputs))
        return tf.truncated_normal_initializer(stddev=stddev)


x = tf.placeholder('float', [None, 784], name='x-input')
y = tf.placeholder('float', [None, 10], name='y-input')

W1 = tf.get_variable("W1", shape=[784, 256], initializer=xavier_init(784, 256))
W2 = tf.get_variable("W2", shape=[256, 256], initializer=xavier_init(256, 256))
W3 = tf.get_variable("W3", shape=[256, 256], initializer=xavier_init(256, 256))
W4 = tf.get_variable("W4", shape=[256, 256], initializer=xavier_init(256, 256))
W5 = tf.get_variable("W5", shape=[256, 10], initializer=xavier_init(256, 10))
b1 = tf.Variable(tf.random_normal([256]))
b2 = tf.Variable(tf.random_normal([256]))
b3 = tf.Variable(tf.random_normal([256]))
b4 = tf.Variable(tf.random_normal([256]))
b5 = tf.Variable(tf.random_normal([10]))

_L1 = tf.nn.relu(tf.add(tf.matmul(x, W1), b1))
L1 = tf.nn.dropout(_L1, dropout_rate)

_L2 = tf.nn.relu(tf.add(tf.matmul(L1, W2), b2))
L2 = tf.nn.dropout(_L2, dropout_rate)

_L3 = tf.nn.relu(tf.add(tf.matmul(L2, W3), b3))
L3 = tf.nn.dropout(_L3, dropout_rate)

_L4 = tf.nn.relu(tf.add(tf.matmul(L3, W4), b4))
L4 = tf.nn.dropout(_L4, dropout_rate)

hypothesis = tf.add(tf.matmul(L4, W5), b5)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(hypothesis, y))
# with_logits: hypothesis의 숫자가 소프트맥스를 취하지 않았다를 말해줌
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
# 가장 좋다고 알려진 옵티마이저

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples / batch_size)

        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys, dropout_rate: 0.7})
            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_ys, dropout_rate: 0.7}) / total_batch

        if epoch % display_step == 0:
            print("Epoch:", "%04d" % (epoch + 1), "cost:", "{:.9f}".format(avg_cost))

        print("optimization finished")

        correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels, dropout_rate: 1}))
