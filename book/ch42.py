# coding=utf-8
from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(threshold=np.nan)

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)


def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


with tf.name_scope('input'):
    x = tf.placeholder('float', [None, 784], name='x-input')
    y_ = tf.placeholder('float', [None, 10], name='y-input')  # 실제 레이블의 확률분포를 담을 플레이스홀더  ex) [  0,   0,   1,   0]

with tf.name_scope('weights'):
    W = tf.Variable(tf.zeros([784, 10]))
    variable_summaries(W)

with tf.name_scope('biases'):
    b = tf.Variable(tf.zeros([10]))
    variable_summaries(b)

with tf.name_scope('weight_image'):
    for i in xrange(10):
        image_shaped_input = tf.reshape(tf.slice(W, [0, i], [784, 1]), [-1, 28, 28, 1]) + tf.slice(b, [i], [1])
        tf.summary.image('weight %d' % i, image_shaped_input, 1)

with tf.name_scope('softmax'):
    y = tf.nn.softmax(tf.matmul(x, W) + b)  # 학습된 값으로 레이블의 확률 분포를 계산할 함수,ex) [0.1, 0.1, 0.6, 0.2]
    tf.summary.histogram('y', y)

with tf.name_scope('cross_entropy'):
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
tf.summary.scalar('cross_entropy', cross_entropy)

with tf.name_scope('accuracy'):
    with tf.name_scope('correct_prediction'):
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    with tf.name_scope('accuracy'):
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
tf.summary.scalar('accuracy', accuracy)

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batchSize = 100

    merged = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("./mnist_logs", sess.graph)

    for i in range(200):
        batch_xs, batch_ys = mnist.train.next_batch(batchSize)  # 학습 데이터에서 n개(튜플)를 뽑아오고 Tuple의 첫번째 두번째를 xs, ys 에 할당
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

        writer.add_summary(sess.run(merged, feed_dict={x: mnist.test.images, y_: mnist.test.labels}), i)
