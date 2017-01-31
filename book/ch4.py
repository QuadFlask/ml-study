# coding=utf-8
from __future__ import print_function
from __future__ import print_function
from __future__ import print_function
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

np.set_printoptions(threshold=np.nan)

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

x = tf.placeholder('float', [None, 784], name='x-input')
y_ = tf.placeholder('float', [None, 10], name='y-input')  # 실제 레이블의 확률분포를 담을 플레이스홀더  ex) [  0,   0,   1,   0]

W1 = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

y = tf.nn.softmax(tf.matmul(x, W1) + b)  # 학습된 값으로 레이블의 확률 분포를 계산할 함수,ex) [0.1, 0.1, 0.6, 0.2]

cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    batchSize = 100

    for i in range(100):
        batch_xs, batch_ys = mnist.train.next_batch(batchSize)  # 학습 데이터에서 n개(튜플)를 뽑아오고 Tuple의 첫번째 두번째를 xs, ys 에 할당

        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

        print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


    # display trained w tensor
    def to_pixels(data):
        return np.array(128 + data * 128, dtype='uint8').reshape((28, 28))


    def get_pixels(i):
        data = (tf.transpose(tf.slice(W1, [0, i], [784, 1])) + b[i]).eval()[0]
        return to_pixels(data)


    pixels = get_pixels(0)
    for i in xrange(9):
        pixels = np.concatenate((pixels, get_pixels(i + 1)), axis=1)

    # plt.title('Label is {label}'.format(label='0'))
    # plt.imshow(pixels, cmap='bwr')
    # plt.show()

    correct_predictions = sess.run(tf.cast(correct_prediction, 'float'),
                                   feed_dict={x: mnist.test.images, y_: mnist.test.labels})

    for i in range(len(correct_predictions) / 10):
        predictions = tf.slice(correct_predictions, [i * 10], [10])

        if tf.equal(tf.reduce_min(predictions).eval(), 0).eval():
            print(predictions.eval())
            print('expected %d' % (tf.argmin(predictions, 0).eval()))
            # TODO 각 레이블별 확률 표시 필요

            pixels = to_pixels(mnist.test.images[i])
            plt.title('test image label is %d' % (tf.argmin(mnist.test.labels[i], 0).eval()))
            plt.imshow(pixels, cmap='gray')
            plt.show()
