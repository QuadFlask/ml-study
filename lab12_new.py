import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import time

x_data = np.array([[
    1, 0, 0, 0,
    0, 1, 0, 0,
    0, 0, 1, 0,
    0, 0, 0, 1
]], dtype='f')

batch_size = 1
time_step_size = 4

rnn_cell = rnn.BasicRNNCell(4)
state = tf.zeros([batch_size, rnn_cell.state_size])
X_split = tf.split(0, time_step_size, x_data)

outputs, state = rnn.rnn(rnn_cell, X_split, state)

logits = tf.reshape(tf.concat(1, outputs), [-1, rnn_size])
targets = tf.reshape(sample[1:], [-1])
W = tf.ones([time_step_size * batch_size])

loss = tf.nn.seq2seq.sequence_loss_by_example([logits], [targets], [W])
cost = tf.reduce_sum(loss) / batch_size
train_op = tf.train.RMSPropOptimizer(0.01, 0.9).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100):
        sess.run(train_op)
        result = sess.run(tf.argmax(logits, 1))
        print(result, [char])