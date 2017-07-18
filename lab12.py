import tensorflow as tf
# from tensorflow.models.rnn import rnn, rnn_cell
import numpy as np

char_rdic = ['h', 'e', 'l', 'o']
char_dic = {w: i for i, w in enumerate(char_rdic)}
print(char_dic)
x_data = np.array([[1, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 1, 0]],
                  dtype="f")

sample = [char_dic[c] for c in "hello"]

char_vocab_size = len(char_dic)
rnn_size = char_vocab_size
time_step_size = 4
batch_size = 1

rnn_cell = tf.nn.rnn_cell.BasicRNNCell(rnn_size)
state = tf.zeros([batch_size, tf.nn.rnn_cell.state_size])
X_split = tf.split(0, time_step_size, x_data)
outputs, state = tf.nn.rnn(rnn_cell, X_split, state)

with tf.Session() as sess:
    # you need to initialize all variables
    sess.run(tf.global_variables_initializer())

    result = sess.run(tf.arg_max(lo))
