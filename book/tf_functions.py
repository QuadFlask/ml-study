# coding=utf-8
# 선형회귀

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    #  sub 두 텐서간 빼기 -> 차원이 작은쪽이 큰쪽으로 맞춰짐(브로드캐스팅)
    print sess.run(tf.sub([4, 5, 6], [1, 2, 3]))
    print sess.run(tf.sub([4], [1, 2, 3]))

    #  reduce_sum 두번째 파라미터의 차원으로 텐서를 더함 -> 그 차원이 없어진다
    print sess.run(tf.reduce_sum(range(1, 11), 0))
    print sess.run(tf.reduce_sum([[1, 2], [3, 4], [5, 6]], 0))

    # argmin, argmax
    print sess.run(tf.argmin([4, 5, 1, 2, 3], 0))
    print sess.run(tf.argmax([4, 5, 1, 2, 3], 0))

    # equal
    print tf.equal([1, 2, 3], [1, 4, 3]).eval()
    print tf.cast(tf.equal([1, 2, 3], [1, 4, 3]), 'float').eval()
    print tf.equal(1, 1).eval()

    print min([1, 2, 3, 1, 0])
    print(tf.reduce_min([1, 3, 1, 0]))
