# coding=utf-8
# 선형회귀

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import seaborn as sns

num_points = 2000
vectors_set = []

for i in xrange(num_points):
    if np.random.random() > 0.5:
        vectors_set.append([np.random.normal(0.0, 0.9), np.random.normal(0.0, 0.9)])
    else:
        vectors_set.append([np.random.normal(3.0, 0.5), np.random.normal(1.0, 0.5)])

# show plot
# df = pd.DataFrame({"x": [v[0] for v in vectors_set],
#                    "y": [v[1] for v in vectors_set]})
# sns.lmplot("x", "y", data=df, fit_reg=False, size=6)
# plt.show()

vectors = tf.constant(vectors_set)
k = 3
centroides = tf.Variable(tf.slice(tf.random_shuffle(vectors), [0, 0], [k, -1]))

expanded_vectors = tf.expand_dims(vectors, 0)
expanded_centroides = tf.expand_dims(centroides, 1)

print expanded_vectors.get_shape()  # (1, 2000, 2)
print expanded_centroides.get_shape()  # (3, 1, 2)
print tf.sub(expanded_vectors, expanded_centroides).get_shape()  # (3, 2000, 2)

assiginments = tf.argmin(tf.reduce_sum(tf.square(tf.sub(expanded_vectors, expanded_centroides)), 2), 0)

print assiginments.get_shape()  # (2000,)

means = tf.concat(0, [
    tf.reduce_mean(tf.gather(vectors, tf.reshape(tf.where(tf.equal(assiginments, c)), [1, -1])), reduction_indices=[1])
    for c in xrange(k)])

update_centroides = tf.assign(centroides, means)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for step in xrange(100):
        _, centroid_values, assignment_values = sess.run([update_centroides, centroides, assiginments])

    data = {"x": [], "y": [], "cluster": []}

    for i in xrange(len(assignment_values)):
        data["x"].append(vectors_set[i][0])
        data["y"].append(vectors_set[i][1])
        data["cluster"].append(assignment_values[i])

    df = pd.DataFrame(data)
    sns.lmplot("x", "y", data=df, fit_reg=False, size=6, hue="cluster", legend=False)
    plt.show()
