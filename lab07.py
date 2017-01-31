import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1

print tf.convert_to_tensor(mnist.train.images).get_shape()
print tf.convert_to_tensor(mnist.train.labels).get_shape()
print tf.convert_to_tensor(mnist.test.images).get_shape()
print tf.convert_to_tensor(mnist.test.labels).get_shape()

x = tf.placeholder("float", [None, 784])
y = tf.placeholder("float", [None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

activation = tf.nn.softmax(tf.matmul(x, W) + b)

cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(activation), reduction_indices=1))
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())

    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)

        for i in range(total_batch):
            batch_xs, batch_yx = mnist.train.next_batch(batch_size)

            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_yx})

            avg_cost += sess.run(cost, feed_dict={x: batch_xs, y: batch_yx}) / total_batch

        if epoch % display_step == 0:
            print "Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(avg_cost)

    print "Optimization finished!"

    correct_prediction = tf.equal(tf.argmax(activation, 1), tf.argmax(y, 1))

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    print "Accuracy: ", accuracy.eval({x: mnist.test.images, y: mnist.test.labels})

    # r = randint(0, mnist.test.num_examples - 1)
    # print "Label:", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1))
    # print "Prediction: ", sess.run(tf.argmax(activation, 1), {x: mnist.test.images[r:r + 1]})
    #
    # plt.imshow(mnist.test.images[r:r + 1])
