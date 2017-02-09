import numpy as np
import random
from common.functions import *
from common.gradient import numerical_gradient

learning_rate = 0.1


def f1(a, x, b):
    return a * x + b


def f1inter(a, b):
    return [0, a, b,
            -b / a, 0,
            0, 0,
            0, b]


def f2(a, x, b, c):
    return a * x ** 2 + b * x + c


def f2inter(a, b, c):
    return [a, b, c,
            (-b - (b * b - 4 * a * c) ** 1 / 2) / (2 * a), 0,
            (-b + (b * b - 4 * a * c) ** 1 / 2) / (2 * a), 0,
            0, c]


class Net:
    def __init__(self, input_size, hidden_size, hidden_size_2, output_size, weight_init_std=0.01):
        self.params = {'W1': weight_init_std * np.random.randn(input_size, hidden_size),
                       'b1': np.zeros(hidden_size),
                       'W2': weight_init_std * np.random.randn(hidden_size, hidden_size_2),
                       'b2': np.zeros(hidden_size_2),
                       'W3': weight_init_std * np.random.randn(hidden_size_2, output_size),
                       'b3': np.zeros(output_size)}

        print(self.params['W1'].shape)
        print(self.params['b1'].shape)
        print(self.params['W2'].shape)
        print(self.params['b2'].shape)
        print(self.params['W3'].shape)
        print(self.params['b3'].shape)

    def predict(self, x):
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = a3

        return y

    def loss(self, x, t):
        y = self.predict(x)
        return np.sum((t - y) ** 2)
        # return 2 ** np.sum(np.abs(t - y)) - 1
        # return np.sum((1 + np.abs(t - y)) ** 2) - 1

    def gradient(self, x, t):
        W1, W2, W3 = self.params['W1'], self.params['W2'], self.params['W3']
        b1, b2, b3 = self.params['b1'], self.params['b2'], self.params['b3']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = a3

        # backward
        dy = (y - t) / batch_num

        grads['W3'] = np.dot(z2.T, dy)
        grads['b3'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W3.T)
        dz1 = sigmoid_grad(a2) * da1
        grads['W2'] = np.dot(z1.T, dz1)
        grads['b2'] = np.sum(dz1, axis=0)

        da2 = np.dot(da1, W2.T)
        dz2 = sigmoid_grad(a1) * da2
        grads['W1'] = np.dot(x.T, dz2)
        grads['b1'] = np.sum(dz2, axis=0)

        return grads


def random_train_batch(count=100):
    batch_data = []
    batch_label = []
    for batch_i in range(count):
        a = random.random() * 10 - 5
        b = random.random() * 10 - 5
        c = random.random() * 10 - 5
        batch_data.append(f1inter(a, b))
        batch_label.append([0, a, b])

        batch_data.append(f2inter(a, b, c))
        batch_label.append([a, b, c])

    return np.array(batch_data), np.array(batch_label)


network = Net(input_size=9, hidden_size=1000, hidden_size_2=125, output_size=3)
train_batch_size = 100

for i in range(30000):
    train_data, train_label = random_train_batch(train_batch_size)

    grad = network.gradient(train_data, train_label)

    for key in ['W1', 'b1', 'W2', 'b2', 'W3', 'b3']:
        network.params[key] -= learning_rate * grad[key]

    if i % 1000 == 0:
        train_loss = network.loss(train_data, train_label)

        test_data, test_label = random_train_batch(1)
        predict = network.predict(test_data)
        test_loss = network.loss(test_data, test_label)

        print(str(i) +
              ' / train loss: ' + str(train_loss / train_batch_size) + ' / test loss: ' + str(test_loss) +
              ' \ntest f(x): ' + str(test_label[0][1]) + ' x + ' + str(test_label[0][2]) +
              ' \ntest f(x): ' + str(test_label[1][0]) + ' x^2 + ' + str(test_label[1][1]) + ' x + ' + str(test_label[1][2]) +
              ' \npredict: ' + str(predict[0]) + ' / ' + str(predict[1]) +
              '\n--------------------------------\n')
