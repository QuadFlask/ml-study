import numpy as np
import random
from common.functions import *
from common.gradient import numerical_gradient

learning_rate = 0.1


def f1(a, x, b):
    return a * x + b


class Net:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.params = {'W1': weight_init_std * np.random.randn(input_size, hidden_size),
                       'b1': np.zeros(hidden_size),
                       'W2': weight_init_std * np.random.randn(hidden_size, output_size),
                       'b2': np.zeros(output_size)}

        print(self.params['W1'].shape)
        print(self.params['b1'].shape)
        print(self.params['W2'].shape)
        print(self.params['b2'].shape)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = a2

        return y

    def loss(self, x, t):
        y = self.predict(x)
        return np.sum((t - y) ** 2)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {'W1': numerical_gradient(loss_W, self.params['W1']),
                 'b1': numerical_gradient(loss_W, self.params['b1']),
                 'W2': numerical_gradient(loss_W, self.params['W2']),
                 'b2': numerical_gradient(loss_W, self.params['b2'])}

        return grads

    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = a2

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads


width_size = 18
half_width_size = int(width_size / 2)


def random_train():
    data = np.zeros(width_size * width_size)
    a = random.randrange(-5, 5)
    b = random.randrange(-5, 5)
    print('fx: ' + str(a) + 'x+' + str(b))
    for x in range(width_size):
        _x = x - half_width_size
        _y = f1(a, _x, b) - half_width_size
        i = _x + half_width_size + (_y + half_width_size) * width_size
        if 0 <= i < width_size * width_size:
            data[i] = 1

    return np.array([data]), np.array([[a, b]])


network = Net(input_size=324, hidden_size=81, output_size=2)

for i in range(50000):
    train_data, train_label = random_train()

    grad = network.gradient(train_data, train_label)

    for key in ['W1', 'b1', 'W2', 'b2']:
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(train_data, train_label)
    print('loss: ' + str(loss))

    test_data, test_label = random_train()
    predict = network.predict(test_data)
    test_acc = 1. - network.loss(test_data, test_label)
    print('predict: ' + str(predict) + ' / test acc: ' + str(test_acc))
