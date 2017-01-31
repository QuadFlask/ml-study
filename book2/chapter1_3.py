# coding=utf-8
# chapter 1

import numpy as np

for i in [1, 2, 3]:
    print(i)


def hello(object):
    print("Hello world " + object + "!")


hello("cat")


class MyClass:
    def __init__(self, name):
        self.name = name
        print("init")

    def hello(self):
        print("hello " + self.name + "!")

    def goodbye(self):
        print("good bye " + self.name + "!")


m = MyClass("tester")
m.hello()
m.goodbye()

x = np.array([1., 2., 3.])
print(x)

y = np.array([2., 4., 6.])

print(x + y)
print(x - y)
print(x * y)
print(x / y)

print(x / 2)
print(x / 2.)

A = np.array([[1, 2], [3, 4]])
print(A)

print(A.shape)

print(A.dtype)

B = np.array([[3, 0], [0, 6]])

print(A + B)
print(A * B)

print(A * 10)

A = np.array([[1, 2], [3, 4]])
B = np.array([10, 20])
print(A * B)

X = np.array([[51, 55], [14, 19], [0, 4]])
print(X)
print(X[0])
print(X[0][1])

for row in X:
    print(row)

X = X.flatten()
print(X)

print(X[np.array([0, 2, 4])])

print(X > 15)

print(X[X > 15])

import matplotlib.pyplot as plt

x = np.arange(0, 6, 0.1)
y1 = np.sin(x)
y2 = np.cos(x)

# plt.plot(x, y1, label="sin")
# plt.plot(x, y2, linestyle="--", label="cos")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.title("sin & cos")
# plt.legend()
# plt.show()

from matplotlib.image import imread

img = imread("/Users/flask/Downloads/a.png")


# plt.imshow(img)
# plt.show()


# chapter 2


def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1 * w1 + x2 * w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1


print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(1, 1))

x = np.array([0, 1])
w = np.array([0.5, 0.5])
b = -0.7
print(w * x)
print(np.sum(w * x))
print(np.sum(w * x) + b)


def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.7
    tmp = np.sum(w * x) + b
    if tmp <= 0:
        return 0
    else:
        return 1


print("=======")
print(AND(0, 0))
print(AND(1, 0))
print(AND(0, 1))
print(AND(1, 1))


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w * x + b)
    if tmp <= 0.5:
        return 0
    else:
        return 1


print("=======")
print(NAND(0, 0))
print(NAND(1, 0))
print(NAND(0, 1))
print(NAND(1, 1))


def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w * x + b)
    if tmp <= 0:
        return 0
    else:
        return 1


print("=======")
print(OR(0, 0))
print(OR(1, 0))
print(OR(0, 1))
print(OR(1, 1))


def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y


print("=======")
print(XOR(0, 0))
print(XOR(1, 0))
print(XOR(0, 1))
print(XOR(1, 1))


# chapter 3

def step_function(x):
    if x > 0:
        return 1
    else:
        return 0


def step_function(x):
    y = x > 0
    return y.astype(np.int)


def step_function(x):
    return np.array(x > 0, dtype=np.int)


x = np.arange(-5, 5, 0.1)
y = step_function(x)

plt.plot(x, y)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


y2 = sigmoid(x)

plt.plot(x, y2)
plt.ylim(-0.1, 1.1)


# plt.show()


def relu(x):
    return np.maximum(0, x)


A = np.array([1, 2, 3, 4])
print(A)
print(np.ndim(A))
print(A.shape)
print(A.shape[0])

B = np.array([[1, 2], [3, 4], [5, 6]])
print(B)
print(np.ndim(B))
print(B.shape)
print(B.shape[0])
print(B.shape[1])

A = np.array([[1, 2], [3, 4]])
print(A.shape)
B = np.array([[5, 6], [7, 8]])
print(B.shape)
print(A * B)
print(np.dot(A, B))  # 내적

A = np.array([[1, 2, 3], [4, 5, 6]])
print(A.shape)
B = np.array([[1, 2], [3, 4], [5, 6]])
print(B.shape)

print(np.dot(A, B))

C = np.array([[1, 2], [3, 4]])
print(C.shape)
print(A.shape)
# print(np.dot(A, C))  # 차원수가 맞지 않아 내적 계산 불가

print("=========3.3.3===========")

X = np.array([1., 0.5])

W1 = np.array([[.1, .3, .5], [.2, .4, .6]])
B1 = np.array([.1, .2, .3])
A1 = np.dot(X, W1) + B1
Z1 = sigmoid(A1)

print(A1)
print(Z1)

W2 = np.array([[.1, .4], [.2, .5], [.3, .6]])
B2 = np.array([.1, .2])
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)

print(A2)
print(Z2)


def identity_function(x):
    return x


W3 = np.array([[.1, .3], [.2, .4]])
B3 = np.array([.1, .2])
A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)

print(Y)

print("-----3.4.3-----")


def init_network():
    network = {}
    network['W1'] = np.array([[.1, .3, .5], [.2, .4, .6]])
    network['b1'] = np.array([.1, .2, .3])
    network['W2'] = np.array([[.1, .4], [.2, .5], [.3, .6]])
    network['b2'] = np.array([.1, .2])
    network['W3'] = np.array([[.1, .3], [.2, .4]])
    network['b3'] = np.array([.1, .2])

    return network


def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)

    return y


network = init_network()
x = np.array([1., .5])
y = forward(network, x)
print(y)

a = np.array([.3, 2.9, 4.0])

exp_a = np.exp(a)
print(exp_a)
sum_exp_a = np.sum(exp_a)
print(sum_exp_a)

y = exp_a / sum_exp_a
print(y)


def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


a = np.array([1010, 1000, 990])


# print(np.exp(a) / np.sum(np.exp(a))) # overflow


def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a - c)
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


a = np.array([.3, 2.9, 4.0])
y = softmax(a)
print(y)
print(np.sum(y))


print("====3.6====")

