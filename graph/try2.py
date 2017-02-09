import numpy as np
import random
from common.functions import *
from common.gradient import numerical_gradient

learning_rate = 0.025


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
        # a = random.random() * 10 - 5
        # b = random.random() * 10 - 5
        # c = random.random() * 10 - 5
        a = random.randrange(-9, 9)
        b = random.randrange(-9, 9)
        c = random.randrange(-9, 9)
        if a == 0:
            a = 1
        batch_data.append(f1inter(a, b))
        batch_label.append([0, a, b])

        batch_data.append(f2inter(a, b, c))
        batch_label.append([a, b, c])

    return np.array(batch_data), np.array(batch_label)


network = Net(input_size=9, hidden_size=729, hidden_size_2=125, output_size=3)
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
              ' \npredict: ' + str(predict[0]) +
              ' \ntest f(x): ' + str(test_label[1][0]) + ' x^2 + ' + str(test_label[1][1]) + ' x + ' + str(test_label[1][2]) +
              ' \npredict: ' + str(predict[1]) +
              '\n--------------------------------\n')


# /usr/local/Cellar/python3/3.6.0/Frameworks/Python.framework/Versions/3.6/bin/python3.6 /Users/flask/Documents/workspace_tf/ml-study/graph/try2.py
# (9, 729)
# (729,)
# (729, 125)
# (125,)
# (125, 3)
# (3,)
# 0 / train loss: 131.14928675 / test loss: 238.514028072
# test f(x): 6 x + -9
# predict: [-0.49178206 -0.92618    -0.56418982]
# test f(x): 6 x^2 + -9 x + -4
# predict: [-0.49190299 -0.92613863 -0.56411964]
# --------------------------------
#
# 1000 / train loss: 0.447101102164 / test loss: 0.139339249541
# test f(x): -4 x + -6
# predict: [  2.60385651e-03  -4.17742019e+00  -6.26805856e+00]
# test f(x): -4 x^2 + -6 x + 7
# predict: [-4.16836873 -6.08693595  7.00965759]
# --------------------------------
#
# 2000 / train loss: 0.092998402749 / test loss: 0.0306104921179
# test f(x): 2 x + 5
# predict: [-0.03260128  1.90464692  5.06357071]
# test f(x): 2 x^2 + 5 x + -8
# predict: [ 2.08577301  5.00382422 -8.09509243]
# --------------------------------
#
# 3000 / train loss: 0.0527982098234 / test loss: 0.0580436070521
# test f(x): 2 x + -3
# predict: [-0.02578883  1.86546509 -2.91355352]
# test f(x): 2 x^2 + -3 x + -1
# predict: [ 1.85419496 -2.90245221 -0.96788728]
# --------------------------------
#
# 4000 / train loss: 0.0576025623232 / test loss: 0.0638844491596
# test f(x): 3 x + -1
# predict: [-0.01076006  2.87032675 -0.95842823]
# test f(x): 3 x^2 + -1 x + 6
# predict: [ 2.97675174 -0.94622032  6.20443231]
# --------------------------------
#
# 5000 / train loss: 0.0943566775727 / test loss: 0.0275819175041
# test f(x): -7 x + 0
# predict: [ 0.0074584  -7.02402039  0.08149828]
# test f(x): -7 x^2 + 0 x + 4
# predict: [-7.13397711  0.0418049   4.02469466]
# --------------------------------
#
# 6000 / train loss: 0.0400816166697 / test loss: 0.00522195768831
# test f(x): -6 x + -1
# predict: [-0.01354474 -6.01333753 -0.96762965]
# test f(x): -6 x^2 + -1 x + 7
# predict: [-6.05423733 -0.97367634  6.9866529 ]
# --------------------------------
#
# 7000 / train loss: 0.032843663997 / test loss: 0.0185040028622
# test f(x): 4 x + -6
# predict: [-0.01643974  4.01070165 -6.08038037]
# test f(x): 4 x^2 + -6 x + 0
# predict: [ 4.00933235 -6.1054973  -0.02101036]
# --------------------------------
#
# 8000 / train loss: 0.0229026075541 / test loss: 0.00676460523564
# test f(x): -3 x + -9
# predict: [ 0.05449699 -3.01092047 -8.99742358]
# test f(x): -3 x^2 + -9 x + 0
# predict: [-2.95862612 -8.96088912  0.02067195]
# --------------------------------
#
# 9000 / train loss: 0.0489384128838 / test loss: 0.00805268898422
# test f(x): -8 x + -2
# predict: [-0.04356078 -8.00348762 -2.01153889]
# test f(x): -8 x^2 + -2 x + 1
# predict: [-8.07197689 -1.9712977   0.99768825]
# --------------------------------
#
# 10000 / train loss: 0.033621889347 / test loss: 0.0140951979362
# test f(x): -8 x + -4
# predict: [ 0.02895387 -8.04028129 -3.92776861]
# test f(x): -8 x^2 + -4 x + 4
# predict: [-8.06946464 -4.00751896  3.96082036]
# --------------------------------
#
# 11000 / train loss: 0.0402005002517 / test loss: 0.0106787265378
# test f(x): -6 x + 1
# predict: [-0.01859966 -5.99806183  0.97259061]
# test f(x): -6 x^2 + 1 x + 5
# predict: [-6.07738161  1.05245319  4.97104319]
# --------------------------------
#
# 12000 / train loss: 0.025366519584 / test loss: 0.0210098799047
# test f(x): 8 x + -4
# predict: [ 0.03256085  7.94135183 -4.01303293]
# test f(x): 8 x^2 + -4 x + 6
# predict: [ 7.88738219 -3.98277017  6.0579704 ]
# --------------------------------
#
# 13000 / train loss: 0.0292375450574 / test loss: 0.00406681088318
# test f(x): -8 x + -2
# predict: [-0.0127869  -8.03549381 -1.98023113]
# test f(x): -8 x^2 + -2 x + -1
# predict: [-8.04623942 -1.99810909 -0.98946303]
# --------------------------------
#
# 14000 / train loss: 0.0108980160308 / test loss: 0.00673006955484
# test f(x): 3 x + -4
# predict: [ -3.89470795e-03   2.97385142e+00  -3.93630072e+00]
# test f(x): 3 x^2 + -4 x + -7
# predict: [ 2.9895801  -3.97058613 -7.03161968]
# --------------------------------
#
# 15000 / train loss: 0.0097993572993 / test loss: 0.170561289968
# test f(x): 8 x + -9
# predict: [ 0.03276098  7.87182144 -8.83721449]
# test f(x): 8 x^2 + -9 x + -6
# predict: [ 7.80129026 -8.71331511 -5.93010466]
# --------------------------------
#
# 16000 / train loss: 0.0179352452286 / test loss: 0.00733113610506
# test f(x): -6 x + -3
# predict: [-0.02321783 -5.95555087 -2.96798901]
# test f(x): -6 x^2 + -3 x + -2
# predict: [-6.03483028 -2.9503892  -1.98917134]
# --------------------------------
#
# 17000 / train loss: 0.0127527269791 / test loss: 0.000915896591815
# test f(x): -6 x + -3
# predict: [ -3.53577832e-03  -6.01061319e+00  -3.00868502e+00]
# test f(x): -6 x^2 + -3 x + 1
# predict: [-6.01825696 -2.98045785  0.99966396]
# --------------------------------
#
# 18000 / train loss: 0.0221853487341 / test loss: 0.0175685293533
# test f(x): -3 x + -2
# predict: [ 0.01789568 -2.97109898 -2.0186065 ]
# test f(x): -3 x^2 + -2 x + 6
# predict: [-3.02043006 -1.96622341  6.12045148]
# --------------------------------
#
# 19000 / train loss: 0.0619233088979 / test loss: 0.00980989915266
# test f(x): -9 x + -4
# predict: [-0.04588418 -9.00506493 -4.01056974]
# test f(x): -9 x^2 + -4 x + -5
# predict: [-9.01495634 -4.08017656 -5.0302522 ]
# --------------------------------
#
# 20000 / train loss: 0.011885550318 / test loss: 0.00640352038783
# test f(x): 1 x + 5
# predict: [ -5.67415809e-04   1.03473704e+00   4.97676237e+00]
# test f(x): 1 x^2 + 5 x + 0
# predict: [  9.94368326e-01   4.93200741e+00   1.35706920e-03]
# --------------------------------
#
# 21000 / train loss: 0.0126130675077 / test loss: 0.00703312849919
# test f(x): 2 x + -5
# predict: [ 0.01140629  2.02262086 -4.9565638 ]
# test f(x): 2 x^2 + -5 x + -3
# predict: [ 1.98518394 -4.93808063 -2.97876101]
# --------------------------------
#
# 22000 / train loss: 0.0118717331063 / test loss: 0.00495553837084
# test f(x): 2 x + -2
# predict: [-0.01611354  2.01378333 -2.01917948]
# test f(x): 2 x^2 + -2 x + -3
# predict: [ 1.93569028 -2.00150177 -3.00025343]
# --------------------------------
#
# 23000 / train loss: 0.0164956363429 / test loss: 0.0147992955212
# test f(x): -1 x + 4
# predict: [ -3.24257601e-03  -1.01806849e+00   4.00004274e+00]
# test f(x): -1 x^2 + 4 x + 6
# predict: [-1.0366037   4.01602856  6.11342647]
# --------------------------------
#
# 24000 / train loss: 0.0196384306215 / test loss: 0.00727209819671
# test f(x): -9 x + 4
# predict: [  8.29972993e-03  -8.95882849e+00   3.99140230e+00]
# test f(x): -9 x^2 + 4 x + -2
# predict: [-8.97545354  3.93715666 -1.97029504]
# --------------------------------
#
# 25000 / train loss: 0.0252454888131 / test loss: 0.0391377330227
# test f(x): 6 x + 0
# predict: [ 0.01541351  6.0085067  -0.01841945]
# test f(x): 6 x^2 + 0 x + -9
# predict: [ 6.07924338 -0.16347923 -9.07405097]
# --------------------------------
#
# 26000 / train loss: 0.00846554759698 / test loss: 0.00434959306818
# test f(x): -2 x + -5
# predict: [ 0.01066225 -2.00857919 -4.93900964]
# test f(x): -2 x^2 + -5 x + 7
# predict: [-2.01030409 -5.01219903  7.0136928 ]
# --------------------------------
#
# 27000 / train loss: 0.0130248289699 / test loss: 0.00600418609481
# test f(x): 3 x + 8
# predict: [ 0.00883483  2.96543708  7.93808754]
# test f(x): 3 x^2 + 8 x + -1
# predict: [ 2.99153249  8.02529411 -0.98632911]
# --------------------------------
#
# 28000 / train loss: 0.0196998248467 / test loss: 0.0294801418766
# test f(x): -9 x + -7
# predict: [-0.05599062 -8.91573136 -7.09956099]
# test f(x): -9 x^2 + -7 x + 4
# predict: [-8.92954781 -7.01312876  4.0647744 ]
# --------------------------------
#
# 29000 / train loss: 0.00922805078495 / test loss: 0.00619081816093
# test f(x): -3 x + 4
# predict: [ -3.35792009e-03  -2.99557961e+00   3.97704928e+00]
# test f(x): -3 x^2 + 4 x + 7
# predict: [-2.996819    3.99662951  7.07491187]
# --------------------------------
#
#
# Process finished with exit code 0
