import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.callbacks import TensorBoard

from keras.models import load_model
from keras.datasets import mnist
from mnist import load_mnist

(X_train, Y_train), (X_test, Y_test) = load_mnist(normalize=True, one_hot_label=True)

batch_size = 128
nb_epoch = 12

model = Sequential()
model.add(Dense(100, input_shape=(784,)))
model.add(Activation("sigmoid"))
model.add(Dense(10))
model.add(Activation("softmax"))

sgd = SGD(lr=0.1)

model.compile(loss='categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])

model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, verbose=1, validation_data=(X_test, Y_test))
score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])

model.save('mlp.h5')
print('saved')