import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
from keras.callbacks import TensorBoard

from keras.models import load_model
from mnist import load_mnist

(X_train, Y_train), (X_test, Y_test) = load_mnist(normalize=True, one_hot_label=True)

nb_filter = 30
kernel_size = (5, 5)
img_rows, img_cols = 28, 28

input_shape = (img_rows, img_cols, 1)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)

model = Sequential()
model.add(Convolution2D(nb_filter, kernel_size[0], kernel_size[1], border_mode='valid', input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(100))
model.add(Activation('relu'))
model.add(Dense(10))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

batch_size = 100
nb_epoch = 5

tensorcallback = TensorBoard(log_dir='./logs', histogram_freq=0, write_graph=True, write_images=False)
callbacks = [tensorcallback]
his = model.fit(X_train, Y_train, batch_size, nb_epoch=nb_epoch, verbose=2, validation_data=(X_test, Y_test), callbacks=callbacks)

model.save('cnn.h5')
