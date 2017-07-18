import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.optimizers import SGD, Adam, RMSprop, Adagrad, Adadelta

from mnist import load_mnist

(X_train, Y_train), (X_test, Y_test) = load_mnist(normalize=True, one_hot_label=True)

from keras.models import load_model

model = load_model('./mlp.h5')

pc = model.predict_classes(X_test[1:100, :], 100)
print("predict class")
print(pc)

pb = model.predict_proba(X_test[0:100, :], 100)
print("predict probability")
print(pb)
