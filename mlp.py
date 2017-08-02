'''Trains a simple deep NN on the binary dataset from Damian.
   This is to test. 
'''

from __future__ import print_function
import numpy as np
#np.random.seed(1337)  # for reproducibility

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from time import time
import matplotlib.pyplot as plt

class PlotWeights(keras.callbacks.Callback):
    def on_batch_end(self, batch, logs={}):
        w = self.model.get_weights()
        plt.plot(batch, w[0][0, 0], 'b.')

    def on_train_end(self, logs={}):
        plt.savefig("test.eps")


def MI_XT(X, T):
  # X: input matrix
  # T: layer output matrix (activations)
  # count words of length N-units, how often do they occur...
  words = set(T)
  nwords = X.shape
  return 12 - len(X)^-1 * np.log2(nwords)


batch_size = 128
nb_classes = 2
nb_epoch = 1

# the data, shuffled and split between train and test sets
data = np.loadtxt('data8.dat', dtype='int')
N = data.shape[0]
print(data.shape)
idx = np.arange(0, data.shape[0])
np.random.shuffle(idx)
data = data[idx, :]
traindata = data[0:round(N * 0.85), :]
testdata = data[round(N * 0.85):, :]

X_train = traindata[:, :-1]
y_train = traindata[:, -1]
X_test = testdata[:, :-1]
y_test = testdata[:, -1]
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

t0 = time()
# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

model = Sequential()
model.add(Dense(12, input_shape=(X_train.shape[1],)))
model.add(Activation('tanh'))
model.add(Dense(10))
model.add(Activation('tanh'))
model.add(Dense(7))
model.add(Activation('tanh'))
model.add(Dense(5))
model.add(Activation('tanh'))
model.add(Dense(4))
model.add(Activation('tanh'))
model.add(Dense(3))
model.add(Activation('tanh'))
model.add(Dense(2))
model.add(Activation('softmax'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.1, momentum=0.93))
#              metrics=['accuracy'])

plotweights = PlotWeights()

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=0, callbacks=[plotweights])


score = model.evaluate(X_test, Y_test, verbose=0)
#print('Test score:', score[0])
#print('Test accuracy:', score[1])
print('Time spent:', time() - t0)
