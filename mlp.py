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


class AnalyzeWeights(keras.callbacks.Callback):
    color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                      '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                      '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                      '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

    def __init__(self):
        #TODO: instead of append to w do reserve needed memory space ahead!!!
        self.w = [None]*7
        self.firstbatch = True

    def on_batch_end(self, batch, logs={}):
        if self.firstbatch:
            self.firstbatch = False
            for layer in range(7):
                self.w[layer] = np.asarray([self.model.get_weights()[layer*2],])
        else:
            for layer in range(7):
                self.w[layer] = np.concatenate((self.w[layer], [self.model.get_weights()[layer*2]]), axis=0)


    def on_train_begin(self, logs={}):
        self.axes = []
        self.figures = []
        configuration = self.model.get_config()
        for layer in range(7):
            nunits = configuration["layers"][layer*2]["output_dim"]
            rows = int(np.ceil(nunits/3))
            f, a = plt.subplots(rows, 3,sharex=True)
            self.axes.append(a.flatten())
            self.figures.append(f)

    def on_train_end(self, logs={}):
        for layer in range(7):
            for unit in range(self.w[layer].shape[2]):
                for link in range(self.w[layer].shape[1]):
                    self.axes[layer][unit].plot(self.w[layer][:, link, unit], \
                                                color=AnalyzeWeights.color_sequence[link])
                    self.axes[layer][unit].xaxis.set_ticks(np.linspace(0, self.w[0].shape[0], 3))
            self.figures[layer].set_dpi = 300
            self.figures[layer].set_size_inches(12,9)
            self.figures[layer].savefig("layer" + str(layer+1) + ".jpg")


def MI_XT(X, T):
  # X: input matrix
  # T: layer output matrix (activations)
  # count words of length N-units, how often do they occur...
  words = set(T)
  nwords = X.shape
  return 12 - len(X)^-1 * np.log2(nwords)


batch_size = 256
nb_classes = 2
nb_epoch = 1000

# the data, shuffled and split between train and test sets
data = np.loadtxt('data9.dat', dtype='int')
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

plotweights = AnalyzeWeights()

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, nb_epoch=nb_epoch,
                    verbose=0, callbacks=[plotweights])


score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score)
#print('Test accuracy:', score[1])
print('Time spent:', time() - t0)
