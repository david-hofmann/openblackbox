'''Trains a simple deep NN on the binary dataset from Damian.
   This is to test. 
'''

from __future__ import print_function
import numpy as np
#np.random.seed(1337)  # for reproducibility

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from time import time
import matplotlib.pyplot as plt

# how to reinstantiate a model:
#config = model.get_config()
#model = Model.from_config(config)
#  or, for Sequential:
#model = Sequential.from_config(config)


class AnalyzeWeights(keras.callbacks.Callback):
    color_sequence = ['#1f77b4', '#aec7e8', '#ff7f0e', '#ffbb78', '#2ca02c',
                      '#98df8a', '#d62728', '#ff9896', '#9467bd', '#c5b0d5',
                      '#8c564b', '#c49c94', '#e377c2', '#f7b6d2', '#7f7f7f',
                      '#c7c7c7', '#bcbd22', '#dbdb8d', '#17becf', '#9edae5']

    def __init__(self, layers, inputdim, niterations):
        self.nlayers = len(layers)
        self.w = []
        self.iter = 0
        self.nlinksperunit = [None]*7
        for i in range(self.nlayers):
            if i == 0:
                self.w.append(np.empty((niterations, inputdim, layers[i])))
                self.nlinksperunit[i] = inputdim
            else:
                self.w.append(np.empty((niterations, layers[i-1], layers[i])))
                self.nlinksperunit[i] = layers[i-1]
        self.nunits = layers[:]

    def on_batch_end(self, batch, logs={}):
        for layer in range(self.nlayers):
            self.w[layer][self.iter, :, :] = np.asarray([self.model.get_weights()[layer * 2], ])
        self.iter += 1


# def MI_XT(X, T):
#   # X: input matrix
#   # T: layer output matrix (activations)
#   # count words of length N-units, how often do they occur...
#   words = set(T)
#   nwords = X.shape
#   return 12 - len(X)^-1 * np.log2(nwords)


batch_size = 256
nb_classes = 2
nb_epoch = 10000
saveplots = True
trainingbatch = 0.85

# the data, shuffled and split between train and test sets
data = np.loadtxt('data9.dat', dtype='int')
N = data.shape[0]
print(data.shape)
idx = np.arange(0, data.shape[0])
np.random.shuffle(idx)
data = data[idx, :]
traindata = data[0:round(N * trainingbatch), :]
testdata = data[round(N * trainingbatch):, :]

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

unitsperlayer = (12, 10, 7, 5, 4, 3, 2)
nlayers = len(unitsperlayer)
(ntrainsamples, inputdim) = X_train.shape
niterations = round(ntrainsamples/batch_size)*nb_epoch

model = Sequential()
for i, n in enumerate(unitsperlayer):
    if i == 0:    # first layer
        model.add(Dense(n, input_shape=(inputdim,)))
        model.add(Activation('tanh'))
    elif i == nlayers-1:    # last layer
        model.add(Dense(n))
        model.add(Activation('softmax'))
    else:
        model.add(Dense(n))
        model.add(Activation('tanh'))

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.1, momentum=0.93), metrics=['accuracy'])

analyzeweights = AnalyzeWeights(unitsperlayer, inputdim, niterations)

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, epochs=nb_epoch,
                    verbose=1, callbacks=[analyzeweights])

print(analyzeweights.iter)
axes = []
autocorr = [None]*nlayers
frac_corr = 10
# compute autocorrelation of weights and plot weight trajectories.
for layer in range(nlayers):
    f, a = plt.subplots(int(np.ceil(unitsperlayer[layer]/3)), 3, sharex=True)
    axes.append(a.flatten())
    autocorr[layer] = np.empty((int(niterations/frac_corr), analyzeweights.nlinksperunit[layer], unitsperlayer[layer]))
    for unit in range(unitsperlayer[layer]):
        for link in range(analyzeweights.nlinksperunit[layer]):
            tmp_w = analyzeweights.w[layer][:, link, unit]
            axes[layer][unit].plot(tmp_w, color=AnalyzeWeights.color_sequence[link])
            axes[layer][unit].xaxis.set_ticks(np.linspace(0, niterations, 3))
            tmp_w = tmp_w[-int(niterations / frac_corr):]
            tmp_w = tmp_w - np.mean(tmp_w)
            autocorr[layer][:, link, unit] = np.correlate(tmp_w, tmp_w, mode='full')[-int(niterations / frac_corr):]
    if saveplots:
        f.set_dpi = 300
        f.set_size_inches(12, 9)
        f.savefig("weights_layer" + str(layer+1) + ".jpg")
    for unit in range(unitsperlayer[layer]):
        axes[layer][unit].cla()
        for link in range(analyzeweights.nlinksperunit[layer]):
            axes[layer][unit].plot(autocorr[layer][:, link, unit], color=AnalyzeWeights.color_sequence[link])
            axes[layer][unit].xaxis.set_ticks(np.linspace(0, int(niterations/frac_corr), 3))
    if saveplots:
        f.savefig("autocorr_layer" + str(layer + 1) + ".jpg")

#TODO: next step is to implement the mean-std graph
# if saveplots:
#     nsamples = int(niterations / nb_epoch)
#     for layer in range(nlayers):
#         for t in range(nb_epoch):
#             meandw[self.iter] = tmp_w[t * nsamples:(t + 1) * nsamples]
#             stddw[self.iter] =

score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print('Time spent:', time() - t0)
