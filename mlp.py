'''Trains a simple deep NN on the binary dataset from Damian.
   This is to test. 
'''

from __future__ import print_function
import numpy as np
#np.random.seed(1337)  # for reproducibility

import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras import backend as K
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

    def __init__(self, layers, inputdim, niterations, act_functor, data):
        super(AnalyzeWeights, self).__init__()
        self.nlayers = len(layers)
        self.data = data
        self.act_func = act_functor
        self.w = []
        self.MI = []
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
            if self.iter % 10 == 0:
                activations = self.act_func([self.data[:, :-1]])
                activations = [np.digitize(a, np.linspace(-1, 1, 30)) for a in activations[:-1]]
                self.MI.append([self.MI_layerwise(data[:, -1], T) for T in activations])
        self.iter += 1


    def coocurrences(self, data):
        dt = np.dtype((np.void, data.dtype.itemsize * data.shape[1]))
        tmp = np.ascontiguousarray(data).view(dt)
        unq, cnt = np.unique(tmp, return_counts=True)
        unq = unq.view(data.dtype).reshape(-1, data.shape[1])
        return dict(zip(tuple(map(tuple, unq)), cnt))


    def MI_layerwise(self, X, act):
        # X: label vector
        # act: layer output matrix (activations)
        # count words of length N-units, how often do they occur...
        N = len(X)
        MI_XT = 12
        NT  = self.coocurrences(act)
        counts = np.asarray(list(NT.values()))
        idx = counts > 1  # log(1) = 0 so remove them, they are many and make for loop slow
        for nT in counts[idx]:
            MI_XT += -N**-1 * nT * np.log2(nT)

        MI_TY = 1
        tmp = act[X.astype(bool), :]
        for (T, nT) in self.coocurrences(tmp).items():
            MI_TY += N**-1 * nT * np.log2(nT/NT[T])
        for (T, nT) in self.coocurrences(act[np.invert(X.astype(bool)), :]).items():
            MI_TY += N**-1 * nT * np.log2(nT/NT[T])

        return (MI_XT, MI_TY)



batch_size = 128
nb_classes = 2
nb_epoch = 1000
saveplots = False
trainingbatch = 0.85

# the data, shuffled and split between train and test sets
data = np.loadtxt('data9.dat', dtype='int')
N = data.shape[0]
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
niterations = int(np.ceil(ntrainsamples/batch_size)*nb_epoch)

model = Sequential()
for i, n in enumerate(unitsperlayer):
    if i == 0:              # first layer
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

# to compute information plane prepare activation functor
inp = model.input                                          # input placeholder
outputs = [layer.output for layer in model.layers if layer.name[:layer.name.index('_')] == 'activation']  # all layer outputs
functor = K.function([inp], outputs)                       # evaluation function

analyzeweights = AnalyzeWeights(unitsperlayer, inputdim, niterations, data=data, act_functor=functor)

history = model.fit(X_train, Y_train,
                    batch_size=batch_size, epochs=nb_epoch,
                    verbose=0, callbacks=[analyzeweights])

t1 = time()
# # compute information plane
# inp = model.input                                           # input placeholder
# outputs = [layer.output for layer in model.layers if layer.name[:layer.name.index('_')] == 'activation']  # all layer outputs
# functor = K.function([inp], outputs )                       # evaluation function

# activations = functor([data[:, :-1]])
# activations = [np.digitize(a, np.linspace(-1, 1, 30)) for a in activations[:-1]]
# MI = [MI_layerwise(data[:, -1], T) for T in activations]

plt.figure()
for MI in analyzeweights.MI:
    for layer in range(len(MI)):
        plt.scatter(MI[layer][0], MI[layer][1], c=AnalyzeWeights.color_sequence[layer])

plt.savefig("IB plot.eps")

t2 = time()
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

if saveplots:
    nsamples = int(niterations / nb_epoch)
    mean_dw = np.empty((nb_epoch-1, nlayers))
    std_dw = np.empty((nb_epoch-1, nlayers))
    mean_w = np.empty((nb_epoch-1, nlayers))
    f, a = plt.subplots(2, 1, sharex=True)
    for layer in range(nlayers):
        tmp_w = np.reshape(analyzeweights.w[layer], (niterations, unitsperlayer[layer]*analyzeweights.nlinksperunit[layer]))
        tmp_dw = np.diff(tmp_w, axis=0)
        for t in range(nb_epoch-1):
            mean_w[t, layer] = np.linalg.norm(np.mean(tmp_w[t * nsamples:(t + 1) * nsamples, :], axis=0))
            mean_dw[t, layer] = np.linalg.norm(np.mean(tmp_dw[t * nsamples:(t + 1) * nsamples, :], axis=0))/\
                                mean_w[t, layer]
            std_dw[t, layer] = np.linalg.norm(np.std(tmp_dw[t * nsamples:(t + 1) * nsamples, :], axis=0))/\
                               mean_w[t, layer]
        a[0].loglog(mean_dw[:, layer], color=AnalyzeWeights.color_sequence[layer])
        a[1].loglog(std_dw[:, layer], '--', color=AnalyzeWeights.color_sequence[layer])
    f.savefig("mean-std_plot.jpg")


score = model.evaluate(X_test, Y_test, verbose=0)
print('Test score:', score[0])
print('Test accuracy:', score[1])
print('Total time spent:', time() - t0, ' MI computation time: ', t2 - t1, 'ANN computation time: ', t1 - t0,
      ' plotting time: ', time() - t2)
