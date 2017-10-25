'''This is a reproduction of the IRNN experiment
with pixel-by-pixel sequential MNIST in
"A Simple Way to Initialize Recurrent Networks of Rectified Linear Units"
by Quoc V. Le, Navdeep Jaitly, Geoffrey E. Hinton

arxiv:1504.00941v2 [cs.NE] 7 Apr 2015
http://arxiv.org/pdf/1504.00941v2.pdf

Optimizer is replaced with RMSprop which yields more stable and steady
improvement.

Reaches 0.93 train/test accuracy after 900 epochs
(which roughly corresponds to 1687500 steps in the original paper.)
'''

from __future__ import print_function

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.layers import SimpleRNN
from keras import initializers
from keras.optimizers import RMSprop
import numpy as np

batch_size = 100
# num_classes = 10
epochs = 20
hidden_units = 100

learning_rate = 1e-4
clip_norm = 100.0

# the data, shuffled and split between train and test sets
# (x_train, y_train), (x_test, y_test) = mnist.load_data()
def gendata(num=10, T=7):
    x = np.zeros((num, T, 2))
    x[:, :, 0] = np.random.rand(num, T)
    for i in range(num):
        x[i, np.random.randint(T, size=2), 1] = 1
    return [x, np.sum(np.multiply(x[:, :, 0], x[:, :, 1]), axis=1, keepdims=True)]

x_train, y_train = gendata(1000, 10)
x_test, y_test = gendata(100, 10)

# x_train = x_train.reshape(x_train.shape[0], -1, 1)
# x_test = x_test.reshape(x_test.shape[0], -1, 1)
# x_train = x_train.astype('float32')
# x_test = x_test.astype('float32')
# x_train /= 255
# x_test /= 255
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

print('Evaluate IRNN...')
model = Sequential()
model.add(SimpleRNN(hidden_units,
                    kernel_initializer=initializers.RandomNormal(stddev=0.001),
                    # bias_initializer='zeros',
                    recurrent_initializer=initializers.Identity(gain=1.0),
                    activation='relu',
                    input_shape=x_train.shape[1:]))
model.add(Dense(1))
model.add(Activation('relu'))
rmsprop = RMSprop(lr=learning_rate)
model.compile(loss='mean_squared_error',
              optimizer=rmsprop)

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=2,
          validation_data=(x_test, y_test))

scores = model.evaluate(x_test, y_test, verbose=0)
print('IRNN test score:', scores)
# print('IRNN test accuracy:', scores[1])
