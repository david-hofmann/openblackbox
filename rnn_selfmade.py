#/usr/bin/python

import numpy as np
from time import time

np.random.seed(0)

class RNNNumpy:

    def __init__(self, timesteps, in_dim, out_dim, batch_size = 10, hidden_dim=100, learningrate=0.001, gradclipthreshold=1):
        # Assign instance variables
        """

        :rtype: object
        """
        self.T = timesteps
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.batch_size = batch_size
        self.lr = learningrate
        self.thresh = gradclipthreshold
        # Initialize the network parameters
        self.U = np.random.uniform(-np.sqrt( 1. / in_dim), np.sqrt( 1. / in_dim), (hidden_dim, in_dim))
        self.V = np.random.uniform(-np.sqrt( 1. / hidden_dim), np.sqrt( 1. / hidden_dim), (out_dim, hidden_dim))
        self.bV = np.zeros((out_dim, 1))
        self.W = np.eye(hidden_dim)
        self.bW = np.zeros((hidden_dim, 1))

    def forward_propagation(self, x):
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((self.T + 1, self.hidden_dim, x.shape[2]))
        # For each time step...
        for t in np.arange(1, self.T + 1):
            s[t] = self.U.dot(x[t-1]) + self.W.dot(s[t-1]) + self.bW
            s[t] = self.rectify(s[t])
        o = self.rectify(self.V.dot(s[-1]) + self.bV)
        return [o, s]

    def rectify(self, x):
        return np.multiply(x, (x > 0))

    def predict(self, x):
        o = self.forward_propagation(x)[0]
        return self.rectify(o)

    def bptt(self, x, y):
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        delta_o = np.multiply(o - y, np.sign(o))
        dLdV = np.dot(delta_o, s[-1].T)
        dLdbV = np.sum(delta_o, axis=1, keepdims=True)
        # Initial delta calculation: dL/dz
        # Backpropagation through time (for at most self.bptt_truncate steps)
        dLdU = np.zeros(self.U.shape)
        dLdW = np.zeros(self.W.shape)
        dLdbW = np.zeros(self.bW.shape)
        delta_t = self.V.T.dot(delta_o)
        for bptt_step in np.arange(self.T):
            # Add to gradients at each previous step
            tmp = np.multiply(delta_t, np.sign(s[-bptt_step - 1]))
            dLdW += np.dot(tmp, s[-bptt_step - 2].T)
            dLdbW += np.sum(tmp, axis=1, keepdims=True)
            dLdU += np.dot(tmp, x[-bptt_step - 1].T)
            # Update delta for next step dL/dz at t-1
            delta_t = self.W.T.dot(tmp)

        return [o, {'dU':self.grad_clip(dLdU/self.batch_size), 'dV':self.grad_clip(dLdV/self.batch_size),
                    'dbV':self.grad_clip(dLdbV/self.batch_size), 'dW':self.grad_clip(dLdW/self.batch_size),
                    'dbW':self.grad_clip(dLdbW/self.batch_size)}]

    def grad_clip(self, g):
        if np.linalg.norm(g) > self.thresh:
            g = self.thresh * g / np.linalg.norm(g)
        return g

    def calculate_mse(self, o, y):
        return 0.5 * np.sum(np.power(o - y, 2)) / o.shape[1]

    def update_weights(self, grads):
        self.U = self.U - self.lr*grads['dU']
        self.W = self.W - self.lr*grads['dW']
        self.bW = self.bW - self.lr*grads['dbW']
        self.V = self.V - self.lr*grads['dV']
        self.bV = self.bV - self.lr*grads['dbV']

    def forward_propagation_general(self, U, V, W, bV, bW, x):
        # The total number of time steps
        T = x.shape[0]
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hidden_dim, x.shape[2]))
        # For each time step...
        for t in np.arange(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = np.dot(U,x[t]) + np.dot(W,s[t-1]) + bW
            s[t] = self.rectify(s[t])

        o = self.rectify(np.dot(V,s[-1]) + bV)
        return [o, s]

    # def gradient_check(self, x, y, grads, epsilon):
    #     # first look at V and dV
    #     V_vector = np.reshape(self.V, (self.V.shape[0] * self.V.shape[1], 1))
    #     dV_vector = np.reshape(grads['dV'], (grads['dV'].shape[0] * grads['dV'].shape[1], 1))
    #     dV_approx = np.zeros(dV_vector.shape)
    #     for i in range(V_vector.shape[0]):
    #         V_minus = np.copy(V_vector)
    #         V_minus[i, 0] = V_minus[i, 0] - epsilon
    #         V_plus = np.copy(V_vector)
    #         V_plus[i, 0] = V_plus[i, 0] + epsilon
    #         ominus, _ = self.forward_propagation_general(self.U, np.reshape(V_minus, self.V.shape), self.W, self.bV,
    #                                                      self.bW, x)
    #         oplus, _ = self.forward_propagation_general(self.U, np.reshape(V_plus, self.V.shape), self.W, self.bV,
    #                                                     self.bW, x)
    #         lossminus = self.calculate_mse(ominus, y)
    #         lossplus = self.calculate_mse(oplus, y)
    #         dV_approx[i, 0] = (lossplus - lossminus) / (2 * epsilon)
    #
    #     errorV = np.linalg.norm(dV_vector - dV_approx) / (np.linalg.norm(dV_vector) + np.linalg.norm(dV_approx))
    #
    #     dbV_approx = np.zeros(self.bV.shape)
    #     for i in range(self.bV.shape[0]):
    #         bV_minus = np.copy(self.bV)
    #         bV_minus[i, 0] = bV_minus[i, 0] - epsilon
    #         bV_plus = np.copy(self.bV)
    #         bV_plus[i, 0] = bV_plus[i, 0] + epsilon
    #         ominus, _ = self.forward_propagation_general(self.U, self.V, self.W, bV_minus, self.bW, x)
    #         oplus, _ = self.forward_propagation_general(self.U, self.V, self.W, bV_plus, self.bW, x)
    #         lossminus = self.calculate_mse(ominus, y)
    #         lossplus = self.calculate_mse(oplus, y)
    #         dbV_approx[i, 0] = (lossplus - lossminus) / (2 * epsilon)
    #
    #     errorbV = np.linalg.norm(grads['dbV'] - dbV_approx) / (
    #     np.linalg.norm(grads['dbV']) + np.linalg.norm(dbV_approx))
    #
    #     # second, look at W and dW
    #     W_vector = np.reshape(self.W, (self.W.shape[0] * self.W.shape[1], 1))
    #     dW_vector = np.reshape(grads['dW'], (grads['dW'].shape[0] * grads['dW'].shape[1], 1))
    #     dW_approx = np.zeros(dW_vector.shape)
    #     for i in range(W_vector.shape[0]):
    #         W_minus = np.copy(W_vector)
    #         W_minus[i, 0] = W_minus[i, 0] - epsilon
    #         W_plus = np.copy(W_vector)
    #         W_plus[i, 0] = W_plus[i, 0] + epsilon
    #         ominus, _ = self.forward_propagation_general(self.U, self.V, np.reshape(W_minus, self.W.shape), self.bV,
    #                                                      self.bW, x)
    #         oplus, _ = self.forward_propagation_general(self.U, self.V, np.reshape(W_plus, self.W.shape), self.bV,
    #                                                     self.bW, x)
    #         lossminus = self.calculate_mse(ominus, y)
    #         lossplus = self.calculate_mse(oplus, y)
    #         dW_approx[i, 0] = (lossplus - lossminus) / (2 * epsilon)
    #
    #     errorW = np.linalg.norm(dW_vector - dW_approx) / (np.linalg.norm(dW_vector) + np.linalg.norm(dW_approx))
    #
    #     dbW_approx = np.zeros(self.bW.shape)
    #     for i in range(self.bW.shape[0]):
    #         bW_minus = np.copy(self.bW)
    #         bW_minus[i, 0] = bW_minus[i, 0] - epsilon
    #         bW_plus = np.copy(self.bW)
    #         bW_plus[i, 0] = bW_plus[i, 0] + epsilon
    #         ominus, _ = self.forward_propagation_general(self.U, self.V, self.W, self.bV, bW_minus, x)
    #         oplus, _ = self.forward_propagation_general(self.U, self.V, self.W, self.bV, bW_plus, x)
    #         lossminus = self.calculate_mse(ominus, y)
    #         lossplus = self.calculate_mse(oplus, y)
    #         dbW_approx[i, 0] = (lossplus - lossminus) / (2 * epsilon)
    #
    #     errorbW = np.linalg.norm(grads['dbW'] - dbW_approx) / (
    #     np.linalg.norm(grads['dbW']) + np.linalg.norm(dbW_approx))
    #
    #     # third, look at W and dW
    #     U_vector = np.reshape(self.U, (self.U.shape[0] * self.U.shape[1], 1))
    #     dU_vector = np.reshape(grads['dU'], (grads['dU'].shape[0] * grads['dU'].shape[1], 1))
    #     dU_approx = np.zeros(dU_vector.shape)
    #     for i in range(U_vector.shape[0]):
    #         U_minus = np.copy(U_vector)
    #         U_minus[i, 0] = U_minus[i, 0] - epsilon
    #         U_plus = np.copy(U_vector)
    #         U_plus[i, 0] = U_plus[i, 0] + epsilon
    #         ominus, _ = self.forward_propagation_general(np.reshape(U_minus, self.U.shape), self.V, self.W, self.bV,
    #                                                      self.bW, x)
    #         oplus, _ = self.forward_propagation_general(np.reshape(U_plus, self.U.shape), self.V, self.W, self.bV,
    #                                                     self.bW, x)
    #         lossminus = self.calculate_mse(ominus, y)
    #         lossplus = self.calculate_mse(oplus, y)
    #         dU_approx[i, 0] = (lossplus - lossminus) / (2 * epsilon)
    #
    #     errorU = np.linalg.norm(dU_vector - dU_approx) / (np.linalg.norm(dU_vector) + np.linalg.norm(dU_approx))
    #
    #     dtheta_vector = np.concatenate((dV_vector, grads['dbV'], dW_vector, dU_vector))
    #     dtheta_approx = np.concatenate((dV_approx, dbV_approx, dW_approx, dU_approx))
    #     error = np.linalg.norm(dtheta_vector - dtheta_approx) / (
    #     np.linalg.norm(dtheta_vector) + np.linalg.norm(dtheta_approx))
    #     return [error, errorU]


def gendata(num=10, T=7):
    x = np.zeros((T, 2, num))
    x[:, 0, :] = np.random.rand(T, num)
    for i in range(num):
        x[np.random.randint(T, size=2), 1, i] = 1
    return [x, np.sum(np.multiply(x[:, 0, :], x[:, 1, :]), axis=0, keepdims=True)]

t_start = time()
inp, outp = gendata(100000, 10)
batch_size = 100
epochs = 20

rnn = RNNNumpy(inp.shape[0], inp.shape[1], outp.shape[0], batch_size=batch_size, gradclipthreshold=100)

for n in range(epochs):
    print("epoch: %i" %n)
    for i in range(int(inp.shape[2] / batch_size)):
        tmp_x = inp[:, :, i*batch_size:(i+1)*batch_size]
        tmp_y = outp[:, i*batch_size:(i+1)*batch_size]
        o, grad = rnn.bptt(tmp_x, tmp_y)
        rnn.update_weights(grad)
    loss = rnn.calculate_mse(o, tmp_y)
    print("Loss is: %f" % loss)

print("elapsed time: %f" % (time() - t_start))