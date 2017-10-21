import numpy as np



class RNNNumpy:

    def __init__(self, word_dim, out_dim, batch_size = 100, hidden_dim=100, bptt_truncate=4):
        # Assign instance variables
        self.word_dim = word_dim
        self.out_dim = out_dim
        self.hidden_dim = hidden_dim
        self.bptt_truncate = bptt_truncate
        self.batch_size = batch_size
        # Randomly initialize the network parameters
        self.U = np.random.uniform(-np.sqrt( 1. / word_dim), np.sqrt( 1. / word_dim), (hidden_dim, word_dim))
        self.V = np.random.uniform(-np.sqrt( 1. / hidden_dim), np.sqrt( 1. / hidden_dim), (out_dim, hidden_dim))
        self.bV = np.zeros((out_dim, 1))
        self.W = np.eye((hidden_dim, hidden_dim))
        self.bW = np.zeros((hidden_dim, 1))

    def forward_propagation(self, x):
        # The total number of time steps
        T = x.shape[0]
        # During forward propagation we save all hidden states in s because need them later.
        # We add one additional element for the initial hidden, which we set to 0
        s = np.zeros((T + 1, self.hidden_dim, x.shape[2]))
        # For each time step...
        for t in np.arange(T):
            # Note that we are indxing U by x[t]. This is the same as multiplying U with a one-hot vector.
            s[t] = self.U.dot(x[t]) + self.W.dot(s[t-1]) + self.bW
            s[t] = self.rectify(s[t])

        o = self.V.dot(s[-2]) + self.bV
        return [o, s]

    def rectify(self, x):
        return np.multiply(x, (x > 0))

    def predict(self, x):
        # Perform forward propagation and return index of the highest score
        o = self.forward_propagation(x)[0]
        return self.rectify(o)

    def bptt(self, x, y):
        T = x.shape[0]
        p = x.shape[2]
        # Perform forward propagation
        o, s = self.forward_propagation(x)
        # We accumulate the gradients in these variables
        delta_o = np.multiply(y - o, np.sign(o, 0))
        dLdV = np.dot(delta_o, s[-2].T)
        dLdbV = np.sum(delta_o, axis=1, keepdims=True)
        # Initial delta calculation: dL/dz
        # Backpropagation through time (for at most self.bptt_truncate steps)
        dLdU = np.zeros(self.U.shape)
        dLdW = np.zeros(self.W.shape)
        dLdbW = np.zeros(self.bW.shape)
        delta_t = self.V.T.dot(delta_o)
        for bptt_step in np.arange(T):
            # print "Backpropagation step t=%d bptt step=%d " % (t, bptt_step)
            # Add to gradients at each previous step
            tmp = np.multiply(delta_t, np.sign(s[-bptt_step - 2]))
            dLdW += np.dot(tmp, s[-bptt_step - 3].T)
            dLdbW += np.sum(tmp, axis=1, keepdims=True)
            dLdU += np.dot(tmp, x[-bptt_step - 1].T)
            # Update delta for next step dL/dz at t-1
            delta_t = self.W.T.dot(tmp)

        return {'dLdU':dLdU/p, 'dLdV':dLdV/p, 'dLdbV':dLdbV/p, 'dLdW':dLdW/p, 'dLdbW':dLdbW/p}


    def calculate_loss(self, o, y):
        return 0.5 * np.sum((o - y)**2) / o.shape[1]

    # def gradient_check(self, x, y, h=0.001, error_threshold=0.01):
    #     # Calculate the gradients using backpropagation. We want to checker if these are correct.
    #     bptt_gradients = self.bptt(x, y)
    #     # List of all parameters we want to check.
    #     model_parameters = ['U', 'V', 'W']
    #     # Gradient check for each parameter
    #     for pidx, pname in enumerate(model_parameters):
    #         # Get the actual parameter value from the mode, e.g. model.W
    #         parameter = operator.attrgetter(pname)(self)
    #         print "Performing gradient check for parameter %s with size %d." % (pname, np.prod(parameter.shape))
    #         # Iterate over each element of the parameter matrix, e.g. (0,0), (0,1), ...
    #         it = np.nditer(parameter, flags=['multi_index'], op_flags=['readwrite'])
    #         while not it.finished:
    #             ix = it.multi_index
    #             # Save the original value so we can reset it later
    #             original_value = parameter[ix]
    #             # Estimate the gradient using (f(x+h) - f(x-h))/(2*h)
    #             parameter[ix] = original_value + h
    #             gradplus = self.calculate_total_loss([x],[y])
    #             parameter[ix] = original_value - h
    #             gradminus = self.calculate_total_loss([x],[y])
    #             estimated_gradient = (gradplus - gradminus)/(2*h)
    #             # Reset parameter to original value
    #             parameter[ix] = original_value
    #             # The gradient for this parameter calculated using backpropagation
    #             backprop_gradient = bptt_gradients[pidx][ix]
    #             # calculate The relative error: (|x - y|/(|x| + |y|))
    #             relative_error = np.abs(backprop_gradient - estimated_gradient)/(np.abs(backprop_gradient) + np.abs(estimated_gradient))
    #             # If the error is to large fail the gradient check
    #             if relative_error > error_threshold:
    #                 print("Gradient Check ERROR: parameter=%s ix=%s" % (pname, ix))
    #                 print("+h Loss: %f" % gradplus)
    #                 print("-h Loss: %f" % gradminus)
    #                 print("Estimated_gradient: %f" % estimated_gradient)
    #                 print("Backpropagation gradient: %f" % backprop_gradient)
    #                 print("Relative Error: %f" % relative_error)
    #                 return
    #             it.iternext()
    #         print("Gradient check for parameter %s passed." % (pname))


def gendata(T=7, num=10):
    x = np.random.rand(num, T)
    y = np.zeros((num, T))
    for i in range(num):
        y[i, np.random.randint(T, size=2)] = 1

    return [x, np.sum(np.multiply(x, y), axis=1)]
