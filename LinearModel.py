import numpy as np


class LinearModel:
    """ A linear regression model with stochastic gradient descent """

    def __init__(self, input_dim, n_action):
        # input_dim is state dimensionality. n_action is output size, or size of action space.
        # Initialize random weight matrix and bias vector of zeros
        self.W = np.random.randn(input_dim, n_action) / np.sqrt(input_dim)
        self.b = np.zeros(n_action)

        # momentum terms (for W and b respectively)
        self.vW = 0
        self.vb = 0

        # Placeholder for losses on each step of gradient descent
        self.losses = []

    def predict(self, X):
        # Takes in 2D array, X of size N x D
        assert (len(X.shape) == 2)
        return X.dot(self.W) + self.b

    def sgd(self, X, Y, learning_rate=0.01, momentum=0.9):
        """Does one Step of Gradient Descent: calculate momentum term, v(t). Then update parameters, W, b.
        :param X: training data
        :param Y: target data
        :param learning_rate / momentum: hyper-parameters

        """
        # make sure X is N x D
        assert (len(X.shape) == 2)

        # Our model is linear regression with multiple outputs. Number of samples = N, number of outputs - K.
        # Then y is of size NxK (i.e. loss is 2D). MSE will be divided by num_values.
        num_values = np.prod(Y.shape)

        # do one step of gradient descent
        Yhat = self.predict(X)

        # Calculate gradients.
        # Note d/dx is the gradient of the loss function. i.e. d/dx (x^2) --> 2x. Could incorporate into
        # learning rate, but this is technically correct.
        gW = 2 * X.T.dot(Yhat - Y) / num_values
        gb = 2 * (Yhat - Y).sum(axis=0) / num_values

        # update momentum terms
        self.vW = momentum * self.vW - learning_rate * gW
        self.vb = momentum * self.vb - learning_rate * gb

        # update params
        self.W += self.vW
        self.b += self.vb

        # Calculate loss for step and append to losses list
        mse = np.mean((Yhat - Y) ** 2)
        self.losses.append(mse)

    def load_weights(self, filepath):
        npz = np.load(filepath)
        self.W = npz['W']
        self.b = npz['b']

    def save_weights(self, filepath):
        np.savez(filepath, W=self.W, b=self.b)