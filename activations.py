import numpy as np

# activation function and its derivative
def sigmoid(x, deriv=False):
    if deriv:
        return np.exp(-x) / (1 + np.exp(-x))**2
    return 1 / (1 + np.exp(-x))


def tanh(x, deriv=False):
    if deriv:
        return 1 - np.tanh(x)**2
    return np.tanh(x)


def relu(x, deriv=False):
    if deriv:
        return np.array(x >= 0).astype('int')
    return np.maximum(x, 0)


def none(x, deriv=False):
    if deriv:
        return 1
    return x
