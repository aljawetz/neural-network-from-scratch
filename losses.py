import numpy as np

# loss function and its derivative
def mse(y_true, y_pred, deriv=False):
  if deriv:
    return 2 * (y_pred - y_true) / y_pred.size
  return np.mean(np.power(y_true - y_pred, 2))

def sse(y_true, y_pred, deriv=False):
  if deriv:
    return y_pred - y_true
  return 0.5 * np.sum(np.power(y_true - y_pred, 2))