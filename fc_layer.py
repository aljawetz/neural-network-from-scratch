from activations import *


class FCLayer:
    def __init__(self, input_size, output_size, activation=none):
        self.input_size = input_size
        self.output_size = output_size
        self.activation = activation

        self.weights = np.random.randn(
            input_size, output_size) / np.sqrt(input_size + output_size)
        self.bias = np.random.randn(
            1, output_size) / np.sqrt(input_size + output_size)

    def forward_propagation(self, input):
        """
        Calculate the predicted output Y = XW + B
        :param input: Given input
        :return: Predicted output
        """
        self.input = input
        self.output = np.dot(self.input, self.weights) + self.bias
        return self.activation(self.output)

    def backward_propagation(self, output_error, learning_rate):
        """
        Update weights and bias given the output_error
        :param output_error: output_error = dE/dY
        :param learning_rate: Step size at each iteration
        :return: input_error = dE/dX
        """
        output_error *= self.activation(self.output, deriv=True)
        # dE/dX = dE/dY @ Wt
        input_error = np.dot(output_error, self.weights.T)
        # dE/dW = Xt @ dE/dY
        weights_error = np.dot(self.input.T, output_error)
        # dE/dB = dE/dY
        bias_error = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * bias_error
        return input_error
