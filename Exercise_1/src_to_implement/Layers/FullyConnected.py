import numpy as np

from .Base import BaseLayer

class FullyConnected(BaseLayer):  # Inherit from the base layer
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.output_size = output_size


        # Initialize weights and biases
        self._weights = np.random.uniform(0, 1, (input_size + 1, output_size)) # (9, 3)
        self._optimizer = None
        self._gradient_weights = None
        self.input_tensor_reuse = None

    def forward(self, input_tensor): # batch size = 7 (batch size, input size) = (4,8)

        ones_column = np.ones((input_tensor.shape[0], 1))

        # Append the ones column to the input tensor
        input_tensor = np.hstack((input_tensor, ones_column))

        self.input_tensor_reuse  = input_tensor

        #X transpose @ weights transpose = Y transpose
        y_prime = input_tensor @ self.weights

        return y_prime

    def backward(self, error_tensor):

        error_previous_layer = error_tensor @ self.weights.T
        error_previous_layer = error_previous_layer[:, :-1]

        self._gradient_weights = self.input_tensor_reuse.T @ error_tensor

        if self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self._gradient_weights)

        return error_previous_layer

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, value):
        self._optimizer = value  # Set the optimizer

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

