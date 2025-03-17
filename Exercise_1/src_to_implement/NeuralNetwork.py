import copy

import numpy as np


class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = []
        self.layers = []
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None
        self.input_tensor = None
        self.prediction_data = None

    def forward(self):
        self.input_tensor, self.label_tensor = self.data_layer.next()  # Fetch new batch of data

        for layer in self.layers:
            self.input_tensor = layer.forward(self.input_tensor)

        self.prediction_data = self.input_tensor

        return self.prediction_data.tolist()

    def backward(self):
        error_tensor = self.loss_layer.backward(self.label_tensor)

        for layer in self.layers[::-1]:
            error_tensor = layer.backward(error_tensor)

    def append_layer(self, layer):
        if layer.trainable:
            layer.optimizer = copy.deepcopy(self.optimizer)

        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):

            self.forward()
            loss_error = self.loss_layer.forward(self.prediction_data, self.label_tensor)

            self.backward()
            self.loss.append(loss_error)

    def test(self, input_tensor):
        for layer in self.layers:
            input_tensor = layer.forward(input_tensor)

        return input_tensor
