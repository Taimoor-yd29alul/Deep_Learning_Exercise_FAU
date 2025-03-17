import numpy as np

from .Base import BaseLayer
class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_tensor = None


    def forward(self, input_tensor):

        self.input_tensor = input_tensor
        output = np.maximum(0,input_tensor) # f(x) = max(0,x)

        return output

    def backward(self, error_tensor):
        relu_derivative = self.input_tensor > 0  #return an array of 0,1
        output = error_tensor * relu_derivative #E(n-1) = En if x> 0 other E(n-1) = 0

        return output
