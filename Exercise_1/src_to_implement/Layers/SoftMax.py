from .Base import BaseLayer
import numpy as np

class SoftMax(BaseLayer):
    def __init__(self):
        BaseLayer.__init__(self)
        self.y_hat = None


    def forward(self, input_tensor):
        max_element_alog_each_row = np.max(input_tensor, axis=1, keepdims=True)

        numerator = np.exp(input_tensor - max_element_alog_each_row)
        denominator = np.sum(numerator, axis=1, keepdims=True)

        # Compute softmax
        softmax = numerator / denominator

        self.y_hat = softmax
        return softmax

    def backward(self, error_tensor):

        softmax_error_tensor_sum = np.sum(self.y_hat * error_tensor, axis=1, keepdims=True)
        output = self.y_hat * (error_tensor - softmax_error_tensor_sum)

        return output
