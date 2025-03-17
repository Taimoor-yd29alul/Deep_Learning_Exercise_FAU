import numpy as np
from .Base import BaseLayer
class Pooling(BaseLayer):
    def __init__(self,stride_shape,pooling_shape):
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.input_shape = None
        self.max_indices = None
        self.trainable= False

    def forward(self,input_tensor):
        self.input_tensor = input_tensor
        m, n = self.pooling_shape  # kernel dimensions
        B, C, input_length_v, input_length_h = input_tensor.shape  # batches, channels, horizontal, vertical

        output_length_v = (input_length_v - m) // self.stride_shape[0] + 1  # vertical
        output_length_h = (input_length_h - n) // self.stride_shape[1] + 1  # horizontal

        output_tensor = np.zeros((B, C, output_length_v, output_length_h))
        self.max_indices = np.zeros((B, C, output_length_v, output_length_h, 2), dtype=int)  # 2: store indices v and h

        for b in range(B):
            for c in range(C):
                for v in range(output_length_v):
                    for h in range(output_length_h):
                        start_v = v * self.stride_shape[0]
                        start_h = h * self.stride_shape[1]
                        end_v = start_v + m
                        end_h = start_h + n
                        pool_slice = input_tensor[b, c, start_v:end_v, start_h:end_h]  # determine window

                        max_index = np.unravel_index(np.argmax(pool_slice),
                                                     pool_slice.shape)  # realtive position in slice
                        self.max_indices[b, c, v, h] = (
                        start_v + max_index[0], start_h + max_index[1])  # total position in whole tensor

                        output_tensor[b, c, v, h] = np.max(pool_slice)

        return output_tensor




    def backward(self, error_tensor):
        B, C, error_length_v, error_length_h = error_tensor.shape
        m, n = self.pooling_shape
        error_output = np.zeros(self.input_tensor.shape)

        for b in range(B):
            for c in range(C):
                for v in range(error_length_v):
                    for h in range(error_length_h):
                        pos_v, pos_h = self.max_indices[b, c, v, h]
                        error_output[b, c, pos_v, pos_h] += error_tensor[b, c, v, h]

        return error_output





