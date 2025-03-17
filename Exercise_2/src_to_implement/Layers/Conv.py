from .Base import BaseLayer
import numpy as np
import math
import copy


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.stride_shape = stride_shape  # single value (1d) or tuple (2d)
        self.convolution_shape = convolution_shape  # 1d: [c,m] ; 2d: [c,m,n]; c: input channels; m,n: spatial extend of filter kernel
        self.num_kernels = num_kernels  # = output channels
        self.trainable = True

        self.weights = np.random.uniform(size=(
        self.num_kernels, *self.convolution_shape))  # [kernels = output channels, input channels, m (if 2d: ,n)]
        self.bias = np.random.uniform(size=(self.num_kernels))  # bias: one for every kernel
        self.input_tensor = None

        self.dl_dw = None  # weight gradient
        self.dl_db = None  # bias gradient
        self.weights_optimizer = None
        self.bias_optimizer = None

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = np.prod(self.convolution_shape)  # [input channels × kernel height × kernel width]
        fan_out = np.prod(
            self.convolution_shape[1:]) * self.num_kernels  # [output channels × kernel height × kernel width]
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        is_1d = len(input_tensor.shape) == 3

        if is_1d:
            batch_size, input_channel, input_width = input_tensor.shape
            kernels_num, _, conv_width = self.weights.shape
            # weights.shape: (output channels = number kernels, input channels, conv width)

            pad_w_left = (conv_width - 1) // 2  # e.g. (4-1)//2 = 1
            pad_w_right = (conv_width - 1) - pad_w_left  # (4-1)-1 = 2
            padded_input = np.pad(input_tensor, ((0, 0), (0, 0), (pad_w_left, pad_w_right)), mode='constant',
                                  constant_values=0)
            # dimensions: batch -> no padding, channels -> no padding, width -> padding

            output_width = (input_width + pad_w_left + pad_w_right - conv_width) // self.stride_shape[
                0] + 1  # e.g. (10+1+2-4)//2 + 1 = 5
            convoluted_output = np.zeros((batch_size, kernels_num, output_width))

            for i in range(batch_size):
                for c in range(kernels_num):
                    for w in range(output_width):
                        w_start = w * self.stride_shape[0]
                        w_end = w_start + conv_width  # determine convolution window of this step
                        window = padded_input[i, :, w_start:w_end]  # ith batch, all channels, window
                        convoluted_output[i, c, w] = np.sum(window * self.weights[c]) + self.bias[
                            c]  # cross correlation

        else:  # 2d
            batch_size, input_channel, input_height, input_width = input_tensor.shape
            kernels_num, _, conv_height, conv_width = self.weights.shape
            # weights.shape: (output channels = number kernels, input channels, conv height, conv width)

            pad_h_top = (conv_height - 1) // 2
            pad_h_bottom = (conv_height - 1) - pad_h_top
            pad_w_left = (conv_width - 1) // 2
            pad_w_right = (conv_width - 1) - pad_w_left
            padded_input = np.pad(input_tensor, ((0, 0), (0, 0), (pad_h_top, pad_h_bottom), (pad_w_left, pad_w_right)),
                                  mode='constant', constant_values=0)
            # dimensions: batch -> no padding, channels -> no padding, height -> padding, width -> padding

            output_height = (input_height + pad_h_top + pad_h_bottom - conv_height) // self.stride_shape[0] + 1
            output_width = (input_width + pad_w_left + pad_w_right - conv_width) // self.stride_shape[1] + 1
            convoluted_output = np.zeros((batch_size, kernels_num, output_height, output_width))

            for i in range(batch_size):
                for c in range(kernels_num):
                    for h in range(output_height):
                        for w in range(output_width):
                            h_start = h * self.stride_shape[0]
                            h_end = h_start + conv_height
                            w_start = w * self.stride_shape[1]
                            w_end = w_start + conv_width
                            window = padded_input[i, :, h_start:h_end, w_start:w_end]
                            convoluted_output[i, c, h, w] = np.sum(window * self.weights[c]) + self.bias[c]
        return convoluted_output

    def backward(self, error_tensor):

        self.dl_dw = np.zeros(self.weights.shape)  # gradient weight
        self.dl_db = np.zeros(self.bias.shape)  # gradient bias

        is_1d = len(error_tensor.shape) == 3

        if is_1d:
            batch_size, kernels_num, error_width = error_tensor.shape
            _, input_channel, conv_width = self.weights.shape  # weights.shape: (output channels, input channels, conv width)
            pad_w = (conv_width - 1) // 2  # e.g. (4-1)//2 = 1
            padded_input = np.pad(self.input_tensor, ((0, 0), (0, 0), (pad_w, pad_w)), mode='constant',
                                  constant_values=0)
            output_error = np.zeros_like(padded_input)

            for i in range(batch_size):
                for c in range(kernels_num):
                    self.dl_db[c] += np.sum(error_tensor[i, c])  # gradient bias
                    for w in range(error_width):
                        w_start = w * self.stride_shape[0]
                        w_end = w_start + conv_width
                        window = padded_input[i, :, w_start:w_end]

                        self.dl_dw[c] += window * error_tensor[i, c, w]  # gradient weight
                        output_error[i, :, w_start:w_end] += self.weights[c] * error_tensor[
                            i, c, w]  # cross correlation

            if pad_w > 0:
                output_error = output_error[:, :,
                               pad_w:-pad_w]  # get rid of all padding, its only there for cross correlation

        else:  # 2d
            batch_size, kernels_num, error_height, error_width = error_tensor.shape
            _, input_channel, conv_height, conv_width = self.weights.shape

            pad_h = (conv_height - 1) // 2
            pad_w = (conv_width - 1) // 2

            padded_input = np.pad(self.input_tensor, ((0, 0), (0, 0), (pad_h, pad_h), (pad_w, pad_w)), mode='constant',
                                  constant_values=0)
            output_error = np.zeros_like(padded_input)

            for i in range(batch_size):
                for c in range(kernels_num):
                    self.dl_db[c] += np.sum(error_tensor[i, c])
                    for h in range(error_height):
                        for w in range(error_width):
                            h_start = h * self.stride_shape[0]
                            h_end = h_start + conv_height
                            w_start = w * self.stride_shape[1]
                            w_end = w_start + conv_width

                            if h_end > padded_input.shape[2] or w_end > padded_input.shape[3]:
                                continue

                            window = padded_input[i, :, h_start:h_end, w_start:w_end]

                            self.dl_dw[c] += window * error_tensor[i, c, h, w]
                            output_error[i, :, h_start:h_end, w_start:w_end] += self.weights[c] * error_tensor[
                                i, c, h, w]

            if pad_h > 0 and pad_w > 0:
                output_error = output_error[:, :, pad_h:-pad_h, pad_w:-pad_w]

        # Ensure updates with optimizers are applied correctly
        if self.weights_optimizer:
            self.weights = self.weights_optimizer.calculate_update(self.weights, self.dl_dw)
        if self.bias_optimizer:
            self.bias = self.bias_optimizer.calculate_update(self.bias, self.dl_db)

        return output_error

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
        self.weights_optimizer = copy.deepcopy(optimizer)
        self.bias_optimizer = copy.deepcopy(optimizer)

    @property
    def gradient_weights(self):
        return self.dl_dw

    @property
    def gradient_bias(self):
        return self.dl_db
