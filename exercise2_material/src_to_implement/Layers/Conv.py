import numpy as np
from .Base import BaseLayer
import copy
from .Initializers import He, UniformRandom, Constant 

class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        self.trainable = True
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.weights = None
        self.bias = None
        self.gradient_weights = None
        self.gradient_bias = None
        self.padding = None
        self.x_padding = None
        self.optimizer_w = None
        self.optimizer_b = None
        self.mark_update = False

        w_init = He() if len(convolution_shape) == 3 else UniformRandom()
        b_init = Constant(0.01)
        self.initialize(w_init, b_init)

    def initialize(self, weights_initializer, bias_initializer):
        if len(self.convolution_shape) == 2:
            c, k_l = self.convolution_shape
            fan_in = c * k_l
            fan_out = self.num_kernels * k_l
            weights_shape = (self.num_kernels,c,k_l)
            self.weights = weights_initializer.initialize(weights_shape, fan_in, fan_out)
            self.bias = bias_initializer.initialize((self.num_kernels,), fan_in, fan_out)

            self.gradient_weights = np.zeros_like(self.weights)
            self.gradient_bias = np.zeros_like(self.bias)


        elif len(self.convolution_shape) == 3:
            c, k_h, k_w = self.convolution_shape
            fan_in = c * k_h * k_w
            fan_out = self.num_kernels * k_h * k_w
            weights_shape = (self.num_kernels, c, k_h, k_w)
            self.weights = weights_initializer.initialize(weights_shape, fan_in, fan_out)
            self.bias = bias_initializer.initialize((self.num_kernels,), fan_in, fan_out)

            self.gradient_weights = np.zeros_like(self.weights)
            self.gradient_bias = np.zeros_like(self.bias)

    def forward(self, input_tensor):
        if len(self.convolution_shape) == 2: #1-D
            # padding
            c, k_l = self.convolution_shape
            pad_left = (k_l -1) // 2 #suit for even kernel
            pad_right = (k_l -1) - pad_left
            input_tensor = np.pad(input_tensor, pad_width=[(0,0),
                                                           (0,0),
                                                           (pad_left,pad_right)]
                                                ,mode="constant")
            self.x_padding = input_tensor
            self.padding = (pad_left, pad_right)
            
            # compute output size
            in_l = input_tensor.shape[2]
            out_l = (in_l - k_l) // self.stride_shape[0] + 1
            output = np.zeros((input_tensor.shape[0], self.num_kernels, out_l), dtype=input_tensor.dtype)
            
            # conv
            for b in range(input_tensor.shape[0]):
                for k in range(self.num_kernels):
                    w_k = self.weights[k]
                    bias_k = self.bias[k]
                    for i in range(out_l):
                        l_start = i * self.stride_shape[0]
                        x_patch = input_tensor[b,:,l_start:l_start+k_l]
                        conv_val = np.sum(x_patch * w_k)
                        output[b,k,i] = conv_val + bias_k
            return output



        elif len(self.convolution_shape) == 3:
            c, k_h, k_w = self.convolution_shape
            pad_top = (k_h - 1) //2
            pad_bot = (k_h - 1) - pad_top
            pad_left = (k_w - 1) //2
            pad_right = (k_w - 1) - pad_left
            input_tensor = np.pad(input_tensor, pad_width=[(0,0),
                                                           (0,0),
                                                           (pad_top,pad_bot),
                                                           (pad_left,pad_right)]
                                                ,mode="constant")
            self.x_padding = input_tensor
            self.padding = (pad_top, pad_bot, pad_left, pad_right)
            
            # compute output size
            s_h, s_w = self.stride_shape
            in_h = input_tensor.shape[2]
            in_w = input_tensor.shape[3]
            out_h = (in_h - k_h) // s_h + 1
            out_w = (in_w - k_w) // s_w + 1
            output = np.zeros((input_tensor.shape[0], self.num_kernels, out_h, out_w))

            # conv
            for b in range(input_tensor.shape[0]):
                for k in range(self.num_kernels):
                    w_k = self.weights[k]
                    bias_k = self.bias[k]
                    for i in range(out_h):
                        h_start = i * s_h
                        h_until = h_start + k_h
                        for j in range(out_w):
                            w_start = j * s_w
                            w_until = w_start + k_w
                            x_patch = input_tensor[b,:, h_start:h_until, w_start:w_until]
                            conv_val = np.sum(x_patch* w_k)
                            output[b,k,i,j] = conv_val + bias_k
            return output

    def backward(self, error_tensor):
        #clearn
        self.gradient_weights = np.zeros_like(self.weights)
        self.gradient_bias = np.zeros_like(self.bias)
        if len(self.convolution_shape) == 2:
            c, k_l = self.convolution_shape

            # grad_bas
            self.gradient_bias = np.sum(error_tensor, axis=(0,2))

            # grad_weights and input
            x_pad = self.x_padding
            L_pad = x_pad.shape[2]
            dx_padded = np.zeros_like(x_pad)
            out_L = error_tensor.shape[2]
            for b in range(error_tensor.shape[0]):
                for k in range(self.num_kernels):
                    for i_out in range(out_L):
                        l_start = i_out * self.stride_shape[0]
                        l_end   = l_start + k_l

                        # 2-a  grad_w
                        self.gradient_weights[k] += (
                            x_pad[b, :, l_start:l_end] *
                            error_tensor[b, k, i_out]
                        )

                        # 2-b  grad_x
                        dx_padded[b, :, l_start:l_end] += (
                            self.weights[k] *
                            error_tensor[b, k, i_out]
                        )
            # strip padding
            pad_left, pad_right = self.padding
            if pad_right == 0:
                dx = dx_padded[:, :, pad_left:]
            else:
                dx = dx_padded[:, :, pad_left:-pad_right]   

        elif len(self.convolution_shape) == 3:
            c, k_h, k_w = self.convolution_shape
            s_h, s_w = self.stride_shape
            pad_top, pad_bot, pad_left, pad_right = self.padding

            # grad_bias
            self.gradient_bias = np.sum(error_tensor, axis=(0, 2, 3))

            # grad_weights and input
            x_pad = self.x_padding
            dx_padded = np.zeros_like(x_pad)
            out_H, out_W = error_tensor.shape[2], error_tensor.shape[3]
            for b in range(error_tensor.shape[0]):
                for k in range(self.num_kernels):
                    for i_out in range(out_H):
                        h_start = i_out * s_h
                        h_end = h_start + k_h
                        for j_out in range(out_W):
                            w_start = j_out * s_w
                            w_end = w_start + k_w

                            # current space
                            delta = error_tensor[b, k, i_out, j_out]

                            # 2-a  grad_w
                            self.gradient_weights[k] += (
                                x_pad[b, :, h_start:h_end, w_start:w_end] * delta
                            )

                            # 2-b  grad_x
                            dx_padded[b, :, h_start:h_end, w_start:w_end] += (
                                self.weights[k] * delta
                            )

            # strip padding
            if pad_bot == 0:
                h_slice = slice(pad_top, None)
            else:
                h_slice = slice(pad_top, -pad_bot)
            if pad_right == 0:
                w_slice = slice(pad_left, None)
            else:
                w_slice = slice(pad_left, -pad_right)
            dx = dx_padded[:, :, h_slice, w_slice]

        if self.optimizer is not None:
            self.weights = self.optimizer_w.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizer_b.calculate_update(self.bias, self.gradient_bias)
            self.mark_update = True

        return dx

    def update_weights(self):
        if not self.mark_update and self.optimizer_w is not None:
            self.weights = self.optimizer_w.calculate_update(self.weights, self.gradient_weights)
            self.bias = self.optimizer_b.calculate_update(self.bias, self.gradient_bias)
    
    @property
    def optimizer(self):
        return self.optimizer_w

    @optimizer.setter
    def optimizer(self, opt):
        if opt is None:
            self.optimizer_w = None
            self.optimizer_b = None
        else:
            # deepcopy -> 独立的动量、二阶矩缓存等 state
            self.optimizer_w = copy.deepcopy(opt)
            self.optimizer_b = copy.deepcopy(opt)