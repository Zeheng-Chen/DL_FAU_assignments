import numpy as np
from .Base import BaseLayer


class FullyConnected(BaseLayer):
    def __init__(self, input_size, output_size):
        self.trainable = True
        self._optimizer = None
        self.gradient_weights = None
        self.gradient_biases = None
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.uniform(low=0.0, high=1.0, size=(output_size, input_size + 1))
        self.biases = np.random.uniform(low=0.0, high=1.0, size=output_size)

    def forward(self, input_tensor):
        # 增加偏置处理项
        bias_term = np.ones((input_tensor.shape[0], 1))
        input_with_bias = np.hstack((input_tensor, bias_term))
        self.last_input = input_with_bias  # 保存完整的输入用于反向传播
        return np.dot(input_with_bias, self.weights.T)

    def backward(self, error_tensor):
        # Calculate gradients
        self.gradient_weights = np.dot(error_tensor.T, self.last_input)
        self.gradient_biases = error_tensor.sum(axis=0)
        self.update_weights()
        return np.dot(error_tensor, self.weights[:, :-1])

    def update_weights(self):
        if self.trainable and self._optimizer:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_weights)
            self.biases = self._optimizer.calculate_update(self.biases, self.gradient_biases)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer
# 通过 getter 和 setter 方法，可以在不改变类的接口的情况下，对属性property的访问进行控制和封装。
