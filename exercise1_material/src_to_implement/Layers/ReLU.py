import numpy as np
from .Base import BaseLayer

class ReLU(BaseLayer):
    def __init__(self):
        super().__init__()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        return np.maximum(0, input_tensor)

    def backward(self, error_tensor):
        return error_tensor * (self.input_tensor > 0)
#一种常用的激活函数，广泛应用于深度学习中的神经网络。
# 它的主要作用是在每一层的线性变换（如全连接层）之后引入非线性，从而使得神经网络能够处理和学习复杂的模式。