import numpy as np
from .Base import BaseLayer

class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.output = None  # 用于存储前向传播的输出


    def forward(self, input_tensor):
        #为了防止数值溢出，减去输入张量中的每一行的最大值。这样可以使得输入张量的值更加接近0，从而避免指数函数导致的数值溢出问题。
        e_x = np.exp(input_tensor - np.max(input_tensor, axis=1, keepdims=True))
        self.output = e_x / np.sum(e_x, axis=1, keepdims=True)
        return self.output

    def backward(self, error_tensor):
        # 计算每个类别的导数
        S = self.output
        # error_tensor 应该和 S 的形状一致
        if S.shape != error_tensor.shape:
            raise ValueError(f"Shape mismatch: S shape {S.shape}, error_tensor shape {error_tensor.shape}")

        # Softmax的梯度公式
        temp = S * (error_tensor - np.sum(S * error_tensor, axis=1, keepdims=True))

        return temp

