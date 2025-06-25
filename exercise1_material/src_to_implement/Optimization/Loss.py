#交叉熵损失用于计算预测值和真实标签之间的损失
import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.output = None

    def forward(self, prediction_tensor, label_tensor):
        epsilon = np.finfo(float).eps
        # 获取浮点类型的最小可表示正数，通常用于避免数值计算中的除零错误或log(0)错误。 epsilon 是一个非常小的正数。

        prediction_tensor = np.clip(prediction_tensor, epsilon, 1 - epsilon)
        # 的值限制在 [epsilon, 1 - epsilon] 范围内。这是为了避免 prediction_tensor 中出现 0 或 1
        # 从而防止计算 log(0) 导致数值不稳定。
        self.output = prediction_tensor
        cross_entropy_loss = -np.sum(label_tensor * np.log(prediction_tensor))
        return cross_entropy_loss



    def backward(self, label_tensor):
        if self.output is None:
            raise ValueError("Forward method must be called before backward.")
        # Calculate gradient, noting that output might be very close to 0 or 1
        grad = -(label_tensor / self.output)
        return grad

