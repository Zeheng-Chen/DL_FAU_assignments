import numpy as np

class Flatten:
    def __init__(self):
        self.input_shape = None

    # compress the other dimension together: (B, C, H, W) -> (B, C*H*W)
    def forward(self,input_tensor):
        self.input_shape = input_tensor.shape
        batch_size = input_tensor.shape[0]
        out = input_tensor.reshape(batch_size, -1)
        return out
    
    def backward(self, error_tensor):
        error_tensor = error_tensor.reshape(self.input_shape)
        return error_tensor
    