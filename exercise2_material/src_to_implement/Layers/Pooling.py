import numpy as np
from .Base import BaseLayer

class Pooling(BaseLayer):
    """
    input  shape : (B, C, H, W)
    output shape : (B, C, out_H, out_W)
    """

    def __init__(self, stride_shape, pooling_shape):
        super().__init__()
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self._input_shape = None
        self._max_mask = None



    def forward(self, input_tensor):
        self._input_shape = input_tensor.shape
        B, C, H, W = self._input_shape
        p_h, p_w = self.pooling_shape
        s_h, s_w = self.stride_shape

        out_h = (H - p_h) // s_h + 1
        out_w = (W - p_w) // s_w + 1
        output = np.empty((B, C, out_h, out_w), dtype=input_tensor.dtype)

        self._max_mask = np.zeros(
            (B, C, out_h, out_w, p_h, p_w), dtype=bool
        )

        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    h0 = i * s_h
                    h1 = h0 + p_h
                    for j in range(out_w):
                        w0 = j * s_w
                        w1 = w0 + p_w

                        window = input_tensor[b, c, h0:h1, w0:w1]  # (p_h, p_w)
                        max_val = np.max(window)
                        output[b, c, i, j] = max_val

                        # record the position of the max
                        max_mask_local = (window == max_val)
                        self._max_mask[b, c, i, j, :, :] = max_mask_local

        return output


    def backward(self, error_tensor):
        B, C, H, W = self._input_shape
        p_h, p_w   = self.pooling_shape
        s_h, s_w   = self.stride_shape
        out_h, out_w = error_tensor.shape[2], error_tensor.shape[3]

        dx = np.zeros((B, C, H, W), dtype=error_tensor.dtype)

        for b in range(B):
            for c in range(C):
                for i in range(out_h):
                    h0 = i * s_h
                    h1 = h0 + p_h
                    for j in range(out_w):
                        w0 = j * s_w
                        w1 = w0 + p_w

                        # go back to the position
                        mask_local = self._max_mask[b, c, i, j]   # (p_h, p_w)
                        dx[b, c, h0:h1, w0:w1] += (
                            mask_local * error_tensor[b, c, i, j]
                        )

        return dx
