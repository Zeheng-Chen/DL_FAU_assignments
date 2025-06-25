import numpy as np

class Constant:
    def __init__(self, w_init = 0.1):
        self.w_init = w_init

    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.full(weights_shape, self.w_init)
        return weights

class UniformRandom:
    def initialize(self, weights_shape, fan_in, fan_out):
        weights = np.random.random(weights_shape)
        return weights
    
class Xavier:
    def initialize(self,weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2/(fan_in + fan_out))
        weights = np.random.normal(0,sigma,weights_shape)
        return weights
        

class He:
    def initialize(self,weights_shape, fan_in, fan_out):
        sigma = np.sqrt(2/ fan_in)
        weights = np.random.normal(0,sigma, weights_shape)
        return weights