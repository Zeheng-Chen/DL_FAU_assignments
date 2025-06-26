# Optimization/Optimizers.py

import numpy as np
EPS = 1e-8

class Sgd:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        return weight_tensor - self.learning_rate * gradient_tensor

class SgdWithMomentum:
    def __init__(self, learning_rate: float, momentum_rate: float):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.intermediate = None
    
    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.intermediate is None:
            self.intermediate = np.zeros_like(weight_tensor)
        
        self.intermediate = self.momentum_rate * self.intermediate - self.learning_rate * gradient_tensor
        return weight_tensor + self.intermediate

    
class Adam:
    
    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.k = 0
        self.v = None
        self.r = None
        

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.v is None:
            self.v = np.zeros_like(weight_tensor)
            self.r = np.zeros_like(weight_tensor)
        self.v = self.mu * self.v + (1-self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1-self.rho) * (gradient_tensor * gradient_tensor)

        self.k += 1
        v_hat = self.v / (1-self.mu**self.k)
        r_hat = self.r / (1-self.rho ** self.k)
        
        return weight_tensor - self.learning_rate * (v_hat / (np.sqrt(r_hat) + EPS))