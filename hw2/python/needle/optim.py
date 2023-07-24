"""Optimization module"""
import needle as ndl
import numpy as np


class Optimizer:
    def __init__(self, params):
        self.params = params

    def step(self):
        raise NotImplementedError()

    def reset_grad(self):
        for p in self.params:
            p.grad = None


class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0):
        super().__init__(params)
        self.lr = lr
        self.momentum = momentum
        self.u = {}
        self.weight_decay = weight_decay

    def step(self):
        ### BEGIN YOUR SOLUTION
        for param in self.params:
            grad = self.momentum * self.u.get(param, 0) + (1 - self.momentum) * (param.grad.detach() + self.weight_decay * param.detach())
            grad = ndl.Tensor(grad, dtype=param.dtype)
            self.u[param] = grad
            param.data -= self.lr * grad
        ### END YOUR SOLUTION


class Adam(Optimizer):
    def __init__(
        self,
        params,
        lr=0.01,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0.0,
    ):
        super().__init__(params)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

        self.m = {}
        self.v = {}

    def step(self):
        ### BEGIN YOUR SOLUTION
        self.t += 1
        for param in self.params:
            m = self.beta1 * self.m.get(param, 0) + (1 - self.beta1) * (param.grad.detach() + self.weight_decay * param.detach())
            v = self.beta2 * self.v.get(param, 0) + (1 - self.beta2) * ((param.grad.detach() + self.weight_decay * param.detach()) ** 2)
            m_bias = m / (1 - self.beta1 ** self.t)
            v_bias = v / (1 - self.beta2 ** self.t)
            self.m[param] = m
            self.v[param] = v
            param.data -= ndl.Tensor(self.lr * m_bias / (v_bias ** 0.5 + self.eps), dtype=param.dtype) 
        ### END YOUR SOLUTION
