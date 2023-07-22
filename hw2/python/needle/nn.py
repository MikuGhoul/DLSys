"""The module.
"""
from typing import List, Callable, Any
from needle.autograd import Tensor
from needle import ops
import needle.init as init
import numpy as np


class Parameter(Tensor):
    """A special kind of tensor that represents parameters."""


def _unpack_params(value: object) -> List[Tensor]:
    if isinstance(value, Parameter):
        return [value]
    elif isinstance(value, Module):
        return value.parameters()
    elif isinstance(value, dict):
        params = []
        for k, v in value.items():
            params += _unpack_params(v)
        return params
    elif isinstance(value, (list, tuple)):
        params = []
        for v in value:
            params += _unpack_params(v)
        return params
    else:
        return []


def _child_modules(value: object) -> List["Module"]:
    if isinstance(value, Module):
        modules = [value]
        modules.extend(_child_modules(value.__dict__))
        return modules
    if isinstance(value, dict):
        modules = []
        for k, v in value.items():
            modules += _child_modules(v)
        return modules
    elif isinstance(value, (list, tuple)):
        modules = []
        for v in value:
            modules += _child_modules(v)
        return modules
    else:
        return []




class Module:
    def __init__(self):
        self.training = True

    def parameters(self) -> List[Tensor]:
        """Return the list of parameters in the module."""
        return _unpack_params(self.__dict__)

    def _children(self) -> List["Module"]:
        return _child_modules(self.__dict__)

    def eval(self):
        self.training = False
        for m in self._children():
            m.training = False

    def train(self):
        self.training = True
        for m in self._children():
            m.training = True

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class Identity(Module):
    def forward(self, x):
        return x


class Linear(Module):
    def __init__(self, in_features, out_features, bias=True, device=None, dtype="float32"):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = init.kaiming_uniform(fan_in=in_features, fan_out=out_features)
        if bias:
            self.bias = init.kaiming_uniform(fan_in=out_features, fan_out=1).reshape((1, out_features))
        else:
            self.bias = None

    def forward(self, X: Tensor) -> Tensor:
        out = ops.matmul(X, self.weight)
        if self.bias is not None:
            out = ops.add(out, self.bias)
        return out



class Flatten(Module):
    def forward(self, X):
        batch = X.shape[0]
        flat = 1
        for i in list(X.shape)[1:]:
            flat *= i
        return ops.reshape(X, (batch, flat))


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return ops.relu(x)


class Sequential(Module):
    def __init__(self, *modules):
        super().__init__()
        self.modules = modules

    def forward(self, x: Tensor) -> Tensor:
        out = x
        for m in self.modules:
            out = m.forward(out)
        return out


class SoftmaxLoss(Module):
    def forward(self, logits: Tensor, y: Tensor):
        y = init.one_hot(np.max(y.cached_data) + 1, y)
        return (ops.summation(ops.logsumexp(logits, (1, ))) - ops.summation((ops.multiply(y, logits)))) / logits.shape[0]


class BatchNorm1d(Module):
    def __init__(self, dim, eps=1e-5, momentum=0.1, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.momentum = momentum
        ### BEGIN YOUR SOLUTION
        self.weight = init.ones(dim)
        self.bias = init.zeros(dim)
        self.running_mean = init.zeros(dim)
        self.running_var = init.ones(dim)
        ### END YOUR SOLUTION


    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            mean_x = ops.divide_scalar(ops.summation(x, 0), x.shape[0])
            mean_x_reshap = ops.reshape(mean_x, (1, mean_x.shape[0]))
            mean_x_broadcast = ops.broadcast_to(mean_x_reshap, x.shape)
            var_x = ops.divide_scalar(ops.summation(ops.power_scalar((x - mean_x_broadcast), 2), 0), x.shape[0])
            var_x_reshape = ops.reshape(var_x, (1, var_x.shape[0]))
            var_x_broadcast = ops.broadcast_to(var_x_reshape, x.shape)
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean_x
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var_x
            return ops.multiply(self.weight, (x - mean_x_broadcast) / ops.power_scalar(var_x_broadcast + self.eps, 0.5)) + self.bias
        else:
            mean_x = self.running_mean
            mean_x_reshap = ops.reshape(mean_x, (1, mean_x.shape[0]))
            mean_x_broadcast = ops.broadcast_to(mean_x_reshap, x.shape)
            var_x = self.running_var
            var_x_reshape = ops.reshape(var_x, (1, var_x.shape[0]))
            var_x_broadcast = ops.broadcast_to(var_x_reshape, x.shape)
            return ops.multiply(self.weight, (x - mean_x_broadcast) / ops.power_scalar(var_x_broadcast + self.eps, 0.5)) + self.bias
        ### END YOUR SOLUTION


class LayerNorm1d(Module):
    def __init__(self, dim, eps=1e-5, device=None, dtype="float32"):
        super().__init__()
        self.dim = dim
        self.eps = eps
        ### BEGIN YOUR SOLUTION
        self.weight = init.ones(dim)
        self.bias = init.zeros(dim)
        ### END YOUR SOLUTION

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        mean_x = ops.divide_scalar(ops.summation(x, 1), x.shape[1])
        mean_x_reshape = ops.reshape(mean_x, (mean_x.shape[0], 1))
        mean_x_broadcast = ops.broadcast_to(mean_x_reshape, x.shape)
        var_x = ops.divide_scalar(ops.summation(ops.power_scalar((x - mean_x_broadcast), 2), 1), x.shape[1])
        var_x_reshap = ops.reshape(var_x, (var_x.shape[0], 1))
        var_x_broadcast = ops.broadcast_to(var_x_reshap, x.shape)
        return ops.multiply(self.weight, (x - mean_x_broadcast) / ops.power_scalar(var_x_broadcast + self.eps, 0.5)) + self.bias
        ### END YOUR SOLUTION


class Dropout(Module):
    def __init__(self, p = 0.5):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        if self.training:
            drop_mask = init.randb(*x.shape, p=self.p) / (1 - self.p)
            return x * drop_mask
        ### END YOUR SOLUTION


class Residual(Module):
    def __init__(self, fn: Module):
        super().__init__()
        self.fn = fn

    def forward(self, x: Tensor) -> Tensor:
        ### BEGIN YOUR SOLUTION
        return self.fn(x) + x
        ### END YOUR SOLUTION



