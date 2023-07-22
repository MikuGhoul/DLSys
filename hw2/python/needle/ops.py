"""Operatpr table."""
# Global operator table.
from numbers import Number
from typing import Optional, List
from .autograd import NDArray
from .autograd import Op, Tensor, Value, TensorOp
from .autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will numpy as the array_api
# to backup our computations, this line will change in later homeworks
import numpy as array_api


class MakeTensorTuple(TensorTupleOp):
    def compute(self, *args) -> tuple:
        return tuple(args)

    def gradient(self, out_grad, node):
        assert isinstance(out_grad, TensorTuple)
        return tuple(*[out_grad[i] for i in range(len(out_grad))])


def make_tuple(*args):
    return MakeTensorTuple()(*args)


class TupleGetItem(TensorOp):
    def __init__(self, index):
        self.index = index

    def __call__(self, a: TensorTuple, fold_const=True) -> Value:
        assert isinstance(a, TensorTuple)
        # constant folding
        if fold_const and isinstance(a.op, MakeTensorTuple):
            return a.inputs[self.index]
        return Tensor.make_from_op(self, [a])

    def compute(self, a):
        return a[self.index]

    def gradient(self, out_grad, node):
        index = self.index
        in_grad = []
        for i, value in enumerate(node.inputs[0]):
            if i != index:
                in_grad.append(zeros_like(value))
            else:
                in_grad.append(out_grad)
        return MakeTensorTuple()(*in_grad)


def tuple_get_item(value, index):
    return TupleGetItem(index)(value)


class FusedAddScalars(TensorTupleOp):
    def __init__(self, c0: float, c1: float):
        self.c0 = c0
        self.c1 = c1

    def compute(self, a):
        return a + self.c0, a + self.c1

    def gradient(self, out_grad, node):
        return out_grad[0] + out_grad[1]


def fused_add_scalars(x, c0, c1):
    return FusedAddScalars(c0, c1)(x)


class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad * rhs, out_grad * lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return (out_grad * self.scalar,)


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        return array_api.power(a, self.scalar)

    def gradient(self, out_grad, node):
        lhs = node.inputs[0]
        return out_grad * self.scalar * array_api.power(lhs, self.scalar - 1)

def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        return a / b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        return out_grad / rhs, out_grad * (- lhs / (rhs * rhs))


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        return a / self.scalar

    def gradient(self, out_grad, node):
        return (out_grad / self.scalar, )


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ax = self.axes
        bx = list(range(len(a.shape)))
        if ax is None:
            t = bx[-1]
            bx[-1] = bx[-2]
            bx[-2] = t
        else:
            bx[ax[0]] = ax[1]
            bx[ax[1]] = ax[0]
        ax = tuple(bx)
        return array_api.transpose(a, ax)

    def gradient(self, out_grad, node):
        return transpose(out_grad, self.axes)


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.reshape(a, self.shape)

    def gradient(self, out_grad, node):
        lhs = node.inputs[0]
        return reshape(out_grad, lhs.shape)


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        return array_api.broadcast_to(a, self.shape)

    def gradient(self, out_grad, node):
        lhs = node.inputs[0]
        if len(out_grad.shape) == len(lhs.shape):
            axes = []
            for i in range(len(lhs.shape)):
                if out_grad.shape[i] != lhs.shape[i]:
                    axes.append(i)
        else:
            if len(lhs.shape) == 0:
                axes = list(range(len(out_grad.shape) - len(lhs.shape)))
            else:
                axes = list(range(len(out_grad.shape) - len(lhs.shape) + 1))
        axes = tuple(axes)
        return reshape(summation(out_grad, axes), lhs.shape)


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        return array_api.sum(a, axis=self.axes)

    def gradient(self, out_grad, node):
        lhs = node.inputs[0]
        g_shape = list(lhs.shape)
        if self.axes is not None:
            for i in tuple([self.axes]):
                g_shape[i] = 1
        else:
            for i in range(len(g_shape)):
                g_shape[i] = 1
        out_grad = reshape(out_grad, g_shape)
        return broadcast_to(out_grad, lhs.shape)


def summation(a, axes=None):
    return Summation(axes)(a)


class MatMul(TensorOp):
    def compute(self, a, b):
        return a@b

    def gradient(self, out_grad, node):
        lhs, rhs = node.inputs
        lg = matmul(out_grad, transpose(rhs))
        rg = matmul(transpose(lhs), out_grad)
        if lg.shape != lhs.shape:
            lg = summation(lg, tuple(range(len(lg.shape) - len(lhs.shape))))
        if rg.shape != rhs.shape:
            rg = summation(rg, tuple(range(len(rg.shape) - len(rhs.shape))))
        return lg, rg



def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        return -a

    def gradient(self, out_grad, node):
        return -out_grad

def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        return array_api.log(a)

    def gradient(self, out_grad, node):
        lhs = node.inputs[0]
        return out_grad / lhs


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        return array_api.exp(a)

    def gradient(self, out_grad, node):
        return out_grad * exp(node.inputs[0])


def exp(a):
    return Exp()(a)


class ReLU(TensorOp):
    def compute(self, a):
        relu_mask = a > 0
        return a * relu_mask

    def gradient(self, out_grad, node):
        lhs = node.inputs[0]
        grad = out_grad.cached_data
        grad[lhs.cached_data<0] = 0
        return Tensor(grad)


def relu(a):
    return ReLU()(a)



class LogSumExp(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, Z):
        maxZ = array_api.max(Z, self.axes, keepdims=True)
        expZ = array_api.exp(Z - maxZ)
        return array_api.log(array_api.sum(expZ, self.axes)) + array_api.max(Z, self.axes,keepdims=False) 

    def gradient(self, out_grad, node):
        lhs = node.inputs[0]
        maxLhs = Tensor(array_api.max(lhs.cached_data, axis=self.axes, keepdims=True))
        vx3 = out_grad / summation(exp(lhs - maxLhs), axes=self.axes)
        g_shape = list(lhs.shape)
        if self.axes:
            for i in self.axes:
                g_shape[i] = 1
        else:
            for i in range(len(g_shape)):
                g_shape[i] = 1
        vx3 = reshape(vx3, g_shape)
        vx2 = broadcast_to(vx3, lhs.shape)
        vx1 = vx2 * exp(lhs - maxLhs)
        return vx1


def logsumexp(a, axes=None):
    return LogSumExp(axes=axes)(a)
