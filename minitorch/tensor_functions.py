"""Implementation of the autodifferentiation Functions for Tensor."""

from __future__ import annotations

import random
from typing import TYPE_CHECKING, Optional, Tuple

import numpy as np

import minitorch

from . import operators
from .autodiff import Context
from .tensor_ops import SimpleBackend, TensorBackend, TensorOps



if TYPE_CHECKING:
    from typing import Any, List, Tuple

    from .tensor import Tensor
    from .tensor_data import UserIndex, UserShape


def wrap_tuple(x: Any) -> tuple:  # type: ignore
    """Turn a possible value into a tuple"""
    if isinstance(x, tuple):
        return x
    return (x,)


# Constructors
class Function:
    @classmethod
    def _backward(cls, ctx: Context, grad_out: Tensor) -> Tuple[Tensor, ...]:
        return wrap_tuple(cls.backward(ctx, grad_out))  # type: ignore

    @classmethod
    def _forward(cls, ctx: Context, *inps: Any) -> Tensor:
        return cls.forward(ctx, *inps)  # type: ignore

    @classmethod
    def apply(cls, *vals: Any) -> Tensor:
        """Call the forward function and track history"""
        raw_vals = []
        need_grad = False

        for v in vals:
            if isinstance(v, minitorch.Tensor):
                if v.requires_grad():
                    need_grad = True
                raw_vals.append(v.detach())
            else:
                # For non-Tensor arguments, append them directly
                raw_vals.append(v)

        # Create the context.
        ctx = Context(not need_grad)

        # Call forward with the variables.
        c = cls._forward(ctx, *raw_vals)

        # Create a new variable from the result with a new history.
        back = None
        if need_grad:
            back = minitorch.History(cls, ctx, vals)
        return minitorch.Tensor(c._tensor, back, backend=c.backend)


class Neg(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        return t1.f.neg_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        return grad_output.f.neg_map(grad_output)


class Inv(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor) -> Tensor:
        ctx.save_for_backward(t1)
        return t1.f.inv_map(t1)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        (t1,) = ctx.saved_values
        return grad_output.f.inv_back_zip(t1, grad_output)


class Add(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        return t1.f.add_zip(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        return grad_output, grad_output


class All(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        # Flatten the tensor and reduce across all elements
        result = a._tensor._storage.all()
        return minitorch.Tensor.make([float(result)], (), backend=a.backend)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[None]:
        # No gradients needed for the all() function
        return None,



class Mul(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        ctx.save_for_backward(a, b)
        # Use TensorOps.zip to get the appropriate function for element-wise multiplication
        return TensorOps.zip(operators.mul)(a, b)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        a, b = ctx.saved_values
        return grad_output * b, grad_output * a


class Sigmoid(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        out = TensorOps.map(operators.sigmoid)(a)
        ctx.save_for_backward(out)
        return out
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        out = ctx.saved_values[0]
        return grad_output * out * (1 - out)


class ReLU(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        ctx.save_for_backward(a)
        return TensorOps.map(operators.relu)(a)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        a = ctx.saved_values[0]
        return TensorOps.zip(operators.relu_back)(grad_output, a)


class Log(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        ctx.save_for_backward(a)
        return TensorOps.map(operators.log)(a)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        a = ctx.saved_values[0]
        return grad_output / a


class Exp(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        out = TensorOps.map(operators.exp)(a)
        ctx.save_for_backward(out)
        return out
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        out = ctx.saved_values[0]
        return grad_output * out


class Sum(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, dim: Optional[Tensor] = None) -> Tensor:
        if dim is not None:
            dim_value = int(dim.item())
            ctx.save_for_backward(a.shape, dim_value)
            return TensorOps.reduce(operators.add)(a, dim_value)
        else:
            ctx.save_for_backward(a.shape, None)
            # Flatten the tensor and reduce over dimension 0
            flat_tensor = a.contiguous().view(int(operators.prod(a.shape)))
            return TensorOps.reduce(operators.add)(flat_tensor, 0)
        
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        shape, dim = ctx.saved_values
        if dim is None:
            # Sum over all dimensions
            # grad_output is a scalar
            # Create a tensor of ones with the same shape as the input
            ones_tensor = zeros(shape, backend=grad_output.backend) + 1.0
            return ones_tensor * grad_output
        else:
            # Sum over specific dimension
            # Reshape grad_output to match the dimensions for broadcasting
            grad_shape = [1] * len(shape)
            grad_shape[dim] = grad_output.shape[0]
            grad_output_reshaped = grad_output.view(*grad_shape)
            # Broadcasting during multiplication will handle dimension expansion
            ones_tensor = zeros(shape, backend=grad_output.backend) + 1.0
            return grad_output_reshaped * ones_tensor



class LT(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        return TensorOps.zip(operators.lt)(a, b)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[None, None]:
        return None, None


class EQ(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        return TensorOps.zip(operators.eq)(a, b)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[None, None]:
        return None, None


class IsClose(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, b: Tensor) -> Tensor:
        return minitorch.TensorOps.zip(operators.is_close)(a, b)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[None, None]:
        return None, None


class Permute(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, order: Tuple[int, ...]) -> Tensor:
        ctx.save_for_backward(order)
        assert len(order) == len(a.shape), "Order must match number of dimensions"
        return Tensor(a._tensor.permute(*order), backend=a.backend)
    
    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        order = ctx.saved_values[0]
        reverse_order = tuple(np.argsort(order))
        return Tensor(grad_output._tensor.permute(*reverse_order), backend=grad_output.backend)



class View(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor, shape: Union[Tuple[int, ...], List[int]]) -> Tensor:
        ctx.save_for_backward(a.shape)
        assert a._tensor.is_contiguous(), "Must be contiguous to view"
        return minitorch.Tensor.make(a._tensor._storage, tuple(shape), backend=a.backend)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        original_shape = ctx.saved_values
        return grad_output.view(original_shape)


class Copy(Function):
    @staticmethod
    def forward(ctx: Context, a: Tensor) -> Tensor:
        """Id function makes contiguous"""
        return a.f.id_map(a)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tensor:
        """Undo"""
        return grad_output


class MatMul(Function):
    @staticmethod
    def forward(ctx: Context, t1: Tensor, t2: Tensor) -> Tensor:
        """Matrix Multiply Forward (module 3)"""
        ctx.save_for_backward(t1, t2)
        return t1.f.matrix_multiply(t1, t2)

    @staticmethod
    def backward(ctx: Context, grad_output: Tensor) -> Tuple[Tensor, Tensor]:
        """Matrix Multiply backward (module 3)"""
        t1, t2 = ctx.saved_values

        def transpose(a: Tensor) -> Tensor:
            order = list(range(a.dims))
            order[-2], order[-1] = order[-1], order[-2]
            return a._new(a._tensor.permute(*order))

        return (
            grad_output.f.matrix_multiply(grad_output, transpose(t2)),
            grad_output.f.matrix_multiply(transpose(t1), grad_output),
        )


# Helpers for Constructing tensors
def zeros(shape: UserShape, backend: TensorBackend = SimpleBackend) -> Tensor:
    """Produce a zero tensor of size shape.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend

    Returns:
    -------
        new tensor

    """
    return minitorch.Tensor.make(
        [0.0] * int(operators.prod(shape)), shape, backend=backend
    )


def rand(
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a random tensor of size shape.

    Args:
    ----
        shape : shape of tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:Tensor : new tensor

    """
    vals = [random.random() for _ in range(int(operators.prod(shape)))]
    tensor = minitorch.Tensor.make(vals, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def _tensor(
    ls: Any,
    shape: UserShape,
    backend: TensorBackend = SimpleBackend,
    requires_grad: bool = False,
) -> Tensor:
    """Produce a tensor with data ls and shape shape.

    Args:
    ----
        ls: data for tensor
        shape: shape of tensor
        backend: tensor backend
        requires_grad: turn on autodifferentiation

    Returns:
    -------
        new tensor

    """
    tensor = minitorch.Tensor.make(ls, shape, backend=backend)
    tensor.requires_grad_(requires_grad)
    return tensor


def tensor(
    ls: Any, backend: TensorBackend = SimpleBackend, requires_grad: bool = False
) -> Tensor:
    """Produce a tensor with data and shape from ls

    Args:
    ----
        ls: data for tensor
        backend : tensor backend
        requires_grad : turn on autodifferentiation

    Returns:
    -------
        :class:Tensor : new tensor

    """

    def shape(ls: Any) -> List[int]:
        if isinstance(ls, (list, tuple)):
            return [len(ls)] + shape(ls[0])
        else:
            return []

    def flatten(ls: Any) -> List[float]:
        if isinstance(ls, (list, tuple)):
            return [y for x in ls for y in flatten(x)]
        else:
            return [ls]

    cur = flatten(ls)
    shape2 = shape(ls)
    return _tensor(cur, tuple(shape2), backend=backend, requires_grad=requires_grad)


# Gradient check for tensors


def grad_central_difference(
    f: Any, *vals: Tensor, arg: int = 0, epsilon: float = 1e-6, ind: UserIndex
) -> float:
    x = vals[arg]
    up = zeros(x.shape)
    up[ind] = epsilon
    vals1 = [x if j != arg else x + up for j, x in enumerate(vals)]
    vals2 = [x if j != arg else x - up for j, x in enumerate(vals)]
    delta: Tensor = f(*vals1).sum() - f(*vals2).sum()

    return delta[0] / (2.0 * epsilon)


def grad_check(f: Any, *vals: Tensor) -> None:
    """Check whether autodiff matches central difference."""
    for x in vals:
        x.requires_grad_(True)
        x.zero_grad_()
    random.seed(10)
    out = f(*vals)
    out.sum().backward()
    err_msg = """

Gradient check error for function %s.

Input %s

Received derivative %f for argument %d and index %s,
but was expecting derivative %f from central difference.

"""

    for i, x in enumerate(vals):
        ind = x._tensor.sample()
        check = grad_central_difference(f, *vals, arg=i, ind=ind)
        assert x.grad is not None
        np.testing.assert_allclose(
            x.grad[ind],
            check,
            1e-2,
            1e-2,
            err_msg=err_msg % (f, vals, x.grad[ind], i, ind, check),
        )