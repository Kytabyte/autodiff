""" Forward/Backward operations applied on variable """
import math
from .var import Var


def add(x1, x2) -> Var:
    return _AddOp(x1, x2)


def sub(x1, x2) -> Var:
    return _SubOp(x1, x2)


def mul(x1, x2) -> Var:
    return _MulOp(x1, x2)


def div(x1, x2) -> Var:
    return _DivOp(x1, x2)


def neg(x) -> Var:
    return _NegOp(x)


def reciprocal(x) -> Var:
    return _DivOp(1, x)


def pow(x, p):
    return _PowOp(x, p)


##########################################
# Actual Implementation of All Operators #
##########################################

# All Operators needs to override two static method
# 1. compute(*inputs: Var or Python Real Number) -> Var
#   Consume a list of Var or Python number to return a val after executing the operator
# 2. gradient(grad: float, output: Var, *inputs: Var or Python Real) -> List[float or None]
#   return the accumulative gradient of output w.r.t all Var inputs. Return None if the input
#   is a Python Number
#
# Every operator will add all inputs to output Var's `_parents`, and add itself to output Var's `_op`
class _Op:
    def __new__(cls, *inputs):
        output = cls.compute(*inputs)
        output._op = cls
        output._parents = inputs
        return output

    @staticmethod
    def compute(*inputs) -> Var:
        raise NotImplementedError

    @staticmethod
    def gradient(grad, output, *inputs):
        raise NotImplementedError


class _AddOp(_Op):

    @staticmethod
    def compute(*inputs) -> Var:
        assert len(inputs) == 2

        x1_val, x2_val = _val(*inputs)
        return Var(x1_val + x2_val)

    @staticmethod
    def gradient(grad, output, *inputs):
        assert len(inputs) == 2

        return tuple(grad if isinstance(x, Var) else None for x in inputs)


class _SubOp(_Op):

    @staticmethod
    def compute(*inputs) -> Var:
        assert len(inputs) == 2

        x1_val, x2_val = _val(*inputs)
        return Var(x1_val - x2_val)

    @staticmethod
    def gradient(grad, output, *inputs):
        assert len(inputs) == 2

        x1, x2 = inputs
        return (grad if isinstance(x1, Var) else None,
                -grad if isinstance(x2, Var) else None)


class _MulOp(_Op):

    @staticmethod
    def compute(*inputs):
        assert len(inputs) == 2

        x1_val, x2_val = _val(*inputs)
        return Var(x1_val * x2_val)

    @staticmethod
    def gradient(grad, output, *inputs):
        assert len(inputs) == 2

        x1, x2 = inputs
        x1_val, x2_val = _val(*inputs)

        return (x2_val * grad if isinstance(x1, Var) else None,
                x1_val * grad if isinstance(x2, Var) else None)


class _DivOp(_Op):

    @staticmethod
    def compute(*inputs):
        assert len(inputs) == 2

        x1_val, x2_val = _val(*inputs)
        if x2_val == 0:
            raise ZeroDivisionError

        return Var(x1_val / x2_val)

    @staticmethod
    def gradient(grad, output, *inputs):
        assert len(inputs) == 2

        x1, x2 = inputs
        x1_val, x2_val = _val(*inputs)
        if x2_val == 0:
            raise ZeroDivisionError

        return (grad / x2_val if isinstance(x1, Var) else None,
                -x1_val * grad / (x2_val * x2_val) if isinstance(x2, Var) else None)


class _NegOp(_Op):

    @staticmethod
    def compute(*inputs) -> Var:
        assert len(inputs) == 1

        x_val, = _val(*inputs)
        return Var(-x_val)

    @staticmethod
    def gradient(grad, output, *inputs):
        assert len(inputs) == 1

        return -grad,


class _PowOp(_Op):

    @staticmethod
    def compute(*inputs) -> Var:
        assert len(inputs) == 2

        # f(x, p) = x^p, where p >= 0 if x == 0
        x_val, p_val = _val(*inputs)
        if x_val == 0 and p_val < 0:
            raise ZeroDivisionError("0 cannot be raised to a negative power")

        return Var(x_val ** p_val)

    @staticmethod
    def gradient(grad, output, *inputs):
        # f(x, p) = x^p
        # df/dx = p * x^(p-1), where x != 0 if p < 1
        #   if x != 0, df/dx = p * y / x
        # df/dp = x^p * ln(p) = y * ln(p), where x >= 0 if p < 0

        assert len(inputs) == 2

        y, x, p = inputs
        y_val, x_val, p_val = _val(output, *inputs)

        # Solve x
        if isinstance(x, Var):
            if x_val == 0 and p_val < 1:
                raise ZeroDivisionError("0 with a power < 1 is not differentiable")
            x_grad = 0 if x_val == 0 else grad * p_val * y_val / x_val
        else:
            x_grad = None

        # Solve p
        if isinstance(p, Var):
            if x_val < 0:
                # p_grad = float('nan')
                raise ArithmeticError("negative raise to a power is not differentiable")
            p_grad = y_val * math.log(x_val)
        else:
            p_grad = None

        return x_grad, p_grad


def _val(*inputs):
    return [x.val if isinstance(x, Var) else x for x in inputs]
