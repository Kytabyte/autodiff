""" Differentiable variables."""

from numbers import Real
import math

from . import op
from .engine import GradEngine


class Var:
    def __init__(self, val, *, ref=None):
        """

        Args:
            val(Python Number): the value of the variable
            ref(str): Optional name of the variable
        """
        self._val = val
        self._ref = ref
        self._grad = None

        self._op = None
        self._parents = ()
        self._engine = GradEngine()

    @property
    def val(self):
        """
        Return the value of Var

        Returns: A Python Number
        """
        return self._val

    @property
    def grad(self):
        """
        Return the gradient of Var

        Returns: A Python Number represents the gradient of this Var. None if
                no gradient
        """
        return self._grad

    @property
    def name(self):
        """
        Return the reference of this Variable.

        Returns: (str) the reference name of this variable. None if this
                variable has no name.
        """
        return self._ref

    def _is_leaf(self):
        # if this variable has parent Var
        return self._op is None

    def __str__(self):
        return f"{type(self).__name__}({self._val}, grad_op={repr(self._op)})"

    __repr__ = __str__

    #############
    # operators #
    #############

    def __add__(self, other):
        return op.add(self, other)

    __radd__ = __add__

    def __sub__(self, other):
        return op.sub(self, other)

    def __rsub__(self, other):
        return op.sub(other, self)

    def __mul__(self, other):
        return op.mul(self, other)

    __rmul__ = __mul__

    def __truediv__(self, other):
        return op.div(self, other)

    def __rtruediv__(self, other):
        return op.div(other, self)

    def __neg__(self):
        return op.neg(self)

    def __pow__(self, power, modulo=None):
        if modulo is not None:
            raise ArithmeticError("'modulo' is not supported not pow function yet")
        return op.pow(self, power)

    ###############
    # comparators #
    ###############

    def __eq__(self, other):
        if isinstance(other, Var):
            return math.isclose(self.val, other.val)
        if isinstance(other, Real):
            return math.isclose(self.val, other)
        return False

    def __ne__(self, other):
        return not self.__eq__(other)

    def __lt__(self, other):
        if not isinstance(other, (Var, Real)):
            raise TypeError(
                f"'<' not supported between instances of '{type(self).__name__}' and '{type(other).__name__}'")
        val = other.val if isinstance(other, Var) else other
        return self.val < val

    def __le__(self, other):
        if not isinstance(other, (Var, Real)):
            raise TypeError(
                f"'<' not supported between instances of '{type(self).__name__}' and '{type(other).__name__}'")
        val = other.val if isinstance(other, Var) else other
        return self.val <= val

    def __gt__(self, other):
        return not self <= other

    def __ge__(self, other):
        return not self < other

    ##########################
    # Gradient Functionality #
    ##########################

    def backward(self, grad=None):
        """
        Do automatic differentiation on this Var, w.r.t all `Var`s this
        variable is dependent on. One can define the base gradient by setting
        `grad`, otherwise the base gradient will be 1.

        After calling gradient(), all `Var`s this var depends on will have
        gradient, by calling var.grad on other `Var`s.

        Args:
            grad: Base gradient to propagate
        """
        engine = self._engine
        if not engine.is_working():
            engine.activate()
            engine.accumulate(1. if grad is None else grad)

        # solve dependency
        for parent in self._parents:
            if isinstance(parent, Var):
                parent._start_task()

        # calculate and propagate gradient
        self._propagate()

    def _start_task(self, task_id=None):
        # TODO: can add task_id to identify current task

        # Whenever this function is called, this node is one
        # of the parent of the callers. engine will increment
        # the dependency that this node depends on.

        engine = self._engine

        # the engine.is_working() has a side functionality of checking
        # if this node is visited or not. If this node is not visited yet,
        # it will propagate _start_task() to its parents. Otherwise, only
        # a dependency will be added.
        if not engine.is_working():
            engine.activate()
            engine.add_dependency()
            for parent in self._parents:
                if isinstance(parent, Var):
                    parent._start_task()
        else:
            engine.add_dependency()

    def _propagate(self):
        # this node has already have all gradient of its dependencies
        # and is ready to propogate the dependency to its parents
        grad = self._engine.get_grad()

        if self._is_leaf():
            # only leaf nodes requires grad, and no need to propagate
            self._grad = grad
        else:
            # calculate gradient of self w.r.t each of self._parents,
            # and propagate the gradient to them
            grads = self._op.gradient(grad, self.val, *self._parents)
            for grad, parent in zip(grads, self._parents):
                if isinstance(parent, Var) and grad is not None:
                    parent._do_task(grad)

        # finish the task
        self._end_task()

    def _do_task(self, grad):
        # accumulate gradient from its dependency, remove one dependency
        # from engine. Propagate the graident if all dependency are resolved.
        engine = self._engine

        engine.accumulate(grad)
        engine.del_dependency()

        if engine.ready_backward():
            self._propagate()

    def _end_task(self):
        # Finish the task
        self._engine.reset()

    def detach(self):
        self._parents = ()
        self._grad = None
        self._end_task()
