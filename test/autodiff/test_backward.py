import math

import autodiff as ad
import autodiff.op as op

import pytest


def test_add():
    # add two vars
    x1, x2 = ad.Var(1), ad.Var(2)
    x3 = x1 + x2
    x3.gradient()

    assert x1.grad == 1
    assert x2.grad == 1

    # add var and pyint
    x1, x2 = ad.Var(1), 2
    x3 = x1 + x2
    x3.gradient()

    assert x1.grad == 1

    x1, x2 = 1, ad.Var(2)
    x3 = x1 + x2
    x3.gradient()

    assert x2.grad == 1


def test_sub():
    # sub two vars
    x1, x2 = ad.Var(1), ad.Var(2)
    x3 = x1 - x2
    x3.gradient()

    assert x1.grad == 1
    assert x2.grad == -1

    # sub var and pyint
    x1, x2 = ad.Var(1), 2
    x3 = x1 - x2
    x3.gradient()

    assert x1.grad == 1

    x1, x2 = 1, ad.Var(2)
    x3 = x1 - x2
    x3.gradient()

    assert x2.grad == -1


def test_mul():
    # mul two vars
    x1, x2 = ad.Var(2), ad.Var(3)
    x3 = x1 * x2
    x3.gradient()

    assert x1.grad == 3
    assert x2.grad == 2

    # add var and variables
    x1, x2 = ad.Var(2), 3
    x3 = x1 * x2
    x3.gradient()

    assert x1.grad == 3

    x1, x2 = 2, ad.Var(3)
    x3 = x1 * x2
    x3.gradient()

    assert x2.grad == 2


def test_truediv():
    # div two vars
    x1, x2 = ad.Var(1), ad.Var(2)
    x3 = x1 / x2
    x3.gradient()

    assert x1.grad == 0.5
    assert x2.grad == -0.25

    # add var and variables
    x1, x2 = ad.Var(1), 2
    x3 = x1 / x2
    x3.gradient()

    assert x1.grad == 0.5

    x1, x2 = 1, ad.Var(2)
    x3 = x1 / x2
    x3.gradient()

    assert x2.grad == -0.25


def test_reciprocal():
    x = ad.Var(2)
    y = op.reciprocal(x)
    y.gradient()

    assert x.grad == -0.25


def test_neg():
    x = ad.Var(2)
    y = -x
    y.gradient()

    assert x.grad == -1

    # advanced
    x1, x2 = ad.Var(1), ad.Var(2)
    x3 = x1 + (-x2)
    x3.gradient()

    assert x1.grad == 1
    assert x2.grad == -1


def test_mixed():
    pass


def test_topo1():
    x1, x2, x3 = ad.Var(1), ad.Var(2), ad.Var(3)
    y1 = x1 * x2
    y2 = x2 + x3
    y3 = y1 / y2
    y3.gradient()

    for var in [x1, x2, x3, y1, y2, y3]:
        assert not var._engine.is_working()

    assert math.isclose(x1.grad, 0.4)
    assert math.isclose(x2.grad, 0.12)
    assert math.isclose(x3.grad, -0.08)


def test_topo2():
    pass
