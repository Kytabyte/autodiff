import autodiff as ad
import autodiff.op as op

import pytest


def test_eq():
    x1 = ad.Var(3)
    x2 = ad.Var(3)
    y1 = 3.
    y2 = 3.

    assert x1 == x2 == y1 == y2


def test_cmp():
    x1, x2 = ad.Var(1), ad.Var(2)
    y = 3

    assert x1 < x2
    assert x2 > x1
    assert x1 < y
    assert y > x1
    assert x2 < y
    assert y > x2


def test_op():
    x1, x2 = ad.Var(1.), ad.Var(2)
    y = 3.

    # add, radd
    x3 = x1 + x2
    assert isinstance(x3, ad.Var)
    assert x3 == ad.Var(3)

    x3 = x1 + y
    assert isinstance(x3, ad.Var)
    assert x3 == ad.Var(4)

    x3 = y + x1
    assert isinstance(x3, ad.Var)
    assert x3 == ad.Var(4)

    # sub, rsub
    x3 = x1 - x2
    assert isinstance(x3, ad.Var)
    assert x3 == ad.Var(-1)

    x3 = x1 - y
    assert isinstance(x3, ad.Var)
    assert x3 == ad.Var(-2)

    x3 = y - x1
    assert isinstance(x3, ad.Var)
    assert x3 == ad.Var(2)

    # mul, rmul
    x3 = x1 * x2
    assert isinstance(x3, ad.Var)
    assert x3 == ad.Var(2)

    x3 = x1 * y
    assert isinstance(x3, ad.Var)
    assert x3 == ad.Var(3)

    x3 = y * x1
    assert isinstance(x3, ad.Var)
    assert x3 == ad.Var(3)

    # truediv, rtruediv
    x3 = x1 / x2
    assert isinstance(x3, ad.Var)
    assert x3 == ad.Var(0.5)

    x3 = x1 / y
    assert isinstance(x3, ad.Var)
    assert x3 == ad.Var(1/3)

    x3 = y / x1
    assert isinstance(x3, ad.Var)
    assert x3 == ad.Var(3)

    with pytest.raises(ZeroDivisionError):
        x3 = x1 / ad.Var(0)

    # neg
    x3 = -x1
    assert isinstance(x3, ad.Var)
    assert x3 == ad.Var(-1)
