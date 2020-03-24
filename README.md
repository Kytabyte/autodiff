# autodiff

This is an eager mode [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) engine written in
Python working on scalar type (Python Real Number).

## Usage

```python
import autodiff as ad

x1, x2 = ad.Var(2), ad.Var(3)
y = x1 * x2

y.backward()

print(x1.grad, x2.grad)
```
```
>>> 3, 2
```

## Requirements

Python 3.6+
