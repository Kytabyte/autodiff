# autodiff

This is a simple eager mode [automatic differentiation](https://en.wikipedia.org/wiki/Automatic_differentiation) engine written in
Python working on scalar type (Python's Real Number).

The motivation to write this project is to give myself a solid understand of how automatic differentiation works in training a neural network. This project also is also good for education to better understand the rationale behind the autodiff.

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
