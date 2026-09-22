# Box-Constrained Optimization

A small Python example for solving box-constrained optimization problems with SciPy's `L-BFGS-B` algorithm.

## Problem Form

This project solves optimization problems of the form:

```text
minimize    f(x)
subject to  lower_bounds <= x <= upper_bounds
```

Each optimization variable can have its own lower and upper bound.

## Features

- Uses `scipy.optimize.minimize`
- Uses the `L-BFGS-B` method for bound-constrained optimization
- Validates input dimensions and bound consistency
- Checks that the initial guess lies inside the feasible region
- Includes a runnable two-variable example

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/jorsacademy/box-constrained-optimization.git
cd box-constrained-optimization
pip install -r requirements.txt
```

## Usage

Run the example:

```bash
python box_optimizer.py
```

The example minimizes:

```text
f(x, y) = (x - 2)^2 + (y - 3)^2
```

subject to:

```text
0 <= x <= 4
1 <= y <= 5
```

The expected solution is approximately:

```text
x = [2.0, 3.0]
objective value = 0.0
```

## Using the Function

```python
from box_optimizer import box_constrained_optimization


def objective(x):
    return (x[0] - 2) ** 2 + (x[1] - 3) ** 2


result = box_constrained_optimization(
    objective_function=objective,
    initial_guess=[0.0, 1.0],
    lower_bounds=[0.0, 1.0],
    upper_bounds=[4.0, 5.0],
)

print(result.x)
print(result.fun)
```

## Requirements

- Python 3.9+
- NumPy
- SciPy

## Algorithm

`L-BFGS-B` is a limited-memory quasi-Newton optimization algorithm designed for problems with simple lower and upper bounds on variables. It is especially useful for optimization problems with many variables where storing a full Hessian approximation would be expensive.
