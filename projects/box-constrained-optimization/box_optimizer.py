import numpy as np
from scipy.optimize import minimize


def box_constrained_optimization(
    objective_function,
    initial_guess,
    lower_bounds,
    upper_bounds,
):
    """
    Perform box-constrained optimization using the L-BFGS-B algorithm.

    Parameters
    ----------
    objective_function : callable
        Function to minimize. It must accept a NumPy array and return
        a scalar value.
    initial_guess : array_like
        Initial point for the optimization.
    lower_bounds : array_like
        Lower bound for each optimization variable.
    upper_bounds : array_like
        Upper bound for each optimization variable.

    Returns
    -------
    scipy.optimize.OptimizeResult
        Optimization result returned by scipy.optimize.minimize.

    Raises
    ------
    ValueError
        If the inputs have incompatible dimensions, invalid bounds,
        or the initial guess is outside the bounds.
    """

    x0 = np.asarray(initial_guess, dtype=float)
    lb = np.asarray(lower_bounds, dtype=float)
    ub = np.asarray(upper_bounds, dtype=float)

    if x0.ndim != 1 or lb.ndim != 1 or ub.ndim != 1:
        raise ValueError(
            "initial_guess and bounds must be one-dimensional."
        )

    if not (x0.shape == lb.shape == ub.shape):
        raise ValueError(
            "initial_guess, lower_bounds, and upper_bounds "
            "must have the same shape."
        )

    if np.any(lb > ub):
        raise ValueError(
            "Each lower bound must be less than or equal "
            "to its upper bound."
        )

    if np.any((x0 < lb) | (x0 > ub)):
        raise ValueError(
            "initial_guess must lie within the provided bounds."
        )

    bounds = list(zip(lb, ub))

    return minimize(
        objective_function,
        x0,
        method="L-BFGS-B",
        bounds=bounds,
    )


if __name__ == "__main__":

    # Minimize:
    #
    # f(x, y) = (x - 2)^2 + (y - 3)^2
    #
    # subject to:
    #
    # 0 <= x <= 4
    # 1 <= y <= 5

    def objective(x):
        return (x[0] - 2) ** 2 + (x[1] - 3) ** 2

    initial_guess = [0.0, 1.0]

    lower_bounds = [0.0, 1.0]
    upper_bounds = [4.0, 5.0]

    result = box_constrained_optimization(
        objective,
        initial_guess,
        lower_bounds,
        upper_bounds,
    )

    print("\nOptimization Result")
    print("-------------------")
    print(f"Solution:        {result.x}")
    print(f"Objective value: {result.fun}")
    print(f"Success:         {result.success}")
    print(f"Message:         {result.message}")
