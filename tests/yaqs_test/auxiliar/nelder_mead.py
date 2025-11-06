#%%
import numpy as np



def nelder_mead_opt(
    func,
    x_start,
    step=0.1,
    tol=1e-6,
    max_iter=500,
    alpha=1.0,    # reflection
    gamma=2.0,    # expansion
    rho=0.5,      # contraction
    sigma=0.5     # shrink
):
    """
    Basic Nelder-Mead optimizer (unconstrained)
    """

    n = len(x_start)
    # Construct the initial simplex
    simplex = np.zeros((n + 1, n))
    simplex[0] = x_start
    for i in range(n):
        y = np.copy(x_start)
        y[i] = y[i] + step
        simplex[i + 1] = y

    # Evaluate function at simplex points
    f_values = np.apply_along_axis(func, 1, simplex)

    for iteration in range(max_iter):
        # Order by function value
        order = np.argsort(f_values)
        simplex = simplex[order]
        f_values = f_values[order]

        # Check convergence (simple criterion)
        if np.std(f_values) < tol:
            break

        # Centroid of best n points
        centroid = np.mean(simplex[:-1], axis=0)

        # Reflection
        x_r = centroid + alpha * (centroid - simplex[-1])
        f_r = func(x_r)

        if f_values[0] <= f_r < f_values[-2]:
            simplex[-1] = x_r
            f_values[-1] = f_r
            continue

        # Expansion
        if f_r < f_values[0]:
            x_e = centroid + gamma * (x_r - centroid)
            f_e = func(x_e)
            if f_e < f_r:
                simplex[-1] = x_e
                f_values[-1] = f_e
            else:
                simplex[-1] = x_r
                f_values[-1] = f_r
            continue

        # Contraction
        x_c = centroid + rho * (simplex[-1] - centroid)
        f_c = func(x_c)
        if f_c < f_values[-1]:
            simplex[-1] = x_c
            f_values[-1] = f_c
            continue

        # Shrink
        for i in range(1, n + 1):
            simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
            f_values[i] = func(simplex[i])

    return {
        "x": simplex[0],
        "fun": f_values[0],
        "nit": iteration + 1,
        "simplex": simplex,
        "success": iteration < max_iter - 1
    }




