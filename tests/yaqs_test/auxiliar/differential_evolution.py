import numpy as np
from multiprocessing import Pool



import numpy as np
from multiprocessing import Pool

def _evaluate_candidate(args):
    f, x, noise_averaging = args
    if noise_averaging > 1:
        return np.mean([f(x) for _ in range(noise_averaging)])
    return f(x)

def differential_evolution_opt(
    f,
    x_low,
    x_up,
    pop_size=20,
    F=0.8,
    Cr=0.7,
    max_iter=100,
    tol=1e-6,
    workers=1,
    noise_averaging=1,
    seed=None,
    verbose=True,
):
    """
    Differential Evolution (DE) for MINIMIZATION of a black-box function,
    with convergence based on fitness and parameter stabilization.

    Args:
        f : Callable[[np.ndarray], float]
            Function to minimize.
        x_low, x_up : float or array-like
            Lower and upper bounds for each dimension.
        tol : float
            Tolerance for fitness change.
        tol_x : float
            Tolerance for x stabilization (mean std across dimensions < tol_x)
    """

    rng = np.random.default_rng(seed)
    x_low = np.array(x_low, dtype=float)
    x_up = np.array(x_up, dtype=float)
    d = len(x_low)

    # Initialize population uniformly within bounds
    pop = rng.uniform(low=x_low, high=x_up, size=(pop_size, d))

    # Helper to evaluate population
    def eval_pop(P):
        if workers > 1:
            with Pool(workers) as pool:
                vals = pool.map(_evaluate_candidate, [(f, x, noise_averaging) for x in P])
        else:
            vals = [_evaluate_candidate((f, x, noise_averaging)) for x in P]
        return np.array(vals)

    fitness = eval_pop(pop)
    best_idx = np.argmin(fitness)
    best_x = pop[best_idx].copy()
    best_f = fitness[best_idx]
    history = [best_f]

    for gen in range(max_iter):
        for i in range(pop_size):
            # Mutation
            idxs = np.arange(pop_size)
            idxs = idxs[idxs != i]
            a, b, c = pop[rng.choice(idxs, 3, replace=False)]

            mutant = a + F * (b - c)
            mutant = np.clip(mutant, x_low, x_up)

            # Crossover
            cross_points = rng.random(d) < Cr
            if not np.any(cross_points):
                cross_points[rng.integers(0, d)] = True
            trial = np.where(cross_points, mutant, pop[i])
            trial = np.clip(trial, x_low, x_up)

            f_trial = _evaluate_candidate((f, trial, noise_averaging))

            # Selection
            if f_trial < fitness[i]:
                pop[i] = trial
                fitness[i] = f_trial
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()

        history.append(best_f)

        # --- Convergence checks ---
        x_stable = np.mean(np.std(pop, axis=0)) < tol  # population spread small

        if x_stable:
            if verbose:
                print(f"Converged  (x stabilized) at iteration {gen}.")
            break


        if f.converged:
            print(f"Average stable at iteration {f.n_eval}.")
            break

    return best_x, best_f, history


