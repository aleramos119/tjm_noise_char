import numpy as np
from multiprocessing import Pool



def _evaluate_candidate(args):
    """Helper to evaluate possibly noisy function multiple times."""
    f, x, noise_averaging = args
    vals = [f(x) for _ in range(noise_averaging)]
    return np.mean(vals)


def differential_evolution_opt(
    f,
    bounds,
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
    Differential Evolution (DE) for MINIMIZATION of a black-box function.

    Args:
        f : Callable[[np.ndarray], float]
            Function to minimize. Must accept 1D NumPy array (x) and return scalar.
        bounds : list of tuples [(x1_min, x1_max), ..., (xd_min, xd_max)]
        pop_size : int
            Population size (recommended: 5–10 × num_dimensions)
        F : float
            Differential weight (0 < F ≤ 2)
        Cr : float
            Crossover probability (0 ≤ Cr ≤ 1)
        max_iter : int
            Maximum iterations (generations)
        tol : float
            Stop if best fitness improvement < tol
        workers : int
            Number of parallel processes for evaluation (1 = serial)
        noise_averaging : int
            If >1, average multiple evaluations per candidate to reduce noise
        seed : int or None
            Random seed for reproducibility
        verbose : bool
            Print progress each iteration

    Returns:
        best_x : np.ndarray
            Best solution found
        best_f : float
            Function value at best_x
        history : list
            Best fitness value at each iteration
    """
    rng = np.random.default_rng(seed)
    d = len(bounds)

    # Initialize population uniformly within bounds
    pop = np.array([rng.uniform(low, high, size=pop_size) for low, high in np.array(bounds)]).T
    # Evaluate population
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
            # Mutation: choose 3 random, distinct indices
            idxs = np.arange(pop_size)
            idxs = idxs[idxs != i]
            a, b, c = pop[rng.choice(idxs, 3, replace=False)]

            # Mutant vector
            mutant = np.clip(a + F * (b - c), [b[0] for b in bounds], [b[1] for b in bounds])

            # Crossover
            cross_points = rng.random(d) < Cr
            if not np.any(cross_points):
                cross_points[rng.integers(0, d)] = True
            trial = np.where(cross_points, mutant, pop[i])

            # Ensure within bounds
            trial = np.clip(trial, [b[0] for b in bounds], [b[1] for b in bounds])

            # Evaluate trial
            f_trial = _evaluate_candidate((f, trial, noise_averaging))

            # Selection
            if f_trial < fitness[i]:
                pop[i] = trial
                fitness[i] = f_trial

                # Update global best
                if f_trial < best_f:
                    best_f = f_trial
                    best_x = trial.copy()

        history.append(best_f)

        if verbose and gen % max(1, max_iter // 20) == 0:
            print(f"Iter {gen:03d}: best f = {best_f:.6f}")

        # Convergence check
        if len(history) > 5 and abs(history[-5] - best_f) < tol:
            if verbose:
                print("Converged.")
            break

    return best_x, best_f, history

