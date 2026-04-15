#%%
import numpy as np
import matplotlib.pyplot as plt

from mqt.yaqs.noise_char.optimization_algorithms.gradient_based.adam import adam_opt

import os
os.environ["JAX_PLATFORMS"] = "cpu"  # prevent GPU auto-detection segfault

import jax
import jax.numpy as jnp

#%%
# Kraus map: rho' = sum_k K_k @ rho @ K_k†
def kraus_map(rho, kraus_ops):
    return sum(K @ rho @ K.conj().T for K in kraus_ops)



# Observable trajectory: returns <O>(t) for each step given gamma
def observable_trajectory(gamma, observable, n_steps=20, rho0=None):
    """
    Evolve a qubit under the amplitude damping channel and return
    the expectation value of `observable` at each time step.

    Args:
        gamma:      decay probability per step, in [0, 1]
        observable: 2x2 Hermitian operator (e.g. sigma_z)
        n_steps:    number of Kraus map applications
        rho0:       initial density matrix (defaults to |+><+|)

    Returns:
        trajectory: jnp.array of shape (n_steps + 1,) with <O>(t)
    """
    K0 = jnp.array([[1.0, 0.0],
                     [0.0, jnp.sqrt(1 - gamma)]])
    K1 = jnp.array([[0.0, jnp.sqrt(gamma)],
                     [0.0, 0.0]])
    kraus_ops = [K0, K1]

    if rho0 is None:
        rho0 = jnp.array([[0.5, 0.5],
                           [0.5, 0.5]])

    def expect(rho):
        return jnp.real(jnp.trace(observable @ rho))

    rho = rho0
    trajectory = [expect(rho)]
    for _ in range(n_steps):
        rho = kraus_map(rho, kraus_ops)
        trajectory.append(expect(rho))

    return jnp.array(trajectory)

#%%
# --- Example usage ---
gamma_ref=0.3

sigma_z = jnp.array([[1.0, 0.0], [0.0, -1.0]])
ref_traj = observable_trajectory(gamma=gamma_ref, observable=sigma_z)

#%%

def _cost_scalar(gamma):
    traj = observable_trajectory(gamma=gamma, observable=sigma_z)
    return jnp.sum((traj - ref_traj) ** 2)

_cost_and_grad = jax.value_and_grad(_cost_scalar)

def cost(gamma):
    cost_value, grad = _cost_and_grad(gamma)

    return cost_value, grad, None


def adam_optimizer(gamma_init, n_iter=200, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    """
    Adam optimizer for the cost function.

    Args:
        gamma_init: initial guess for gamma
        n_iter:     number of iterations
        lr:         learning rate (alpha)
        beta1:      exponential decay rate for 1st moment
        beta2:      exponential decay rate for 2nd moment
        eps:        numerical stability constant

    Returns:
        gamma:      optimized gamma
        history:    list of (cost_value, gamma) per iteration
    """
    gamma = float(gamma_init)
    m, v = 0.0, 0.0
    history = []

    for t in range(1, n_iter + 1):
        cost_value, grad, _ = cost(gamma)
        g = float(grad)

        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * g ** 2

        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)

        gamma = gamma - lr * m_hat / (v_hat ** 0.5 + eps)

        history.append((float(cost_value), gamma))

        print(f"Iter {t:4d} | cost = {float(cost_value):.6e} | gamma = {gamma:.6f}")

    return gamma, history


#%%
gamma_init = 0.7
gamma_opt, history = adam_optimizer(gamma_init)
print(f"True gamma: {gamma_ref}, Recovered gamma: {gamma_opt:.4f}")




# %%

numpy_his=np.array(history)
plt.plot(numpy_his[:,1])
# %%
