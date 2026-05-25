#%%
import sys

import numpy as np

from mqt.yaqs.noise_char.loss import LossClass
from mqt.yaqs.noise_char.propagation import Propagator

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable

from mqt.yaqs.core.libraries.gate_library import X, Y, Z

#%%

L = 2
T = 2
J = 1
g = 1
dt = 0.1
max_bond_dim = 8
threshold = 1e-6
order = 1
num_traj = 100
n_neumann = 2
epsilon = 1e-2

gamma_reference = 0.01

#%%

H_0 = MPO.ising(L, J, g)

init_state = MPS(L, state='zeros')

obs_list = (
    [Observable(X(), site) for site in range(L)]
    + [Observable(Y(), site) for site in range(L)]
    + [Observable(Z(), site) for site in range(L)]
)

sim_params = AnalogSimParams(
    observables=obs_list,
    elapsed_time=T,
    dt=dt,
    num_traj=num_traj,
    max_bond_dim=max_bond_dim,
    threshold=threshold,
    order=order,
    sample_timesteps=True,
)

#%%

ref_proc = (
    [{"name": "pauli_x", "sites": [i], "strength": gamma_reference} for i in range(L)]
    + [{"name": "pauli_y", "sites": [i], "strength": gamma_reference} for i in range(L)]
    + [{"name": "pauli_z", "sites": [i], "strength": gamma_reference} for i in range(L)]
)
ref_noise_model = CompactNoiseModel(ref_proc)

ref_propagator = Propagator(
    sim_params=sim_params,
    hamiltonian=H_0,
    compact_noise_model=ref_noise_model,
    init_state=init_state,
)

ref_propagator.set_observable_list(obs_list)

print("Computing reference trajectory...")
ref_propagator.run(ref_noise_model, n_neumann)
ref_traj = ref_propagator.obs_traj
print("Done.")

#%%

gamma_eval = np.array([0.02] * (3 * L))

eval_proc = (
    [{"name": "pauli_x", "sites": [i], "strength": gamma_eval[i]}         for i in range(L)]
    + [{"name": "pauli_y", "sites": [i], "strength": gamma_eval[i + L]}   for i in range(L)]
    + [{"name": "pauli_z", "sites": [i], "strength": gamma_eval[i + 2*L]} for i in range(L)]
)
eval_noise_model = CompactNoiseModel(eval_proc)

opt_propagator = Propagator(
    sim_params=sim_params,
    hamiltonian=H_0,
    compact_noise_model=eval_noise_model,
    init_state=init_state,
    compute_gradient_obs=True,
)

#%%

scan_index = int(sys.argv[1]) if len(sys.argv) > 1 else 0

loss = LossClass(
    ref_traj=ref_traj,
    propagator=opt_propagator,
    num_traj=lambda _: num_traj,
    n_neumann=n_neumann,
    epsilon=epsilon,
    compare_gradients=True,
)

print(f"\nEvaluating at x = {gamma_eval}")
f, grad, sim_time = loss(gamma_eval)

print(f"\nloss = {f:.6e}  (sim_time = {sim_time:.1f}s)")
print("\nGradient comparison:")
print(f"  {'param':<8} {'analytical':>14} {'numeric':>14} {'rel_err':>12}")
for i in range(len(gamma_eval)):
    rel_err = abs(loss.analytical_grad[i] - loss.numeric_grad[i]) / (abs(loss.numeric_grad[i]) + 1e-30)
    print(f"  x[{i}]     {loss.analytical_grad[i]:>14.6e} {loss.numeric_grad[i]:>14.6e} {rel_err:>12.3e}")

#%%

gamma_scan = np.arange(0, 0.055, 0.005)
scan_results = np.zeros((len(gamma_scan), 2))

scan_loss = LossClass(
    ref_traj=ref_traj,
    propagator=opt_propagator,
    num_traj=lambda _: num_traj,
    n_neumann=n_neumann,
)

print(f"\nScanning gamma[{scan_index}] from {gamma_scan[0]:.3f} to {gamma_scan[-1]:.3f}...")
for k, g_val in enumerate(gamma_scan):
    x_scan = gamma_eval.copy()
    x_scan[scan_index] = g_val
    f_scan, _, _ = scan_loss(x_scan)
    scan_results[k] = [g_val, f_scan]
    print(f"  gamma={g_val:.4f}  loss={f_scan:.6e}")

scan_file = f"loss_scan_gamma{scan_index}.txt"
np.savetxt(scan_file, scan_results, header=f"gamma[{scan_index}]  loss", fmt="%.6f  %.6e")
print(f"\nScan saved to {scan_file}")

# %%
