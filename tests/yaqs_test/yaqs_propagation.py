#%%
import numpy as np
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from mqt.yaqs.noise_char.propagation import Propagator
from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import CompactNoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
from mqt.yaqs.core.libraries.gate_library import X, Y, Z


def run_single_trajectory(args):
    k, L, work_dir, noise_model = args
    print(f"[traj {k}] Starting ...", flush=True)

    T = 6
    J = 1
    g = 1
    dt = 0.1
    max_bond_dim = 8
    threshold = 1e-6
    order = 1

    H_0=MPO.ising(L, J, g)

    init_state = MPS(L, state='zeros')

    obs_list = (
        [Observable(X(), site) for site in range(L)] +
        [Observable(Y(), site) for site in range(L)] +
        [Observable(Z(), site) for site in range(L)]
    )

    sim_params = AnalogSimParams(
        observables=obs_list, elapsed_time=T, dt=dt, num_traj=1,
        max_bond_dim=max_bond_dim, threshold=threshold, order=order,
        sample_timesteps=True,
    )

    propagator = Propagator(
        sim_params=sim_params,
        hamiltonian=H_0,
        compact_noise_model=noise_model,
        init_state=init_state,
    )
    propagator.set_observable_list(obs_list)
    propagator.run(noise_model, 1)
    output_file = Path(work_dir) / f"traj_{k}.txt"
    propagator.write_traj(output_file)
    print(f"[traj {k}] Written to {output_file}", flush=True)


if __name__ == '__main__':

    L        = int(sys.argv[1])
    N        = int(sys.argv[2])
    work_dir = sys.argv[3]
    gamma_x  = float(sys.argv[4])
    gamma_y  = float(sys.argv[5])
    gamma_z  = float(sys.argv[6])
    gamma_ct = float(sys.argv[7])
    n_cpus   = int(sys.argv[8])
    neighbor_radius = 4

    work_dir_path = Path(work_dir)
    work_dir_path.mkdir(parents=True, exist_ok=True)

    ## Defining the noise model
    proc = [
        {"name": "pauli_x", "sites": [i for i in range(L)], "strength": gamma_x},
        {"name": "pauli_y", "sites": [i for i in range(L)], "strength": gamma_y},
        {"name": "pauli_z", "sites": [i for i in range(L)], "strength": gamma_z},
    ]
    for r in range(1, neighbor_radius + 1):
        proc += [{"name": "crosstalk_zz", "sites": [[i, i + r] for i in range(L - r)], "strength": gamma_ct}]
    noise_model = CompactNoiseModel(proc)

    ## Writing reference gammas to file
    np.savetxt(work_dir + "/gammas.txt", noise_model.strength_list, header="##", fmt="%.6f")

    print(f"Computing {N} trajectories on {n_cpus} CPUs ...")

    args = [(k, L, work_dir, noise_model) for k in range(N)]

    with ProcessPoolExecutor(max_workers=n_cpus) as executor:
        for _ in executor.map(run_single_trajectory, args):
            pass

    print(f"Written {N} individual trajectories to {work_dir}.")
