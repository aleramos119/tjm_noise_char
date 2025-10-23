#%%
import matplotlib.pyplot as plt
import numpy as np

from mqt.yaqs.noise_char.propagation import PropagatorWithGradients
from mqt.yaqs.noise_char.characterizer import Characterizer



from pathlib import Path



from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel, CompactNoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable


from mqt.yaqs.core.libraries.gate_library import X, Y, Z, Create, Destroy

# from auxiliar.write import *

import sys



def write_gammas( noise_model: CompactNoiseModel, file_name):
    
    gammas = []

    for proc in noise_model.compact_processes:

        gammas.append(proc["strength"])
    

    np.savetxt(file_name, gammas, header="##", fmt="%.6f")


#%%


#%%


if __name__ == '__main__':
#%%


    work_dir=f"test/characterizer/"

    work_dir_path = Path(work_dir)

    work_dir_path.mkdir(parents=True, exist_ok=True)



    ## Defining Hamiltonian and observable list
    L=3

    J=1
    g=0.5


    H_0 = MPO()
    H_0.init_ising(L, J, g)


    # Define the initial state
    init_state = MPS(L, state='zeros')


    # obs_list = [Observable(X(), site) for site in range(L)]  + [Observable(Y(), site) for site in range(L)] + [Observable(Z(), site) for site in range(L)]
    obs_list = [Observable(Y(), site) for site in range(L)]



    #%%
    ## Defining simulation parameters

    T=3

    dt=0.1

    N=1

    max_bond_dim=8

    threshold=1e-6

    order=2

    sim_params = AnalogSimParams(observables=obs_list, elapsed_time=T, dt=dt, num_traj=N, max_bond_dim=max_bond_dim, threshold=threshold, order=order, sample_timesteps=True)





    #%%


    #%%
    ## Defining reference noise model and reference trajectory
    gamma_rel = 0.1

    gamma_deph = 0.1
    # ref_noise_model =  CompactNoiseModel([{"name": "lowering", "sites": [i for i in range(L)], "strength": gamma_rel}] + [{"name": "pauli_z", "sites": [i for i in range(L)], "strength": gamma_deph}])
    ref_noise_model =  CompactNoiseModel( [{"name": "pauli_z", "sites": [i for i in range(L)], "strength": gamma_deph} ])

    # ref_noise_model =  CompactNoiseModel([{"name": noise_operator, "sites": [i], "strength": gamma_rel} for i in range(L)] )


    ## Writing reference gammas to file
    np.savetxt(work_dir + "gammas.txt", ref_noise_model.strength_list, header="##", fmt="%.6f")


    propagator = PropagatorWithGradients(
        sim_params=sim_params,
        hamiltonian=H_0,
        compact_noise_model=ref_noise_model,
        init_state=init_state
    )

    propagator.set_observable_list(obs_list)


    print("Computing reference trajectory ... ")

    propagator.run(ref_noise_model)

    ref_traj = propagator.obs_traj


    print("Reference trajectory computed.")




    




    # for i in range(len(propagator.obs_traj)):
    #     plt.plot( sim_params.times, propagator.obs_array[i], label=f"obs_{str(i)}_gamma_deph_{gamma_deph}")

    # plt.legend()
    # plt.savefig(f"L_{L}_N_{N}_gamma_rel_{gamma_rel}_gamma_deph_{gamma_deph}.png")
    # plt.show()



    #%% Optimizing the model
    gamma_rel_guess=0.6
    gamma_deph_guess=0.4
    sim_params.num_traj=int(1)

    # guess_noise_model =  CompactNoiseModel([{"name": "lowering", "sites": [i for i in range(L)], "strength": gamma_rel_guess} ] + [{"name": "pauli_z", "sites": [i for i in range(L)], "strength": gamma_deph_guess} ])
    guess_noise_model =  CompactNoiseModel( [{"name": "pauli_z", "sites": [i for i in range(L)], "strength": gamma_deph_guess} ])
    # guess_noise_model =  CompactNoiseModel([{"name": noise_operator, "sites": [i], "strength": gamma_rel_guess} for i in range(L)] )


    characterizer = Characterizer(
        sim_params=sim_params,
        hamiltonian=H_0,
        init_guess=guess_noise_model,
        init_state=init_state,
        ref_traj=ref_traj,
        work_dir=work_dir
    )


    print("Optimizing ... ")

    characterizer.adam_optimize(max_iterations=100)


