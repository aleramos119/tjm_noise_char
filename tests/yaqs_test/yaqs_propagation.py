#%%
import numpy as np

from mqt.yaqs.noise_char.propagation import Propagator, PropagatorWithGradients
# from auxiliar.scikit_tt_propagator_with_gradients import PropagatorWithGradients

from mqt.yaqs.noise_char.optimization import LossClass


from mqt.yaqs.noise_char.characterizer import Characterizer



from pathlib import Path



from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import  CompactNoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable


from mqt.yaqs.core.libraries.gate_library import X, Y, Z, Create, Destroy

import sys

#%%


#%%


if __name__ == '__main__':

    T=6

    obs="Z"

    noise="X"


    L=int(sys.argv[1])



    N=int(sys.argv[2])



    work_dir=sys.argv[3]

    work_dir_path = Path(work_dir)

    work_dir_path.mkdir(parents=True, exist_ok=True)



    ## Defining Hamiltonian and observable list
    

    J=1
    g=1


    H_0 = MPO()
    H_0.init_ising(L, J, g)


    # Define the initial state
    init_state = MPS(L, state='zeros')


    if obs=="X":
        obs=X()
    
    if obs=="Y":
        obs=Y()

    if obs=="Z":
        obs=Z()

    if noise=="X":
        noise_operator="pauli_x"
    
    if noise=="Y":
        noise_operator="pauli_y"

    if noise=="Z":
        noise_operator="pauli_z"


    obs_list = [Observable(X(), site) for site in range(L)]  + [Observable(Y(), site) for site in range(L)] + [Observable(Z(), site) for site in range(L)]
    # obs_list = [Observable(obs, site) for site in range(L)]


    #%%
    ## Defining simulation parameters

    dt=0.1

    max_bond_dim=8

    threshold=1e-6

    order=1

    sim_params = AnalogSimParams(observables=obs_list, elapsed_time=T, dt=dt, num_traj=N, max_bond_dim=max_bond_dim, threshold=threshold, order=order, sample_timesteps=True)





    #%%


    #%%
    ## Defining reference noise model and reference trajectory
    gamma_reference = 0.01
    # ref_noise_model =  CompactNoiseModel([{"name": "lowering", "sites": [i for i in range(L)], "strength": gamma_rel}] + [{"name": "pauli_z", "sites": [i for i in range(L)], "strength": gamma_deph}])
    # ref_noise_model =  CompactNoiseModel( [{"name": noise_operator, "sites": [i for i in range(L)], "strength": gamma_reference} ])
    ref_noise_model =  CompactNoiseModel([{"name": "pauli_x", "sites": [i for i in range(L)], "strength": gamma_reference} ] + 
                                         [{"name": "pauli_y", "sites": [i for i in range(L)], "strength": gamma_reference} ] +
                                         [{"name": "pauli_z", "sites": [i for i in range(L)], "strength": gamma_reference} ])


    # ref_noise_model =  CompactNoiseModel([{"name": noise_operator, "sites": [i], "strength": gamma_rel} for i in range(L)] )


    ## Writing reference gammas to file
    np.savetxt(work_dir + "/gammas.txt", ref_noise_model.strength_list, header="##", fmt="%.6f")


    propagator = Propagator(
        sim_params=sim_params,
        hamiltonian=H_0,
        compact_noise_model=ref_noise_model,
        init_state=init_state
    )

    propagator.set_observable_list(obs_list)


    print("Computing reference trajectory ... ")

    propagator.run(ref_noise_model)

    ref_traj = propagator.obs_traj

    propagator.write_traj(work_dir_path/"ref_traj.txt")

    print("Reference trajectory computed.")



