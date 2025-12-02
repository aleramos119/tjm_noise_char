#%%
import numpy as np

from mqt.yaqs.noise_char.propagation import PropagatorWithGradients, Propagator
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

def lineal_function_100_2(i):
    return 100 + 2*i

def lineal_function_400_2(i):
    return 400 + 2*i

def lineal_function_800_2(i):
    return 800 + 2*i


def lineal_function_1000_2(i):
    return 1000 + 2*i

def lineal_function_4000(i):
    return 4000

def lineal_function_1000(i):
    return 1000
#%%


if __name__ == '__main__':

    L=int(sys.argv[1])

    const = float(sys.argv[2])

    method = sys.argv[3]

    params = sys.argv[4]

    x_lim = float(sys.argv[5])

    work_dir=sys.argv[6]

    work_dir_path = Path(work_dir)

    work_dir_path.mkdir(parents=True, exist_ok=True)



    ## Defining Hamiltonian and observable list

    T=6

    J=1
    g=1


    H_0 = MPO()
    H_0.init_ising(L, J, g)


    # Define the initial state
    init_state = MPS(L, state='zeros')


    obs_list = [Observable(X(), site) for site in range(L)]  + [Observable(Y(), site) for site in range(L)] + [Observable(Z(), site) for site in range(L)]



    #%%
    ## Defining simulation parameters

    dt=0.1

    max_bond_dim=8

    threshold=1e-6

    order=1

    sim_params = AnalogSimParams(observables=obs_list, elapsed_time=T, dt=dt, num_traj=4000, max_bond_dim=max_bond_dim, threshold=threshold, order=order, sample_timesteps=True)





    #%%


    #%%
    ## Defining reference noise model and reference trajectory
    gamma_reference = 0.01
    if params=="d_3":
        ref_noise_model =  CompactNoiseModel( [{"name": "pauli_x", "sites": [i for i in range(L)], "strength": gamma_reference} ] 
                                            + [{"name": "pauli_y", "sites": [i for i in range(L)], "strength": gamma_reference} ] 
                                            + [{"name": "pauli_z", "sites": [i for i in range(L)], "strength": gamma_reference} ])
    if params=="d_3L":
        ref_noise_model =  CompactNoiseModel( [{"name": "pauli_x", "sites": [i], "strength": gamma_reference} for i in range(L)]
                                            + [{"name": "pauli_y", "sites": [i], "strength": gamma_reference} for i in range(L)] 
                                            + [{"name": "pauli_z", "sites": [i], "strength": gamma_reference} for i in range(L)])


    d = len(ref_noise_model.strength_list)

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

    propagator.write_traj(work_dir_path / "ref_traj.txt")

    print("Reference trajectory computed.")




    #%% Optimizing the model
    x_low=np.array([0]*d)
    x_up=np.array([x_lim]*d)


    gamma_guess=x_low + np.random.rand(*x_low.shape) * (x_up - x_low)


    if params=="d_3":
        guess_noise_model =  CompactNoiseModel( [{"name": "pauli_x", "sites": [i for i in range(L)], "strength": gamma_guess[0]} ] 
                                            + [{"name": "pauli_y", "sites": [i for i in range(L)], "strength": gamma_guess[1]} ] 
                                            + [{"name": "pauli_z", "sites": [i for i in range(L)], "strength": gamma_guess[2]} ])
    if params=="d_3L":
        guess_noise_model =  CompactNoiseModel( [{"name": "pauli_x", "sites": [i], "strength": gamma_guess[i]} for i in range(L)]
                                            + [{"name": "pauli_y", "sites": [i], "strength": gamma_guess[i+L]} for i in range(L)] 
                                            + [{"name": "pauli_z", "sites": [i], "strength": gamma_guess[i+2*L]} for i in range(L)])


    opt_propagator = Propagator(
        sim_params=sim_params,
        hamiltonian=H_0,
        compact_noise_model=guess_noise_model,
        init_state=init_state
    )



    n_t = len(sim_params.times)

    n_obs = len(obs_list)

    N=int(np.ceil(const/(n_t*n_obs)))
    
    def n_traj_func(i):
        return N


    loss=LossClass(
            ref_traj=ref_traj, propagator=opt_propagator, num_traj = n_traj_func, working_dir=work_dir, print_to_file=True
        )


    characterizer = Characterizer(
        propagator=opt_propagator,
        init_guess=guess_noise_model,
        loss=loss,
    )


    print("Optimizing ... ")

    if method == "cma":

        characterizer.cma_optimize(x_low=x_low, x_up = x_up)

    if method == "bayesian":
        
        characterizer.bayesian_optimize(x_low=x_low, x_up = x_up, n_init=3*d)

    if method == "adam":

        characterizer.adam_optimize(x_low=x_low, x_up = x_up)
    
    if method == "gradient_descent":

        characterizer.gradient_descent_optimize(x_low=x_low, x_up = x_up)  

    if method == "mcmc":

        characterizer.mcmc_optimize(x_low=x_low, x_up = x_up)

    print("Optimization complete.")


