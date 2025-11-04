#%%
import numpy as np

from mqt.yaqs.noise_char.propagation import PropagatorWithGradients
# from auxiliar.scikit_tt_propagator_with_gradients import PropagatorWithGradients

from mqt.yaqs.noise_char.optimization import LossClass


from mqt.yaqs.noise_char.characterizer import Characterizer



from pathlib import Path



from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import  CompactNoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable


from mqt.yaqs.core.libraries.gate_library import X, Y, Z, Create, Destroy


from scipy.optimize import minimize
from scipy.optimize._optimize import _minimize_neldermead
#%%


#%%


if __name__ == '__main__':



    work_dir=f"test/nelder_mead_opt"

    work_dir_path = Path(work_dir)

    work_dir_path.mkdir(parents=True, exist_ok=True)

    for file in work_dir_path.iterdir():  # iterate over all items in folder
        if file.is_file():         # only remove files, not subfolders
            file.unlink()   



    ## Defining Hamiltonian and observable list
    L=2

    J=1
    g=1


    H_0 = MPO()
    H_0.init_ising(L, J, g)


    # Define the initial state
    init_state = MPS(L, state='zeros')


    # obs_list = [Observable(X(), site) for site in range(L)]  + [Observable(Y(), site) for site in range(L)] + [Observable(Z(), site) for site in range(L)]
    obs_list = [Observable(Y(), site) for site in range(1)]

    noise_operator = "pauli_z"


    #%%
    ## Defining simulation parameters

    T=6

    dt=0.1

    N=1000

    max_bond_dim=8

    threshold=1e-6

    order=1

    sim_params = AnalogSimParams(observables=obs_list, elapsed_time=T, dt=dt, num_traj=N, max_bond_dim=max_bond_dim, threshold=threshold, order=order, sample_timesteps=True)





    #%%


    #%%
    ## Defining reference noise model and reference trajectory
    gamma_reference = 0.1
    # ref_noise_model =  CompactNoiseModel([{"name": "lowering", "sites": [i for i in range(L)], "strength": gamma_rel}] + [{"name": "pauli_z", "sites": [i for i in range(L)], "strength": gamma_deph}])
    ref_noise_model =  CompactNoiseModel( [{"name": noise_operator, "sites": [i for i in range(1)], "strength": gamma_reference} ])

    # ref_noise_model =  CompactNoiseModel([{"name": noise_operator, "sites": [i], "strength": gamma_rel} for i in range(L)] )


    ## Writing reference gammas to file
    np.savetxt(work_dir + "/gammas.txt", ref_noise_model.strength_list, header="##", fmt="%.6f")


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
    gamma_guess=0.3
    sim_params.num_traj=int(100)

    # guess_noise_model =  CompactNoiseModel([{"name": "lowering", "sites": [i for i in range(L)], "strength": gamma_rel_guess} ] + [{"name": "pauli_z", "sites": [i for i in range(L)], "strength": gamma_deph_guess} ])
    guess_noise_model =  CompactNoiseModel( [{"name": noise_operator, "sites": [i for i in range(1)], "strength": gamma_guess} ])
    # guess_noise_model =  CompactNoiseModel([{"name": noise_operator, "sites": [i], "strength": gamma_rel_guess} for i in range(L)] )


    opt_propagator = PropagatorWithGradients(
        sim_params=sim_params,
        hamiltonian=H_0,
        compact_noise_model=guess_noise_model,
        init_state=init_state
    )

    loss=LossClass(
            ref_traj=ref_traj, traj_gradients=opt_propagator, working_dir=work_dir, 
            print_to_file=True, return_gradients=False
        )


    # res = minimize(loss, guess_noise_model.strength_list, method='Nelder-Mead',
    #            options={'disp': True, 
    #                     'adaptive': True,
    #                     'xatol': 1e-10,        # tolerance in x
    #                     'fatol': 1e-10,   
    #                     'maxiter': 100})
    

    
    res = _minimize_neldermead(
        loss, guess_noise_model.strength_list,
        options={
            'maxiter': 1000,
            'xatol': 1e-6,
            'fatol': 1e-6,
            'alpha': 1.0,   # reflection
            'gamma': 2.0,   # expansion
            'rho': 0.8,     # contraction (higher = slower shrink)
            'sigma': 0.8,   # shrink (higher = slower reduction)
            'disp': True,
        }
    )


