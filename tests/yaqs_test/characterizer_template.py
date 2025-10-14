#%%
import matplotlib.pyplot as plt
import numpy as np

from mqt.yaqs.noise_char.propagation import PropagatorWithGradients
from mqt.yaqs.noise_char.characterizer import Characterizer



from pathlib import Path



from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable


from mqt.yaqs.core.libraries.gate_library import X, Y, Z, Create, Destroy

# from auxiliar.write import *

import sys



def write_gammas( noise_model: NoiseModel, file_name):
    
    gammas = []

    for proc in noise_model.processes:

        gammas.append(proc["strength"])
    

    np.savetxt(file_name, gammas, header="##", fmt="%.6f")


#%%


#%%


if __name__ == '__main__':
#%%

    observable_list=[X(), Y(), Z()]

    noise_list=[ "pauli_x", "pauli_y", "pauli_z", "lowering", "raising"]


    for obs_operator in observable_list:
        for noise_operator in noise_list:

            work_dir=f"test/characterizer/scan/init_state_zeros/observable_{obs_operator.name}_noise_{noise_operator}"

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


            #obs_list = [Observable(X(), site) for site in range(L)]  + [Observable(Y(), site) for site in range(L)] + [Observable(Z(), site) for site in range(L)]
            obs_list = [Observable(obs_operator, site) for site in range(L)]



            #%%
            ## Defining simulation parameters

            T=5

            dt=0.1

            N=1000

            max_bond_dim=8

            threshold=1e-6

            order=2

            sim_params = AnalogSimParams(observables=obs_list, elapsed_time=T, dt=dt, num_traj=N, max_bond_dim=max_bond_dim, threshold=threshold, order=order, sample_timesteps=True)





            #%%


            #%%
            ## Defining reference noise model and reference trajectory
            gamma_rel = 0.1

            gamma_deph = 0.1
            # ref_noise_model =  NoiseModel([{"name": "lowering", "sites": [i], "strength": gamma_rel} for i in range(L)] + [{"name": "pauli_z", "sites": [i], "strength": gamma_deph} for i in range(L)])
            # ref_noise_model =  NoiseModel( [{"name": "pauli_z", "sites": [i], "strength": gamma_deph} for i in range(L)])

            ref_noise_model =  NoiseModel([{"name": noise_operator, "sites": [i], "strength": gamma_rel} for i in range(L)] )



            propagator = PropagatorWithGradients(
                sim_params=sim_params,
                hamiltonian=H_0,
                noise_model=ref_noise_model,
                init_state=init_state
            )

            propagator.set_observable_list(obs_list)

            propagator.run(ref_noise_model)

            ref_traj = propagator.obs_traj


            write_gammas(ref_noise_model, work_dir + "/gammas.txt")


            




            # for i in range(len(propagator.obs_traj)):
            #     plt.plot( sim_params.times, propagator.obs_array[i], label=f"obs_{str(i)}_gamma_deph_{gamma_deph}")

            # plt.legend()
            # plt.savefig(f"L_{L}_N_{N}_gamma_rel_{gamma_rel}_gamma_deph_{gamma_deph}.png")
            # plt.show()



            #%% Optimizing the model
            gamma_rel_guess=0.6
            gamma_deph_guess=0.4
            sim_params.num_traj=int(80)

            # guess_noise_model =  NoiseModel([{"name": "lowering", "sites": [i], "strength": gamma_rel_guess} for i in range(L)] + [{"name": "pauli_z", "sites": [i], "strength": gamma_deph_guess} for i in range(L)])
            # guess_noise_model =  NoiseModel( [{"name": "pauli_z", "sites": [i], "strength": gamma_deph_guess} for i in range(L)])
            guess_noise_model =  NoiseModel([{"name": noise_operator, "sites": [i], "strength": gamma_rel_guess} for i in range(L)] )


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


#%%
# on = Z() 
# lk = Destroy()

# res = lk.dag() * on * lk - 0.5 * on * lk.dag() * lk - 0.5 * lk.dag() * lk * on


# rho0= np.array([[1, 0], [0, 0]])
# rho1= np.array([[0, 0], [0, 1]])


# expected_value_0= np.trace(res.matrix @ rho0)
# expected_value_1= np.trace(res.matrix @ rho1)



# print("Result matrix: ",res.matrix, " Expected_value_0: ", expected_value_0, " Expected_value_1: ", expected_value_1 )
#%%

 
    # %%

    # import qutip as qt
    # t = np.arange(0, T + dt, dt) 

    # n_t = len(t)

    # '''QUTIP Initialization + Simulation'''

    # # Define Pauli matrices
    # sx = qt.sigmax()
    # sy = qt.sigmay()
    # sz = qt.sigmaz()

    # # Construct the Ising Hamiltonian
    # H = 0
    # for i in range(L-1):
    #     H += -J * qt.tensor([sz if n==i or n==i+1 else qt.qeye(2) for n in range(L)])
    # for i in range(L):
    #     H += -g * qt.tensor([sx if n==i else qt.qeye(2) for n in range(L)])



    # # Construct collapse operators
    # c_ops = []
    # gammas = []

    # # Relaxation operators
    # for i in range(L):
    #     c_ops.append(np.sqrt(gamma_rel[i]) * qt.tensor([qt.destroy(2) if n==i else qt.qeye(2) for n in range(L)]))
    #     gammas.append(gamma_rel)

    # # Dephasing operators
    # for i in range(L):
    #     c_ops.append(np.sqrt(gamma_deph[i]) * qt.tensor([sz if n==i else qt.qeye(2) for n in range(L)]))
    #     gammas.append(gamma_deph)

    # #c_ops = [rel0, rel1, rel2,... rel(L-1), deph0, deph1,..., deph(L-1)]

    # # Initial state
    # psi0 = qt.tensor([qt.basis(2, 0) for _ in range(L)])



    # # Create obs_list based on the observables in sim_params_class.observables
    # obs_list = []


    # for obs_type in sim_params_class.observables:
    #     if obs_type.lower() == 'x':
    #         # For each site, create the measurement operator for 'x'
    #         obs_list.extend([qt.tensor([sx if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)])
    #     elif obs_type.lower() == 'y':
    #         obs_list.extend([qt.tensor([sy if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)])
    #     elif obs_type.lower() == 'z':
    #         obs_list.extend([qt.tensor([sz if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)])





    # jump_site_list = [ qt.destroy(2)  ,  sz]

    # obs_site_list = [sx, sy, sz]


    # A_kn_site_list = []


    # n_jump_site = len(jump_site_list)
    # n_obs_site = len(obs_site_list)


    # for lk in jump_site_list:
    #     for on in obs_site_list:
    #         for k in range(L):
    #             A_kn_site_list.append( qt.tensor([  lk.dag()*on*lk  -  0.5*on*lk.dag()*lk  -  0.5*lk.dag()*lk*on   if n == k else qt.qeye(2) for n in range(L)]) )



    # new_obs_list = obs_list + A_kn_site_list

    # n_obs= len(obs_list)
    # n_jump= len(c_ops)

    #     # Exact Lindblad solution
    # result_lindblad = qt.mesolve(H, psi0, t, c_ops, new_obs_list, progress_bar=True)

    # exp_vals = []
    # for i in range(len(new_obs_list)):
    #     exp_vals.append(result_lindblad.expect[i])



    # # Separate original and new expectation values from result_lindblad.
    # n_obs = len(obs_list)  # number of measurement operators (should be L * n_types)
    # original_exp_vals = exp_vals[:n_obs]
    # new_exp_vals = exp_vals[n_obs:]  # these correspond to the A_kn operators



    # # Compute the integral of the new expectation values to obtain the derivatives
    # d_On_d_gk = [ trapezoidal(new_exp_vals[i],t)  for i in range(len(A_kn_site_list)) ]

    # d_On_d_gk = np.array(d_On_d_gk).reshape(n_jump_site, n_obs_site, L, n_t)
    # original_exp_vals = np.array(original_exp_vals).reshape(n_obs_site, L, n_t)


    # avg_min_max_traj_time = [None, None, None]


    # return t, original_exp_vals, d_On_d_gk, avg_min_max_traj_time