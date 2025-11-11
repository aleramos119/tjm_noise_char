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

import sys

#%%


#%%


if __name__ == '__main__':

    T=sys.argv[1]

    obs=sys.argv[2]

    noise=sys.argv[3]





    work_dir=f"test/gamma_scan_T_{T}_gamma_ref_0.01_obs_{obs}_noise_{noise}"

    work_dir_path = Path(work_dir)

    work_dir_path.mkdir(parents=True, exist_ok=True)



    ## Defining Hamiltonian and observable list
    L=2

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


    # obs_list = [Observable(X(), site) for site in range(L)]  + [Observable(Y(), site) for site in range(L)] + [Observable(Z(), site) for site in range(L)]
    obs_list = [Observable(obs, site) for site in range(L)]


    #%%
    ## Defining simulation parameters

    dt=0.1

    N=4000

    max_bond_dim=8

    threshold=1e-6

    order=1

    sim_params = AnalogSimParams(observables=obs_list, elapsed_time=T, dt=dt, num_traj=N, max_bond_dim=max_bond_dim, threshold=threshold, order=order, sample_timesteps=True)





    #%%


    #%%
    ## Defining reference noise model and reference trajectory
    gamma_reference = 0.01
    # ref_noise_model =  CompactNoiseModel([{"name": "lowering", "sites": [i for i in range(L)], "strength": gamma_rel}] + [{"name": "pauli_z", "sites": [i for i in range(L)], "strength": gamma_deph}])
    ref_noise_model =  CompactNoiseModel( [{"name": noise_operator, "sites": [i for i in range(L)], "strength": gamma_reference} ])

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
    gamma_guess=0.06
    sim_params.num_traj=int(4000)

    # guess_noise_model =  CompactNoiseModel([{"name": "lowering", "sites": [i for i in range(L)], "strength": gamma_rel_guess} ] + [{"name": "pauli_z", "sites": [i for i in range(L)], "strength": gamma_deph_guess} ])
    guess_noise_model =  CompactNoiseModel( [{"name": noise_operator, "sites": [i for i in range(L)], "strength": gamma_guess} ])
    # guess_noise_model =  CompactNoiseModel([{"name": noise_operator, "sites": [i], "strength": gamma_rel_guess} for i in range(L)] )


    opt_propagator = PropagatorWithGradients(
        sim_params=sim_params,
        hamiltonian=H_0,
        compact_noise_model=guess_noise_model,
        init_state=init_state
    )

    loss=LossClass(
            ref_traj=ref_traj, traj_gradients=opt_propagator, working_dir=work_dir, print_to_file=True
        )


    characterizer = Characterizer(
        traj_gradients=opt_propagator,
        init_guess=guess_noise_model,
        loss=loss,
    )


    #%%

    # print("Optimizing ... ")

    # characterizer.gradient_descent_optimize(max_iterations=20)

    # print("Optimization complete.")




#%%


# from mqt.yaqs.noise_char.optimization import trapezoidal
# import matplotlib.pyplot as plt

# obs=1

# noise = 1
# plt.plot(propagator.times, propagator.obs_traj[obs].results,'o')

# # %%

# int_obst = trapezoidal(propagator.obs_traj[obs].results, propagator.times)
# plt.plot(propagator.times, propagator.d_on_d_gk_array[noise,obs],'o', label="d_on_d_gk")
# plt.plot(propagator.times, -2*int_obst,'x-', label="obs_int")
# plt.legend()

# # %%
# x_guess=np.array([0.06,0.06])

# loss_value, grad, sim_time=loss(x_guess)
# # %%
# print(loss, grad)
# # %%
# obs=0

# noise = 0
# plt.plot(loss.traj_gradients.times, loss.obs_array[obs],'o', label="obs traj")
# plt.plot(loss.traj_gradients.times, loss.ref_traj_array[obs],'x', label="ref traj")
# plt.plot(loss.traj_gradients.times, loss.obs_array[obs]-loss.ref_traj_array[obs],'x', label="diff")
# plt.plot(loss.traj_gradients.times, loss.d_on_d_gk[noise,obs],'-', label="d_on_d_gk")
# # plt.plot(loss.traj_gradients.times, -2*trapezoidal(loss.obs_array[obs], loss.traj_gradients.times),'x', label="int obs traj")

# inside_sum = 2*(loss.obs_array[obs]-loss.ref_traj_array[obs])*loss.d_on_d_gk[noise,obs]
# plt.plot(loss.traj_gradients.times, inside_sum, 'o',label="inside sum" )


# plt.legend(loc='lower left')



    # %%

    gamma_list = np.array([0.001*1.3**i for i in range(27)]
)

    loss_list=[]

    grad_list=[]

    obs_array_list=[]

    d_on_list=[]

    np.savetxt(work_dir + "/gamma_list.txt", gamma_list, header="##", fmt="%.6f")


    np.savetxt(work_dir + f"/ref_traj.txt", loss.ref_traj_array, header="##", fmt="%.6f")


    for i,gamma in enumerate(gamma_list):

        loss_value, grad, sim_time=loss(np.array([gamma]))

        loss_list.append(loss_value)
        grad_list.append(grad[0])

        obs_array_list.append(loss.obs_array)

        d_on_list.append(loss.d_on_d_gk)

        np.savetxt(work_dir + f"/obs_array_{i}.txt", loss.obs_array, header="##", fmt="%.6f")
        np.savetxt(work_dir + f"/d_on_list_{i}.txt", loss.d_on_d_gk.reshape(1*1,len(loss.traj_gradients.times)), header="##", fmt="%.6f")





    np.savetxt(work_dir + "/loss_list.txt", loss_list, header="##", fmt="%.6f")

    np.savetxt(work_dir + "/grad_list.txt", grad_list, header="##", fmt="%.6f")







# %%
# %%
