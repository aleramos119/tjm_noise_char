
#%%
import matplotlib.pyplot as plt
import numpy as np

import time

from yaqs.noise_char.optimization import *
from yaqs.noise_char.propagation import *


#%%
sim_params = SimulationParameters()
t, qt_ref_traj, d_On_d_gk=qutip_traj(sim_params)


#%%
alg_list=["ADAM","BFGS"]

N_list=[250,500,1000,2000,4000]

N_list=[1,2,3]


max_iter=3

time_list=np.zeros([len(alg_list),len(N_list)])
iter_list=np.zeros([len(alg_list),len(N_list)])
error_list=np.zeros([len(alg_list),len(N_list)])
time_per_iter_list=np.zeros([len(alg_list),len(N_list)])

time_file="time.txt"
iter_file="iter.txt"
error_file="error.txt"
time_per_iter_file="time_per_iter.txt"


#%%

def append_to_file(file_path: str, row: list) -> None:
    """Appends a row to a file.

    Args:
        file_path (str): The path to the file.
        row (list): The row to append, as a list of values.
    """
    with open(file_path, 'a') as file:
        file.write('    '.join(map(str, row)) + '\n')


#%%

for i,N in enumerate(N_list):
    initial_params = SimulationParameters()
    initial_params.gamma_rel = 0.05
    initial_params.gamma_deph = 0.4
    initial_params.N = N

    start_time=time.time()

    adam_loss_history, adam_gr_history, adam_gd_history, adam_dJ_dgr_history, adam_dJ_dgd_history = ADAM_gradient_descent(initial_params, qt_ref_traj, tjm_traj, learning_rate=0.2, max_iterations=max_iter,tolerance=1e-8, beta1=0.5)

    adam_iter=len(adam_loss_history)

    end_time=time.time()

    time_list[0,i]=(end_time-start_time)/60

    time_per_iter_list[0,i]=time_list[0,i]/adam_iter

    iter_list[0,i]=adam_iter

    error_list[0,i]=(adam_gr_history[-1] - sim_params.gamma_rel)**2 + (adam_gd_history[-1] - sim_params.gamma_deph)**2



    start_time=time.time()

    bfgs_loss_history, bfgs_gr_history, bfgs_gd_history, bfgs_dJ_dgr_history, bfgs_dJ_dgd_history = BFGS(initial_params, qt_ref_traj, tjm_traj, learning_rate=0.2, max_iterations=max_iter,tolerance=1e-8)

    bfgs_iter=len(bfgs_loss_history)

    end_time=time.time()

    time_list[1,i]=(end_time-start_time)/60

    time_per_iter_list[1,i]=time_list[1,i]/bfgs_iter

    iter_list[1,i]=bfgs_iter

    error_list[1,i]=(bfgs_gr_history[-1] - sim_params.gamma_rel)**2 + (bfgs_gd_history[-1] - sim_params.gamma_deph)**2



    append_to_file(time_file,[N, time_list[0,i], time_list[1,i]])
    append_to_file(iter_file,[N, iter_list[0,i], iter_list[1,i]])
    append_to_file(error_file,[N, error_list[0,i], error_list[1,i]])
    append_to_file(time_per_iter_file,[N, time_per_iter_list[0,i], time_per_iter_list[1,i]])


    np.savetxt(f"adam_loss_{N}.txt", adam_loss_history, fmt="%.2f", delimiter="    ", header="")
    np.savetxt(f"bfgs_loss_{N}.txt", bfgs_loss_history, fmt="%.2f", delimiter="    ", header="")


    np.savetxt(f"adam_gr_{N}.txt", adam_gr_history, fmt="%.2f", delimiter="    ", header="")
    np.savetxt(f"bfgs_gr_{N}.txt", bfgs_gr_history, fmt="%.2f", delimiter="    ", header="")


    np.savetxt(f"adam_gd_{N}.txt", adam_gd_history, fmt="%.2f", delimiter="    ", header="")
    np.savetxt(f"bfgs_gd_{N}.txt", bfgs_gd_history, fmt="%.2f", delimiter="    ", header="")


















# #%%





# #%%
# plt.plot(np.log10(loss_history), label='log(J)')
# plt.legend()

# #%%
# plt.plot(gr_history,gd_history,label='gamma_relaxation')
# # plt.plot(gd_history, label='gamma_dephasing')
# plt.axhline(y=0.1, color='r', linestyle='--', label='gamma_reference')
# plt.legend()


# # # %%
# # n_jump=len(d_On_d_gk)
# # n_obs=len(d_On_d_gk[0])

# # lk_list=["d"+str(i) for i in range(sim_params.L)]+["sz"+str(i) for i in range(sim_params.L)]
# # on_list=["sx"+str(i) for i in range(sim_params.L)]+["sy"+str(i) for i in range(sim_params.L)]+["sz"+str(i) for i in range(sim_params.L)]


# # %matplotlib qt
# # for i in [0]:
# #     for j in [4]:
# #         if max(abs(d_On_d_gk[i][j]))>1e-5:
# #           plt.plot(d_On_d_gk[i][j], label=lk_list[i]+"_"+on_list[j]+",  L"+str(i)+"_O"+str(j))
# # plt.xlabel('Time')
# # plt.ylabel('d_On_d_gk')
# # plt.legend()
# # plt.show()
# # # %%

# # # %%


# # %%

# loss, gamma_rel, gamma_deph = np.genfromtxt('data.txt', unpack=True)

# # %%


# plt.plot(loss, label='log(J)')
# plt.legend()
# # %%

# plt.plot(gamma_rel,label='gamma_relaxation')
# plt.plot(gamma_deph, label='gamma_dephasing')
# plt.axhline(y=0.1, color='r', linestyle='--', label='gamma_reference')
# plt.legend()
# # %%
# initial_params = SimulationParameters()

# # %%
# initial_params.gamma_list
# # %%
# initial_params.gamma_rel=0.3

# initial_params.gamma_list
# # %%

# N_list=[100,200,400,800,1600,3200]
# # N_list=[100,200]


# n_jump_site, n_obs_site, L, n_t=d_On_d_gk.shape
# n_N=len(N_list)


# error=np.zeros([n_obs_site,L,n_N])

# error_d_On_d_gk=np.zeros([n_jump_site, n_obs_site, L, n_N])


# for i,N in enumerate(N_list):
#     sim_params.N=N


#     t, traj_N, d_On_d_gk_N= tjm_traj(sim_params)

#     for j in range(n_obs_site):
#         for k in range(L):
#             error[j,k,i]=np.linalg.norm(traj_N[j,k]-qt_ref_traj[j,k])

#     for j in range(n_jump_site):
#         for k in range(n_obs_site):
#             for l in range(L):
#                 error_d_On_d_gk[j,k,l,i]=np.linalg.norm(d_On_d_gk_N[j,k,l]-d_On_d_gk[j,k,l])



# #%%
# sim_params = SimulationParameters()
# sim_params.N=3200
# t, traj_N, d_On_d_gk_N= tjm_traj(sim_params)

# #%%
# %matplotlib qt
# for i in [0]:
#     for j in [0]:
#         plt.plot(t,qt_ref_traj[i,j],'o-', label='O'+str(i)+'_L'+str(j))
#         plt.plot(t,traj_N[i,j],'-', label='O'+str(i)+'_L'+str(j))

# plt.legend()

# #%%
# for i in range(n_obs_site):
#     for j in range(L):
#         plt.plot(N_list,np.log10(error[i,j]),'o-', label='O'+str(i)+'_L'+str(j))
#         # plt.plot(t,qt_ref_traj[i,j],'o-', label='O'+str(i)+'_L'+str(j))
#         # plt.plot(t,traj_N[i,j],'-', label='O'+str(i)+'_L'+str(j))

# plt.legend()


# #%%
# for i in range(n_jump_site):
#     for j in range(n_obs_site):
#         for k in range(L):
#             plt.plot(N_list,error_d_On_d_gk[i,j,k],'o-', label='jump_'+str(i)+'_O'+str(j)+'_L'+str(k))
# plt.legend()
# # %%
# print("Error shape",error.shape)
# print("Trajectory shape",qt_ref_traj.shape)
# print("Norm shape: ",np.linalg.norm(qt_ref_traj-qt_ref_traj, axis=2).shape)
# # %%

# import time
# start = time.time()

# sim_params = SimulationParameters()
# sim_params.N=3200
# t, traj_N, d_On_d_gk_N= scikit_tt_traj(sim_params)

# end = time.time()

# print("Time elapsed: ",end - start)
# # %%
