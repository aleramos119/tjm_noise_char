
#%%
import matplotlib.pyplot as plt
import numpy as np

import time

from mqt.yaqs.noise_char.optimization import *
from mqt.yaqs.noise_char.propagation import *


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











