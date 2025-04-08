
#%%
import matplotlib.pyplot as plt
import numpy as np

import time

from mqt.yaqs.noise_char.optimization import *
from mqt.yaqs.noise_char.propagation import *

import os

#%%
sim_params = SimulationParameters()
t, qt_ref_traj, d_On_d_gk=qutip_traj(sim_params)


#%%
alg_list=["adam"]

N_list=[250,500,1000,2000,4000]

N_list=[1,2,3]


max_iter=3

time_list=np.zeros([len(N_list),len(alg_list)])
iter_list=np.zeros([len(N_list),len(alg_list)])
error_list=np.zeros([len(N_list),len(alg_list)])
time_per_iter_list=np.zeros([len(N_list),len(alg_list)])
min_log_loss_list=np.zeros([len(N_list),len(alg_list)])

#%%
folder="test/"

os.makedirs(folder, exist_ok=True)

time_file=folder + "time.txt"
iter_file=folder + "iter.txt"
error_file=folder + "error.txt"
min_log_loss_file=folder + "min_log_loss.txt"
time_per_iter_file=folder + "time_per_iter.txt"

file_list=[time_file,iter_file,error_file,min_log_loss_file,time_per_iter_file]

for file_name in file_list:
    if os.path.exists(file_name):
        os.remove(file_name)



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
files_label="# N   " + "   ".join(alg_list)  + "\n"

for file_name in file_list:
    with open(file_name, 'w') as file:
        file.write(files_label)



for i,N in enumerate(N_list):
    for j,alg in enumerate(alg_list):

        initial_params = SimulationParameters()
        initial_params.gamma_rel = 0.05
        initial_params.gamma_deph = 0.4
        initial_params.N = N

        start_time=time.time()

        if alg=="adam":
            loss_history, gr_history, gd_history, dJ_dgr_history, dJ_dgd_history = ADAM_gradient_descent(initial_params, qt_ref_traj, qutip_traj, learning_rate=0.2, max_iterations=max_iter,tolerance=1e-8, beta1=0.5, file_name=folder + f"adam_log_{N}.txt")
        elif alg=="bfgs":
            loss_history, gr_history, gd_history, dJ_dgr_history, dJ_dgd_history = BFGS(initial_params, qt_ref_traj, qutip_traj, learning_rate=0.2, max_iterations=max_iter,tolerance=1e-8, file_name=folder + f"bfgs_log_{N}.txt")

        end_time=time.time()

        iter=len(loss_history)

        time_list[i,j]=(end_time-start_time)/60

        time_per_iter_list[i,j]=time_list[i,j]/iter

        iter_list[i,j]=iter

        error_list[i,j]=(gr_history[-1] - sim_params.gamma_rel)**2 + (gd_history[-1] - sim_params.gamma_deph)**2

        min_log_loss_list[i,j]=np.log10(min(loss_history))


    append_to_file(time_file,[N] + list(time_list[i,:]))
    append_to_file(iter_file,[N] + list(iter_list[i,:]))
    append_to_file(error_file,[N] + list(error_list[i,:]))
    append_to_file(time_per_iter_file,[N] + list(time_per_iter_list[i,:]))
    append_to_file(min_log_loss_file,[N] + list(min_log_loss_list[i,:]))




# %%
