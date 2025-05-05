
#%%
import matplotlib.pyplot as plt
import numpy as np

import time

from mqt.yaqs.noise_char.optimization import *
from mqt.yaqs.noise_char.propagation import *

import os
import sys
#%%
N = int(sys.argv[1])
alg=sys.argv[2]

#%%
N=10
alg="adam"
#%%
sim_params = SimulationParameters()
t, qt_ref_traj, d_On_d_gk=qutip_traj(sim_params)


#%%
max_iter=3

#%%
folder=f"test1/{alg}/"

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
files_label="# N   " + f"{alg}   "  + "\n"

for file_name in file_list:
    with open(file_name, 'w') as file:
        file.write(files_label)




initial_params = SimulationParameters()
initial_params.gamma_rel = 0.05
initial_params.gamma_deph = 0.4
initial_params.N = N

start_time=time.time()

if alg=="adam":
    loss_history, gr_history, gd_history, dJ_dgr_history, dJ_dgd_history = ADAM_gradient_descent(initial_params, qt_ref_traj, qutip_traj, learning_rate=0.2, max_iterations=max_iter,tolerance=1e-8, beta1=0.5, file_name=folder + f"adam_log_{N}.txt")
elif alg=="bfgs":
    loss_history, gr_history, gd_history, dJ_dgr_history, dJ_dgd_history = BFGS(initial_params, qt_ref_traj, qutip_traj, learning_rate=0.2, max_iterations=max_iter,tolerance=1e-8, file_name=folder + f"bfgs_log_{N}.txt")
elif alg=="UCB":
    loss_history, gr_history, gd_history = bayesian_optimization(initial_params, qt_ref_traj, qutip_traj,[(0.01,0.2),(0.01,0.2)], acquisition="UCB", n_init=40, max_iterations=100, tolerance=1e-8, beta=50, num_restarts=20, raw_samples=20, file_name=" ", device="cpu", loss_std=6e-5, dJ_d_gr_std=0.0065, dJ_d_gd_std=0.025)


end_time=time.time()

iter=len(loss_history)

append_to_file(time_file,[N] + [(end_time-start_time)/60])
append_to_file(iter_file,[N] + [iter])
append_to_file(error_file,[N] + [(gr_history[-1] - sim_params.gamma_rel)**2 + (gd_history[-1] - sim_params.gamma_deph)**2])
append_to_file(time_per_iter_file,[N] + [(end_time-start_time)/60/iter])
append_to_file(min_log_loss_file,[N] + [np.log10(min(loss_history))])




# %%
