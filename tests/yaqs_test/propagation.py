#%%
import numpy as np
from mqt.yaqs.noise_char.propagation import *


import sys

args = sys.argv[1:]



# args=["results/cpu_traj_scan/cpu_traj_scan/32_cpus_4096_traj", 124]

folder = args[0]

ntraj = args[1]



sim_params = SimulationParameters()
sim_params.N = int(ntraj)

t, qt_ref_traj, d_On_d_gk=tjm_traj(sim_params)


#%%
qt_ref_traj_reshaped = qt_ref_traj.reshape(-1, *qt_ref_traj.shape[2:]).T

np.savetxt(f"{folder}/qt_ref_traj.txt", qt_ref_traj_reshaped )


