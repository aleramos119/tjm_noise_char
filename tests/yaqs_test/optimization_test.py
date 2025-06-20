
#%%
import matplotlib.pyplot as plt
import numpy as np


from mqt.yaqs.noise_char.optimization import *
from mqt.yaqs.noise_char.propagation import *

import sys
import os
#
args = sys.argv[1:]

# args=["test/reset", 100, 3, "True"]

folder = args[0]

ntraj = int(args[1])

L = int(args[2])

restart = args[3].lower() == "true" if len(args) > 3 else False


order = int(args[4])

threshold = float(args[5])





#%%

sim_params = SimulationParameters()
sim_params.T = 1
sim_params.L = L
sim_params.N = 4096



t, qt_ref_traj, d_On_d_gk=tjm_traj(sim_params)


qt_ref_traj_reshaped = qt_ref_traj.reshape(-1, qt_ref_traj.shape[-1])

qt_ref_traj_with_t=np.concatenate([np.array([t]), qt_ref_traj_reshaped], axis=0)


#%%

if not restart:

    header =   "t  " +  "  ".join([obs+str(i)   for obs in sim_params.observables for i in range(sim_params.L) ])

    if not os.path.exists(folder):
        os.makedirs(folder)


    ref_traj_file= f"{folder}/ref_traj.txt"

    np.savetxt(ref_traj_file, qt_ref_traj_with_t.T, header=header, fmt='%.6f')

#%%


#%%
sim_params.N = ntraj
sim_params.order = order
sim_params.threshold = threshold

loss_function=loss_class(sim_params, qt_ref_traj, tjm_traj, print_to_file=True)


loss_function.set_file_name(f"{folder}/loss_x_history", reset=not restart)

x0 = np.array([0.3,0.25])



#%%
loss_function.reset()
loss_history, x_history, x_avg_history, t_opt, exp_val_traj= ADAM_loss_class(loss_function, x0, alpha=0.02, max_iterations=1000, threshhold = 5e-4, max_n_convergence = 50, tolerance=1e-8, beta1 = 0.5, beta2 = 0.99, epsilon = 1e-8, restart=restart)#, Ns=10e5)


# %%

exp_val_traj_reshaped = exp_val_traj.reshape(-1, exp_val_traj.shape[-1])

exp_val_traj_with_t=np.concatenate([np.array([t_opt]), exp_val_traj_reshaped], axis=0)



opt_traj_file= f"{folder}/opt_traj.txt"

np.savetxt(opt_traj_file, exp_val_traj_with_t.T, header=header, fmt='%.6f')




# %%
