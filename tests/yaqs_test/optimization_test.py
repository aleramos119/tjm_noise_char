
#%%
import matplotlib.pyplot as plt
import numpy as np


from mqt.yaqs.noise_char.optimization import *
from mqt.yaqs.noise_char.propagation import *

import sys
import os
#
# args = sys.argv[1:]

args=["test/reset", 100, 3, "True", "1", "1e-4"]

folder = args[0]

ntraj = int(args[1])

L = int(args[2])

restart = args[3].lower() == "true" if len(args) > 3 else False


order = int(args[4])

threshold = float(args[5])





#%%

if restart:
    gammas = np.genfromtxt(f"{folder}/gammas.txt", skip_header=1)
    gamma_rel = gammas[:L]
    gamma_deph = gammas[L:]

else:
    gamma_rel=np.random.rand(L)
    gamma_deph=np.random.rand(L)



sim_params = SimulationParameters(L, gamma_rel, gamma_deph)
sim_params.T = 5
sim_params.N = 4096



t, qt_ref_traj, d_On_d_gk=qutip_traj(sim_params)


qt_ref_traj_reshaped = qt_ref_traj.reshape(-1, qt_ref_traj.shape[-1])

qt_ref_traj_with_t=np.concatenate([np.array([t]), qt_ref_traj_reshaped], axis=0)


#%%

header =   "t  " +  "  ".join([obs+str(i)   for obs in sim_params.observables for i in range(sim_params.L) ])
gamma_header = "  ".join([f"gr_{i+1}" for i in range(L)] + [f"gd_{i+1}" for i in range(L)])



if not restart:


    if not os.path.exists(folder):
        os.makedirs(folder)


    ref_traj_file= f"{folder}/ref_traj.txt"

    np.savetxt(ref_traj_file, qt_ref_traj_with_t.T, header=header, fmt='%.6f')


    # Save gamma values next to each other with appropriate header
    gamma_file = f"{folder}/gammas.txt"
    gamma_data = np.hstack([gamma_rel, gamma_deph])
    np.savetxt(gamma_file, gamma_data.reshape(1, -1), header=gamma_header, fmt='%.6f')



#%%
sim_params.N = ntraj
sim_params.order = order
sim_params.threshold = threshold

loss_function=loss_class_nd(sim_params, qt_ref_traj, qutip_traj, print_to_file=True)


loss_function.set_file_name(f"{folder}/loss_x_history", reset=not restart)

x0 = np.random.rand(2*sim_params.L)



#%%
loss_function.reset()
loss_history, x_history, x_avg_history, t_opt, exp_val_traj= ADAM_loss_class(loss_function, x0, alpha=0.1, max_iterations=100, threshhold = 1e-3, max_n_convergence = 20, tolerance=1e-8, beta1 = 0.5, beta2 = 0.99, epsilon = 1e-8, restart=restart)#, Ns=10e5)


# %%

exp_val_traj_reshaped = exp_val_traj.reshape(-1, exp_val_traj.shape[-1])

exp_val_traj_with_t=np.concatenate([np.array([t_opt]), exp_val_traj_reshaped], axis=0)



opt_traj_file= f"{folder}/opt_traj.txt"

np.savetxt(opt_traj_file, exp_val_traj_with_t.T, header=header, fmt='%.6f')


# %%









L=100
ntraj=1024
x_avg_file="test/reset/loss_x_history.txt"
gammas_file="test/reset/gammas.txt"

data = np.genfromtxt(x_avg_file, skip_header=1)
gammas=np.genfromtxt(gammas_file, skip_header=1)


nt,cols = data.shape

d=cols-2

L=d//2

for i in range(d):
    plt.plot(data[:, 0], data[:, 2 + i], label=f"$\\gamma_{{{i+1}}}$")
    plt.axhline(gammas[i], color=plt.gca().lines[-1].get_color(), linestyle='--', linewidth=2)


plt.xlabel("Iterations")
plt.ylabel(r"$\gamma$")
plt.legend()




# %%

plt.plot(loss_function.diff_avg_history)
# %%
loss_function.diff_avg_history
# %%
