
#%%
import matplotlib.pyplot as plt
import numpy as np


from mqt.yaqs.noise_char.optimization import *
from mqt.yaqs.noise_char.propagation import *

from auxiliar.write import *

import sys
import os


import psutil
import threading
from datetime import datetime
import pandas as pd


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"


#%%
stop_event = threading.Event()




#
args = sys.argv[1:]

#args=["test/optimization", 20, 2, "False", "1", "1e-4", "2", "scikit_tt", "krylov_5", "6"]

folder = args[0]

ntraj = int(args[1])

L = int(args[2])

restart = args[3].lower() == "true" if len(args) > 3 else False


order = int(args[4])

threshold = float(args[5])

dimensions = args[6]




method = args[7]

solver = args[8]


allocated_cpus = int(args[9])



print("Input parameters:")
print(f"folder = {folder}")
print(f"ntraj = {ntraj}")
print(f"L = {L}")
print(f"restart = {restart}")
print(f"order = {order}")
print(f"threshold = {threshold}")
print(f"dimensions = {dimensions}")
print(f"method = {method}")
print(f"solver = {solver}")
print(f"allocated_cpus = {allocated_cpus}")




pid = os.getpid()


log_file = folder+"/self_memory_log.csv"

# Start memory logging in a background thread
logger_thread = threading.Thread(target=log_memory, args=(pid, log_file, 30,stop_event), daemon=True)
logger_thread.start()






## Defining the gammas
if restart:
    gammas = np.genfromtxt(f"{folder}/gammas.txt", skip_header=1)

    if dimensions == "2":
        gamma_rel=gammas[0]
        gamma_deph=gammas[1]

    if dimensions == "2L":
        gamma_rel = gammas[:L]
        gamma_deph = gammas[L:]

else:
    if dimensions == "2":
        gamma_rel=0.1
        gamma_deph=0.1

    if dimensions == "2L":
        gamma_rel=np.random.rand(L)
        gamma_deph=np.random.rand(L)
    

if method == "tjm":
    traj_function = tjm_traj

if method == "scikit_tt":
    traj_function = scikit_tt_traj




## Computing reference trajectory 
print("Running ref traj")
sim_params = SimulationParameters(L, gamma_rel, gamma_deph)
sim_params.T = 5
sim_params.N = 4096
sim_params.order = order
sim_params.threshold = threshold
sim_params.req_cpus = allocated_cpus - 1
sim_params.set_solver("tdvp"+str(order),solver)


t, qt_ref_traj, d_On_d_gk, avg_min_max_traj_time=traj_function(sim_params)




## Saving reference trajectory and gammas
gamma_header = "  ".join([f"gr_{i+1}" for i in range(L)] + [f"gd_{i+1}" for i in range(L)])



if not restart:


    if not os.path.exists(folder):
        os.makedirs(folder)


    write_ref_traj(t, qt_ref_traj, f"{folder}/ref_traj.txt")


    # Save gamma values next to each other with appropriate header
    gamma_file = f"{folder}/gammas.txt"
    gamma_data = np.hstack([gamma_rel, gamma_deph])
    np.savetxt(gamma_file, gamma_data.reshape(1, -1), header=gamma_header, fmt='%.6f')






## Defining the loss function and initial parameters
sim_params.N = ntraj

if dimensions == "2":

    loss_function=loss_class_2d(sim_params, qt_ref_traj, traj_function, print_to_file=True)
    x0 = np.random.rand(2)

if dimensions == "2L":

    loss_function=loss_class_nd(sim_params, qt_ref_traj, traj_function, print_to_file=True)
    x0 = np.random.rand(2*sim_params.L)

loss_function.set_file_name(f"{folder}/loss_x_history", reset=not restart)





## Running the optimization
print("running optimzation !!!")
loss_function.reset()
loss_history, x_history, x_avg_history, t_opt, opt_traj= ADAM_loss_class(loss_function, x0, alpha=0.1, max_iterations=500, threshhold = 1e-3, max_n_convergence = 20, tolerance=1e-8, beta1 = 0.5, beta2 = 0.99, epsilon = 1e-8, restart=restart)#, Ns=10e5)




write_ref_traj(t_opt, opt_traj, f"{folder}/opt_traj.txt" )



stop_event.set()

logger_thread.join()



# Wait briefly to ensure logger finishes last write
time.sleep(1)

#%%


# %%









# L=100
# ntraj=1024
# x_avg_file="test/optimization/loss_x_history.txt"
# gammas_file="test/optimization/gammas.txt"

# data = np.genfromtxt(x_avg_file, skip_header=1)
# gammas=np.genfromtxt(gammas_file, skip_header=1)


# nt,cols = data.shape

# d=cols-2

# L=d//2

# for i in range(d):
#     plt.plot(data[:, 0], data[:, 2 + i], label=f"$\\gamma_{{{i+1}}}$")
#     plt.axhline(gammas[i], color=plt.gca().lines[-1].get_color(), linestyle='--', linewidth=2)


# plt.xlabel("Iterations")
# plt.ylabel(r"$\gamma$")
# plt.legend()




# %%

# plt.plot(loss_function.diff_avg_history)
# # %%
# loss_function.diff_avg_history
# # %%


# %%
# %matplotlib qt

# mem_usage = pd.read_csv(f"test/optimization_klotz2/self_memory_log.csv")

# plt.plot(np.array(mem_usage['ram_GB']), label="Memory Usage (GB)")


# #%%
# garbage = np.genfromtxt(f"test/optimization_klotz/garbage.txt", skip_header=1)

# plt.plot(garbage[:,1], label="Memory Usage (GB)")

# # %%
# ref_traj = np.genfromtxt("results/optimization/d_2L/L_5/ntraj_512/ref_traj.txt", skip_header=1)
# opt_traj = np.genfromtxt("results/optimization/d_2L/L_5/ntraj_512/opt_traj.txt", skip_header=1)

# # Compare trajectories for each observable
# plt.figure(figsize=(10, 6))
# for i in range(1, ref_traj.shape[1]):  # skip time column
#     plt.plot(ref_traj[:, 0], ref_traj[:, i], label=f"Ref obs {i}")
#     plt.plot(opt_traj[:, 0], opt_traj[:, i], '--', color=plt.gca().lines[-1].get_color(), label=f"Opt obs {i}")

# plt.xlabel("Time")
# plt.ylabel("Observable value")
# plt.legend()
# plt.title("Reference vs Optimized Trajectories")
# plt.tight_layout()
# plt.show()