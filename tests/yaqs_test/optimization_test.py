
#%%
import matplotlib.pyplot as plt
import numpy as np


from mqt.yaqs.noise_char.optimization import *
from mqt.yaqs.noise_char.propagation import *

import sys
import os


import psutil
import threading
from datetime import datetime
import pandas as pd

#%%
stop_event = threading.Event()
def log_memory(pid, log_file, interval=0.1):
    process = psutil.Process(pid)
    with open(log_file, "w") as f:
        f.write("timestamp,ram_GB\n")
    try:
        while not stop_event.is_set():
            # Get memory usage of the main process
            total_mem_bytes = process.memory_info().rss

            # Add memory usage of all child processes (recursively)
            for child in process.children(recursive=True):
                try:
                    total_mem_bytes += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass  # child might have exited

            mem_gb = total_mem_bytes / 1024 / 1024 / 1024  # Convert to GB
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(log_file, "a") as f:
                f.write(f"{timestamp},{mem_gb:.2f}\n")

            time.sleep(interval)
    except Exception as e:
        pass  # Silently ignore exceptions when stopping


#%%






#
# args = sys.argv[1:]

args=["test/optimization", 100, 2, "False", "1", "1e-4", "2L"]

folder = args[0]

ntraj = int(args[1])

L = int(args[2])

restart = args[3].lower() == "true" if len(args) > 3 else False


order = int(args[4])

threshold = float(args[5])

dimensions = args[6]



pid = os.getpid()


log_file = folder+"/self_memory_log.csv"

# Start memory logging in a background thread
logger_thread = threading.Thread(target=log_memory, args=(pid, log_file), daemon=True)
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
        gamma_rel=np.random.rand()
        gamma_deph=np.random.rand()

    if dimensions == "2L":
        gamma_rel=np.random.rand(L)
        gamma_deph=np.random.rand(L)
    




## Computing reference trajectory 

sim_params = SimulationParameters(L, gamma_rel, gamma_deph)
sim_params.T = 5
sim_params.N = 1024



t, qt_ref_traj, d_On_d_gk=tjm_traj(sim_params)


qt_ref_traj_reshaped = qt_ref_traj.reshape(-1, qt_ref_traj.shape[-1])

qt_ref_traj_with_t=np.concatenate([np.array([t]), qt_ref_traj_reshaped], axis=0)




## Saving reference trajectory and gammas
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






## Defining the loss function and initial parameters
sim_params.N = ntraj
sim_params.order = order
sim_params.threshold = threshold




if dimensions == "2":

    loss_function=loss_class_2d(sim_params, qt_ref_traj, tjm_traj, print_to_file=True)
    x0 = np.random.rand(2)

if dimensions == "2L":

    loss_function=loss_class_nd(sim_params, qt_ref_traj, tjm_traj, print_to_file=True)
    x0 = np.random.rand(2*sim_params.L)

loss_function.set_file_name(f"{folder}/loss_x_history", reset=not restart)





## Running the optimization
loss_function.reset()
loss_history, x_history, x_avg_history, t_opt, exp_val_traj= ADAM_loss_class(loss_function, x0, alpha=0.1, max_iterations=500, threshhold = 1e-3, max_n_convergence = 20, tolerance=1e-8, beta1 = 0.5, beta2 = 0.99, epsilon = 1e-8, restart=restart)#, Ns=10e5)






## Saving the optimization results
exp_val_traj_reshaped = exp_val_traj.reshape(-1, exp_val_traj.shape[-1])

exp_val_traj_with_t=np.concatenate([np.array([t_opt]), exp_val_traj_reshaped], axis=0)



opt_traj_file= f"{folder}/opt_traj.txt"

np.savetxt(opt_traj_file, exp_val_traj_with_t.T, header=header, fmt='%.6f')



stop_event.set()
logger_thread.join()

#%%


# %%









# L=100
# ntraj=1024
# x_avg_file="test/reset/loss_x_history.txt"
# gammas_file="test/reset/gammas.txt"

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




# # %%

# plt.plot(loss_function.diff_avg_history)
# # %%
# loss_function.diff_avg_history
# # %%


# %%
%matplotlib qt

mem_usage = pd.read_csv(f"test/optimization_klotz2/self_memory_log.csv")

plt.plot(np.array(mem_usage['ram_GB']), label="Memory Usage (GB)")


#%%
garbage = np.genfromtxt(f"test/optimization_klotz/garbage.txt", skip_header=1)

plt.plot(garbage[:,1], label="Memory Usage (GB)")

# %%
ref_traj = np.genfromtxt("results/optimization/d_2L/L_5/ntraj_512/ref_traj.txt", skip_header=1)
opt_traj = np.genfromtxt("results/optimization/d_2L/L_5/ntraj_512/opt_traj.txt", skip_header=1)

# Compare trajectories for each observable
plt.figure(figsize=(10, 6))
for i in range(1, ref_traj.shape[1]):  # skip time column
    plt.plot(ref_traj[:, 0], ref_traj[:, i], label=f"Ref obs {i}")
    plt.plot(opt_traj[:, 0], opt_traj[:, i], '--', color=plt.gca().lines[-1].get_color(), label=f"Opt obs {i}")

plt.xlabel("Time")
plt.ylabel("Observable value")
plt.legend()
plt.title("Reference vs Optimized Trajectories")
plt.tight_layout()
plt.show()