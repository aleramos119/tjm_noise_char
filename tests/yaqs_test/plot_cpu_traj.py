
#%%
import matplotlib.pyplot as plt
import numpy as np
import os

import pandas as pd

#%%
L=100

cpu_list_initial=list(range(3, 50))  # Initial list of CPUs

folder="results/cpu_traj_scan"

method_list=[
            #  "method_tjm/order_2/threshold_1e-6", 
            #  "method_tjm/order_1/threshold_1e-4", 

            #  "method_tjm_new_calc/solver_exact/order_1/threshold_1e-4", 

            #  "method_tjm_new_calc_1/solver_exact/order_1/threshold_1e-4", 
             "method_scikit_tt_new_calc_big/solver_krylov_5/order_1/threshold_1e-4",
             "method_scikit_tt_new_calc_1/solver_krylov_5/order_1/threshold_1e-4",

            #  "method_scikit_tt_new_calc/solver_exact/order_1/threshold_1e-4",
            #  "method_scikit_tt_new_calc/solver_krylov_5/order_1/threshold_1e-4",


             "method_scikit_tt_klotz/solver_krylov_5/order_1/threshold_1e-4",

            #     "method_tjm_klotz/solver_exact/order_1/threshold_1e-4",



             ]



ntraj=512


#%%
%matplotlib qt
fig, axs = plt.subplots(1, 2, figsize=(12, 5))


for method in method_list:
    cpu_mem = []
    cpu_time = []

    print(method)
    for cpu in cpu_list_initial:
        mem_file = f"{folder}/{method}/{L}_sites/{cpu}_cpus/{ntraj}_traj/self_memory_log.csv"
        time_file = f"{folder}/{method}/{L}_sites/{cpu}_cpus/{ntraj}_traj/time_sec.txt"

        if os.path.exists(mem_file):
            data = pd.read_csv(mem_file).values 
            if data.shape[1] > 3: 
                cpu_mem.append([cpu, max(data[:, 3])]) 
            else: 
                cpu_mem.append([cpu, max(data[:, 1])]) 

        if os.path.exists(time_file):
            cpu_time.append([cpu, np.loadtxt(time_file) / 60 / 60])

    cpu_mem = np.array(cpu_mem)
    cpu_time = np.array(cpu_time)



    axs[0].plot(cpu_mem[:, 0], cpu_mem[:, 1], '-o', label=method)



    axs[1].plot(cpu_time[:, 0], cpu_time[:, 1],'o-', label=method)



axs[0].set_xlabel("CPUs")
axs[0].set_ylabel("Max Memory (GB)")
axs[1].set_xlabel("CPUs")
axs[1].set_ylabel("Time (hours)")
plt.tight_layout()
plt.legend()
plt.show()
#%%
mem_file
# %%

method="method_scikit_tt_new_calc_1/solver_krylov_5/order_1/threshold_1e-4"
L=100
ntraj=512
cpu=3
mem_file = f"{folder}/{method}/{L}_sites/{cpu}_cpus/{ntraj}_traj/self_memory_log.csv"

data=pd.read_csv(mem_file).values[:, 1]

plt.plot(data, label=f"{method} {L}_sites {ntraj}_traj {cpu}_cpus")
# %%
method="method_tjm_new_calc/solver_exact/order_1/threshold_1e-4"
L=100
ntraj=512
cpu=34

cpu_file = f"{folder}/{method}/{L}_sites/{cpu}_cpus/{ntraj}_traj/cpu_usage.csv"


cpu_usage_df = pd.read_csv(cpu_file)

# Pivot the data to have CPUs as columns and timestamps as rows
cpu_usage_pivot = cpu_usage_df.pivot(index='timestamp', columns='cpu', values='usr')

# Plot the CPU usage for each CPU
plt.figure(figsize=(12, 6))
for cpu in cpu_usage_pivot.columns:
    plt.plot(np.array(cpu_usage_pivot[cpu]), label=f"CPU {cpu}")

plt.xlabel("Time (seconds)")
plt.ylabel("CPU Usage (%)")
plt.title(f"CPU Usage Over Time for {cpu} CPUs")
plt.legend(loc="upper left", ncol=2)

# %%



print(cpu_usage_df.columns.tolist())
# %%

fig, axs = plt.subplots(1, 2, figsize=(14, 5))
col = 1
ntraj_list = [1024, 2048, 4096, 8192]

for ntraj in ntraj_list:
    traj_file = f"results/cpu_traj_scan/method_tjm_new_calc_omp_1/solver_exact/order_1/threshold_1e-4/10_sites/96_cpus/{ntraj}_traj/ref_traj.txt"
    data = np.loadtxt(traj_file)
    axs[0].plot(data[:, 0], data[:, col], label=f"tjm - {ntraj} trajs")

axs[0].set_title("tjm_new_calc_omp_1")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Value")
axs[0].legend()

for ntraj in ntraj_list:
    traj_file = f"results/cpu_traj_scan/method_scikit_tt_new_calc_omp_1/solver_krylov_5/order_1/threshold_1e-4/10_sites/96_cpus/{ntraj}_traj/ref_traj.txt"
    data = np.loadtxt(traj_file)
    axs[1].plot(data[:, 0], data[:, col], label=f"scikit_tt - {ntraj} trajs")

axs[1].set_title("scikit_tt_new_calc_omp_1")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Value")
axs[1].legend()

plt.tight_layout()
plt.show()
col=1


#%%

ntraj_list = [1024, 2048, 4096, 8192]
time_list = []
for ntraj in ntraj_list:
    time_file = f"results/cpu_traj_scan/method_scikit_tt_new_calc_omp_1/solver_krylov_5/order_1/threshold_1e-4/10_sites/96_cpus/{ntraj}_traj/time_sec.txt"
    time=np.loadtxt(time_file)
    time_list.append(time / 60)  # convert seconds to minutes

plt.plot(ntraj_list, time_list, 'o-', label=f"scikit_tt")
   

time_list = []
for ntraj in ntraj_list:
    time_file = f"results/cpu_traj_scan/method_tjm_new_calc_omp_1/solver_exact/order_1/threshold_1e-4/10_sites/96_cpus/{ntraj}_traj/time_sec.txt"
    time=np.loadtxt(time_file)
    time_list.append(time / 60)  # convert seconds to minutes
plt.plot(ntraj_list, time_list, 'o-', label=f"tjm")

plt.xlabel("Number of Trajectories")
plt.ylabel("Time (min)")
plt.legend()
# %%
time_list
# %%
ntraj_list
# %%
