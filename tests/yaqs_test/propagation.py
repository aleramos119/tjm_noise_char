#%%
import numpy as np
from mqt.yaqs.noise_char.propagation import *


import sys
import time
import pandas as pd

#%%

args = sys.argv[1:]



args=["results/cpu_traj_scan/cpu_traj_scan/32_cpus_4096_traj", 124]

folder = args[0]

ntraj = args[1]


folder="results/cpu_traj_scan/"+folder


sim_params = SimulationParameters()
sim_params.N = int(ntraj)

sim_params.max_bond_dim = 2

start_time = time.time()

t, qt_ref_traj, d_On_d_gk=tjm_traj(sim_params)


end_time = time.time()

#%%
qt_ref_traj_reshaped = qt_ref_traj.reshape(-1, *qt_ref_traj.shape[2:]).T

np.savetxt(f"{folder}/qt_ref_traj.txt", qt_ref_traj_reshaped )


duration = end_time - start_time
with open(f"{folder}/time_sec.txt", "w") as f:
    f.write(f"{duration}\n")






#%%
cpu_list=[8, 12, 16, 20, 32, 36, 64, 68]

sites=4

ntraj=4096




time_list=[]

mem_list=[]


for cpu in cpu_list:

    folder=f"results/cpu_traj_scan/{sites}_sites/{cpu}_cpus/{ntraj}_traj/"

    time_file= f"{folder}/time_sec.txt"
    mem_file= f"{folder}/mem_usage.csv"

    time_list.append(np.loadtxt(time_file))
    mem_list.append(pd.read_csv(mem_file).values[-1])


plt.plot(cpu_list, time_list,'-o')
plt.xlabel("Number of CPUs")
plt.ylabel("Time (seconds)")

#%%
plt.plot(cpu_list, mem_list,'-o')
plt.xlabel("Number of CPUs")
plt.ylabel("Memory Usage (MB)")


# %%
import matplotlib.pyplot as plt

ncpus=20

csv_file=f"/home/aramos/Dokumente/Work/Simulation of Open Quantum Systems/tjm_noise_char/tests/yaqs_test/results/cpu_traj_scan/4_sites/{ncpus}_cpus/4096_traj/cpu_usage.csv"

cpu_usage_df = pd.read_csv(csv_file)

# Pivot the data to have CPUs as columns and timestamps as rows
cpu_usage_pivot = cpu_usage_df.pivot(index='timestamp', columns='cpu', values='usr')

# Plot the CPU usage for each CPU
plt.figure(figsize=(12, 6))
for cpu in cpu_usage_pivot.columns:
    plt.plot(np.array(cpu_usage_pivot[cpu]), label=f"CPU {cpu}")

plt.xlabel("Time (seconds)")
plt.ylabel("CPU Usage (%)")
plt.title(f"CPU Usage Over Time for {ncpus} CPUs")
plt.legend(loc="upper left", ncol=2)
# plt.grid()
# plt.tight_layout()
plt.show()
# %%
np.array(cpu_usage_pivot[cpu]).shape
# %%

mem_usage = pd.read_csv(f"/home/aramos/Dokumente/Work/Simulation of Open Quantum Systems/tjm_noise_char/tests/yaqs_test/results/cpu_traj_scan/4_sites/{ncpus}_cpus/4096_traj/mem_usage.csv")

plt.plot(np.array(mem_usage/1024/1024))


# %%
mem_usage
# %%
