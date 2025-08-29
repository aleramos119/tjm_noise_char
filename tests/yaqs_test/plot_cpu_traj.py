
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
ntraj_list = [1024]

for ntraj in ntraj_list:
    traj_file = f"results/cpu_traj_scan/method_tjm_new_calc_omp_1/solver_exact/order_1/threshold_1e-4/10_sites/96_cpus/{ntraj}_traj/ref_traj.txt"
    data = np.loadtxt(traj_file)
    axs[0].plot(data[:, 0], data[:, col], label=f"tjm - {ntraj} trajs")

axs[0].set_title("tjm_new_calc_omp_1")
axs[0].set_xlabel("Time")
axs[0].set_ylabel("Value")
axs[0].legend()

for ntraj in ntraj_list:
    traj_file = f"results/cpu_traj_scan/method_scikit_tt_serial_test/solver_exact/order_1/threshold_1e-4/10_sites/32_cpus/{ntraj}_traj/gamma_0.1/gamma_0.1/ref_traj.txt"
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

ntraj_list = [1024]
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



fig, axs = plt.subplots(1, 2, figsize=(14, 5))

col = 2
ntraj_list=[1024,2048,4096]

ntraj=1024

folder=f"test/propagation"
folder=f"results/cpu_traj_scan/method_scikit_tt_comment_test/solver_exact/order_1/threshold_1e-4/10_sites/32_cpus/{ntraj}_traj"
traj_file = folder + f"/ref_traj.txt"
data = np.loadtxt(traj_file)
axs[0].plot(data[:, 0], data[:, col], label=f"tjm  N={ntraj}")
axs[0].set_title(f"tjm N={ntraj}")
axs[0].legend()


# traj_file = f"results/cpu_traj_scan/method_scikit_tt_new_calc_omp_1/solver_exact/order_1/threshold_1e-4/10_sites/32_cpus/{ntraj}_traj/ref_traj.txt"
# data = np.loadtxt(traj_file)
# axs[1].plot(data[:, 0], data[:, col], 'o-', label="ref_traj")





for i in range(ntraj):
    traj_file = folder + f"/res_traj_{i}.txt"
    data = np.loadtxt(traj_file)
    axs[1].plot(data[:, 0], data[:, col],linestyle=':', color = "gray", linewidth=1, alpha=0.7)



traj_file = folder + f"/avg_traj.txt"
data = np.loadtxt(traj_file)
axs[1].plot(data[:, 0], data[:, col], label="avg_traj", linewidth=3)



axs[1].set_title(f"scikit_tt exact solver N={ntraj}")
axs[1].set_xlabel("Time")
axs[1].set_ylabel("Value")
axs[1].legend()


plt.tight_layout()
plt.show()
col=1

# %%
#### Plot gamma_rel gamma_deph scan 


%matplotlib qt

import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


g_rel_list=[0.05,0.1,0.2,0.4,0.6,0.8]

g_deph_list=[0.05,0.1,0.2,0.4,0.6,0.8]










# default column to start with
col_init = 15


# Initial parameters
g_rel_init = g_rel_list[0]
g_deph_init = g_deph_list[0]

def load_traj(g_rel, g_deph):
    """Load trajectory file for given gamma_rel and gamma_deph."""
    folder = f"results/cpu_traj_scan/method_tjm_gamma_scan/solver_exact/order_1/threshold_1e-4/100_sites/32_cpus/512_traj/gamma_rel_{g_rel}/gamma_deph_{g_deph}/"
    file_path = os.path.join(folder, "ref_traj.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    return np.genfromtxt(file_path, skip_header=1)

# Initial parameters
g_rel_index_init = 0
g_deph_index_init = 0
traj = load_traj(g_rel_list[g_rel_index_init], g_deph_list[g_deph_index_init])

# Plot setup
fig, ax = plt.subplots(figsize=(8, 5))
plt.subplots_adjust(bottom=0.3)
l, = ax.plot(traj[:, 0], traj[:, col_init], label="Trajectory", linestyle='--')

ax.set_xlabel("Time")
ax.set_ylabel(f"Value (col={col_init})")
ax.set_title("Trajectory vs gamma_rel & gamma_deph")
ax.set_xlim(0,5)
ax.set_ylim(0,1)
ax.legend()

# Sliders (index-based)
ax_slider_grel = plt.axes([0.2, 0.15, 0.6, 0.03])
slider_grel = Slider(ax_slider_grel, 'gamma_rel idx', 0, len(g_rel_list)-1, 
                     valinit=g_rel_index_init, valstep=1)

ax_slider_gdeph = plt.axes([0.2, 0.05, 0.6, 0.03])
slider_gdeph = Slider(ax_slider_gdeph, 'gamma_deph idx', 0, len(g_deph_list)-1, 
                      valinit=g_deph_index_init, valstep=1)

def update(val):
    g_rel = g_rel_list[int(slider_grel.val)]
    g_deph = g_deph_list[int(slider_gdeph.val)]
    try:
        traj = load_traj(g_rel, g_deph)
        l.set_xdata(traj[:, 0])
        l.set_ydata(traj[:, col_init])
        ax.set_ylabel(f"Value (col={col_init})")
        ax.set_title(f"gamma_rel={g_rel}, gamma_deph={g_deph}")
        ax.relim()
        ax.autoscale_view()
        ax.set_xlim(0,5)
        ax.set_ylim(0,1)

        fig.canvas.draw_idle()
    except FileNotFoundError:
        pass  # Ignore if file is missing

slider_grel.on_changed(update)
slider_gdeph.on_changed(update)

plt.show()







# %%
## Plot loss function scan

import matplotlib.pyplot as plt
import numpy as np


g_rel_list=[0.05,0.1,0.2,0.4,0.6,0.8]

g_deph_list=[0.05,0.1,0.2,0.4,0.6,0.8]


n_g_rel=len(g_rel_list)
n_g_deph=len(g_deph_list)


L=100
ntraj=512

folder=f"results/cpu_traj_scan/method_tjm_gamma_scan/solver_exact/order_1/threshold_1e-4/{L}_sites/32_cpus/{ntraj}_traj"


def compute_loss(ref_traj,traj):
    diff = ref_traj - traj
    loss = np.sum(diff**2)
    return loss


def load_traj(g_rel,g_deph, folder):
    traj_folder = f"{folder}/gamma_rel_{g_rel}/gamma_deph_{g_deph}/"

    traj_file=traj_folder+"ref_traj.txt"

    if os.path.exists(traj_file):
    
        traj_data = np.genfromtxt(traj_file)

    else:

        print(f"File {traj_file} not found!!!")


    t=traj_data[:,0]

    n_t=len(t)

    n_obs_site=(len(traj_data[0])-1)//L

    traj = traj_data[:,1:].reshape(n_obs_site,L,n_t)

    return traj


def scan_loss(ref_traj, folder):

    g_rel_plt=[]

    g_deph_plt=[]

    loss_plt=[]

    for i,g_rel in enumerate(g_rel_list):
        for j,g_deph in enumerate(g_deph_list):

            traj=load_traj(g_rel,g_deph, folder)

            loss_plt.append( compute_loss(ref_traj,traj))

            g_rel_plt.append(g_rel)
            g_deph_plt.append(g_deph)


    return g_rel_plt, g_deph_plt, loss_plt
    

def plot_loss(g_rel_plt, g_deph_plt, loss_plt, folder):

    plt.figure(figsize=(8, 6))
    contour = plt.tricontourf(g_rel_plt, g_deph_plt, loss_plt, levels=100, cmap='viridis', vmin=0, vmax=max(loss_plt))
    plt.xlabel("gamma_rel")
    plt.ylabel("gamma_deph")
    # Add colorbar with label
    cbar = plt.colorbar(contour)
    cbar.set_label("Loss")
    output_file =folder + "/loss.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()


max_loss=[]
min_loss=[]


for g_rel in g_rel_list:
    for g_deph in g_deph_list:


        ref_traj=load_traj(g_rel,g_deph, folder)

        g_rel_plt, g_deph_plt, loss_plt= scan_loss(ref_traj, folder)

        max_loss.append(max(loss_plt))
        min_loss.append(min(loss_plt))



        plot_loss(g_rel_plt, g_deph_plt, loss_plt, f"{folder}/gamma_rel_{g_rel}/gamma_deph_{g_deph}/")



# %%



fig, axs = plt.subplots(1, 1, figsize=(7, 5))
col = 1

ntraj = 2048


traj_file=f"results/cpu_traj_scan/method_qutip/solver_exact/order_1/threshold_1e-4/5_sites/10_cpus/1_traj/gamma_rel_0.1/gamma_deph_0.1/ref_traj.txt"
data = np.loadtxt(traj_file)
axs.plot(data[:, 0], data[:, col], label=f"qutip")


traj_file=f"results/cpu_traj_scan/method_tjm/solver_exact/order_1/threshold_1e-4/5_sites/10_cpus/{ntraj}_traj/gamma_rel_0.1/gamma_deph_0.1/ref_traj.txt"
data = np.loadtxt(traj_file)
axs.plot(data[:, 0], data[:, col], label=f"tjm - {ntraj} trajs")


traj_file=f"results/cpu_traj_scan/method_scikit_tt/solver_exact/order_1/threshold_1e-4/5_sites/10_cpus/{ntraj}_traj/gamma_rel_0.1/gamma_deph_0.1/ref_traj.txt"
data = np.loadtxt(traj_file)
axs.plot(data[:, 0], data[:, col], label=f"scikit-tt - {ntraj} trajs")


traj_file=f"results/cpu_traj_scan/method_scikit_tt/solver_exact/order_2/threshold_1e-6/5_sites/10_cpus/{ntraj}_traj/gamma_rel_0.1/gamma_deph_0.1/ref_traj.txt"
data = np.loadtxt(traj_file)
axs.plot(data[:, 0], data[:, col], label=f"scikit-tt order_2- {ntraj} trajs")


axs.set_xlabel("Time")
axs.set_ylabel("Value")
axs.legend()
# %%
