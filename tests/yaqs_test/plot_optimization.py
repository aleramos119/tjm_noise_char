
#%%
import matplotlib.pyplot as plt
import numpy as np
import os

import pandas as pd
import glob
from matplotlib.widgets import Slider

#%%
L_list_initial=[10,20,40, 80, 100]

folder="results/optimization/d_2/"


ntraj_list=[512, 1024]

ntraj=512


#%%
### Plotting the error vs L for different ntraj values
plt.rcParams.update({'axes.linewidth': 1.2})
plt.rcParams.update({'font.size': 12, 'lines.linewidth': 2, 'lines.markersize': 6})

plt.figure(figsize=(9, 6))


for ntraj in ntraj_list:

    error_list=[]
    L_list = []



    for L in L_list_initial:

        x_avg_file=folder + f"L_{L}/ntraj_{ntraj}/loss_x_history_avg.txt"

        print(x_avg_file)

        if os.path.exists(x_avg_file):

            data = np.genfromtxt(x_avg_file, skip_header=1)

            if len(data) > 1:

                error_list.append(max(abs(data[-1,-2:]-0.1)))

                L_list.append(L)

    plt.plot(L_list, np.log10(np.array(error_list)), marker='o', label=f"N_traj={ntraj}")

plt.xlabel(r"N")
plt.ylabel(r"$log( e_{max})$")
plt.legend()
plt.savefig(f"{folder}/error_vs_L_15x5.pdf", dpi=300, bbox_inches='tight')




# %%
### Plotting the average gamma values over iterations


L=120
ntraj=256
max_bond_dim=8
gamma="random"
gamma_0="random"

folder = f"results/optimization/method_tjm_exact_opt_script_test/max_bond_dim_{max_bond_dim}/d_2/gamma_{gamma}/gamma_0_{gamma_0}/L_{L}/ntraj_{ntraj}/"
x_avg_file=folder + f"loss_x_history.txt"

data = np.genfromtxt(x_avg_file, skip_header=1)

gammas_file=folder + f"gammas.txt"
gammas = np.genfromtxt(gammas_file, skip_header=1)

d=len(gammas)

for i in np.random.choice(range(d), size=2, replace=False):
    plt.plot(data[:, 0], data[:, 2 + i], label=f"$\\gamma_{{{i+1}}}$")
    plt.axhline(gammas[i], color=plt.gca().lines[-1].get_color(), linestyle='--', linewidth=2)


plt.xlabel("Iterations")
plt.ylabel(r"$\gamma$")
plt.legend()
plt.show()
# plt.savefig(f"{folder}/gamma_avg_vs_iterations_L_{L}_ntraj_{ntraj}.pdf", dpi=300, bbox_inches='tight')

#%%

data

#%%
# Plot the loss from loss_x_history.txt

L=80
ntraj=512

folder = f"results/optimization/method_tjm_exact/d_2_test/L_{L}/ntraj_{ntraj}/"
loss_file = folder + f"loss_x_history.txt"

if os.path.exists(loss_file):
    data = np.genfromtxt(loss_file, skip_header=1)
    plt.figure(figsize=(8, 5))
    plt.plot(data[:, 0], np.log10(data[:, 1]), label="Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss vs Iterations")
    plt.legend()
    plt.show()
else:
    print("Loss file not found.")






#%%

#### Plot reference and optimized trajectory


col=1


L=80
ntraj=512

folder = f"results/optimization/method_tjm_exact/d_2_test/L_{L}/ntraj_{ntraj}/"

ref_traj_file = folder + f"ref_traj.txt"
opt_traj_file = folder + f"opt_traj.txt"

if os.path.exists(ref_traj_file) and os.path.exists(opt_traj_file):
    ref_traj = np.genfromtxt(ref_traj_file, skip_header=1)
    opt_traj = np.genfromtxt(opt_traj_file, skip_header=1)

    plt.figure(figsize=(8, 5))
    plt.plot(ref_traj[:, 0], ref_traj[:, col], label="Reference Trajectory")
    plt.plot(opt_traj[:, 0], opt_traj[:, col], label="Optimized Trajectory", linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Trajectory Value")
    plt.legend()
    plt.title("Comparison of Reference and Optimized Trajectories")
    plt.show()
else:
    print("Reference or optimized trajectory file not found.")



#%%

loss=np.sum((ref_traj-opt_traj)**2)
print(loss)




#%%

#### Plot reference and optimized trajecotry
folder="results/optimization/d_2/"
L=100
ntraj=512


ref_traj_file = folder + f"L_{L}/ntraj_{ntraj}/ref_traj.txt"
opt_traj_file = folder + f"L_{L}/ntraj_{ntraj}/opt_traj.txt"

if os.path.exists(ref_traj_file):
    ref_traj = np.genfromtxt(ref_traj_file, skip_header=1)
    # opt_traj = np.genfromtxt(opt_traj_file, skip_header=1)

    plt.figure(figsize=(8, 5))
    plt.plot(ref_traj[:, 0], ref_traj[:, 1], label="Reference Trajectory")
    # plt.plot(opt_traj[:, 0], opt_traj[:, 1], label="Optimized Trajectory", linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Trajectory Value")
    plt.legend()
    plt.title("Comparison of Reference and Optimized Trajectories")
    plt.show()
else:
    print("Reference or optimized trajectory file not found.")





#%%

#### Plot reference and opptimized trajecotry
folder="results/optimization/d_2/"
L=100
ntraj=512


ref_traj_file = f"results/cpu_traj_scan/method_scikit_tt_new_calc_omp_1/solver_krylov_5/order_1/threshold_1e-4/{L}_sites/96_cpus/{ntraj}_traj/qt_ref_traj.txt"

if os.path.exists(ref_traj_file):
    ref_traj = np.genfromtxt(ref_traj_file, skip_header=1)
    # opt_traj = np.genfromtxt(opt_traj_file, skip_header=1)

    plt.figure(figsize=(8, 5))
    plt.plot(ref_traj[:, 0], ref_traj[:, 1], label="Reference Trajectory")
    # plt.plot(opt_traj[:, 0], opt_traj[:, 1], label="Optimized Trajectory", linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Trajectory Value")
    plt.legend()
    plt.title("Comparison of Reference and Optimized Trajectories")
    plt.show()
else:
    print("Reference or optimized trajectory file not found.")


#%%
ref_traj.shape



# %%

# Folder and file pattern

%matplotlib qt

col=300

L=120
ntraj=256
max_bond_dim=12
gamma="random"
gamma_0="random"

folder = f"results/optimization/method_tjm_exact_opt_script_test/max_bond_dim_{max_bond_dim}/d_2/gamma_{gamma}/gamma_0_{gamma_0}/L_{L}/ntraj_{ntraj}/"


file_pattern = folder + f"opt_traj_*.txt"

# Find all matching files and sort by index
opt_traj_files = sorted(glob.glob(file_pattern), key=lambda x: int(x.split('_')[-1].split('.')[0]))
if not opt_traj_files:
    print("No opt_traj_{i}.txt files found.")
else:
    # Load all trajectories
    opt_trajs = [np.genfromtxt(f, skip_header=1) for f in opt_traj_files]


    ref_traj = np.genfromtxt(folder + f"ref_traj.txt", skip_header=1)

    # Initial plot
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.subplots_adjust(bottom=0.2)
    ax.plot(ref_traj[:, 0], ref_traj[:, col], label="Reference Trajectory")
    l, = ax.plot(opt_trajs[0][:, 0], opt_trajs[0][:, col], label="Optimized Trajectory", linestyle='--')

    ax.set_xlabel("Time")
    ax.set_ylabel("Trajectory Value")
    ax.set_title("Optimized Trajectory (select index with slider)")
    ax.legend()
    

    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Index', 0, len(opt_trajs)-1, valinit=0, valstep=1)

    def update(val):
        idx = int(slider.val)
        l.set_ydata(opt_trajs[idx][:, col])
        l.set_xdata(opt_trajs[idx][:, 0])
        # ax.relim()
        # ax.autoscale_view()
        fig.canvas.draw_idle()
        ax.legend()

    slider.on_changed(update)
    plt.show()
# %%
ref_traj.shape
# %%

### Plot error gammas


L=5
ntraj_list=[512, 1024, 2048, 4096, 8192]

error_list=[]

for ntraj in ntraj_list:

    folder = f"results/optimization/method_tjm_exact/d_2L_test/L_{L}/ntraj_{ntraj}/"
    x_avg_file=folder + f"loss_x_history_avg.txt"

    data = np.genfromtxt(x_avg_file, skip_header=1)

    gammas_file=folder + f"gammas.txt"
    gammas = np.genfromtxt(gammas_file, skip_header=1)

    d=len(gammas)

    error_list.append(np.log10(max(abs(data[-1,2:]-gammas))))
    


plt.plot(ntraj_list, error_list,'o-', label=f"L={L}")


#%%


data[-1, 2:].shape



# %%
