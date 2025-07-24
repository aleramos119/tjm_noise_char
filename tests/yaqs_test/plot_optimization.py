
#%%
import matplotlib.pyplot as plt
import numpy as np
import os

import pandas as pd

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


L=10
ntraj=512

folder = "test/optimization/"
x_avg_file=folder + f"loss_x_history.txt"

data = np.genfromtxt(x_avg_file, skip_header=1)

gammas_file=folder + f"gammas.txt"
gammas = np.genfromtxt(gammas_file, skip_header=1)

d=len(gammas)

for i in range(d):
    plt.plot(data[:, 0], data[:, 2 + i], label=f"$\\gamma_{{{i+1}}}$")
    plt.axhline(gammas[i], color=plt.gca().lines[-1].get_color(), linestyle='--', linewidth=2)


plt.xlabel("Iterations")
plt.ylabel(r"$\gamma$")
plt.legend()
plt.show()
# plt.savefig(f"{folder}/gamma_avg_vs_iterations_L_{L}_ntraj_{ntraj}.pdf", dpi=300, bbox_inches='tight')




#%%
# Plot the loss from loss_x_history.txt

L = 10
ntraj = 512

folder = "test/optimization/"
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

L=10
ntraj=512


col=3


folder="test/optimization/"

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
