
#%%
import matplotlib.pyplot as plt
import numpy as np
import os

#%%
L_list_initial=[10,20,40,80,100]

folder="results/optimization/d_2_ntraj0_8192/"


ntraj_list=[128,256,512,1024]

ntraj=512




plt.rcParams.update({'axes.linewidth': 1.2})
plt.rcParams.update({'font.size': 12, 'lines.linewidth': 2, 'lines.markersize': 6})


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
plt.savefig(f"{folder}/error_vs_L.pdf", dpi=300, bbox_inches='tight')




# %%

L=100
ntraj=1024
x_avg_file=folder + f"L_{L}/ntraj_{ntraj}/loss_x_history_avg.txt"

data = np.genfromtxt(x_avg_file, skip_header=1)


plt.plot(data[:,0], data[:,-2], label=r"$\gamma_{r}$")
plt.plot(data[:,0], data[:,-1], label=r"$\gamma_{d}$")
plt.xlabel("Iterations")
plt.ylabel(r"$\gamma$")
plt.legend()
plt.axhline(0.1, color='gray', linestyle='--', linewidth=2, label='0.1')
plt.savefig(f"{folder}/gamma_avg_vs_iterations_L_{L}_ntraj_{ntraj}.pdf", dpi=300, bbox_inches='tight')



# %%
L=100
ntraj=1024
x_avg_file=folder + f"L_{L}/ntraj_{ntraj}/loss_x_history.txt"

data = np.genfromtxt(x_avg_file, skip_header=1)


plt.plot(data[:,0], data[:,-2], label=r"$\gamma_{r}$")
plt.plot(data[:,0], data[:,-1], label=r"$\gamma_{d}$")
plt.xlabel("Iterations")
plt.ylabel(r"$\gamma$")
plt.legend()
plt.axhline(0.1, color='gray', linestyle='--', linewidth=2, label='0.1')
plt.savefig(f"{folder}/gamma_vs_iterations_L_{L}_ntraj_{ntraj}.pdf", dpi=300, bbox_inches='tight')


# %%







#%%

L=10
ntraj=1024
x_avg_file=folder + f"L_{L}/ntraj_{ntraj}/loss_x_history.txt"

data = np.genfromtxt(x_avg_file, skip_header=1)


plt.plot(data[:,0], data[:,-2], label=r"$\gamma_{r}$")
plt.plot(data[:,0], data[:,-1], label=r"$\gamma_{d}$")
plt.xlabel("Iterations")
plt.ylabel(r"$\gamma$")
plt.legend()
plt.axhline(0.1, color='gray', linestyle='--', linewidth=2, label='0.1')
plt.savefig(f"{folder}/gamma_vs_iterations_L_{L}_ntraj_{ntraj}.pdf", dpi=300, bbox_inches='tight')






# %%
