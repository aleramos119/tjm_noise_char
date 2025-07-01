
#%%
import matplotlib.pyplot as plt
import numpy as np
import os

import pandas as pd

#%%
L_list_initial=[10,50,100]

folder="results/optimization/d_2L/"


ntraj_list=[512]

ntraj=512


#%%

plt.rcParams.update({'axes.linewidth': 1.2})
plt.rcParams.update({'font.size': 12, 'lines.linewidth': 2, 'lines.markersize': 6})

plt.figure(figsize=(15, 5))


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

L=5
ntraj=512
x_avg_file=folder + f"L_{L}/ntraj_{ntraj}/loss_x_history_avg.txt"

data = np.genfromtxt(x_avg_file, skip_header=1)

gammas_file=folder + f"L_{L}/ntraj_{ntraj}/gammas.txt"
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



# %%










# %%
%matplotlib inline
L=5
ntraj=512
mem_usage = pd.read_csv(f"results/optimization/d_2L/L_{L}/ntraj_{ntraj}/sstat_log.csv")

plt.figure()
plt.plot(mem_usage.iloc[:, -1])
plt.xlabel("Step")
plt.ylabel("Memory Usage")
plt.title("Memory Usage Over Steps")
plt.show()
# plt.savefig(f"{folder}/mem_usage_L_{L}_ntraj_{ntraj}.pdf", dpi=300, bbox_inches='tight')
# %%
