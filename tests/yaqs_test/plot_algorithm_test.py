
#%%
import matplotlib.pyplot as plt
import numpy as np

#%%
N_list=[500,1000,2000,4000,8000]
samp_list=[50]


folder="results/algorithm_test/"

folder_list=[ "BFGS_test", "ADAM_test","GPModelWithDerivatives_UCB_test", "GPModel_UCB_test" ]

alg_name_list=["Secant_Penalized BFGS", "ADAM", "BO GP_Derivatives_Kernel", "BO GP_Additive_Kernel"]



file="time_error_vs_d.txt"

column_names=["d" , "min_error", "avg_error" , "max_error", "n_iter" ]


col1=column_names.index("d")
col2=column_names.index("max_error")


plt.rcParams.update({'axes.linewidth': 1.2})
plt.rcParams.update({'font.size': 12, 'lines.linewidth': 2, 'lines.markersize': 6})
for i,folder_name in enumerate(folder_list):
    data = np.genfromtxt(f"{folder}/{folder_name}/{file}", skip_header=1)

    x_values = data[:, col1]
    y_values = data[:, col2]
    plt.plot(x_values, np.log10(y_values), label=alg_name_list[i], marker='o')

plt.xlabel(column_names[col1])
plt.ylabel(r"$log( e_{max})$")
plt.legend()
plt.savefig(f"{folder}/time_error_vs_d.pdf", dpi=300, bbox_inches='tight')




# %%
