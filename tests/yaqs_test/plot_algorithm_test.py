
#%%
import matplotlib.pyplot as plt
import numpy as np

#%%
N_list=[500,1000,2000,4000,8000]
samp_list=[50]


folder="results/"

folder_list=["ADAM_test","sweep_GPModelWithDerivatives_UCB_test","GPModelWithDerivatives_UCB_test", "GPModelWithDerivatives_LEI_test","GPModel_LEI_test",  "GPModel_UCB_test" ]

file="time_error_vs_d.txt"

column_names=["d" , "min_error", "avg_error" , "max_error", "n_iter" ]


col1=column_names.index("d")
col2=column_names.index("max_error")


for folder_name in folder_list:
    data = np.genfromtxt(f"{folder}/{folder_name}/{file}", skip_header=1)

    x_values = data[:, col1]
    y_values = data[:, col2]
    plt.plot(x_values, np.log10(y_values), label=folder_name)

plt.xlabel(column_names[col1])
plt.ylabel(column_names[col2])
plt.legend()
plt.show()


# %%
