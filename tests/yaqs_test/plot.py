
#%%
import matplotlib.pyplot as plt
import numpy as np

#%%
header="#  Iter    Loss    Log10(Loss)    Gamma_rel    Gamma_deph "


#,"secant_penalized_bfgs_500_8000_max_iter_60"

folder_list=["adam_bfgs_500_8000_max_iter_60","adam_bfgs_loss_test_500_8000_max_iter_60","adam_bfgs_rate_test_500_8000_max_iter_60"]

file_list=["iter.txt","time.txt","min_log_loss.txt","error.txt","time_per_iter.txt"]

traj_list=[500,1000,2000,4000,8000]

alg_list=["adam","bfgs"]

#%%

for folder in folder_list:
    for file in file_list:
        filename = folder + "/" + file
        data = np.genfromtxt(filename, skip_header=1)
        
        with open(filename, 'r') as f:
            column_names = f.readline().strip().split()

        for i in range(1,data.shape[1]):        
            plt.plot(data[:, 0], data[:, i], '-o', label=column_names[i+1])
        plt.xlabel(column_names[1])
        plt.ylabel(file.replace(".txt", ""))
        plt.title(f"Plot for {file} in {folder}")
        plt.legend()
        plt.savefig(f"{folder}/{file.replace('.txt', '.png')}")
        plt.close()

    for traj in [8000]:

        filename_adam= folder + "/" + "adam_log_" + str(traj) + ".txt"
        filename_bfgs= folder + "/" + "bfgs_log_" + str(traj) + ".txt"
        data_adam = np.genfromtxt(filename_adam, skip_header=1)
        data_bfgs = np.genfromtxt(filename_bfgs, skip_header=1)

        with open(filename_adam, 'r') as f:
                column_names = f.readline().strip().split()

        plt.plot(data_adam[:, 0], data_adam[:, 2], '-o', label="adam"+column_names[3])
        plt.plot(data_bfgs[:, 0], data_bfgs[:, 2], '-o', label="bfgs"+column_names[3])
        plt.xlabel(column_names[1])
        plt.ylabel(column_names[3])
        plt.title(f"Plot for {alg} error in {folder}")
        plt.legend()
        plt.savefig(f"{folder}/{alg}_{traj}_{column_names[3]}.png")
        plt.close()


        for alg in alg_list:
            filename = folder + "/" + alg + "_log_" + str(traj) + ".txt"
            data = np.genfromtxt(filename, skip_header=1)
            
            with open(filename, 'r') as f:
                column_names = f.readline().strip().split()

            plt.plot(data[:, 0], data[:, 3], '-o', label=column_names[4])
            plt.plot(data[:, 0], data[:, 4], '-o', label=column_names[5])
            plt.xlabel(column_names[1])
            plt.axhline(y=0.1, color='r', linestyle='--')
            plt.title(f"Plot for {alg} error in {folder}")
            plt.legend()
            plt.savefig(f"{folder}/{alg}_{traj}_gamma.png")
            plt.close()


#%%


plt.plot(data[:,0],data[:,2],'-o')
plt.xlabel("Iteration")


# %%

plt.plot(data[:,0],data[:,3:5],'-o')
plt.axhline(y=0.1, color='r', linestyle='--', label='y=0.1')
plt.xlabel("Iteration")

# %%
%matplotlib qt
filename_2 = "adam_bfgs_500_8000_max_iter_60/error.txt"

data_2=np.genfromtxt(filename_2, skip_header=1)


plt.plot(data_2[:,0],data_2[:,1],'-o',label='adam')

plt.plot(data_2[:,0],data_2[:,2],'-o',label='bfgs')
plt.legend()
plt.xlabel("Iteration")
# %%
data_2
# %%
data.shape
# %%
filename
# %%
column_names

# %%
