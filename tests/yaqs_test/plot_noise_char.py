
#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

#%%
N_list=[500,1000,2000,4000,8000]
samp_list=[50]

file_list=["loss.txt","gamma_rel.txt","gamma_deph.txt"]



#%%
func_list=["mean","std"]

for N in N_list:


    folder=f"noise_char/N_{N}/samples_50/"

    for file in file_list:

        full_file=folder+file
        data=np.genfromtxt(full_file)

        for func in func_list:

            if func=="mean":
                data_1=np.array([[data[i,0],data[i,1], np.mean(data[i,2:])] for i in range(len(data))])

            if func=="std":
                data_1=np.array([[data[i,0],data[i,1], np.std(data[i,2:])] for i in range(len(data))])
        
            plt.figure(figsize=(8, 6))
            contour = plt.tricontourf(data_1[:,0], data_1[:,1], data_1[:,2], levels=100, cmap='viridis')
            plt.colorbar(contour, label=f"{func}")
            plt.xlabel("X-axis (Column 1)")
            plt.ylabel("Y-axis (Column 2)")
            plt.title(f"{file.split('.')[0]}")
            output_file = f"{folder}{file.split('.')[0]}_{func}.png"
            plt.savefig(output_file, dpi=300, bbox_inches='tight')
            plt.close()


#%%


for file in file_list:
    std_list_max=[]
    std_list_min=[]
    std_list_avg=[]

    for N in N_list:
        folder=f"noise_char/N_{N}/samples_50/"

        full_file=folder+file
        data=np.genfromtxt(full_file)

        std_list_max.append(np.max(np.std(data[:,2:],axis=1)))
        std_list_min.append(np.min(np.std(data[:,2:],axis=1)))
        std_list_avg.append(np.mean(np.std(data[:,2:],axis=1)))

    plt.figure(figsize=(8, 6))
    plt.plot(N_list,std_list_max,'o-',label='max')
    plt.plot(N_list,std_list_min,'o-',label='min')
    plt.plot(N_list,std_list_avg,'o-',label='avg')
    plt.legend()
    plt.xlabel("N")
    plt.ylabel("std")
    plt.title(f"{file.split('.')[0]}")
    output_file = f"noise_char/{file.split('.')[0]}_std.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    

# %%
full_file=f"noise_char/N_{4000}/samples_50/gamma_deph.txt"

data=np.genfromtxt(full_file)


plt.figure(figsize=(8, 6))
plt.hist(data[0, 2:], bins=10, color='blue', alpha=0.7)
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.title("Distribution of Values")
plt.grid(True)
plt.tight_layout()
# %%



# %%


L=3

n_obs=3*L
n_t=61


ntraj=3370


full_data=np.zeros((n_t,n_obs,ntraj))
for i in range(1,ntraj+1):
    traj=f"results/propagation/yaqs/L_{L}/run_{i}/ref_traj.txt"
    data=np.genfromtxt(traj)[:,1:]
    
    full_data[:,:,i-1]=data

time=np.genfromtxt(traj)[:,0]

# %%
obs_idx = 0     # 0 <= obs_idx < full_data.shape[1]

sample_list=[1, 10, 100, 1000, 2000, 3000]

time_list=[0, 15, 30, 60]

for n_samples in sample_list:
    for time_idx in time_list:

        print(f"n_samples={n_samples}, time_idx={time_idx}")

        full_data_avg = np.zeros_like(full_data)
        rng = np.random.default_rng(42)  # change or remove seed for different draws

        for j in range(ntraj):
            idx = rng.choice(ntraj, size=n_samples, replace=True)
            # average over the sampled trajectories (axis=2)
            full_data_avg[:, :, j] = np.mean(full_data[:, :, idx], axis=2)

        # now full_data_avg has shape (n_t, n_obs, ntraj) with the averaged data


        # select indices (0-based). change these to the desired time and observable indices


        # extract data over trajectories for the chosen time and observable
        samples = full_data_avg[time_idx, obs_idx, :].ravel()

        # plot histogram (normalized to a probability density)
        plt.figure(figsize=(8, 6))
        n_bins = 30
        counts, bins, patches = plt.hist(samples, bins=n_bins, density=True, alpha=0.6, color='C0', label='Histogram')

        # try to add a KDE if scipy is available
        try:
            kde = gaussian_kde(samples)
            x = np.linspace(bins[0], bins[-1], 400)
            plt.plot(x, kde(x), color='C1', lw=2, label='KDE')
        except Exception:
            pass

        plt.xlabel("Observable value")
        plt.ylabel("Probability density")
        plt.title(f"Distribution at time {time[time_idx]}, t_max={time[-1]}, obs {obs_idx}, N_traj={n_samples}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # save figure
        output_file = f"results/propagation/yaqs/plots/L_{L}/distribution_t_{time[time_idx]}_obs_{obs_idx}_ntraj_{n_samples}.png"
        print(f"Saving to {output_file}")
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
# %%
