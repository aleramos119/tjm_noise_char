
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
import os
import numpy as np

L=3
n_obs_L=3
n_obs=n_obs_L*L
n_t=61


traj_indexes = []   

for i in range(1, 10000 + 1):
    traj = f"results/propagation/yaqs/L_{L}/run_{i}/ref_traj.txt"
    if not os.path.isfile(traj):
        print(f"Skipping missing file: {traj}")
        continue

    traj_indexes.append(i)


ntraj=len(traj_indexes)
full_data=np.zeros((n_t,n_obs,ntraj))
j=0
for i in traj_indexes:
    traj = f"results/propagation/yaqs/L_{L}/run_{i}/ref_traj.txt"
    
    if not os.path.isfile(traj):
        print(f"Skipping missing file: {traj}")
        continue
    
    data = np.genfromtxt(traj)[:, 1:]
    full_data[:, :, j] = data
    time=np.genfromtxt(traj)[:,0]
    j += 1

print(f"Loaded {ntraj} trajectories.")


split_data = full_data.reshape(n_t, n_obs_L, L, ntraj)
ref_traj = np.mean(full_data, axis=2)

#%%
split_data.shape
# %%
obs_idx = 0     # 0 <= obs_idx < full_data.shape[1]

sample_list=[1, 10, 20, 80]

# sample_list=[1, 10, 100]

time_list=[15, 30, 60]

for n_samples in sample_list:

    rng = np.random.default_rng(42)  # change or remove seed for different draws

    n_samp_avg = 10000
    full_data_avg = np.zeros((n_t, n_obs, n_samp_avg))
    for j in range(n_samp_avg):
        idx = rng.choice(ntraj, size=n_samples, replace=True)
        # average over the sampled trajectories (axis=2)
        full_data_avg[:, :, j] = np.mean(full_data[:, :, idx], axis=2)


    for time_idx in time_list:

        print(f"n_samples={n_samples}, time_idx={time_idx}")

        

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

        plt.xlim(0,0.5)
        plt.ylim(0,35)
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


#%%

import matplotlib.pyplot as plt

# compose a grid of saved distribution images
# relies on sample_list, time_list, time, L, obs_idx being defined above

n_rows = len(sample_list)
n_cols = len(time_list)

fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows))

# normalize axes to 2D array
if n_rows == 1 and n_cols == 1:
    axes = np.array([[axes]])
elif n_rows == 1:
    axes = np.array([axes])
elif n_cols == 1:
    axes = np.array([[ax] for ax in axes])

for i, n_samples in enumerate(sample_list):
    for j, time_idx in enumerate(time_list):
        ax = axes[i, j]
        # filename format used when saving individual plots
        fname = f"results/propagation/yaqs/plots/L_{L}/distribution_t_{time[time_idx]}_obs_{obs_idx}_ntraj_{n_samples}.png"
        if os.path.exists(fname):
            img = plt.imread(fname)
            ax.imshow(img)
            ax.axis("off")
            ax.set_title(f"t = {time[time_idx]}, N_traj = {n_samples}")

        else:
            ax.text(0.5, 0.5, f"Missing\n{os.path.basename(fname)}", ha="center", va="center")
            ax.set_xticks([])
            ax.set_yticks([])


plt.suptitle(f"Composed distributions — L={L}, obs={obs_idx}", fontsize=14)
plt.tight_layout(rect=[0, 0, 1, 0.96])

out_dir = f"results/propagation/yaqs/plots/L_{L}/"
os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, f"composed_distributions_obs_{obs_idx}.png")
plt.savefig(out_file, dpi=300, bbox_inches="tight")
plt.close()

# %%
n_samples=10


rng = np.random.default_rng(42)  # change or remove seed for different draws

n_samp_avg = 10000
split_data_avg = np.zeros((n_t, n_obs_L, L, n_samp_avg))
for j in range(n_samp_avg):
    idx = rng.choice(ntraj, size=n_samples, replace=True)
    # average over the sampled trajectories (axis=2)
    split_data_avg[:, :, :, j] = np.mean(split_data[:, :, :, idx], axis=3)

delta_data = split_data_avg**2


#%%
import matplotlib.pyplot as plt

%matplotlib qt

obs_idx = 2
L_index = 0
time_idx = 60


# flattened_data = delta_data.reshape(n_t* n_obs, n_samp_avg)
transposed_data = delta_data.transpose(0,1,2,3)
final_data = transposed_data.reshape(n_t * n_obs_L * L, n_samp_avg)

C = np.abs(np.cov(final_data))



plt.imshow(C)                  # show matrix (default colormap)
plt.colorbar()                 # add color scale
plt.title("Covariance Matrix")
plt.xlabel("Variables")
plt.ylabel("Variables")
# plt.savefig(f"results/propagation/yaqs/plots/L_{L}/correlation_matrix_obs_{obs_idx}_ntraj_{n_samples}.png", dpi=300, bbox_inches='tight')
plt.show()

# %%
time_i= 27
time_j= 35
obs_idx_i= 1
obs_idx_j= 1


plt.plot(full_data_avg[time_i,obs_idx,:], full_data_avg[time_j,obs_idx,:], 'o', alpha=0.5)
# plt.xlim(0.3, 0.8)
# plt.ylim(0.3, 0.8)
# %%
