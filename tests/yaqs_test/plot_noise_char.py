
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



##############################################################
##   Load trajectories
##############################################################
import os
import numpy as np
import matplotlib.pyplot as plt

def load_traj(method, L):  

    n_obs_L=3
    n_obs=n_obs_L*L
    n_t=61


    traj_indexes = []   

    for i in range(1, 10000 + 1):
        traj = f"results/propagation/{method}/L_{L}/run_{i}/ref_traj.txt"
        if not os.path.isfile(traj):
            # print(f"Skipping missing file: {traj}")
            continue

        traj_indexes.append(i)

    ntraj=len(traj_indexes)
    full_data=np.zeros((n_t,n_obs,ntraj))
    j=0
    for i in traj_indexes:
        traj = f"results/propagation/{method}/L_{L}/run_{i}/ref_traj.txt"
        
        if not os.path.isfile(traj):
            # print(f"Skipping missing file: {traj}")
            continue
        
        data = np.genfromtxt(traj)[:, 1:]
        full_data[:, :, j] = data
        time=np.genfromtxt(traj)[:,0]
        j += 1

    print(f"Loaded {ntraj} trajectories.")


    split_data = full_data.reshape(n_t, n_obs_L, L, ntraj)
    ref_traj = np.mean(split_data, axis=3)

    return split_data, ref_traj, time

# %%

########## Loads reference trajectories


L=3
n_obs_L=3
n_obs=n_obs_L*L
n_t=61



split_data, ref_traj, time = load_traj("yaqs", L)
split_data_scikit, ref_traj_scikit, time_scikit = load_traj("scikit_tt",L)

n_t, n_obs_L, L, ntraj = split_data.shape


#%%



#%%

#### Plots reference trajectories
obs_idx=0
plt.plot(ref_traj[:,obs_idx], 'o-', label=f"yaqs L={L} obs_{obs_idx}" )
plt.plot(ref_traj_scikit[:,obs_idx], 'x-', label=f"scikit_tt L={L} obs_{obs_idx}")
plt.legend()
plt.savefig(f"results/propagation/ref_traj_L_{L}_obs_{obs_idx}.png", dpi=300, bbox_inches='tight')
plt.show()
#%%


# %%
obs_idx = 0     # 0 <= obs_idx < full_data.shape[1]

L_index=0

sample_list=[1, 10, 20, 80]

# sample_list=[1, 10, 100]

time_list=[15, 30, 60]

for n_samples in sample_list:

    rng = np.random.default_rng(42)  # change or remove seed for different draws

    n_samp_avg = 10000
    full_data_avg = np.zeros((n_t, n_obs_L, L, n_samp_avg))
    for j in range(n_samp_avg):
        idx = rng.choice(ntraj, size=n_samples, replace=True)
        # average over the sampled trajectories (axis=2)
        full_data_avg[:, :, :, j] = np.mean(split_data[:, :, :, idx], axis=2)


    for time_idx in time_list:

        print(f"n_samples={n_samples}, time_idx={time_idx}")

        

        # now full_data_avg has shape (n_t, n_obs, ntraj) with the averaged data


        # select indices (0-based). change these to the desired time and observable indices


        # extract data over trajectories for the chosen time and observable
        samples = full_data_avg[time_idx, obs_idx, L_index, :].ravel()

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
        plt.title(f"Distribution at time {time[time_idx]}, t_max={time[-1]}, obs {obs_idx}, L_index={L_index}, N_traj={n_samples}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()

        # save figure
        output_file = f"results/propagation/yaqs/plots/L_{L}/distribution_t_{time[time_idx]}_obs_{obs_idx}_L_index_{L_index}_ntraj_{n_samples}.png"
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
n_samples=500


L=50
n_obs_L=3
n_obs=n_obs_L*L
n_t=61


method="yaqs"


split_data, ref_traj, time = load_traj("yaqs", L)

n_t, n_obs_L, L, ntraj = split_data.shape


rng = np.random.default_rng(42)  # change or remove seed for different draws

n_samp_avg = 1000
split_data_avg = np.zeros((n_t, n_obs_L, L, n_samp_avg))
for j in range(n_samp_avg):
    idx = rng.choice(ntraj, size=n_samples, replace=True)
    # average over the sampled trajectories (axis=2)
    split_data_avg[:, :, :, j] = np.mean(split_data[:, :, :, idx], axis=3)



delta_data = split_data_avg**2


#%%
import matplotlib.pyplot as plt
import matplotlib as mpl

%matplotlib qt

obs_idx = 2
L_index = 0
time_idx = 60


# flattened_data = delta_data.reshape(n_t* n_obs, n_samp_avg)
transposed_data = delta_data.transpose(2,1,0,3)[:,0,29,:]
final_data = transposed_data.reshape(L, n_samp_avg)

C = np.abs(np.cov(final_data))

# --- Publication-quality plotting for covariance matrix ---
# Set scientific plotting defaults
mpl.rcParams.update({
    'figure.figsize': (5, 4),
    'axes.linewidth': 1.5,
    'axes.labelsize': 16,
    'axes.titlesize': 17,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'lines.linewidth': 2,
    'lines.markersize': 7,
    'font.family': 'serif',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(C, vmin=0, vmax=1e-4, origin="lower")  # set origin lower so i and i' increase to right and up
cbar = plt.colorbar(im, format="%.1e", ax=ax)
cbar.ax.tick_params(labelsize=13)  # Set colorbar tick font size

# Remove the exponent only at the top of the colorbar
cbar.ax.yaxis.get_offset_text().set_visible(False)

ax.set_title(r"$|Cov(Y_{i}, Y_{i'})|$", fontsize=17)
ax.set_xlabel(r"$i$", labelpad=4)
ax.set_ylabel(r"$i'$", labelpad=4)
# Show top and right border (make sure they're visible)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
# Do not set title; leave for caption
plt.tight_layout()
plt.savefig(f"results/propagation/yaqs/plots/L_{L}/correlation_matrix_ntraj_{n_samples}.pdf", dpi=600, bbox_inches="tight", transparent=True)
plt.close(fig)

#%%
import gc
gc.collect()


# %%
time_i= 27
time_j= 35
obs_idx_i= 1
obs_idx_j= 1


plt.plot(full_data_avg[time_i,obs_idx,:], full_data_avg[time_j,obs_idx,:], 'o', alpha=0.5)
# plt.xlim(0.3, 0.8)
# plt.ylim(0.3, 0.8)




# %%
##############################################################
##   Loss Distribution
##############################################################


L_list=[3, 20, 50, 100]

sample_list=[100, 250, 500, 1000]


method="yaqs"

n_samp_avg=5000

rel_err=np.zeros((len(L_list), len(sample_list)))

rng = np.random.default_rng(42)  # change or remove seed for different draws


for i, L in enumerate(L_list):

    split_data, ref_traj, time = load_traj(method, L)

    n_t, n_obs_L, L, ntraj_max = split_data.shape

    # Add new axis to ref_traj to enable broadcasting: (n_t, n_obs_L, L) -> (n_t, n_obs_L, L, 1)
    ref_traj_expanded = ref_traj[..., np.newaxis]  # or ref_traj[:, :, :, np.newaxis]
    loss_data = np.sum((split_data - ref_traj_expanded)**2, axis=(0,1,2))



    for j, n_samples in enumerate(sample_list):


        idx = rng.choice(ntraj_max, size=(n_samp_avg, n_samples), replace=True)

        loss_data_samples = np.mean(loss_data[idx], axis=1)


        rel_err[i, j] = np.std(loss_data_samples)/np.mean(loss_data_samples)







# %%


for i, L in enumerate(L_list):
    plt.plot(sample_list, rel_err[i, :], 'o-', label=f"L={L}")
plt.xlabel("Number of trajectories")
plt.ylabel("Relative error (Loss)")
plt.legend()

plt.savefig(f"results/propagation/yaqs/plots/rel_err_vs_ntraj.png", dpi=300, bbox_inches='tight')
# %%

# --- Publication-quality plotting for relative error vs L ---
import matplotlib as mpl

# Set scientific plotting defaults
mpl.rcParams.update({
    'figure.figsize': (5, 4),
    'axes.linewidth': 1.5,
    'axes.labelsize': 16,
    'axes.titlesize': 17,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'lines.linewidth': 2,
    'lines.markersize': 7,
    'font.family': 'serif',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

fig, ax = plt.subplots(figsize=(5, 4))
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
for i, ntraj in enumerate(sample_list):
    ax.plot(
        L_list, rel_err[:,i], 
        'o-', 
        label=r"$N_{traj}$="+f"{ntraj}",
        color=color_cycle[i % len(color_cycle)],
    )

ax.set_xlabel(r"$N_L$", labelpad=4)
ax.set_ylabel(r"$\varepsilon_{rel} ( \mathcal{J} )$", labelpad=4)
ax.legend(frameon=False, loc='best', handlelength=2)
# Show top and right border (make sure they're visible)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
# Do not set title; leave for caption
plt.tight_layout()
plt.savefig(f"results/propagation/yaqs/plots/rel_err_vs_L.pdf", dpi=600, bbox_inches="tight", transparent=True)
plt.close(fig)



# %%
%matplotlib inline
# Create interpolated function from rel_err (for irregular grids)
from scipy.interpolate import CloughTocher2DInterpolator, LinearNDInterpolator, RBFInterpolator

# Prepare data for irregular grid interpolation
# Create meshgrid of all (sample_list, L_list) combinations
X, Y = np.meshgrid(sample_list, L_list)
# Flatten to create points array: each row is (n_samples, L)
points = np.column_stack([X.ravel(), Y.ravel()])
# Flatten rel_err to match the points
values = rel_err.ravel()

# Choose interpolation order:
# 'linear' - order 1, faster but less smooth
# 'cubic' - order 3, smoother (CloughTocher)
# 'rbf' - radial basis function, can use different kernels
interp_order = 'linear'  # Change to 'linear' or 'rbf' if desired

if interp_order == 'linear':
    interp_base = LinearNDInterpolator(points, values)
elif interp_order == 'cubic':
    interp_base = CloughTocher2DInterpolator(points, values)
elif interp_order == 'rbf':
    # RBF interpolation with different kernels: 'linear', 'thin_plate_spline', 'cubic', 'quintic', 'gaussian'
    # For 'gaussian' kernel, epsilon must be specified (controls width of Gaussian)
    # Scale-invariant kernels (cubic, linear, thin_plate_spline, quintic) don't need epsilon
    # Calculate epsilon based on data scale, or use a fixed value
    # epsilon ~ typical distance between points
    x_range = np.max(sample_list) - np.min(sample_list)
    y_range = np.max(L_list) - np.min(L_list)
    epsilon = np.sqrt(x_range**2 + y_range**2) / len(points)  # adaptive epsilon
    # Or use a fixed value: epsilon = 1.0
    interp_base = RBFInterpolator(points, values, kernel='thin_plate_spline')
else:
    raise ValueError(f"Unknown interpolation order: {interp_order}. Choose 'linear', 'cubic', or 'rbf'")

# Create a wrapper function that handles different calling conventions
def interp_rel_err(x, y):
    """
    Interpolate rel_err at given (x, y) points.
    
    Parameters
    ----------
    x : scalar or array-like
        sample_list values (number of trajectories)
    y : scalar or array-like
        L_list values (system sizes)
    
    Returns
    -------
    scalar or array
        Interpolated rel_err values
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Handle broadcasting for arrays
    if x.ndim == 0 and y.ndim == 0:
        # Both scalars
        query_points = np.array([[x, y]])
    elif x.ndim == 0:
        # x is scalar, y is array
        query_points = np.column_stack([np.full_like(y, x), y])
    elif y.ndim == 0:
        # y is scalar, x is array
        query_points = np.column_stack([x, np.full_like(x, y)])
    else:
        # Both arrays - create meshgrid
        X, Y = np.meshgrid(x, y)
        query_points = np.column_stack([X.ravel(), Y.ravel()])
        result = interp_base(query_points)
        return result.reshape(len(y), len(x))
    
    result = interp_base(query_points)
    return result[0] if result.size == 1 else result

# Example usage:
# rel_err_interpolated = interp_rel_err(n_samples, L)  # returns scalar or array
# For example:
# rel_err_interpolated = interp_rel_err(375, 35)  # interpolate for n_samples=375, L=35

site_points=np.linspace(100, 1000, 100)
site_values=interp_rel_err(site_points,50)


plt.plot(site_points, site_values, '-')
plt.plot(sample_list, rel_err[2, :], 'o')
plt.show()
# %%
sample_list
# %%
interp_rel_err(547,40)
# %%
