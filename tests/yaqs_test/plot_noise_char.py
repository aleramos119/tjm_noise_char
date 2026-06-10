
#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde


##############################################################
##   Load trajectories
##############################################################
import os
import numpy as np
import matplotlib.pyplot as plt

def load_traj(folder, L):
    import glob

    n_obs_L = 3

    traj_files = sorted(glob.glob(os.path.join(folder, "run_*", "ref_traj.txt")))
    if not traj_files:
        traj_files = sorted(glob.glob(os.path.join(folder, "traj_*.txt")))

    ntraj = len(traj_files)
    if ntraj == 0:
        raise FileNotFoundError(f"No trajectory files found in {folder}")

    first = np.genfromtxt(traj_files[0])
    n_t = first.shape[0]
    n_obs = n_obs_L * L
    time = first[:, 0]

    full_data = np.zeros((n_t, n_obs, ntraj))
    for j, traj in enumerate(traj_files):
        full_data[:, :, j] = np.genfromtxt(traj)[:, 1:]

    print(f"Loaded {ntraj} trajectories from {folder}.")

    split_data = full_data.reshape(n_t, n_obs_L, L, ntraj)
    ref_traj = np.mean(split_data, axis=3)

    return split_data, ref_traj, time

#%%
split_data, ref_traj, time = load_traj("/home/ale/Documents/Work/simulation_of_open_quantum_systems/tjm_noise_char/tests/yaqs_test/results/propagation/yaqs/L_10", 10)
#%%


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

n_samp_avg = 5000
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
plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams["mathtext.fontset"] = "cm"

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(C, vmin=0, vmax=1e-4, origin="lower")  # set origin lower so i and i' increase to right and up
cbar = plt.colorbar(im, format="%.1e", ax=ax)
cbar.ax.tick_params(labelsize=13)  # Set colorbar tick font size

# Remove the exponent only at the top of the colorbar
cbar.ax.yaxis.get_offset_text().set_visible(False)

ax.set_title(r"$|$"+"Cov"+r"$(Y_{i}, Y_{i'})|$", fontsize=17)
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


L_list=[10, 20, 40, 80, 160]

sample_list=[125, 250, 500, 1000]


method="yaqs"

n_samp_avg=5000

rel_err=np.zeros((len(L_list), len(sample_list)))

loss_array=np.zeros((len(L_list), len(sample_list)))

std_array=np.zeros((len(L_list), len(sample_list)))

loss_data_all=np.zeros((len(L_list), len(sample_list), n_samp_avg))



rng = np.random.default_rng(42)  # change or remove seed for different draws


for i, L in enumerate(L_list):

    split_data, ref_traj, time = load_traj(method, L)

    n_t, n_obs_L, L, ntraj_max = split_data.shape

    # split_data_avg = np.zeros((n_t, n_obs_L, L, n_samp_avg))



    for j, n_samples in enumerate(sample_list):

        loss_data_samples = np.zeros(n_samp_avg)

        # idx shape: (n_samp_avg, n_samples)
        idx = rng.choice(ntraj_max, size=(n_samp_avg, n_samples), replace=True)

        # Process in chunks to avoid allocating (n_t, n_obs_L, L, n_samp_avg, n_samples) at once
        max_mem_bytes = 1024 * 1024**2  # 500 MB per chunk
        chunk_size = max(1, int(max_mem_bytes / (n_t * n_obs_L * L * n_samples * 8)))
        for start in range(0, n_samp_avg, chunk_size):
            end = min(start + chunk_size, n_samp_avg)
            # shape: (n_t, n_obs_L, L, end-start, n_samples) -> mean -> (n_t, n_obs_L, L, end-start)
            chunk_avg = np.mean(split_data[:, :, :, idx[start:end]], axis=4)
            loss_data_samples[start:end] = np.sum(
                (chunk_avg - ref_traj[:, :, :, np.newaxis])**2, axis=(0, 1, 2)
            ) / (n_t * n_obs_L * L)


        loss_data_all[i, j] = loss_data_samples

        rel_err[i, j] = np.std(loss_data_samples)/np.mean(loss_data_samples)

        loss_array[i, j] = np.mean(loss_data_samples)

        std_array[i, j] = np.std(loss_data_samples)

#%%
std_array[-1, -1]
#%%
plt.plot(loss_data_samples,'o')

#%%
std_log_array = np.std(np.log10(np.sqrt(loss_data_all)), axis=2)


# %%
%matplotlib qt
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

plt.rc('text', usetex=True)
plt.rc('font', family='serif')
plt.rcParams["mathtext.fontset"] = "cm"

fig, ax = plt.subplots(figsize=(5, 4))
color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
for i, ntraj in enumerate(sample_list):
    ax.plot(
        L_list, std_log_array[:,i], 
        'o-', 
        label=r"$N_{\mathrm{traj}}$="+f"{ntraj}",
        color=color_cycle[i % len(color_cycle)],
    )

ax.set_xlabel(r"$N_{\mathrm{site}}$", labelpad=4)
ax.set_ylabel(r"$\sigma ( J )$", labelpad=4)
ax.legend(frameon=False, loc='best', handlelength=2)
# Show top and right border (make sure they're visible)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
# Do not set title; leave for caption
# plt.tight_layout()
plt.savefig(f"results/propagation/yaqs/plots/std_vs_L.pdf", dpi=600, bbox_inches="tight", transparent=True)
# plt.close(fig)

# %%


for i, L in enumerate(L_list):
    plt.plot(sample_list, rel_err[i, :], 'o-', label=r"$N_{\mathrm{site}}=$" + f"{L}")
plt.xlabel(r"$N_{\mathrm{traj}}$", labelpad=4)
plt.ylabel(r"$\varepsilon_{\mathrm{rel}} ( J )$", labelpad=4)
plt.legend()

plt.savefig(f"results/propagation/yaqs/plots/rel_err_vs_ntraj.pdf", dpi=600, bbox_inches='tight')


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
import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# Example data
x_data = L_list
y_data = rel_err[:,1]

# Define model function
def model(x, a):
    return a/np.sqrt(x)

# Fit model
params, covariance = curve_fit(model, x_data, y_data)

a=params[0]

print(a)
# Plot result
x_fit = np.linspace(10, 160, 100)
y_fit = model(x_fit, 1.2)

plt.scatter(x_data, y_data, label="data")
plt.plot(x_fit, y_fit, label="fit")
plt.legend()
plt.show()
# %%

# loss_data_all=np.zeros((len(L_list), len(sample_list), n_samp_avg))
for i in range(5):
    data=np.log10(np.sqrt(loss_data_all[i,0]))
    std=np.std(data)
    plt.plot(data, label=f"std={std}")

plt.legend()
# %%
np.std(np.log10(np.sqrt(loss_data_all[0,0])))
# %%
std_list=[]
for L in L_list:
    loss=np.genfromtxt(f"/home/aramos/Dokumente/Work/simulation_of_open_quantum_systems/tjm_noise_char/tests/yaqs_test/results/characterizer_gradient_free/loss_scale_True_reduced/module_yaqs/method_cma/params_d_3/const_4e6/L_{L}/xlim_0.1/loss_x_history.txt")[:,1]
    std_list.append(np.std(loss[-200:]))
# %%
%matplotlib qt
plt.plot(L_list,std_list,'o-')
# %%

# %%
std_list