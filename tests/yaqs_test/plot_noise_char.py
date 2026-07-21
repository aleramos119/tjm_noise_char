
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

    traj_data = np.transpose(split_data,(2,1,0,3))

    ref_traj = np.mean(traj_data, axis=3)

    return traj_data, ref_traj, time


def sample_y_ijk(traj_data, ref_traj, ntraj_size, n_samples, rng=None):
    """Draw samples of Y_ijk = (X_ijk - ref_traj_ijk)^2.

    X_ijk is the average of traj_data over a random subset of `ntraj_size`
    trajectories. Indices i, j, k correspond to L, n_obs_L and n_t.

    traj_data : (L, n_obs_L, n_t, ntraj)
    ref_traj  : (L, n_obs_L, n_t)
    returns   : (L, n_obs_L, n_t, n_samples)
    """
    if rng is None:
        rng = np.random.default_rng()

    L, n_obs_L, n_t, ntraj = traj_data.shape

    # W[t, s] = number of times trajectory t is drawn in sample s (bootstrap).
    W = rng.multinomial(ntraj_size, np.full(ntraj, 1.0 / ntraj),
                        size=n_samples).T          # (ntraj, n_samples)

    # X[..., s] = average of traj_data over sample s's draws.
    x = traj_data @ W / ntraj_size                 # (L, n_obs_L, n_t, n_samples)
    y = (x - ref_traj[..., None]) ** 2


    cost=y.mean(axis=(0, 1, 2))

    return x, y, cost

def compute_stats(traj_data, ref_traj, ntraj_size, n_samples, rng=None,
                  chunk_size=250):
    """Memory-efficient bootstrap statistics for the cost-function bound.

    Computes, over the bootstrap sample axis, everything the analysis needs
    without ever materializing the full (L, n_obs_L, n_t, n_samples) arrays:

        cost      : (n_samples,)  per-sample mean of Y_ijk
        c_1mP_bar : (1/N) sum_ijk c_ijk (1 - P_i(t_k))

    Here (see main.tex, Sec. "Estimation of the Standard Deviation of the
    Cost-Function"):

        c_ijk     = (M_ijk + |mu_ijk| + 2 |mu_ijk - mu^ref_ijk|)^2   (M=1, Pauli)
        P_i(t_k)  = (1 + <X_i>^2 + <Y_i>^2 + <Z_i>^2) / 2            local purity

    P_i is the purity of the single-site reduced state, reconstructed from the
    Bloch vector (the three Pauli reference means stored along n_obs_L).

    Samples are drawn in chunks of `chunk_size`, so peak memory scales with
    chunk_size instead of n_samples.
    """
    if rng is None:
        rng = np.random.default_rng()

    L, n_obs_L, n_t, ntraj = traj_data.shape
    p = np.full(ntraj, 1.0 / ntraj)

    cost = np.empty(n_samples)
    sum_x = np.zeros((L, n_obs_L, n_t))     # running sum   of X over samples

    start = 0
    while start < n_samples:
        m = min(chunk_size, n_samples - start)

        # W[t, s] = number of times trajectory t is drawn in sample s.
        W = rng.multinomial(ntraj_size, p, size=m).T       # (ntraj, m)

        x = traj_data @ W / ntraj_size                     # (L, n_obs_L, n_t, m)

        cost[start:start + m] = ((x - ref_traj[..., None]) ** 2).mean(axis=(0, 1, 2))
        sum_x += x.sum(axis=3)

        start += m

    mu = sum_x / n_samples                       # E[X_ijk]

    # c_ijk = (M + |mu| + 2 |mu - mu^ref|)^2, with M = 1 for Pauli observables.
    c = (1.0 + np.abs(mu) + 2.0 * np.abs(mu - ref_traj)) ** 2      # (L, n_obs_L, n_t)

    # Local single-site purity P_i(t_k) = (1 + |Bloch vector|^2) / 2, using the
    # three Pauli reference means as the Bloch vector components.
    P = 0.5 * (1.0 + (ref_traj ** 2).sum(axis=1))                 # (L, n_t)

    # Proper average of the product c_ijk (1 - P_i(t_k)) over sites/obs/times.
    c_1mP_bar = (c * (1.0 - P[:, None, :])).mean()

    return cost, c_1mP_bar


def compute_constant(x, ref_traj):
    """Compute the constants sigma_max and c_max from the X_ijk samples.

    x        : (L, n_obs_L, n_t, n_samples)  samples of X_ijk
    ref_traj : (L, n_obs_L, n_t)             reference means mu^ref_ijk

    sigma_max = max_ijk std(X_ijk)
    c_max     = max_ijk ( 1 + |mu_ijk| + 2 |mu^ref_ijk| )
    """
    mu = x.mean(axis=3)                 # (L, n_obs_L, n_t)  expectation of X_ijk
    sigma = x.std(axis=3)              # (L, n_obs_L, n_t)  std of X_ijk

    sigma_max = sigma.max()
    c_max = (1.0 + np.abs(mu) + 2.0 * np.abs(ref_traj)).max()

    sigma_max_2=sigma_max**2

    return sigma_max_2, c_max


def bound(nsite, ntraj, c_1mP_bar, ell):
    """Variance bound from main.tex, Eq. (sigma_J_bound):

        Var(J) <= C / (N_traj N_site),   C = 2 (2 ell + 1) * c_1mP_bar,

    where c_1mP_bar = (1/N) sum_ijk c_ijk (1 - P_i(t_k)) and ell is the maximum
    covariance distance. Returns the variance bound (take sqrt for sigma).
    """
    C = 2.0 * (2 * ell + 1) * c_1mP_bar

    return C / (nsite * ntraj)



def compute_cov_mat(y, j, k, jp=None, kp=None):
    """|Cov(Y_ijk, Y_i'j'k')| over the sample axis, with jk and j'k' fixed.

    y  : (L, n_obs_L, n_t, n_samples)
    j, k   : fixed observable / time indices for the first variable (i axis)
    jp, kp : fixed indices for the second variable (i' axis); default to j, k

    returns : (L, L) matrix C, with C[i, i'] = |Cov(Y_ijk, Y_i'j'k')|
    """
    if jp is None:
        jp = j
    if kp is None:
        kp = k

    a = y[:, j, k, :]      # (L, n_samples)  -> i axis
    b = y[:, jp, kp, :]    # (L, n_samples)  -> i' axis

    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)

    n_samples = y.shape[3]
    C = a @ b.T / (n_samples - 1)        # (L, L) cross-covariance

    return np.abs(C)


def scan_max_cov_distance(y, threshold):
    """Scan over (j, k, j', k') for the longest-reaching site correlation.

    For every choice of observable/time indices (j, k) and (j', k') the
    correlation matrix R[i, i'] = |Corr(Y_ijk, Y_i'j'k')| is formed over the
    sample axis. The "reach" of that matrix is the largest site separation
    |i - i'| at which |R[i, i']| still exceeds `threshold`.

    Correlation (covariance normalised by the two standard deviations) is used
    instead of raw covariance because the Y_ijk magnitudes are tiny and vary
    across indices, which makes an absolute covariance cutoff impractical.

    The function returns the (j, k, j', k') whose matrix has the largest reach.

    y         : (L, n_obs_L, n_t, n_samples)
    threshold : correlation in [0, 1] defining a 'significant' correlation

    returns   : dict with
        j, k, jp, kp : the selected indices
        dist         : the maximum |i - i'| above threshold for that matrix
        C            : the (L, L) covariance matrix |Cov(Y_ijk, Y_i'j'k')|
        R            : the (L, L) correlation matrix |Corr(Y_ijk, Y_i'j'k')|
    """
    L, n_obs_L, n_t, n_samples = y.shape
    P = n_obs_L * n_t

    # Flatten (j, k) -> p and center over the sample axis once.
    A = y.reshape(L, P, n_samples)
    A = A - A.mean(axis=2, keepdims=True)            # (L, P, n_samples)

    # std[i, p] of each Y_ip over the sample axis (constant series -> 0 corr).
    std = np.sqrt(np.einsum('ips,ips->ip', A, A) / (n_samples - 1))   # (L, P)
    std = np.where(std > 0, std, np.inf)

    idx = np.arange(L)
    dist = np.abs(idx[:, None] - idx[None, :])        # (L, L) site separations

    best = {"dist": -1, "j": 0, "k": 0, "jp": 0, "kp": 0, "C": None, "R": None}
    for p in range(P):
        a = A[:, p, :]                                # (L, n_samples) -> i axis
        # C[q, i, i'] = Cov(Y_ip, Y_i'q) for all q at once.
        C = np.einsum('is,jqs->qij', a, A) / (n_samples - 1)          # (P, L, L)
        # Normalise to correlation: divide by std_p[i] * std_q[i'].
        denom = std[:, p][None, :, None] * std.T[:, None, :]          # (P, L, L)
        R = np.abs(C) / denom

        # Largest separation above threshold for each q (-1 if none qualify).
        reach = np.where(R > threshold, dist[None], -1).reshape(P, -1).max(axis=1)
        q = int(reach.argmax())

        if reach[q] > best["dist"]:
            j, k = divmod(p, n_t)
            jp, kp = divmod(q, n_t)
            best = {"dist": int(reach[q]), "j": j, "k": k, "jp": jp, "kp": kp,
                    "C": np.abs(C[q]), "R": R[q].copy()}

    return best


def plot_cov_mat(C, vmax=1e-4, save_path=None):
    """Publication-quality plot of an absolute covariance matrix |Cov(Y_i, Y_i')|.

    C         : (L, L) matrix to display
    vmax      : upper limit of the color scale (lower limit fixed at 0)
    save_path : if given, save the figure there (pdf); otherwise show it
    """
    import matplotlib.pyplot as plt
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
    im = ax.imshow(C, vmin=0, vmax=vmax, origin="lower")  # origin lower so i, i' increase to right and up
    cbar = plt.colorbar(im, format="%.1e", ax=ax)
    cbar.ax.tick_params(labelsize=13)  # Set colorbar tick font size

    # Remove the exponent only at the top of the colorbar
    cbar.ax.yaxis.get_offset_text().set_visible(False)

    ax.set_title(r"$|$"+"Cov"+r"$(Y_{i}, Y_{i'})|$", fontsize=17)
    ax.set_xlabel(r"$i$", labelpad=4)
    ax.set_ylabel(r"$i'$", labelpad=4)
    # Show top and right border
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight", transparent=True)
        plt.close(fig)
    else:
        plt.show()

    return fig, ax


def plot_var_vs_L(L_list, var_array, sample_list, save_path=None):
    """Plot the variance of J versus the number of sites, one curve per N_traj.

    L_list     : list of N_site values (x axis)
    var_array  : (len(L_list), len(sample_list)) variance of J
    sample_list: list of N_traj values (one curve each)
    save_path  : if given, save the figure (pdf); otherwise show it
    """
    import matplotlib.pyplot as plt
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
            L_list, var_array[:, i],
            'o-',
            label=r"$N_{\mathrm{traj}}$="+f"{ntraj}",
            color=color_cycle[i % len(color_cycle)],
        )

    ax.set_xlabel(r"$N_{\mathrm{site}}$", labelpad=4)
    ax.set_ylabel(r"$\mathrm{Var}( J )$", labelpad=4)
    ax.legend(frameon=False, loc='best', handlelength=2)
    # Show top and right border
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight", transparent=True)
        plt.close(fig)
    else:
        plt.show()

    return fig, ax



# %%
##############################################################
##   Loss Distribution
##############################################################


L_list=[10, 20, 40, 80, 160]

ntraj_list=[125, 250, 500, 1000]

n_samp_avg=5000

var_array=np.zeros((len(L_list), len(ntraj_list)))

c_1mP_array=np.zeros((len(L_list), len(ntraj_list)))

rng = np.random.default_rng(42)


for i, L in enumerate(L_list):

    folder=f"/home/aramos/Dokumente/Work/simulation_of_open_quantum_systems/tjm_noise_char/tests/yaqs_test/results/propagation/yaqs/L_{L}/"


    traj_data, ref_traj, time = load_traj(folder, L)

    for j, ntraj in enumerate(ntraj_list):

        cost, c_1mP_bar = compute_stats(traj_data, ref_traj, ntraj,
                                        n_samp_avg, rng=rng)

        var_array[i, j] = cost.var()

        c_1mP_array[i, j] = c_1mP_bar


#%%
# --- Variance vs L: measured vs bound (main.tex, ell=4) ---
ell = 4

bound_array = np.zeros((len(L_list), len(ntraj_list)))
for i, L in enumerate(L_list):
    for j, ntraj in enumerate(ntraj_list):
        bound_array[i, j] = bound(L, ntraj, c_1mP_array[i, j], ell)


import matplotlib as mpl

mpl.rcParams.update({
    'figure.figsize': (5, 4),
    'axes.linewidth': 1.5,
    'axes.labelsize': 16,
    'axes.titlesize': 17,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 11,
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
for j, ntraj in enumerate(ntraj_list):
    color = color_cycle[j % len(color_cycle)]
    # measured standard deviation
    ax.plot(L_list, np.sqrt(var_array[:, j]), 'o-', color=color,
            label=r"$N_{\mathrm{traj}}$="+f"{ntraj}")
    # bound (ell=4)
    ax.plot(L_list, np.sqrt(bound_array[:, j]), '--', color=color)

ax.set_xlabel(r"$N_{\mathrm{site}}$", labelpad=4)
ax.set_ylabel(r"$\sigma( J )$", labelpad=4)
# ax.set_yscale('log')
# solid = measured, dashed = bound
# ax.plot([], [], 'k-', label="measured")
ax.plot([], [], 'k--', label=r"bound ($\ell="+f"{ell}$)")
ax.legend(frameon=False, loc='best', handlelength=2)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
plt.tight_layout()
plt.savefig("results/propagation/yaqs/plots/std_vs_L_bound.pdf",
            dpi=600, bbox_inches="tight", transparent=True)


# %%
##############################################################
##   Longest-reaching covariance over (j, k, j', k')  (L = 40)
##############################################################

L_cov = 40
ntraj_cov = 1000        # bootstrap subset size for X_ijk
n_samp_cov = 5000       # number of bootstrap samples

rng = np.random.default_rng(42)

folder = f"/home/ale/Documents/Work/simulation_of_open_quantum_systems/tjm_noise_char/tests/yaqs_test/results/propagation/yaqs/L_{L_cov}/"
traj_data, ref_traj, time = load_traj(folder, L_cov)

_, y, _ = sample_y_ijk(traj_data, ref_traj, ntraj_cov, n_samp_cov, rng=rng)


#%%
threshold = 0.3e-8         # |correlation| in [0, 1] defining a 'significant' correlation

best = scan_max_cov_distance(y, threshold)

print(f"Largest correlation reach above |corr| = {threshold:g}:")
print(f"  (j, k)   = ({best['j']}, {best['k']})")
print(f"  (j', k') = ({best['jp']}, {best['kp']})")
print(f"  max |i - i'| = {best['dist']}")

save_path = f"results/propagation/yaqs/plots/cov_mat_maxdist_L{L_cov}.pdf"
plot_cov_mat(best["C"], vmax=best["C"].max(), save_path=save_path)



# %%

cov_mat=np.zeros([L_cov,L_cov])

for i in range(L_cov):
    for j in range(L_cov):
        cov_mat[i,j]=abs(np.cov(y[i,0,41,:],y[j,0,45,:])[0,1])

plt.imshow(cov_mat,vmax=1e-8)

# %%
plt.plot([abs(np.cov(y[20,0,43,:],y[i,0,46,:])[0,1]) for i in range(40)])
plt.ylim([0,2e-8])
# %%
