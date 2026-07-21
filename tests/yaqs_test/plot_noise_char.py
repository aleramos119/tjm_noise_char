
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
    """Memory-efficient version of sample_y_ijk + compute_constant.

    Computes, over the bootstrap sample axis, everything the analysis needs
    without ever materializing the full (L, n_obs_L, n_t, n_samples) arrays:

        cost        : (n_samples,)  per-sample mean of Y_ijk
        sigma_max_2 : max_ijk Var(X_ijk)
        c_max       : max_ijk (1 + |mu_ijk| + 2 |mu^ref_ijk|)
        sharp_avg   : (1/N) sum_ijk sigma_ijk^2 (c_ijk - sigma_ijk^2), the
                      average of the sharp bound on Var(Y_ijk), with
                      c_ijk = (1 + |mu_ijk| + 2 |mu_ijk - mu^ref_ijk|)^2

    Samples are drawn in chunks of `chunk_size`, so peak memory scales with
    chunk_size instead of n_samples.
    """
    if rng is None:
        rng = np.random.default_rng()

    L, n_obs_L, n_t, ntraj = traj_data.shape
    p = np.full(ntraj, 1.0 / ntraj)

    cost = np.empty(n_samples)
    sum_x = np.zeros((L, n_obs_L, n_t))     # running sum   of X over samples
    sumsq_x = np.zeros((L, n_obs_L, n_t))   # running sum   of X^2 over samples

    start = 0
    while start < n_samples:
        m = min(chunk_size, n_samples - start)

        # W[t, s] = number of times trajectory t is drawn in sample s.
        W = rng.multinomial(ntraj_size, p, size=m).T       # (ntraj, m)

        x = traj_data @ W / ntraj_size                     # (L, n_obs_L, n_t, m)

        cost[start:start + m] = ((x - ref_traj[..., None]) ** 2).mean(axis=(0, 1, 2))
        sum_x += x.sum(axis=3)
        sumsq_x += (x ** 2).sum(axis=3)

        start += m

    mu = sum_x / n_samples                       # E[X_ijk]
    var_x = sumsq_x / n_samples - mu ** 2        # Var[X_ijk]  (ddof=0, matches std())

    sigma_max_2 = var_x.mean()
    c_max = (1.0 + np.abs(mu) + 2.0 * np.abs(ref_traj)).mean()

    c_sharp = (1.0 + np.abs(mu) + 2.0 * np.abs(mu - ref_traj)) ** 2
    sharp_avg = (var_x * (c_sharp - var_x)).mean()

    return cost, sigma_max_2, c_max, sharp_avg


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


def baund(nsite, ntraj, c_max, d):

    return c_max*(2**d+1)/(nsite*ntraj)



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


def variance_J(y, abs_cov=False, chunk_size=1000):
    """Var(J) = (1/N^2) sum_{ijk, i'j'k'} Cov(Y_ijk, Y_i'j'k').

    y : (L, n_obs_L, n_t, n_samples)
    abs_cov : if True, sum |Cov(Y_ijk, Y_i'j'k')| instead. This is an upper
        bound on Var(J), not Var(J) itself, and requires the pairwise
        covariances explicitly; they are formed in row blocks of
        `chunk_size` to keep memory at chunk_size * N instead of N^2.

    N = L * n_obs_L * n_t. Without abs_cov the full double sum over the N^2
    covariances is evaluated without forming the N x N covariance matrix,
    using sum_{a,b} Cov(Y_a, Y_b) = Var(sum_a Y_a): center each Y_ijk over
    the sample axis, sum over ijk, and take the second moment over samples.
    """
    L, n_obs_L, n_t, n_samples = y.shape
    N = L * n_obs_L * n_t

    yc = y - y.mean(axis=3, keepdims=True)       # center over samples

    if abs_cov:
        Yc = yc.reshape(N, n_samples)
        cov_sum = 0.0
        for start in range(0, N, chunk_size):
            block = Yc[start:start + chunk_size]             # (m, n_samples)
            cov_sum += np.abs(block @ Yc.T).sum()            # sum |cov| rows
        cov_sum /= n_samples - 1
    else:
        s = yc.sum(axis=(0, 1, 2))               # (n_samples,) sum of centered Y
        cov_sum = (s @ s) / (n_samples - 1)      # sum_{ab} Cov(Y_a, Y_b)

    return cov_sum / N**2


def variance_J_bound(y, d, eps):
    """Bound on Var(J) = (1/N^2) sum_{ijk, i'j'k'} |Cov(Y_ijk, Y_i'j'k')| using

        |Cov(Y_ijk, Y_i'j'k')| <= sqrt(Var[Y_ijk] Var[Y_i'j'k'])  if |i-i'| <= d
                                  eps^(|i-i'|-d)                   if |i-i'| >  d

    y   : (L, n_obs_L, n_t, n_samples)
    d   : site separation up to which the Cauchy-Schwarz bound is used
    eps : decay base of the covariance bound beyond distance d

    The near part factorizes over sites: with S_i = sum_jk sqrt(Var[Y_ijk]),
    it is sum_{|i-i'|<=d} S_i S_i'. The far part is independent of (j, k),
    contributing (n_obs_L*n_t)^2 * sum_{|i-i'|>d} eps^(|i-i'|-d).
    """
    L, n_obs_L, n_t, n_samples = y.shape
    N = L * n_obs_L * n_t
    P = n_obs_L * n_t

    sigma = y.std(axis=3, ddof=1)                # (L, n_obs_L, n_t)
    S = sigma.sum(axis=(1, 2))                   # (L,) per-site sum of stds

    idx = np.arange(L)
    dist = np.abs(idx[:, None] - idx[None, :])   # (L, L) site separations

    near = ((dist <= d) * np.outer(S, S)).sum()
    far = (np.where(dist > d, float(eps) ** np.maximum(dist - d, 1), 0.0)).sum() * P**2

    return (near + far) / N**2


def variance_J_bound_max(x, ref_traj, d, eps):
    """Bound on Var(J) = (1/N^2) sum_{ijk, i'j'k'} |Cov(Y_ijk, Y_i'j'k')| using

        |Cov(Y_ijk, Y_i'j'k')| <= sigma_max^2 c_max   if |i-i'| <= d
                                  eps^(|i-i'|-d)      if |i-i'| >  d

    with (main.tex, Lemma var_y_bound)
        sigma_max^2 = max_ijk Var[X_ijk]
        c_max       = max_ijk (M_ijk + |mu_ijk| + 2 |mu_ref_ijk|)^2,  M_ijk = 1

    x        : (L, n_obs_L, n_t, n_samples)  samples of X_ijk
    ref_traj : (L, n_obs_L, n_t)             reference means mu_ref_ijk
    d        : site separation up to which the constant bound is used
    eps      : decay base of the covariance bound beyond distance d

    The near part is constant per pair: sigma_max^2 c_max times the
    N_site(2d+1) - d(d+1) site pairs with |i-i'| <= d, times (n_obs_L*n_t)^2.
    The far part contributes (n_obs_L*n_t)^2 * sum_{|i-i'|>d} eps^(|i-i'|-d).
    """
    L, n_obs_L, n_t, n_samples = x.shape
    N = L * n_obs_L * n_t
    P = n_obs_L * n_t

    sigma_max_2 = x.var(axis=3, ddof=1).max()
    mu = x.mean(axis=3)
    c_max = ((1.0 + np.abs(mu) + 2.0 * np.abs(ref_traj)) ** 2).max()

    idx = np.arange(L)
    dist = np.abs(idx[:, None] - idx[None, :])   # (L, L) site separations

    near = sigma_max_2 * c_max * (dist <= d).sum() * P**2
    far = (np.where(dist > d, float(eps) ** np.maximum(dist - d, 1), 0.0)).sum() * P**2

    return (near + far) / N**2


def variance_J_bound_avg(x, ref_traj, d, eps):
    """Bound on Var(J) = (1/N^2) sum_{ijk, i'j'k'} |Cov(Y_ijk, Y_i'j'k')| using

        |Cov(Y_ijk, Y_i'j'k')| <= (1/N) sum_ijk sigma_ijk^2 c_ijk   if |i-i'| <= d
                                  eps^(|i-i'|-d)                    if |i-i'| >  d

    Same as variance_J_bound_max but with the per-pair constant replaced by
    the average over ijk of sigma_ijk^2 c_ijk, where

        sigma_ijk^2 = Var[X_ijk]
        c_ijk       = (M_ijk + |mu_ijk| + 2 |mu_ref_ijk|)^2,  M_ijk = 1

    x        : (L, n_obs_L, n_t, n_samples)  samples of X_ijk
    ref_traj : (L, n_obs_L, n_t)             reference means mu_ref_ijk
    d        : site separation up to which the constant bound is used
    eps      : decay base of the covariance bound beyond distance d
    """
    L, n_obs_L, n_t, n_samples = x.shape
    N = L * n_obs_L * n_t
    P = n_obs_L * n_t

    var_x = x.var(axis=3, ddof=1)                # (L, n_obs_L, n_t) sigma_ijk^2
    mu = x.mean(axis=3)
    c = (1.0 + np.abs(mu) + 2.0 * np.abs(ref_traj)) ** 2

    avg = (var_x * c).mean()                     # (1/N) sum_ijk sigma_ijk^2 c_ijk

    idx = np.arange(L)
    dist = np.abs(idx[:, None] - idx[None, :])   # (L, L) site separations

    near = avg * (dist <= d).sum() * P**2
    far = (np.where(dist > d, float(eps) ** np.maximum(dist - d, 1), 0.0)).sum() * P**2

    return (near + far) / N**2


def variance_J_bound_avg_sharp(x, ref_traj, d, eps):
    """Bound on Var(J) like variance_J_bound_avg, but using the sharper
    per-index bound on Var[Y_ijk]:

        Var[Y_ijk] <= sigma_ijk^2 [ (M_ijk + |mu_ijk| + 2 |mu_ijk - mu_ref_ijk|)^2
                                    - sigma_ijk^2 ],   M_ijk = 1

    The near part (|i-i'| <= d) uses the average over ijk of that bound;
    the far part is eps^(|i-i'|-d) as before.

    x        : (L, n_obs_L, n_t, n_samples)  samples of X_ijk
    ref_traj : (L, n_obs_L, n_t)             reference means mu_ref_ijk
    d        : site separation up to which the constant bound is used
    eps      : decay base of the covariance bound beyond distance d
    """
    L, n_obs_L, n_t, n_samples = x.shape
    N = L * n_obs_L * n_t
    P = n_obs_L * n_t

    var_x = x.var(axis=3, ddof=1)                # (L, n_obs_L, n_t) sigma_ijk^2
    mu = x.mean(axis=3)
    c = (1.0 + np.abs(mu) + 2.0 * np.abs(mu - ref_traj)) ** 2

    avg = (var_x * (c - var_x)).mean()           # (1/N) sum_ijk sharp bound

    idx = np.arange(L)
    dist = np.abs(idx[:, None] - idx[None, :])   # (L, L) site separations

    near = avg * (dist <= d).sum() * P**2
    far = (np.where(dist > d, float(eps) ** np.maximum(dist - d, 1), 0.0)).sum() * P**2

    return (near + far) / N**2


def scan_max_cov_distance(y, threshold):
    """Scan over (j, k, j', k') for the longest-reaching site covariance.

    For every choice of observable/time indices (j, k) and (j', k') the
    covariance matrix C[i, i'] = |Cov(Y_ijk, Y_i'j'k')| is formed over the
    sample axis. The "reach" of that matrix is the largest site separation
    |i - i'| at which |C[i, i']| still exceeds `threshold`.

    The function returns the (j, k, j', k') whose matrix has the largest reach.

    y         : (L, n_obs_L, n_t, n_samples)
    threshold : absolute covariance defining a 'significant' covariance

    returns   : dict with
        j, k, jp, kp : the selected indices
        dist         : the maximum |i - i'| above threshold for that matrix
        C            : the (L, L) covariance matrix |Cov(Y_ijk, Y_i'j'k')|
    """
    L, n_obs_L, n_t, n_samples = y.shape
    P = n_obs_L * n_t

    # Flatten (j, k) -> p and center over the sample axis once.
    A = y.reshape(L, P, n_samples)
    A = A - A.mean(axis=2, keepdims=True)            # (L, P, n_samples)

    idx = np.arange(L)
    dist = np.abs(idx[:, None] - idx[None, :])        # (L, L) site separations

    best = {"dist": -1, "j": 0, "k": 0, "jp": 0, "kp": 0, "C": None}
    for p in range(P):
        a = A[:, p, :]                                # (L, n_samples) -> i axis
        # C[q, i, i'] = Cov(Y_ip, Y_i'q) for all q at once.
        C = np.abs(np.einsum('is,jqs->qij', a, A) / (n_samples - 1))  # (P, L, L)

        # Largest separation above threshold for each q (-1 if none qualify).
        reach = np.where(C > threshold, dist[None], -1).reshape(P, -1).max(axis=1)
        q = int(reach.argmax())

        if reach[q] > best["dist"]:
            j, k = divmod(p, n_t)
            jp, kp = divmod(q, n_t)
            best = {"dist": int(reach[q]), "j": j, "k": k, "jp": jp, "kp": kp,
                    "C": C[q].copy()}

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

cmax_array=np.zeros((len(L_list), len(ntraj_list)))

sharp_array=np.zeros((len(L_list), len(ntraj_list)))

rng = np.random.default_rng(42)


for i, L in enumerate(L_list):

    folder=f"/home/ale/Documents/Work/simulation_of_open_quantum_systems/tjm_noise_char/tests/yaqs_test/results/propagation/yaqs/L_{L}/"


    traj_data, ref_traj, time = load_traj(folder, L)

    for j, ntraj in enumerate(ntraj_list):

        cost, sigma_max_2, c_max, sharp_avg = compute_stats(traj_data, ref_traj, ntraj,
                                                            n_samp_avg, rng=rng)

        var_array[i, j] = cost.var()

        cmax_array[i, j] = c_max

        sharp_array[i, j] = sharp_avg


#%%
# --- Variance vs L: measured vs sharp bound (d=4) ---
d = 4

# Var(J) <= sharp_avg * (L(2d+1) - d(d+1)) / L^2   (eps -> 0 limit)
bound_array = np.zeros((len(L_list), len(ntraj_list)))
for i, L in enumerate(L_list):
    for j, ntraj in enumerate(ntraj_list):
        bound_array[i, j] = sharp_array[i, j] * (L * (2 * d + 1) - d * (d + 1)) / L**2


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
    # bound (d=4)
    ax.plot(L_list, np.sqrt(bound_array[:, j]), '--', color=color)

ax.set_xlabel(r"$N_{\mathrm{site}}$", labelpad=4)
ax.set_ylabel(r"$\sigma( J )$", labelpad=4)
ax.set_yscale('log')
# solid = measured, dashed = bound
# ax.plot([], [], 'k-', label="measured")
ax.plot([], [], 'k--', label=f"sharp bound ($d={d}$)")
ax.legend(frameon=False, loc='best', handlelength=2)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
plt.tight_layout()
plt.savefig("results/propagation/yaqs/plots/std_vs_L_sharp_bound.pdf",
            dpi=600, bbox_inches="tight", transparent=True)


# %%
##############################################################
##   Longest-reaching covariance over (j, k, j', k')  (L = 40)
##############################################################

L_cov = 40
ntraj_cov = 1000        # bootstrap subset size for X_ijk
n_samp_cov = 10000       # number of bootstrap samples

rng = np.random.default_rng(42)

folder = f"/home/ale/Documents/Work/simulation_of_open_quantum_systems/tjm_noise_char/tests/yaqs_test/results/propagation/yaqs/L_{L_cov}/"
traj_data, ref_traj, time = load_traj(folder, L_cov)

x, y, cost = sample_y_ijk(traj_data, ref_traj, ntraj_cov, n_samp_cov, rng=rng)
#%%
##############################################################
##   Var(J) = (1/N^2) sum_{ijk, i'j'k'} Cov(Y_ijk, Y_i'j'k')
##############################################################

d_bound = 4              # site separation covered by the Cauchy-Schwarz part
eps_bound = 0         # decay base of the covariance bound beyond d

var_J = variance_J(y, abs_cov=True)
var_J_bound = variance_J_bound(y, d_bound, eps_bound)
var_J_bound_max = variance_J_bound_max(x, ref_traj, d_bound, eps_bound)
var_J_bound_avg = variance_J_bound_avg(x, ref_traj, d_bound, eps_bound)
var_J_bound_avg_sharp = variance_J_bound_avg_sharp(x, ref_traj, d_bound, eps_bound)


print(f"Std(J) from max bound (d={d_bound}, eps={eps_bound:g}) : {np.sqrt(var_J_bound_max):.6e}")
print(f"Std(J) from avg bound (d={d_bound}, eps={eps_bound:g}) : {np.sqrt(var_J_bound_avg):.6e}")
print(f"Std(J) from sharp avg bound (d={d_bound}, eps={eps_bound:g}) : {np.sqrt(var_J_bound_avg_sharp):.6e}")
print(f"Std(J) from bound (d={d_bound}, eps={eps_bound:g}) : {np.sqrt(var_J_bound):.6e}")
print(f"Std(J) from covariance sum : {np.sqrt(var_J):.6e}")
print(f"Std(J) from cost samples   : {cost.std(ddof=1):.6e}")
#%%
##############################################################
##   Var(Y_ijk) vs its sharp bound sigma_ijk^2 (c_ijk - sigma_ijk^2)
##############################################################

var_y = y.var(axis=3, ddof=1).ravel()                    # V[Y_ijk]
mu_x = x.mean(axis=3)
var_x = x.var(axis=3, ddof=1)
c_sharp = (1.0 + np.abs(mu_x) + 2.0 * np.abs(mu_x - ref_traj)) ** 2
bound_y = (var_x * (c_sharp - var_x)).ravel()

fig, ax = plt.subplots(figsize=(5, 4))
idx_flat = np.arange(var_y.size)
ax.plot(idx_flat, bound_y/var_y, lw=0.8, color='C1',
        label=r"$\sigma_{ijk}^2(c_{ijk}-\sigma_{ijk}^2)\,/\,\mathrm{V}[Y_{ijk}]$")
# ax.plot(idx_flat, var_y, lw=0.8, color='C0',
#         label=r"$\mathrm{V}[Y_{ijk}]$")
ax.set_yscale('log')
ax.set_xlabel(r"flattened index $(i,j,k)$", labelpad=4)
ax.set_ylabel(r"$\mathrm{V}[Y_{ijk}]$", labelpad=4)
ax.legend(frameon=False, loc='best', handlelength=2)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
plt.tight_layout()
plt.show()

nz = var_y > 0
print(f"bound holds for all ijk : {np.all(var_y <= bound_y)}")
print(f"min bound/Var ratio     : {(bound_y[nz] / var_y[nz]).min():.3g}")
#%%


#%%
threshold = 2e-9         # absolute |covariance| defining a 'significant' covariance

best = scan_max_cov_distance(y, threshold)

print(f"Largest covariance reach above |cov| = {threshold:g}:")
print(f"  (j, k)   = ({best['j']}, {best['k']})")
print(f"  (j', k') = ({best['jp']}, {best['kp']})")
print(f"  max |i - i'| = {best['dist']}")

save_path = f"results/propagation/crosstalk_zz_model/yaqs/cov_mat_L{L_cov}_threshold_{threshold}_ntraj_{ntraj_cov}.pdf"
plot_cov_mat(best["C"], vmax=best["C"].max(), save_path=save_path)



# %%

cov_mat=np.zeros([L_cov,L_cov])

for i in range(L_cov):
    for j in range(L_cov):
        cov_mat[i,j]=abs(np.cov(y[i,0,36,:],y[j,0,40,:])[0,1])

plt.imshow(cov_mat)
plt.colorbar()

# %%
plt.plot([abs(np.cov(y[20,0,36,:],y[i,0,40,:])[0,1]) for i in range(40)])
# plt.ylim([0,2e-8])
# %%
