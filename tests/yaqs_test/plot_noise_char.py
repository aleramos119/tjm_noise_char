
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


def bound_waterfall(traj_data, ref_traj, ntraj_size, n_samples, ell, rng=None):
    """Decompose the cost-function variance bound into its approximation stages.

    Each stage corresponds to one inequality used in main.tex, Sec.
    "Estimation of the Standard Deviation of the Cost-Function". Every value is
    an upper bound on Var[J] evaluated at a single (N_site, N_traj); comparing
    consecutive stages shows how much each approximation inflates the bound.

    Stages (a = (i, j, k) is the flattened site/observable/time index,
    N = N_site N_ob N_time, band = { (a, b) : |i_a - i_b| <= ell }):

        0  exact          Var[J]                         = (1/N^2) sum_ab Cov(Y_a,Y_b)
        1  triangle       (1/N^2) sum_ab |Cov(Y_a,Y_b)|
        2  finite dist    (1/N^2) sum_band |Cov(Y_a,Y_b)|         (drop |i-i'|>ell tail)
        3  Cauchy-Schwarz (1/N^2) sum_band sqrt(V[Y_a] V[Y_b])
        4  AM-GM + count  (2l+1)/N_site * mean_a V[Y_a]
        5  Lemma V[Y]     (2l+1)/N_site * mean_a c_a sigma_a^2    (V[Y_a] <= c_a sigma_a^2)
        6  purity         2(2l+1)/(N_site N_traj) * mean_a c_a (1-P_i)   (final bound)

    Returns an ordered dict {stage_label: variance_value} (take sqrt for sigma).
    """
    x, y, cost = sample_y_ijk(traj_data, ref_traj, ntraj_size, n_samples, rng=rng)

    L, n_obs_L, n_t, ns = y.shape
    N = L * n_obs_L * n_t
    N_traj = ntraj_size

    # Flatten (i, j, k) -> a. reshape keeps order (L, n_obs_L, n_t), so the site
    # index i is the slowest-varying axis.
    Yf = y.reshape(N, ns)
    site = np.repeat(np.arange(L), n_obs_L * n_t)          # (N,)  site of each a

    VY = Yf.var(axis=1, ddof=1)                            # (N,) Var[Y_a]

    # Full covariance matrix of Y over the sample axis.
    Yc = Yf - Yf.mean(axis=1, keepdims=True)
    Cov = Yc @ Yc.T / (ns - 1)                             # (N, N)

    band = np.abs(site[:, None] - site[None, :]) <= ell    # (N, N) near-diagonal mask

    # Per-a constants for the Lemma / purity stages.
    mu = x.mean(axis=3)                                    # (L, n_obs_L, n_t)
    sigma2 = x.var(axis=3, ddof=1)                         # sigma_a^2 = Var[X_a]
    c = (1.0 + np.abs(mu) + 2.0 * np.abs(mu - ref_traj)) ** 2
    P = 0.5 * (1.0 + (ref_traj ** 2).sum(axis=1))          # (L, n_t) local purity
    one_mP = np.broadcast_to((1.0 - P)[:, None, :], (L, n_obs_L, n_t))

    c_flat = c.reshape(N)
    csig2 = (c * sigma2).reshape(N)
    c_1mP = (c * 2.0 * one_mP / N_traj).reshape(N)

    root = np.sqrt(VY)
    pref = (2 * ell + 1) / L                               # (2l+1)/N_site

    stages = {}
    stages["exact Var[J]"]        = cost.var(ddof=1)
    stages["triangle |Cov|"]      = np.abs(Cov).sum() / N ** 2
    stages["finite dist (band)"]  = np.abs(Cov)[band].sum() / N ** 2
    stages["Cauchy-Schwarz"]      = (np.outer(root, root) * band).sum() / N ** 2
    stages["AM-GM + band count"]  = pref * VY.mean()
    stages["Lemma V[Y]<=c s^2"]   = pref * csig2.mean()
    stages["purity (final bound)"] = pref * c_1mP.mean()

    return stages


def plot_bound_waterfall(stages, L, ntraj, save_path=None):
    """Bar chart of sigma = sqrt(Var-bound) at each approximation stage.

    Each bar is annotated with the multiplicative inflation factor relative to
    the previous stage, so the approximation that raises the bound the most is
    immediately visible.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams.update({
        'axes.linewidth': 1.5,
        'axes.labelsize': 15,
        'axes.titlesize': 15,
        'xtick.labelsize': 11,
        'ytick.labelsize': 12,
        'font.family': 'serif',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams["mathtext.fontset"] = "cm"

    # LaTeX-safe display labels (usetex is on), in the fixed stage order.
    display = {
        "exact Var[J]":         r"exact $\mathrm{Var}[\hat{J}]$",
        "triangle |Cov|":       r"triangle $|\mathrm{Cov}|$",
        "finite dist (band)":   r"finite dist.\ (band)",
        "Cauchy-Schwarz":       r"Cauchy--Schwarz",
        "AM-GM + band count":   r"AM--GM $+$ band count",
        "Lemma V[Y]<=c s^2":    r"Lemma $\mathrm{V}[Y]\leq c\sigma^2$",
        "purity (final bound)": r"purity (final)",
    }
    keys = list(stages.keys())
    labels = [display.get(k, k) for k in keys]
    vals = np.array([stages[k] for k in keys])
    sig = np.sqrt(vals)                       # sigma at each stage
    factors = sig[1:] / sig[:-1]              # step-to-step inflation of sigma

    fig, ax = plt.subplots(figsize=(8, 4.5))
    xpos = np.arange(len(labels))
    ax.bar(xpos, sig, color="#4C72B0", edgecolor="k", linewidth=1.0)
    ax.set_yscale('log')

    for i, s in enumerate(sig):
        txt = f"{s:.2e}"
        if i > 0:
            txt += "\n" + rf"($\times{factors[i-1]:.2g}$)"
        ax.text(xpos[i], s, txt, ha='center', va='bottom', fontsize=9)

    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, rotation=30, ha='right')
    ax.set_ylabel(r"$\sigma(\hat{J})$ bound")
    ax.set_title(rf"Bound waterfall ($N_{{\mathrm{{site}}}}={L}$, "
                 rf"$N_{{\mathrm{{traj}}}}={ntraj}$)")
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight", transparent=True)
        plt.close(fig)
    else:
        plt.show()

    return fig, ax


def plot_var_y_vs_bound(traj_data, ref_traj, ntraj_size, n_samples,
                        i_site=None, j_obs=0, rng=None, save_path=None):
    """Compare the measured V[Y_ijk] with its Lemma bound c_ijk * sigma_ijk^2.

    This is the step (Lemma var_y_bound in main.tex) that dominates the looseness
    of the cost-function bound. Left panel: both quantities vs time for a fixed
    (site, observable). Right panel: distribution of the ratio
    (c_ijk sigma_ijk^2) / V[Y_ijk] over all sites/observables/times.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams.update({
        'axes.linewidth': 1.5,
        'axes.labelsize': 15,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 11,
        'lines.linewidth': 2,
        'font.family': 'serif',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams["mathtext.fontset"] = "cm"

    x, y, _ = sample_y_ijk(traj_data, ref_traj, ntraj_size, n_samples, rng=rng)
    L, n_obs_L, n_t, ns = y.shape

    VY = y.var(axis=3, ddof=1)                                   # (L, n_obs, n_t)
    mu = x.mean(axis=3)
    sigma2 = x.var(axis=3, ddof=1)
    c = (1.0 + np.abs(mu) + 2.0 * np.abs(mu - ref_traj)) ** 2
    bound_y = c * sigma2                                         # c_ijk sigma_ijk^2

    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = (bound_y / VY).ravel()                          # looseness factor (inf where V[Y]=0)

    if i_site is None:
        i_site = L // 2
    t = np.arange(n_t)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: V[Y] vs its bound over time, for one (site, observable).
    ax0.plot(t, VY[i_site, j_obs], 'o-', color="#4C72B0", ms=4,
             label=r"$\mathrm{V}[Y_{ijk}]$ (measured)")
    ax0.plot(t, bound_y[i_site, j_obs], 's--', color="#C44E52", ms=4,
             label=r"$c_{ijk}\,\sigma_{ijk}^2$ (bound)")
    ax0.set_yscale('log')
    ax0.set_xlabel(r"time index $k$")
    ax0.set_ylabel(r"variance of $Y_{ijk}$")
    ax0.set_title(rf"$i={i_site}$, $j={j_obs}$")
    ax0.legend(frameon=False, handlelength=2)
    ax0.spines['top'].set_visible(True)
    ax0.spines['right'].set_visible(True)

    # Right: distribution of the looseness ratio over all indices.
    finite = np.isfinite(ratio) & (ratio > 0)
    r = ratio[finite]
    ax1.hist(np.log10(r), bins=50, color="#4C72B0", edgecolor="k", linewidth=0.5)
    med = np.median(r)
    ax1.axvline(np.log10(med), color="#C44E52", lw=2,
                label=rf"median $= {med:.0f}\times$")
    ax1.set_xlabel(r"$\log_{10}\left( c_{ijk}\sigma_{ijk}^2 \,/\, \mathrm{V}[Y_{ijk}] \right)$")
    ax1.set_ylabel("count")
    ax1.set_title("looseness of the Lemma bound")
    ax1.legend(frameon=False)
    ax1.spines['top'].set_visible(True)
    ax1.spines['right'].set_visible(True)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight", transparent=True)
        plt.close(fig)
    else:
        plt.show()

    print(f"Lemma bound / V[Y] ratio: median={med:.1f}, "
          f"mean={r.mean():.1f}, min={r.min():.2f}, max={r.max():.1f}")

    return fig, (ax0, ax1)


def plot_bound_comparison(traj_data, ref_traj, ntraj_size, n_samples,
                          i_site=None, j_obs=0, kappa=3.0, delta=0.0,
                          delta_in_sigma=False, rng=None, save_path=None):
    """Compare the old (sigma^2) and new (sigma^4) bounds on V[Y_ijk].

    Old bound (main.tex, Lemma var_y_bound):
        V[Y] <= c sigma^2,        c = (M + |mu| + 2|mu - mu_ref|)^2, M = 1

    New bound (exact identity V[Y] = (mu4 - sigma^4) + 4 d mu3 + 4 d^2 sigma^2
    with d = mu - mu_ref, closed with a kurtosis bound mu4 <= kappa sigma^4 and
    |mu3| <= sqrt(kappa) sigma^3):
        V[Y] <= (kappa - 1) sigma^4 + 4 sqrt(kappa) |d| sigma^3 + 4 d^2 sigma^2

    `delta` shifts the reference, mu_ref -> ref_traj + delta, creating a genuine
    model-reference mismatch d. This matters: the bootstrap reference is the mean
    of the very trajectories being resampled, so d = 0 identically at delta = 0,
    which is the ONLY regime where the bound is genuinely quartic. As shown in
    variance_bound_tightening.tex (Sec. "Is a bound proportional to sigma^4
    possible?"), for any fixed d != 0 the term 4 d^2 sigma^2 is linear in sigma^2
    and dominates the quartic 2 sigma^4 once |d| > sigma/sqrt(2); since
    sigma ~ 1/sqrt(N_traj), adding trajectories destroys the quartic regime.
    Set delta_in_sigma=True to give `delta` in units of the typical sigma, so
    delta=1 sits at the crossover and delta=10 is deep in the sigma^2 regime.

    Left panel: both bounds vs the measured V[Y] over time for one
    (site, observable), with the two competing terms 2 sigma^4 and 4 d^2 sigma^2
    shown separately. Right panel: distribution of each bound's looseness ratio
    (bound / measured) over all sites/observables/times.
    """
    import matplotlib.pyplot as plt
    import matplotlib as mpl

    mpl.rcParams.update({
        'axes.linewidth': 1.5,
        'axes.labelsize': 15,
        'axes.titlesize': 14,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 10,
        'lines.linewidth': 2,
        'font.family': 'serif',
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
    })
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    plt.rcParams["mathtext.fontset"] = "cm"

    x, _, _ = sample_y_ijk(traj_data, ref_traj, ntraj_size, n_samples, rng=rng)
    L, n_obs_L, n_t, ns = x.shape

    mu = x.mean(axis=3)
    sigma2 = x.var(axis=3, ddof=1)                   # sigma^2

    # Typical sigma, used both to interpret delta_in_sigma and to report d/sigma.
    live = sigma2 > sigma2.max() * 1e-6              # drop t=0 where sigma = 0
    sigma_typ = float(np.sqrt(np.median(sigma2[live])))

    delta_abs = delta * sigma_typ if delta_in_sigma else float(delta)

    # Shift the reference to create a real model-reference mismatch d.
    ref_shift = ref_traj + delta_abs
    y = (x - ref_shift[..., None]) ** 2
    VY = y.var(axis=3, ddof=1)                       # measured V[Y_ijk]
    d = mu - ref_shift                               # mismatch d = mu - mu_ref

    c = (1.0 + np.abs(mu) + 2.0 * np.abs(d)) ** 2
    bound_old = c * sigma2                                        # ~ sigma^2
    term_quartic = (kappa - 1.0) * sigma2 ** 2                    # ~ sigma^4
    term_offset = 4.0 * d ** 2 * sigma2                           # ~ sigma^2
    bound_new = (term_quartic
                 + 4.0 * np.sqrt(kappa) * np.abs(d) * sigma2 ** 1.5
                 + term_offset)

    with np.errstate(divide='ignore', invalid='ignore'):
        r_old = (bound_old / VY).ravel()
        r_new = (bound_new / VY).ravel()

    if i_site is None:
        i_site = L // 2
    t = np.arange(n_t)

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(10, 4))

    # Left: measured V[Y] against both bounds.
    # Measured drawn thick underneath; the new bound lies almost on top of it.
    ax0.plot(t, VY[i_site, j_obs], '-', color="#4C72B0", lw=4.5, alpha=0.8,
             label=r"$\mathrm{V}[Y_{ijk}]$ (measured)")
    ax0.plot(t, bound_old[i_site, j_obs], 's--', color="#C44E52", ms=4,
             label=r"old: $c_{ijk}\sigma_{ijk}^2$")
    ax0.plot(t, bound_new[i_site, j_obs], '^--', color="#55A868", ms=3.5,
             lw=1.5, label=r"new: $(\kappa-1)\sigma_{ijk}^4 + \ldots$")
    # The two competing terms of the new bound: which one dominates is the point.
    ax0.plot(t, term_quartic[i_site, j_obs], ':', color="#55A868", lw=1.5,
             label=r"\quad $(\kappa-1)\sigma_{ijk}^4$ only")
    if delta_abs != 0.0:
        ax0.plot(t, term_offset[i_site, j_obs], ':', color="#8172B2", lw=1.5,
                 label=r"\quad $4d_{ijk}^2\sigma_{ijk}^2$ only")
    ax0.set_yscale('log')
    ax0.set_xlabel(r"time index $k$")
    ax0.set_ylabel(r"variance of $Y_{ijk}$")
    ax0.set_title(rf"$i={i_site}$, $j={j_obs}$, "
                  rf"$\delta/\sigma={delta_abs/sigma_typ:.2f}$")
    ax0.legend(frameon=False, handlelength=2)
    ax0.spines['top'].set_visible(True)
    ax0.spines['right'].set_visible(True)

    # Right: looseness distribution of both bounds.
    stats = {}
    for r, col, lab in ((r_old, "#C44E52", "old ($\\sigma^2$)"),
                        (r_new, "#55A868", "new ($\\sigma^4$)")):
        good = np.isfinite(r) & (r > 0)
        rg = r[good]
        med = np.median(rg)
        stats[lab] = (med, rg.mean(), rg.min(), rg.max())
        ax1.hist(np.log10(rg), bins=50, color=col, edgecolor="k",
                 linewidth=0.4, alpha=0.65,
                 label=rf"{lab}, median $={med:.3g}\times$")
        ax1.axvline(np.log10(med), color=col, lw=2)

    ax1.axvline(0.0, color='k', ls=':', lw=1.5)     # bound == measured
    ax1.set_xlabel(r"$\log_{10}\left( \mathrm{bound} \,/\, \mathrm{V}[Y_{ijk}] \right)$")
    ax1.set_ylabel("count")
    ax1.set_title("looseness of each bound")
    ax1.legend(frameon=False)
    ax1.spines['top'].set_visible(True)
    ax1.spines['right'].set_visible(True)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=600, bbox_inches="tight", transparent=True)
        plt.close(fig)
    else:
        plt.show()

    dom = np.median(term_offset[live]) / np.median(term_quartic[live])
    print(f"delta = {delta_abs:.4g}  (delta/sigma = {delta_abs/sigma_typ:.3g}, "
          f"typical sigma = {sigma_typ:.3g})")
    print(f"  4d^2sigma^2 / (kappa-1)sigma^4 = {dom:.3g}  -> "
          f"{'quartic regime' if dom < 1 else 'sigma^2 regime'}")
    for lab, (med, mean, lo, hi) in stats.items():
        print(f"{lab:16s} bound/V[Y]: median={med:.4g}  mean={mean:.4g}  "
              f"min={lo:.4g}  max={hi:.4g}")

    return fig, (ax0, ax1)


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
##   Bound waterfall: contribution of each approximation
##   (single N_site, N_traj) — which approximation raises it most?
##############################################################

L_wf = 20            # keep small: stage 1-3 build a full (N, N) covariance
ntraj_wf = 500       # bootstrap subset size (N_traj)
n_samp_wf = 5000     # number of bootstrap samples
ell = 4              # maximum covariance distance

rng = np.random.default_rng(42)

folder = f"/home/aramos/Dokumente/Work/simulation_of_open_quantum_systems/tjm_noise_char/tests/yaqs_test/results/propagation/yaqs/L_{L_wf}/"
traj_data, ref_traj, time = load_traj(folder, L_wf)

stages = bound_waterfall(traj_data, ref_traj, ntraj_wf, n_samp_wf, ell, rng=rng)

print(f"\nBound waterfall (N_site={L_wf}, N_traj={ntraj_wf}, ell={ell}):")
prev = None
for name, v in stages.items():
    s = np.sqrt(v)
    fac = "" if prev is None else f"  (x{s/prev:.3g})"
    print(f"  {name:24s}  sigma = {s:.3e}{fac}")
    prev = s

save_path = f"results/propagation/yaqs/plots/bound_waterfall_L{L_wf}_ntraj{ntraj_wf}.pdf"
plot_bound_waterfall(stages, L_wf, ntraj_wf, save_path=save_path)


#%%
##############################################################
##   V[Y_ijk] vs its Lemma bound c_ijk sigma_ijk^2
##   (the approximation that dominates the bound's looseness)
##############################################################

L_vy = 20            # system size
ntraj_vy = 500       # bootstrap subset size (N_traj)
n_samp_vy = 5000     # number of bootstrap samples
i_site_vy = L_vy // 2   # site to show in the left panel
j_obs_vy = 0            # observable to show (0=X, 1=Y, 2=Z)

rng = np.random.default_rng(42)

folder = f"/home/aramos/Dokumente/Work/simulation_of_open_quantum_systems/tjm_noise_char/tests/yaqs_test/results/propagation/yaqs/L_{L_vy}/"
traj_data, ref_traj, time = load_traj(folder, L_vy)

save_path = f"results/propagation/yaqs/plots/var_y_vs_bound_L{L_vy}_ntraj{ntraj_vy}.pdf"
plot_var_y_vs_bound(traj_data, ref_traj, ntraj_vy, n_samp_vy,
                    i_site=i_site_vy, j_obs=j_obs_vy, rng=rng, save_path=save_path)


#%%
##############################################################
##   Old (sigma^2) vs new (sigma^4) bound on V[Y_ijk]
##############################################################

L_cmp = 20            # system size
ntraj_cmp = 500       # bootstrap subset size (N_traj)
n_samp_cmp = 5000     # number of bootstrap samples
i_site_cmp = L_cmp // 2  # site shown in the left panel
j_obs_cmp = 0            # observable shown (0=X, 1=Y, 2=Z)
# Kurtosis bound. kappa=3 is the Gaussian / large-N_traj sample-mean limit and
# reproduces V[Y] to ~2% on average, but it is NOT a strict upper bound: the
# measured kurtosis reaches ~5.4 at some indices, so kappa=3 is violated there
# (min ratio ~0.5). Use kappa >= 6 for a rigorous bound on this data (still
# ~2.5x from tight, vs ~4.7e4x for the old sigma^2 bound).
kappa_cmp = 3.0


rng = np.random.default_rng(42)

folder = f"/home/aramos/Dokumente/Work/simulation_of_open_quantum_systems/tjm_noise_char/tests/yaqs_test/results/propagation/yaqs/L_{L_cmp}/"
traj_data, ref_traj, time = load_traj(folder, L_cmp)

#%%
# Model-reference mismatch d, in units of the typical sigma (delta_in_sigma=True).
# delta=0  : d = 0 identically (bootstrap artifact) -- the ONLY quartic regime.
# delta=1  : the crossover, 4 d^2 sigma^2 == 2 sigma^4.
# delta=10 : deep in the sigma^2 regime, quartic term irrelevant (~200x smaller).
delta_cmp = 10.0

save_path = (f"results/propagation/yaqs/plots/bound_comparison_L{L_cmp}"
             f"_ntraj{ntraj_cmp}_delta{delta_cmp:g}.pdf")
plot_bound_comparison(traj_data, ref_traj, ntraj_cmp, n_samp_cmp,
                      i_site=i_site_cmp, j_obs=j_obs_cmp, kappa=kappa_cmp,
                      delta=delta_cmp, delta_in_sigma=True,
                      rng=rng, save_path=save_path)


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
