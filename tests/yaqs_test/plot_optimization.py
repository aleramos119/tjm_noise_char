
#%%
import matplotlib.pyplot as plt
import numpy as np
import os

import pandas as pd
import glob
from matplotlib.widgets import Slider

# from mqt.yaqs.noise_char.optimization import trapezoidal


def plot_gamma_optimization(folder: str) -> None:
    """Plot the optimization history of gamma parameters from a given folder.

    Parameters
    ----------
    folder : str
        The folder containing the optimization data files.
    """
    file_list = ["/loss_x_history", "/loss_x_history_avg"]

    max_diff = None

    print("Folder:", folder)

    if os.path.isfile(folder + file_list[0] + ".txt"):

        for file in file_list:

            x_avg_file = folder + file + ".txt"
                        
            gammas_file = folder + "/gammas.txt"

            data = np.genfromtxt(x_avg_file, skip_header=1, ndmin=2)
            gammas = np.array(np.genfromtxt(gammas_file, skip_header=1, ndmin=1))

            d = len(gammas)

            # --- Publication-quality plotting for gamma optimization history ---
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

            # Gamma trace plot
            fig, ax = plt.subplots(figsize=(5, 4))
            color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
            for i in range(d):
                ax.plot(
                    data[:, 0], data[:, 2 + i], 
                    label=rf"$\gamma_{{{i+1}}}$",
                    color=color_cycle[i % len(color_cycle)],
                )
            ax.axhline(
                gammas[0], 
                color="black",
                linestyle='--', linewidth=2,
                alpha=0.7
            )
            ax.set_xlabel("Iterations", labelpad=4)
            ax.set_ylabel(r"$\gamma$", labelpad=4)
            ax.legend(frameon=False, loc='best', handlelength=2)
            # Show top and right border (make sure they're visible)
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            # Remove grid
            # Do not set title; leave for caption
            plt.tight_layout()
            plt.savefig(folder + file + ".pdf", dpi=600, bbox_inches="tight", transparent=True)
            plt.close(fig)

            max_diff = max(abs(np.mean(data[10:, 2:2 + d], axis=0) - gammas))

            # Publication-quality plotting for loss optimization history
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            ax2.plot(
                data[:, 0], np.log10(np.sqrt(data[:, 1])), color="tab:blue"
            )
            ax2.set_xlabel("Iterations", labelpad=4)
            ax2.set_ylabel(r"$\log_{10}\left(\sqrt{J}\right)$", labelpad=4)
            ax2.legend(frameon=False, loc="best")
            ax2.spines['top'].set_visible(True)
            ax2.spines['right'].set_visible(True)
            # Remove grid
            plt.tight_layout()
            plt.savefig(folder + "/loss.pdf", dpi=600, bbox_inches="tight", transparent=True)
            plt.close(fig2)



    return max_diff


#%%
L_list_initial=[10,20,40, 80, 100]

folder="results/optimization/d_2/"


ntraj_list=[512, 1024]

ntraj=512


#%%
### Plotting the error vs L for different ntraj values
plt.rcParams.update({'axes.linewidth': 1.2})
plt.rcParams.update({'font.size': 12, 'lines.linewidth': 2, 'lines.markersize': 6})

plt.figure(figsize=(9, 6))


for ntraj in ntraj_list:

    error_list=[]
    L_list = []



    for L in L_list_initial:

        x_avg_file=folder + f"L_{L}/ntraj_{ntraj}/loss_x_history_avg.txt"

        print(x_avg_file)

        if os.path.exists(x_avg_file):

            data = np.genfromtxt(x_avg_file, skip_header=1)

            if len(data) > 1:

                error_list.append(max(abs(data[-1,-2:]-0.1)))

                L_list.append(L)

    plt.plot(L_list, np.log10(np.array(error_list)), marker='o', label=f"N_traj={ntraj}")

plt.xlabel(r"N")
plt.ylabel(r"$log( e_{max})$")
plt.legend()
plt.savefig(f"{folder}/error_vs_L_15x5.pdf", dpi=300, bbox_inches='tight')




# %%
### Plotting the average gamma values over iterations


L=120
ntraj=256
max_bond_dim=8
gamma="random"
gamma_0="random"

folder = f"results/optimization/method_tjm_exact_opt_script_test/max_bond_dim_{max_bond_dim}/d_2/gamma_{gamma}/gamma_0_{gamma_0}/L_{L}/ntraj_{ntraj}/"
x_avg_file=folder + f"loss_x_history.txt"

data = np.genfromtxt(x_avg_file, skip_header=1)

gammas_file=folder + f"gammas.txt"
gammas = np.genfromtxt(gammas_file, skip_header=1)

d=len(gammas)

for i in np.random.choice(range(d), size=2, replace=False):
    plt.plot(data[:, 0], data[:, 2 + i], label=f"$\\gamma_{{{i+1}}}$")
    plt.axhline(gammas[i], color=plt.gca().lines[-1].get_color(), linestyle='--', linewidth=2)


plt.xlabel("Iterations")
plt.ylabel(r"$\gamma$")
plt.legend()
plt.show()
# plt.savefig(f"{folder}/gamma_avg_vs_iterations_L_{L}_ntraj_{ntraj}.pdf", dpi=300, bbox_inches='tight')

#%%

data

#%%
# Plot the loss from loss_x_history.txt

L=80
ntraj=512

folder = f"results/optimization/method_tjm_exact/d_2_test/L_{L}/ntraj_{ntraj}/"
loss_file = folder + f"loss_x_history.txt"

if os.path.exists(loss_file):
    data = np.genfromtxt(loss_file, skip_header=1)
    plt.figure(figsize=(8, 5))
    plt.plot(data[:, 0], np.log10(data[:, 1]), label="Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Loss vs Iterations")
    plt.legend()
    plt.show()
else:
    print("Loss file not found.")






#%%

#### Plot reference and optimized trajectory


col=1


L=80
ntraj=512

folder = f"results/optimization/method_tjm_exact/d_2_test/L_{L}/ntraj_{ntraj}/"

ref_traj_file = folder + f"ref_traj.txt"
opt_traj_file = folder + f"opt_traj.txt"

if os.path.exists(ref_traj_file) and os.path.exists(opt_traj_file):
    ref_traj = np.genfromtxt(ref_traj_file, skip_header=1)
    opt_traj = np.genfromtxt(opt_traj_file, skip_header=1)

    plt.figure(figsize=(8, 5))
    plt.plot(ref_traj[:, 0], ref_traj[:, col], label="Reference Trajectory")
    plt.plot(opt_traj[:, 0], opt_traj[:, col], label="Optimized Trajectory", linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Trajectory Value")
    plt.legend()
    plt.title("Comparison of Reference and Optimized Trajectories")
    plt.show()
else:
    print("Reference or optimized trajectory file not found.")



#%%

loss=np.sum((ref_traj-opt_traj)**2)
print(loss)




#%%

#### Plot reference and optimized trajecotry
folder="results/optimization/d_2/"
L=100
ntraj=512


ref_traj_file = folder + f"L_{L}/ntraj_{ntraj}/ref_traj.txt"
opt_traj_file = folder + f"L_{L}/ntraj_{ntraj}/opt_traj.txt"

if os.path.exists(ref_traj_file):
    ref_traj = np.genfromtxt(ref_traj_file, skip_header=1)
    # opt_traj = np.genfromtxt(opt_traj_file, skip_header=1)

    plt.figure(figsize=(8, 5))
    plt.plot(ref_traj[:, 0], ref_traj[:, 1], label="Reference Trajectory")
    # plt.plot(opt_traj[:, 0], opt_traj[:, 1], label="Optimized Trajectory", linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Trajectory Value")
    plt.legend()
    plt.title("Comparison of Reference and Optimized Trajectories")
    plt.show()
else:
    print("Reference or optimized trajectory file not found.")





#%%

#### Plot reference and opptimized trajecotry
folder="results/optimization/d_2/"
L=100
ntraj=512


ref_traj_file = f"results/cpu_traj_scan/method_scikit_tt_new_calc_omp_1/solver_krylov_5/order_1/threshold_1e-4/{L}_sites/96_cpus/{ntraj}_traj/qt_ref_traj.txt"

if os.path.exists(ref_traj_file):
    ref_traj = np.genfromtxt(ref_traj_file, skip_header=1)
    # opt_traj = np.genfromtxt(opt_traj_file, skip_header=1)

    plt.figure(figsize=(8, 5))
    plt.plot(ref_traj[:, 0], ref_traj[:, 1], label="Reference Trajectory")
    # plt.plot(opt_traj[:, 0], opt_traj[:, 1], label="Optimized Trajectory", linestyle='--')
    plt.xlabel("Time")
    plt.ylabel("Trajectory Value")
    plt.legend()
    plt.title("Comparison of Reference and Optimized Trajectories")
    plt.show()
else:
    print("Reference or optimized trajectory file not found.")


#%%
ref_traj.shape



# %%

# Folder and file pattern

%matplotlib qt

col=1

L=120
ntraj=256
max_bond_dim=12
gamma="random"
gamma_0="random"

# folder = f"results/optimization/method_tjm_exact_opt_script_test/max_bond_dim_{max_bond_dim}/d_2/gamma_{gamma}/gamma_0_{gamma_0}/L_{L}/ntraj_{ntraj}/"
folder = f"test/gradient_descent_T_4_characterizer/"


file_pattern = folder + f"opt_traj_*.txt"

# Find all matching files and sort by index
opt_traj_files = sorted(glob.glob(file_pattern), key=lambda x: int(x.split('_')[-1].split('.')[0]))
if not opt_traj_files:
    print("No opt_traj_{i}.txt files found.")
else:
    # Load all trajectories
    opt_trajs = [np.genfromtxt(f, skip_header=1) for f in opt_traj_files]


    ref_traj = np.genfromtxt(folder + f"ref_traj.txt", skip_header=1)

    # Initial plot
    fig, ax = plt.subplots(figsize=(8, 5))
    plt.subplots_adjust(bottom=0.2)
    ax.plot(ref_traj[:, 0], ref_traj[:, col], label="Reference Trajectory")
    l, = ax.plot(opt_trajs[0][:, 0], opt_trajs[0][:, col], label="Optimized Trajectory", linestyle='--')

    ax.set_xlabel("Time")
    ax.set_ylabel("Trajectory Value")
    ax.set_title("Optimized Trajectory (select index with slider)")
    ax.legend()
    

    # Slider
    ax_slider = plt.axes([0.2, 0.05, 0.6, 0.03])
    slider = Slider(ax_slider, 'Index', 0, len(opt_trajs)-1, valinit=0, valstep=1)

    def update(val):
        idx = int(slider.val)
        l.set_ydata(opt_trajs[idx][:, col])
        l.set_xdata(opt_trajs[idx][:, 0])
        # ax.relim()
        # ax.autoscale_view()
        fig.canvas.draw_idle()
        ax.legend()

    slider.on_changed(update)
    plt.show()
# %%
ref_traj.shape
# %%

### Plot error gammas


L=5
ntraj_list=[512, 1024, 2048, 4096, 8192]

error_list=[]

for ntraj in ntraj_list:

    folder = f"results/optimization/method_tjm_exact/d_2L_test/L_{L}/ntraj_{ntraj}/"
    x_avg_file=folder + f"loss_x_history_avg.txt"

    data = np.genfromtxt(x_avg_file, skip_header=1)

    gammas_file=folder + f"gammas.txt"
    gammas = np.genfromtxt(gammas_file, skip_header=1)

    d=len(gammas)

    error_list.append(np.log10(max(abs(data[-1,2:]-gammas))))
    


plt.plot(ntraj_list, error_list,'o-', label=f"L={L}")



#%%


#%%

obs_list = [ "pauli_z","XYZ"]
noise_list = ["pauli_x", "pauli_y", "pauli_z","XYZ"]

sites=1
ntraj=400


diff_array=np.full((len(obs_list), len(noise_list)), np.nan)

for i,obs in enumerate(obs_list):
    for j,noise in enumerate(noise_list):


        if noise==obs and noise!="XYZ":
            continue

        if obs == "XYZ" and noise=="pauli_z":
            continue


        folder=f"results/characterizer/sites_{sites}/num_traj_{ntraj}/parameters_global/init_state_zeros/observable_{obs}/noise_{noise}/"

        diff_max=plot_gamma_optimization(folder)

        diff_array[i,j]=diff_max


plt.figure(figsize=(8, 6))
im = plt.imshow(diff_array, origin='lower', cmap='viridis', aspect='auto')
cbar = plt.colorbar(im)
cbar.set_label("max |mean(gamma) - gamma_opt|")
plt.xticks(np.arange(len(noise_list)), noise_list, rotation=45, ha='right')
plt.yticks(np.arange(len(obs_list)), obs_list)
plt.xlabel("Noise")
plt.ylabel("Observable")
plt.title("Max difference between averaged and optimized gammas")

# Annotate cells with values
vmax = np.nanmax(diff_array)
for i in range(diff_array.shape[0]):
    for j in range(diff_array.shape[1]):
        val = diff_array[i, j]
        color = "white" if (not np.isnan(vmax) and val > 0.5 * vmax) else "black"
        plt.text(j, i, f"{val:.2e}", ha="center", va="center", color=color, fontsize=9)

plt.tight_layout()
plt.savefig(f"results/characterizer/sites_{sites}/num_traj_{ntraj}/diff_gamma_obs_vs_noise_sites_{sites}_ntraj_{ntraj}.pdf", dpi=300, bbox_inches='tight')
plt.show()





# %%

folder="test/nelder_mead_opt/"

plot_gamma_optimization(folder)
# %%

obs_y_traj_file="test/obs_y_characterizer/ref_traj.txt"

z1_traj_file="results/characterizer/sites_2/num_traj_400/parameters_global/init_state_zeros/observable_pauli_y/noise_pauli_z/ref_traj.txt"


obs_y_traj=np.genfromtxt(obs_y_traj_file)
z1_traj=np.genfromtxt(z1_traj_file)

i=1
plt.plot(obs_y_traj[:,0], obs_y_traj[:,i],'x',label="obs_y_"+str(i) )
plt.plot(z1_traj[:,0], z1_traj[:,i],'x',label="z1_"+str(i) )
plt.legend()
plt.show()


# %%

work_dir="test/gamma_scan_T_4_gamma_ref_0.01/"
loss_list=np.genfromtxt(work_dir + "/loss_list.txt")
grad_list=np.genfromtxt(work_dir + "/grad_list.txt")
gamma_list=np.genfromtxt(work_dir + "/gamma_list.txt")

num_grad=np.gradient(loss_list, gamma_list)

# %%
%matplotlib qt
plt.plot(gamma_list, loss_list,'o', label="loss")
plt.plot(gamma_list, grad_list,'x', label="grad")
plt.plot(gamma_list, num_grad, label="numeric gradient")
plt.plot(gamma_list, num_grad/grad_list, label="relation")
plt.grid(True)
plt.legend()


# %%
work_dir="test/gamma_scan_T_6/"
i=4
%matplotlib qt
for i in range(1):
    d_on_d_gk=np.genfromtxt(work_dir + f"/d_on_list_{i}.txt", ndmin = 2)
    obs_array=np.genfromtxt(work_dir + f"/obs_array_{i}.txt", ndmin = 2)
    ref_traj_data=np.genfromtxt(work_dir + f"/ref_traj.txt", ndmin = 2)

    t=ref_traj_data[:,0]

    ref_traj=ref_traj_data[:,1:].T

    diff=obs_array-ref_traj


    inside_sum=2*diff[0]*d_on_d_gk[0]

    sumation=[]
    total=0

    for j in range(len(diff[0])):
        total=total+inside_sum[j]
        sumation.append(total)


    if i==0:
        sumation0=sumation

    result=np.array(sumation)

    plt.plot(t,diff[0],'x', label=f"diff {i}")
    plt.plot(t,ref_traj[0],'-', label=f"ref_traj {i}")
    plt.plot(t,obs_array[0],'x', label=f"obs_array {i}")
    plt.plot(t,inside_sum,'o-', label=f"inside sum {i}")
    plt.plot(t,sumation,'o-', label=f"sumation {i}")




    plt.plot(t,d_on_d_gk[0],'x', label=f"d_on_d_gk {i}")
    plt.plot(t,-2*trapezoidal(obs_array[0], t),'-', label=f"trapezoidal {i}")
plt.grid(True)
plt.legend()

#%%
ref_traj.shape


# %%
plt.plot(t,sumation0)
# %%

# %%

%matplotlib qt
def loss(obs_traj, ref_traj, sigma, exponent=2):

    diff = obs_traj-ref_traj

    error = np.random.normal(loc=0.0, scale=sigma, size=ref_traj.shape)

    diff_err = diff + error


    result = np.sum(diff_err**exponent)
    return result

def loss_err(obs_traj, ref_traj, sigma, exponent=2):

    diff = obs_traj-ref_traj

    error = np.random.normal(loc=0.0, scale=sigma)

    diff_err = diff


    result = np.sum(diff_err**exponent) + np.abs(error)
    return result

obs="Z"

noise="Z"


N_list=[4000]

op_list=["X", "Y", "Z"]


for N in N_list:
    for noise in op_list:

        T=6

        data_dir = f"test/loss_scan/gamma_ref_0.01/N_{N}/T_{T}/obs_{obs}/noise_{noise}/"


        gamma_list=np.genfromtxt(data_dir + "gamma_list.txt")



        ref_traj = np.genfromtxt(data_dir + "ref_traj.txt")

        exponent=2

        sigma = 0.00

        # sigma_values=[0.01, 0.02, 0.03, 0.04, 0.05]



        # for sigma in sigma_values:
        loss_list = []

        for i in range(len(gamma_list)):

            obs_traj = np.genfromtxt(data_dir + f"obs_array_{i}.txt")

            loss_list.append(loss_err(obs_traj, ref_traj, sigma, exponent=exponent))


        plt.plot(gamma_list,loss_list, 'x-', label=f"sigma_{sigma}, N_{N}/T_{T}/obs_{obs}/noise_{noise} ")
plt.legend()
# %%



%matplotlib qt
method_list = ["bo_ucb", "bo_ei", "bo_pi", "cma", "diff_evol", "nelder_mead"]

d_list = [i+1 for i in range(1,25)]

std=0.04

for method in method_list:

    error_list=[]

    for d in d_list:

        file=f"test/optimization_comparisson/opt_{method}/std_{std}/d_{d}/rel_error.txt"

        iterations = np.genfromtxt(f"test/optimization_comparisson/opt_{method}/std_{std}/d_{d}/loss_x_history.txt")[-1,0]

        error = np.genfromtxt(file)

        error_list.append(error*iterations)

    
    plt.plot(d_list,error_list,'x-',label=method)


plt.title(f"std_{std}")
plt.grid(True)
plt.legend()



#%%
import re
from pathlib import Path

def get_total_duration_hours(log_dir):
    """
    Search for '*.log' files in the given directory,
    extract the last 'Total duration:' line,
    and return the total duration in seconds.
    """

    log_dir = Path(log_dir)
    pattern = re.compile(
        r"Total duration:\s*(\d+)\s*days?,\s*(\d+)\s*hours?,\s*(\d+)\s*minutes?,\s*(\d+)\s*seconds?"
    )

    last_match = None

    # Search through all *.log files
    for logfile in sorted(log_dir.glob("*.log")):
        with logfile.open() as f:
            for line in f:
                match = pattern.search(line)
                if match:
                    last_match = match

    if last_match:
        days, hours, minutes, seconds = map(int, last_match.groups())
        total_seconds = days*86400 + hours*3600 + minutes*60 + seconds
        return total_seconds/3600

    return None



L=100
N=500

cpu_list=[20,40,80,160]

time_list=[]


for cpu in cpu_list:

    folder=f"results/yaqs_time_test/L_{L}/N_{N}/ncpus_{cpu}"

    time_list.append(get_total_duration_hours(folder))


plt.plot(cpu_list, time_list, 'o-', label=f"L_{L}/N_{N}")

plt.legend()


# %%

for current_dir, subdirs, files in os.walk("results/characterizer_gradient_free/loss_scale_True_reduced"):
        # If the directory has no subdirectories, treat it as a leaf node
        if not subdirs:
            plot_gamma_optimization(current_dir)

# %%


i_list= np.array(range(0,1000))

i0=100

n_1=i0 + i_list

n_3=i0 + 2*i_list


n_2=i0 + 200*np.log10(i_list)
plt.plot(i_list,n_1,label="lineal")
plt.plot(i_list,n_2,label="log")
plt.plot(i_list,n_3,label="2*lineal")

plt.legend()
plt.grid(True)



# %%
import matplotlib.pyplot as plt
import numpy as np
L=10
N_list=[50, 100, 200, 500, 1000]

for N in N_list:
    directory=f"test/loss_scan/gamma_ref_0.01_d_3/L_{L}/N_{N}/T_6/obs_Z/noise_X/"
    gamma_list = np.genfromtxt(directory + f"gamma_list.txt")
    loss_list = np.genfromtxt(directory + f"loss_list.txt")

    plt.plot(gamma_list, loss_list, 'o-')
    plt.xlim(0, 0.1)
    plt.ylim(-0.2, 50)
    plt.savefig(directory + "loss_vs_gamma.pdf", dpi=300, bbox_inches='tight')
    plt.close()
# %%



import itertools
from pathlib import Path
import matplotlib.pyplot as plt
from pdf2image import convert_from_path

# Parameter lists
# method="cma"
# list1 = [0.01, 0.04, 0.08, 0.16]
# par1_name="sigma0"
# list2 = [4,8,16]
# par2_name="popsize"
# ntraj=1000


method="bayesian"
list1 = [0.0001, 0.001, 0.005]
par1_name="std"
list2 = [0.1,0.5,1,4,8]
par2_name="beta"
ntraj=2000


# Path where your PDFs are located

# Create 3x3 figure
fig, axes = plt.subplots(len(list1), len(list2), figsize=(12, 12))

# Iterate through the 9 combinations
for (p1, p2), ax in zip(itertools.product(list1, list2), axes.flatten()):

    folder=Path(f"test/{method}_parameter_test/method_{method}/ntraj_{ntraj}/{par1_name}_{p1}/{par2_name}_{p2}") 

    pdf_path = folder / "loss_x_history.pdf"

    print(pdf_path)

    try:
        # Convert first (and only) page of the PDF to an image
        img = convert_from_path(str(pdf_path), first_page=1, last_page=1)[0]

        # Show on the axis
        ax.imshow(img)
        ax.axis("off")
        ax.set_title(f"p1={p1}, p2={p2}")

    except Exception as e:
        ax.text(0.5, 0.5, f"Missing\n{pdf_path.name}", ha="center", va="center")
        ax.axis("off")

plt.tight_layout()
plt.savefig(f"test/{method}_parameter_test/method_{method}/ntraj_{ntraj}/{method}_parameter_scan_ntraj_{ntraj}.png", dpi=300, bbox_inches='tight')
plt.show()
plt.close()
# %%
L=10
n_t=60
n_obs=3*L
const=4e6
N=int(np.ceil(const/(n_t*n_obs)))
print(N)
# %%

from matplotlib.gridspec import GridSpec

def plot_optimization_grid(L1: int, L2: int, module: str, method: str, params: str, 
                           const: str, xlim: float, output_file: str = None, 
                           traj_col: int = 1) -> None:
    """
    Create a 3x2 grid plot showing loss, gamma optimization history, and trajectory comparison
    for two different L values. All y-axes within a row are set to the same range.
    Removes x-labels from the first row.
    """
    import matplotlib as mpl

    # Set scientific plotting defaults
    mpl.rcParams.update({
        'figure.figsize': (10, 12),
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

    fig, axes = plt.subplots(
    3, 2,
    figsize=(10, 12),
    sharey='row'
)  # Ensure shared Y axes per row




    L_list = [L1, L2]

    # Data collectors for ylim computation
    row1_y = []
    row2_y = []
    row3_y = []

    # To keep track of which columns have data (for later y axis normalization)
    row1_axes, row2_axes, row3_axes = [], [], []

    all_data = {}

    # PRELOAD all data to determine axis scales first
    for col_idx, L in enumerate(L_list):
        folder = f"results/characterizer_gradient_free/loss_scale_True_reduced/module_{module}/method_{method}/params_{params}/const_{const}/L_{L}/xlim_{xlim}/"

        loss_x_file = os.path.join(folder, "loss_x_history.txt")
        gammas_file = os.path.join(folder, "gammas.txt")
        ref_traj_file = os.path.join(folder, "ref_traj.txt")
        opt_traj_files = sorted(glob.glob(os.path.join(folder, "opt_traj_*.txt")))
        opt_traj_file = opt_traj_files[0] if opt_traj_files else None

        if not os.path.exists(loss_x_file):
            print(f"Warning: {loss_x_file} does not exist, skipping L={L}")
            continue

        data = np.genfromtxt(loss_x_file, skip_header=1, ndmin=2)
        gammas = np.array(np.genfromtxt(gammas_file, skip_header=1, ndmin=1)) if os.path.exists(gammas_file) else None
        d = len(gammas) if gammas is not None else 0

        # Store so we don't reload
        all_data[L] = {
            'data': data,
            'gammas': gammas,
            'd': d,
            'ref_traj_file': ref_traj_file,
            'opt_traj_file': opt_traj_file
        }

        # Row 1
        row1_y.extend(np.log10(np.sqrt(data[:, 1])))
        row1_axes.append(col_idx)

        # Row 2
        if d > 0:
            for i in range(d):
                row2_y.extend(list(data[:, 2 + i]))
            row2_axes.append(col_idx)

        # Row 3
        if os.path.exists(ref_traj_file) and opt_traj_file and os.path.exists(opt_traj_file):
            ref_traj = np.genfromtxt(ref_traj_file, skip_header=1)
            opt_traj = np.genfromtxt(opt_traj_file, skip_header=1)
            if ref_traj.shape[1] > traj_col and opt_traj.shape[1] > traj_col:
                row3_y.extend(ref_traj[:, traj_col])
                row3_y.extend(opt_traj[:, traj_col])
                row3_axes.append(col_idx)

    # Compute global y-limits for each row (if any data exists)
    if row1_y:
        y1min, y1max = np.min(row1_y), np.max(row1_y)
        margin1 = 0.05 * (y1max - y1min) if y1max > y1min else 0.1
        y1lim = (y1min - margin1, y1max + margin1)
    else:
        y1lim = None
    if row2_y:
        y2min, y2max = np.min(row2_y), np.max(row2_y)
        margin2 = 0.05 * (y2max - y2min) if y2max > y2min else 0.1
        y2lim = (y2min - margin2, y2max + margin2)
    else:
        y2lim = None
    if row3_y:
        y3min, y3max = np.min(row3_y), np.max(row3_y)
        margin3 = 0.05 * (y3max - y3min) if y3max > y3min else 0.1
        y3lim = (y3min - margin3, y3max + margin3)
    else:
        y3lim = None

    for col_idx, L in enumerate(L_list):
        if L not in all_data:
            continue
        this = all_data[L]
        data = this['data']
        gammas = this['gammas']
        d = this['d']
        ref_traj_file = this['ref_traj_file']
        opt_traj_file = this['opt_traj_file']

        # Row 1: Loss plot
        ax = axes[0, col_idx]
        loss_vals = np.log10(np.sqrt(data[:, 1]))
        ax.plot(data[:, 0], loss_vals, color="tab:blue")
        # Remove x axis label for the first row
        ax.set_xlabel("")
        ax.tick_params(labelbottom=False)
        if col_idx == 0:
            ax.set_ylabel(r"$\log_{10}\left(\sqrt{J}\right)$", labelpad=4)
            # ax.tick_params(left=True, labelleft=True)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        # if col_idx == 1:
        #     ax.set_yticklabels([])
        #     ax.set_yticks([])
        #     ax.set_ylabel("")
        #     ax.tick_params(left=False, labelleft=False)

        # Set global y-limits for row 1
        if y1lim:
            ax.set_ylim(y1lim)

        # Row 2: Gamma optimization history
        ax = axes[1, col_idx]
        color_cycle = plt.rcParams['axes.prop_cycle'].by_key()['color']
        if d > 0:
            for i in range(d):
                ax.plot(
                    data[:, 0], data[:, 2 + i], 
                    label=rf"$\gamma_{{{i+1}}}$",
                    color=color_cycle[i % len(color_cycle)],
                )
            if gammas is not None and len(gammas) > 0:
                ax.axhline(
                    gammas[0], 
                    color="black",
                    linestyle='--', linewidth=2,
                    alpha=0.7
                )
        ax.set_xlabel("Iterations", labelpad=4)
        if col_idx == 0:
            ax.set_ylabel(r"$\gamma$", labelpad=4)
            # ax.tick_params(left=True, labelleft=True)
        ax.legend(frameon=False, loc='best', handlelength=2, fontsize=11)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        # if col_idx == 1:
        #     ax.set_yticklabels([])
        #     ax.set_yticks([])
        #     ax.set_ylabel("")
        #     ax.tick_params(left=False, labelleft=False)
        # elif col_idx == 0:
        #     # Explicitly keep ticks visible for first column
        #     ax.tick_params(left=True, labelleft=True)
        # Set global y-limits for row 2
        if y2lim:
            ax.set_ylim(y2lim)

        # Row 3: Trajectory comparison
        ax = axes[2, col_idx]
        if os.path.exists(ref_traj_file) and opt_traj_file and os.path.exists(opt_traj_file):
            ref_traj = np.genfromtxt(ref_traj_file, skip_header=1)
            opt_traj = np.genfromtxt(opt_traj_file, skip_header=1)

            # Ensure we have enough columns
            if ref_traj.shape[1] > traj_col and opt_traj.shape[1] > traj_col:
                ax.plot(ref_traj[:, 0], ref_traj[:, traj_col], 
                        label=r"$\langle O_{11} \rangle^{(ref)}$", color="tab:blue")
                ax.plot(opt_traj[:, 0], opt_traj[:, traj_col], 
                        label=r"$\langle O_{11} \rangle^{(opt)}$", 
                        linestyle='--', color="tab:orange")
  
                ax.legend(frameon=False, loc='best', handlelength=2)
        else:
            ax.text(0.5, 0.5, "Trajectory files\nnot found", 
                   ha="center", va="center", transform=ax.transAxes)

        ax.set_xlabel("t", labelpad=4)
        if col_idx == 0:
            ax.set_ylabel(r"$\langle O_{11} \rangle$", labelpad=4)
            ax.tick_params(left=True, labelleft=True)
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        # if col_idx == 1:
        #     ax.set_yticklabels([])
        #     ax.set_yticks([])
        #     ax.set_ylabel("")
        #     ax.tick_params(left=False, labelleft=False)
        # elif col_idx == 0:
        #     # Explicitly keep ticks visible for first column
        #     ax.tick_params(left=True, labelleft=True)
        # Set global y-limits for row 3
        if y3lim:
            ax.set_ylim(y3lim)

    # Add column labels
    for col_idx, L in enumerate(L_list):
        axes[0, col_idx].text(0.5, 1.05, f"$({chr(97+col_idx)}) \\quad N_L={L}$", 
                             transform=axes[0, col_idx].transAxes,
                             ha='center', va='bottom', fontsize=17)

    plt.tight_layout()

    if output_file is None:
        folder1 = f"results/characterizer_gradient_free/loss_scale_True_reduced/module_{module}/method_{method}/params_{params}/const_{const}/"
        output_file = os.path.join(folder1, f"optimization_grid_L{L1}_L{L2}.pdf")

    plt.savefig(output_file, dpi=600, bbox_inches='tight', transparent=True)
    plt.close(fig)


# %%
# Example usage:
plot_optimization_grid(L1=10, L2=160, module="yaqs", method="cma", params="d_3", const="4e6", xlim=0.1)
# %%
