
#%%
import matplotlib.pyplot as plt
import numpy as np
import os

import pandas as pd
import glob
from matplotlib.widgets import Slider

# from mqt.yaqs.noise_char.optimization import trapezoidal

def moving_average(data, window):
    """
    Compute moving average along the first dimension (nt) for all columns (ng).
    
    Args:
        data: numpy array of shape (nt, ng)
        window: size of the moving average window
    
    Returns:
        numpy array of shape (nt - window + 1, ng) with moving averages
    """
    nt, ng = data.shape
    result = np.zeros((nt - window + 1, ng))
    
    for i in range(nt - window + 1):
        result[i] = np.mean(data[i:i + window], axis=0)
    
    return result



def find_convergence_index(data, threshold):
    """
    Compute the standard deviation of data with a sliding window and find the first index
    where all standard deviations are below a threshold.
    
    Args:
        data: numpy array of shape (nt, ng)
        threshold: standard deviation threshold
    
    Returns:
        int: the first index i where all std(data[i:]) are less than threshold,
                or -1 if no such index exists
    """
    nt, ng = data.shape
    
    for i in range(nt):
        window_data = data[i:, :]
        stds = np.std(window_data, axis=0)
        
        if np.all(stds < threshold):
            return i
    
    return -1


#%%
module="yaqs"
params = "d_3"

method_list = ["cma"]

if params == "d_3":
    L_list_initial = [10,20,40,80,160]

if params == "d_3L":
    L_list_initial = [2,4,8,16]


xlim = 0.1

const_list = ["4e6"]
std_conv=0.001

# --- Publication-quality plotting ---
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

for const in const_list:
    for method in method_list:
        error_list = []
        L_list = []
        for L in L_list_initial:
            directory = f"results/characterizer_gradient_free/loss_scale_True_reduced/module_{module}/method_{method}/params_{params}/const_{const}/L_{L}/xlim_{xlim}/"
            file = directory + "loss_x_history.txt"


            print(f"Doing module={module}, method={method}, params={params}, const={const}, L={L}")


            if not os.path.exists(file):
                print(f"File {file} does not exist")
                continue

            L_list.append(L)
            data = np.genfromtxt(file)[:,2:]
            gammas=np.genfromtxt(directory + "gammas.txt")
            d=data.shape[1]//2
            data = data[:,:d]

            conv_indx=find_convergence_index(data,std_conv)


            mean_gammas = np.mean(data[conv_indx:], axis=0)

            all_differences = [ abs(mean_gammas[i] - gammas[i])  for i in range(d) ]

            all_errors = [ abs(mean_gammas[i] - gammas[i])/gammas[i]  for i in range(d) ]

            

            error = np.max(all_errors)

            # print(f"L={L}, method={method}, const={const}, conv_indx={conv_indx}, error={error}, mean_gammas={mean_gammas}, gammas={gammas}")


            error_list.append(error)

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(L_list, error_list, marker='o', label=f"{method}_{params}_const_{const}")
        ax.set_xlabel("L", labelpad=4)
        ax.set_ylabel("Relative error", labelpad=4)
        ax.legend(frameon=False, loc='best', handlelength=2)
        # Show top and right border
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        # Remove grid
        plt.tight_layout()
        plt.savefig(f"results/characterizer_gradient_free/error_vs_L_loss_scale_{module}_{params}.pdf", dpi=600, bbox_inches='tight', transparent=True)
        plt.close(fig)



# %%
#%%
module="yaqs"
params = "d_3L"

method_list = ["cma"]

if params == "d_3":
    L_list_initial = [10,20,40,80,160]

if params == "d_3L":
    L_list_initial = [2,4,8,16]


xlim = 0.1

const_list = ["4e6"]
std_conv=0.001

# --- Publication-quality plotting ---
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

for const in const_list:
    for method in method_list:
        error_list = []
        L_list = []
        for L in L_list_initial:
            directory = f"results/characterizer_gradient_free/loss_scale_True_reduced/module_{module}/method_{method}/params_{params}/const_{const}/L_{L}/xlim_{xlim}/"
            file = directory + "loss_x_history.txt"


            print(f"Doing module={module}, method={method}, params={params}, const={const}, L={L}")


            if not os.path.exists(file):
                print(f"File {file} does not exist")
                continue

            L_list.append(L)
            loss = np.genfromtxt(file)[:,1][-1]

            error_list.append(np.log10(np.sqrt(loss)))

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(L_list, error_list, marker='o')
        ax.set_xlabel(r"$L$", labelpad=4)
        ax.set_ylabel(r"$\log_{10}\left( \sqrt{J} \right)$", labelpad=4)
        # Show top and right border
        ax.spines['top'].set_visible(True)
        ax.spines['right'].set_visible(True)
        # Format y ticks with less significant digits
        yticks = ax.get_yticks()
        reduced_yticks = yticks[::2]
        ax.set_yticks(reduced_yticks)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.2g}'.format(y)))
        plt.tight_layout()
        plt.savefig(f"results/characterizer_gradient_free/loss_vs_L_loss_scale_{module}_{params}.pdf", dpi=600, bbox_inches='tight', transparent=True)
        plt.close(fig)
# %%


