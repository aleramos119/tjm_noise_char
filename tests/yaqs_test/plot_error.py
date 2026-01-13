
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
module="scikit"
params = "d_3L"

method_list = ["cma","mcmc"]

if params == "d_3":
    L_list_initial = [10,20,40,80,160]

if params == "d_3L":
    L_list_initial = [2,4,8,16]


xlim = 0.1

const_list = ["4e6"]
std_conv=0.001


for const in const_list:
    for method in method_list:
        error_list = []
        L_list = []
        for L in L_list_initial:
            directory = f"results/characterizer_gradient_free/module_{module}_reduced/method_{method}/params_{params}/const_{const}/L_{L}/xlim_{xlim}/"
            file = directory + "loss_x_history.txt"

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

            print(f"L={L}, method={method}, const={const}, conv_indx={conv_indx}, error={error}, mean_gammas={mean_gammas}, gammas={gammas}")


            error_list.append(error)

        plt.plot(L_list, error_list, marker='o', label=f"{method}_{params}_const_{const}")

plt.title(f"Relative error vs L for different const values")
plt.xlabel("L")
plt.ylabel("Relative error")
plt.legend()

plt.savefig(f"results/characterizer_gradient_free/error_vs_L_{module}_{params}.pdf", dpi=300, bbox_inches='tight')
plt.close()



# %%
