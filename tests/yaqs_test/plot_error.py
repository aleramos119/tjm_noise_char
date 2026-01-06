
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
method_list = ["cma","mcmc"]
L_list = [10,20,40,80,160]
params = "d_3"
xlim = 0.1
const="4e6"

std_conv=0.001
 

for method in method_list:
    error_list = []
    for L in L_list:
        directory = f"results/characterizer_gradient_free/method_{method}_reduced/params_{params}/const_{const}/L_{L}/xlim_{xlim}/"
        file = directory + "loss_x_history.txt"
        data = np.genfromtxt(file)[:,2:]
        gammas=np.genfromtxt(directory + "gammas.txt")
        d=data.shape[1]//2
        data = data[:,:d]

        conv_indx=find_convergence_index(data,std_conv)

        print(f"L={L}, method={method}, conv_indx={conv_indx}")

        mean_gammas = np.mean(data[conv_indx:], axis=0)

        error = np.max([ abs(mean_gammas[i] - gammas[i])/gammas[i]  for i in range(d) ])

        error_list.append(error)

    plt.plot(L_list, error_list, marker='o', label=f"{method}_{params}_const_{const}")
plt.legend()




# %%
