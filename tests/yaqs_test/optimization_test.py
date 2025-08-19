#%%
import matplotlib.pyplot as plt
import numpy as np


from mqt.yaqs.noise_char.optimization import *
from mqt.yaqs.noise_char.propagation import *

from auxiliar.write import *

import sys
import os


import psutil
import threading
from datetime import datetime
import pandas as pd


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"



def print_gammas(gamma_rel, gamma_deph,params, sim_params=None):

    print("Gamma_rel: ",gamma_rel)
    print("Gamma_deph: ",gamma_deph)


    gammas = np.genfromtxt(f"{params["folder"]}/gammas.txt", skip_header=1)

    print("Folder gammas:", gammas)


    if sim_params!=None:
        print("Sim params gamma_rel: ", sim_params.gamma_rel)
        print("Sim params gamma_deph: ", sim_params.gamma_deph)






#%%

def set_gammas(params):

    sites=params["sites"]

    # If it's a restart job gammas will be read from the file
    if params["restart"]:

        gammas = np.genfromtxt(f"{params["folder"]}/gammas.txt", skip_header=1)

        if params["dim"] == "2":
            gamma_rel=gammas[0]
            gamma_deph=gammas[1]

        if params["dim"] == "2L":
            gamma_rel = gammas[:sites]
            gamma_deph = gammas[sites:]

    # If it's not a restart job gammas will be read from the file
    else:
        if params["dim"] == "2":
            if params["g_rel"]=="random":
                gamma_rel=np.random.rand()
            if params["g_rel"]!="random":
                gamma_rel=float(params["g_rel"])

            if params["g_deph"]=="random":
                gamma_deph=np.random.rand()
            if params["g_deph"]!="random":
                gamma_deph=float(params["g_deph"])


        if params["dim"] == "2L":

            if params["g_rel"]=="random":
                gamma_rel=list(np.random.rand(sites))
            if params["g_rel"]!="random":
                gamma_rel=[float(params["g_rel"])]*sites

            if params["g_deph"]=="random":
                gamma_deph=list(np.random.rand(sites))
            if params["g_deph"]!="random":
                gamma_deph=[float(params["g_deph"])]*sites
            
            
    if not params["restart"]:

        if not os.path.exists(params["folder"]):
                os.makedirs(params["folder"])


        gamma_header = "  ".join([f"gr_{i+1}" for i in range(sites)] + [f"gd_{i+1}" for i in range(sites)])
        gamma_file = f"{params["folder"]}/gammas.txt"
        gamma_data = np.hstack([gamma_rel, gamma_deph])
        np.savetxt(gamma_file, gamma_data.reshape(1, -1), header=gamma_header, fmt='%.6f')

    
    return gamma_rel, gamma_deph




def running_ref_traj(params,sim_params,traj_function):


    ## Computing reference trajectory 
    print("Running ref traj")
    


    if params["restart"]:
        data = np.genfromtxt(f"{params["folder"]}/ref_traj.txt", skip_header=1)

        t=data[:,0]
        n_t=len(t)

        n_obs_site=(len(data[0,0])-1)//params["sites"]

        qt_ref_traj=data[:,1:].reshape(n_obs_site, params["sites"], n_t)



    if not params["restart"]:

        t, qt_ref_traj, _, _=traj_function(sim_params)
        write_ref_traj(t, qt_ref_traj, f"{params["folder"]}/ref_traj.txt")

    
    return t, qt_ref_traj



def set_sim_params(gamma_rel, gamma_deph,params):

    sim_params = SimulationParameters(params["sites"], gamma_rel, gamma_deph)
    sim_params.T = params["T"]
    sim_params.N = params["ntraj_0"]
    sim_params.order = params["order"]
    sim_params.threshold = params["threshold"]
    sim_params.req_cpus = params["alloc_cpus"] - 1  ### I remove 1 cpu for the thread that logs the memory
    sim_params.max_bond_dim = params["max_bond_dim"]
    sim_params.set_solver("tdvp"+str(params["order"]),params["solver"])

    sim_params.set_gammas(gamma_rel, gamma_deph)


    return sim_params



def set_loss_function(params, sim_params, qt_ref_traj, traj_function):
    sites=params["sites"]

    ## Defining the loss function and initial parameters
    sim_params.N = params["ntraj"]

    if params["dim"] == "2":

        loss_function=loss_class_2d(sim_params, qt_ref_traj, traj_function, print_to_file=True)


        if params["g_rel_0"]=="random":
            gamma_rel_0=np.random.rand()
        if params["g_rel_0"]!="random":
            gamma_rel_0=float(params["g_rel_0"])

        if params["g_deph_0"]=="random":
            gamma_deph_0=np.random.rand()
        if params["g_deph_0"]!="random":
            gamma_deph_0=float(params["g_deph_0"])

        x0 = [gamma_rel_0, gamma_deph_0]

    if params["dim"] == "2L":

        loss_function=loss_class_nd(sim_params, qt_ref_traj, traj_function, print_to_file=True)


        if params["g_rel_0"]=="random":
            gamma_rel_0=list(np.random.rand(sites))
        if params["g_rel_0"]!="random":
            gamma_rel_0=[float(params["g_rel_0"])]*sites

        if params["g_deph_0"]=="random":
            gamma_deph_0=list(np.random.rand(sites))
        if params["g_deph_0"]!="random":
            gamma_deph_0=[float(params["g_deph_0"])]*sites


        x0 = gamma_rel_0 + gamma_deph_0

    loss_function.set_file_name(f"{params["folder"]}/loss_x_history", reset=not params["restart"])



    return loss_function, x0





#%%

default_params = {
    "folder": ".",
    "ntraj": 10,
    "ntraj_0": 10,
    "sites": 3,
    "restart": False,
    "order": 1,
    "threshold": 1e-4,
    "dim": "2",
    "method": "tjm",
    "solver": "exact",
    "alloc_cpus": 1,
    "max_bond_dim": 8,
    "g_rel" : "random",
    "g_deph" : "random",
    "g_rel_0" : "random",
    "g_deph_0" : "random",
    "T": 5
}



#%%
## Parsing and printing parameters

params=parse_cli_kwargs(default_params)
params["dim"] = str(params["dim"])
print(params)



#%%
# Start memory logging in a background thread
# stop_event = threading.Event()
# pid = os.getpid()

# log_file = params["folder"]+"/self_memory_log.csv"

# logger_thread = threading.Thread(target=log_memory, args=(pid, log_file, 30,stop_event), daemon=True)
# logger_thread.start()




#%%

print("Setting gammas")

gamma_rel,gamma_deph = set_gammas(params)

print_gammas(gamma_rel, gamma_deph,params, sim_params=None)

#%%



if params["method"] == "tjm":
    traj_function = tjm_traj

if params["method"] == "scikit_tt":
    traj_function = scikit_tt_traj

if params["method"] == "qutip":
    traj_function = qutip_traj



print("Setting sim_params")


sim_params=set_sim_params(gamma_rel, gamma_deph, params)

print_gammas(gamma_rel, gamma_deph,params, sim_params)



print("Running Ref traj")

t, qt_ref_traj = running_ref_traj(params,sim_params,traj_function)

print_gammas(gamma_rel, gamma_deph,params, sim_params)




print("Setting Loss_function")


loss_function, x0 = set_loss_function(params, sim_params, qt_ref_traj, traj_function)

print_gammas(gamma_rel, gamma_deph,params, sim_params)





## Running the optimization
print("running optimzation !!!")
loss_function.reset()
loss_history, x_history, x_avg_history, t_opt, opt_traj= ADAM_loss_class(loss_function, x0, alpha=0.07, max_iterations=2, threshhold = 1e-3, max_n_convergence = 50, tolerance=1e-8, beta1 = 0.5, beta2 = 0.99, epsilon = 1e-7, restart=params["restart"])#, Ns=10e5)

print_gammas(gamma_rel, gamma_deph,params, sim_params=sim_params)



write_ref_traj(t_opt, opt_traj, f"{params["folder"]}/opt_traj.txt" )



# stop_event.set()

# logger_thread.join()



# # Wait briefly to ensure logger finishes last write
# time.sleep(1)

# #%%


# # %%








# x_avg_file="test/optimization/loss_x_history.txt"
# gammas_file="test/optimization/gammas.txt"

# data = np.genfromtxt(x_avg_file, skip_header=1)
# gammas=np.genfromtxt(gammas_file, skip_header=1)


# nt,cols = data.shape

# d=(cols-2)//2

# L=d//2

# for i in range(d):
#     plt.plot(data[:, 0], data[:, 2 + i], label=f"$\\gamma_{{{i+1}}}$")
#     plt.axhline(gammas[i], color=plt.gca().lines[-1].get_color(), linestyle='--', linewidth=2)


# plt.xlabel("Iterations")
# plt.ylabel(r"$\gamma$")
# plt.legend()
# plt.show()

# #%%
# data.shape




	
