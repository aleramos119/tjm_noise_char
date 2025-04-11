
#%%
import numpy as np
from mqt.yaqs.noise_char.optimization import loss_function
from mqt.yaqs.noise_char.propagation import SimulationParameters, qutip_traj, tjm_traj
import sys
import os

import numpy as np
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# Initialize simulation parameters
# Get N from the first argument in the command line



sim_params = SimulationParameters()
sim_params.N = int(sys.argv[1])

num_samples = int(sys.argv[2])


# sim_params.N=2
# num_samples=2


gamma_list=[0.01, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]


folder=f"noise_char/N_{str(sim_params.N)}/samples_{str(num_samples)}/"


t, qt_ref_traj, d_On_d_gk = qutip_traj(sim_params)



#%%

# Create the output folder if it doesn't exist

def write_line_to_file(file_path, line):
    """
    Writes a list of values as a single line to a file. If the file does not exist, it creates it.

    Parameters:
    file_path (str): The path to the file.
    line (list): The list of values to write to the file.
    """
    with open(file_path, "a") as file:
        file.write("    ".join(map(str, line)) + "\n")




# Top-level helper to avoid lambda
def compute_wrapper(args, sim_params, qt_ref_traj, qutip_traj,num_samples):
    gamma_rel, gamma_deph = args
    return compute_for_params(gamma_rel, gamma_deph, sim_params, qt_ref_traj, qutip_traj,num_samples)



# Wrap the computation in a function
def compute_for_params(gamma_rel, gamma_deph, sim_params_base, qt_ref_traj, qutip_traj,num_samples):
    from copy import deepcopy
    sim_params = deepcopy(sim_params_base)
    sim_params.gamma_rel = gamma_rel
    sim_params.gamma_deph = gamma_deph

    losses = []
    gradients = []

    for _ in range(num_samples):
        loss, _, grad = loss_function(sim_params, qt_ref_traj, qutip_traj)
        losses.append(loss)
        gradients.append(grad)

    losses = np.array(losses)
    gradients = np.array(gradients)

    return (gamma_rel, gamma_deph, losses, gradients)

# Run in parallel
def run_parallel(sim_params, qt_ref_traj, qutip_traj, gamma_list, num_samples, folder):

    tasks = [(gamma_rel, gamma_deph) for gamma_rel in gamma_list for gamma_deph in gamma_list]


    wrapped_func = partial(compute_wrapper, sim_params=sim_params, qt_ref_traj=qt_ref_traj, qutip_traj=qutip_traj, num_samples=num_samples)


    with ProcessPoolExecutor() as executor:
        results = executor.map( wrapped_func, tasks)


    
    if not os.path.exists(folder):
        os.makedirs(folder)

    with open(folder+"loss.txt", "w") as f:
        pass

    with open(folder+"gamma_rel.txt", "w") as f:
        pass

    with open(folder+"gamma_deph.txt", "w") as f:
        pass


    for gamma_rel, gamma_deph, losses, gradients in results:
        write_line_to_file(folder+"loss.txt", [gamma_rel, gamma_deph]+list(losses))
        write_line_to_file(folder+"gamma_rel.txt", [gamma_rel, gamma_deph]+list(gradients[:,0]))
        write_line_to_file(folder+"gamma_deph.txt", [gamma_rel, gamma_deph]+list(gradients[:,1]))


#%%

run_parallel(sim_params, qt_ref_traj, tjm_traj, gamma_list, num_samples, folder)




# %%
