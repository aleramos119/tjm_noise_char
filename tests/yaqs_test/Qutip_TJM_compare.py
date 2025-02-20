
#%%
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from yaqs.core.data_structures.networks import MPO, MPS
from yaqs.core.data_structures.noise_model import NoiseModel
from yaqs.core.data_structures.simulation_parameters import Observable, PhysicsSimParams

from yaqs import Simulator


import time

#%%
import importlib
import yaqs

from yaqs.noise_char.optimization import *
from yaqs.noise_char.propagation import *

importlib.reload(yaqs.noise_char.optimization)
importlib.reload(yaqs.noise_char.propagation)


#%%

# Generate reference trajectory
sim_params = SimulationParameters()

t, qt_ref_traj, A_kn_exp_vals=qutip_traj(sim_params)


#%%

# Perform gradient descent

initial_params = SimulationParameters()
initial_params.gamma_rel = 0.3
initial_params.gamma_deph = 0.4


loss_history, gr_history, gd_history, dJ_dgr_history, dJ_dgd_history = BFGS_optimization(initial_params, qt_ref_traj, qutip_traj, learning_rate=0.2, max_iterations=200,tolerance=1e-8)



#%%
plt.plot(np.log(loss_history), label='log(J)')
plt.legend()

#%%
plt.plot(gr_history,label='gamma_relaxation')
plt.plot(gd_history, label='gamma_dephasing')
plt.axhline(y=0.1, color='r', linestyle='--', label='gamma_reference')
plt.legend()


# %%

# %%
