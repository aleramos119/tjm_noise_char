
#%%
import matplotlib.pyplot as plt
import numpy as np


from mqt.yaqs.noise_char.optimization import *
from mqt.yaqs.noise_char.propagation import *


#%%
sim_params = SimulationParameters()

t, qt_ref_traj, d_On_d_gk=qutip_traj(sim_params)



#%%
initial_params = SimulationParameters()
initial_params.gamma_rel = 0.15
initial_params.gamma_deph = 0.2
initial_params.N = 200

loss_history, gr_history, gd_history, dJ_dgr_history, dJ_dgd_history = BFGS(initial_params, qt_ref_traj, qutip_traj, learning_rate=0.2, max_iterations=100,tolerance=1e-8, file_name='test.txt')


# %%


plt.plot(np.log10(loss_history), label='log(J)')
plt.legend()

#%%
plt.plot(gr_history,label='gamma_relaxation')
plt.plot(gd_history, label='gamma_dephasing')
plt.axhline(y=0.1, color='r', linestyle='--', label='gamma_reference')
plt.legend()

# %%
