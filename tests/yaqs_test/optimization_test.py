
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

# loss_history, gr_history, gd_history, dJ_dgr_history, dJ_dgd_history = BFGS(initial_params, qt_ref_traj, qutip_traj, learning_rate=0.2, max_iterations=50,tolerance=1e-40,loss_std=6e-5, dJ_d_gr_std=0.0065, dJ_d_gd_std=0.025)#, Ns=10e5)


# loss_history, gr_history, gd_history, dJ_dgr_history, dJ_dgd_history = Secant_Penalized_BFGS(initial_params, qt_ref_traj, qutip_traj, learning_rate=0.2, max_iterations=200,tolerance=1e-40, alpha=20, Ns=1e2, N0=1e-10, loss_std=6e-5, dJ_d_gr_std=0.0065, dJ_d_gd_std=0.025)#, Ns=10e5)


#loss_history, gr_history, gd_history, dJ_dgr_history, dJ_dgd_history = ADAM_gradient_descent(initial_params, qt_ref_traj, qutip_traj, learning_rate=0.2, max_iterations=200,tolerance=1e-40, beta1=0.5, alpha=0.9, loss_std=6e-5, dJ_d_gr_std=0.0065, dJ_d_gd_std=0.025)#, Ns=10e5)


loss_history, gr_history, gd_history = bayesian_optimization(initial_params, qt_ref_traj, qutip_traj,[(0.01,0.2),(0.01,0.2)], n_init=10, max_iterations=50, tolerance=1e-8, beta=1, num_restarts=20, raw_samples=20, file_name=" ", device="cpu", loss_std=6e-5, dJ_d_gr_std=0.0065, dJ_d_gd_std=0.025)
# %%

%matplotlib qt
plt.plot(np.log10(loss_history), label='log(J)')
plt.legend()

#%%
plt.plot(gr_history,label='gamma_relaxation')
plt.plot(gd_history, label='gamma_dephasing')
plt.axhline(y=0.1, color='r', linestyle='--', label='gamma_reference')
plt.legend()

# %%

# %%

# %%
