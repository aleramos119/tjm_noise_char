
#%%
import numpy as np
from mqt.yaqs.noise_char.optimization import loss_function
from mqt.yaqs.noise_char.propagation import SimulationParameters, qutip_traj, tjm_traj

# Initialize simulation parameters
sim_params = SimulationParameters()
sim_params.gamma_rel = 0.1
sim_params.gamma_deph = 0.1
sim_params.N = 100

#%%
# Generate reference trajectory
t, qt_ref_traj, d_On_d_gk = qutip_traj(sim_params)

# Number of samples for analysis
num_samples = 5


#%%
# Arrays to store loss and gradient values
losses = []
gradients = []

# Compute loss and gradients multiple times
for _ in range(num_samples):
    loss, _, grad = loss_function(sim_params, qt_ref_traj, tjm_traj)
    losses.append(loss)
    gradients.append(grad)

# Convert gradients to a numpy array for easier analysis
gradients = np.array(gradients)

# Analyze standard deviation
loss_std = np.std(losses)
grad_std = np.std(gradients, axis=0)

# Print results
print(f"Standard deviation of loss: {loss_std}")
print(f"Standard deviation of gradients: {grad_std}")
# %%
