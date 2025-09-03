#%%
import matplotlib.pyplot as plt
import numpy as np

from mqt.yaqs.noise_char.propagation import PropagatorWithGradients
from mqt.yaqs.noise_char.characterizer import Characterizer



from pathlib import Path



from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable


from mqt.yaqs.core.libraries.gate_library import X, Y, Z

from auxiliar.write import *

import sys




#%%
## Defining Hamiltonian and observable list
L=3

J=1
g=1


H_0 = MPO()
H_0.init_ising(L, J, g)


# Define the initial state
init_state = MPS(L, state='zeros')


obs_list = [Observable(X(), site) for site in range(L)]  + [Observable(Y(), site) for site in range(L)] + [Observable(Z(), site) for site in range(L)]

print("Reftraj results:", obs_list[0].results)




#%%
## Defining simulation parameters

T=5

dt=0.1

N=1000

max_bond_dim=8

threshold=1e-6

order=2

sim_params = AnalogSimParams(observables=obs_list, elapsed_time=T, dt=dt, num_traj=N, max_bond_dim=max_bond_dim, threshold=threshold, order=order, sample_timesteps=True)





#%%
## Defining reference noise model and reference trajectory
gamma_rel = 0.1
gamma_deph = 0.1
sites_list = [i for i in range(L)]
ref_noise_model =  NoiseModel([{"name": "lowering", "sites": sites_list, "strength": gamma_rel},{"name": "pauli_z", "sites": sites_list, "strength": gamma_deph}])



propagator = PropagatorWithGradients(
    sim_params=sim_params,
    hamiltonian=H_0,
    noise_model=ref_noise_model,
    init_state=init_state
)

propagator.set_observable_list(obs_list)

propagator.run(ref_noise_model)

ref_traj = propagator.obs_traj



#%%
for i in range(len(ref_traj)):
    plt.plot( ref_traj[i].results, label=str(i))

#%%
## Defining reference noise model and reference trajectory
gamma_rel = 0.1
gamma_deph = 0.2
sites_list = [i for i in range(L)]
ref_noise_model =  NoiseModel([{"name": "lowering", "sites": sites_list, "strength": 0.1},{"name": "pauli_z", "sites": sites_list, "strength": 0.1}])



propagator = PropagatorWithGradients(
    sim_params=sim_params,
    hamiltonian=H_0,
    noise_model=ref_noise_model,
    init_state=init_state
)

propagator.set_observable_list(obs_list)

propagator.run(ref_noise_model)

ref_traj = propagator.obs_traj

print("Reftraj results:", ref_traj[0].results[3])


#%% Optimizing the model
guess_noise_model =  NoiseModel([{"name": "lowering", "sites": sites_list, "strength": 0.4},{"name": "pauli_z", "sites": sites_list, "strength": 0.5}])

characterizer = Characterizer(
    sim_params=sim_params,
    hamiltonian=H_0,
    init_guess=guess_noise_model,
    init_state=init_state,
    ref_traj=ref_traj,
    work_dir="test/characterizer"
)


characterizer.adam_optimize()


#%%
print(characterizer.optimal_model.processes)


#%%


# %%
