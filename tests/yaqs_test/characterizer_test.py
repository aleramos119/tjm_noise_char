#%%
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from mqt.yaqs.noise_char.propagation import PropagatorWithGradients
from mqt.yaqs.noise_char.characterizer import Characterizer

from mqt.yaqs.noise_char.propagation import flatten_noise_model


from pathlib import Path



from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable


from mqt.yaqs.core.libraries.gate_library import X, Y, Z

from auxiliar.write import *

import sys

if __name__ == '__main__':


#%%
## Defining Hamiltonian and observable list
    L=3

    J=1
    g=0.5


    H_0 = MPO()
    H_0.init_ising(L, J, g)


    # Define the initial state
    init_state = MPS(L, state='zeros')


    obs_list = [Observable(X(), site) for site in range(L)]  + [Observable(Y(), site) for site in range(L)] + [Observable(Z(), site) for site in range(L)]


    #%%
    ## Defining simulation parameters

    T=5

    dt=0.1

    N=int(sys.argv[1])

    max_bond_dim=8

    threshold=1e-6

    order=2

    sim_params = AnalogSimParams(observables=obs_list, elapsed_time=T, dt=dt, num_traj=N, max_bond_dim=max_bond_dim, threshold=threshold, order=order, sample_timesteps=True, show_progress=True)





    #%%
    ## Defining reference noise model and reference trajectory
    gamma_rel = 0.1
    gamma_deph = 0.1
    sites_list = [i for i in range(L)]
    ref_noise_model =  NoiseModel([{"name": "lowering", "sites": [i], "strength": gamma_rel} for i in range(L)] + [{"name": "pauli_z", "sites": [i], "strength": gamma_deph} for i in range(L)])

    #%%
    from mqt.yaqs import simulator
    simulator.run(init_state, H_0, sim_params, ref_noise_model, parallel=True)


    #%%
    for i in range(len(obs_list)):
        plt.plot( sim_params.times, sim_params.observables[i].results, label=str(i))

    plt.legend()
    plt.show()
#%%



# from typing import Any
# from unittest.mock import patch

# import numpy as np

# from mqt.yaqs import simulator
# from mqt.yaqs.analog.analog_tjm import analog_tjm_1, analog_tjm_2, initialize, step_through
# from mqt.yaqs.core.data_structures.networks import MPO, MPS
# from mqt.yaqs.core.data_structures.noise_model import NoiseModel
# from mqt.yaqs.core.data_structures.simulation_parameters import AnalogSimParams, Observable
# from mqt.yaqs.core.libraries.gate_library import X, Z
# import matplotlib.pyplot as plt

# import sys

# if __name__ == '__main__':

#     ntraj = int(sys.argv[1])

#     L = 3
#     H = MPO()
#     H.init_ising(L, 1.0, 1.0)
#     state = MPS(L, state="zeros")
#     sim_params = AnalogSimParams(
#         observables=[Observable(Z(), site) for site in range(L)],
#         elapsed_time=1,
#         dt=0.05,
#         num_traj=ntraj,
#         max_bond_dim=8,
#         order=2,
#         sample_timesteps=True,
#         show_progress=False,
#     )


#     print("Ntraj = ", ntraj)

#     gamma_rel = 0.1
#     gamma_deph = 0.1
#     sites_list = [i for i in range(L)]
#     noise =  NoiseModel([{"name": "lowering", "sites": [i], "strength": gamma_rel} for i in range(L)] + [{"name": "pauli_z", "sites": [i], "strength": gamma_deph} for i in range(L)])

#     # Run simulation
#     print("Running simulation . ...")
#     simulator.run(state, H, sim_params, noise, parallel=True)


#     print("Plotting . ...")


#     for i in range(L):
#         plt.plot(sim_params.observables[i].results, label=f"obs {i}")
#     plt.legend()
#     plt.show()

# %%
