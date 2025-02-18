
#%%
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt

from yaqs.core.data_structures.networks import MPO, MPS
from yaqs.core.data_structures.noise_model import NoiseModel
from yaqs.core.data_structures.simulation_parameters import Observable, PhysicsSimParams

from yaqs import Simulator
from dataclasses import dataclass

import time




@dataclass
class SimulationParameters:
    T: float = 1
    dt: float = 0.1
    L: int = 4
    J: float = 1
    g: float = 0.5
    gamma_rel: float = 0.1
    gamma_deph: float = 0.1


def qutip_traj(sim_params_class: SimulationParameters):

    T = sim_params_class.T
    dt = sim_params_class.dt
    L = sim_params_class.L
    J = sim_params_class.J
    g = sim_params_class.g
    gamma_rel = sim_params_class.gamma_rel
    gamma_deph = sim_params_class.gamma_deph


    t = np.arange(0, T + dt, dt) 

    '''QUTIP Initialization + Simulation'''

    # Define Pauli matrices
    sx = qt.sigmax()
    sy = qt.sigmay()
    sz = qt.sigmaz()

    # Construct the Ising Hamiltonian
    H = 0
    for i in range(L-1):
        H += -J * qt.tensor([sz if n==i or n==i+1 else qt.qeye(2) for n in range(L)])
    for i in range(L):
        H += -g * qt.tensor([sx if n==i else qt.qeye(2) for n in range(L)])



    # Construct collapse operators
    c_ops = []
    gammas = []

    # Relaxation operators
    for i in range(L):
        c_ops.append(np.sqrt(gamma_rel) * qt.tensor([qt.destroy(2) if n==i else qt.qeye(2) for n in range(L)]))
        gammas.append(gamma_rel)

    # Dephasing operators
    for i in range(L):
        c_ops.append(np.sqrt(gamma_deph) * qt.tensor([sz if n==i else qt.qeye(2) for n in range(L)]))
        gammas.append(gamma_deph)

    # Initial state
    psi0 = qt.tensor([qt.basis(2, 0) for _ in range(L)])

    # Define measurement operators
    sx_list = [qt.tensor([sx if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]
    sy_list = [qt.tensor([sy if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]
    sz_list = [qt.tensor([sz if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]

    obs_list = sx_list  + sy_list + sz_list


    # Create new set of observables by multiplying every operator in obs_list with every operator in c_ops
    A_kn_list= []
    for i,c_op in enumerate(c_ops):
        for obs in obs_list:
            A_kn_list.append(  (1/gammas[i]) * (c_op.dag()*obs*c_op  -  0.5*obs*c_op.dag()*c_op  -  0.5*c_op.dag()*c_op*obs)   )



    new_obs_list = obs_list + A_kn_list




    n_obs= len(obs_list)
    n_jump= len(c_ops)


    # Exact Lindblad solution
    result_lindblad = qt.mesolve(H, psi0, t, c_ops, new_obs_list, progress_bar=True)

    exp_vals = []
    for i in range(len(new_obs_list)):
        exp_vals.append(result_lindblad.expect[i])
    

    # Separate original and new expectation values
    original_exp_vals = exp_vals[:n_obs]
    new_exp_vals = exp_vals[n_obs:]

    # Reshape new_exp_vals to be a list of lists with dimensions n_jump times n_obs
    A_kn_exp_vals = [new_exp_vals[i * n_obs:(i + 1) * n_obs] for i in range(n_jump)]

    return t, original_exp_vals, A_kn_exp_vals
    



def tjm(sim_params_class: SimulationParameters, N=1000):

    T = sim_params_class.T
    dt = sim_params_class.dt
    L = sim_params_class.L
    J = sim_params_class.J
    g = sim_params_class.g
    gamma_rel = sim_params_class.gamma_rel
    gamma_deph = sim_params_class.gamma_deph


    t = np.arange(0, T + dt, dt) 


    # Define the system Hamiltonian
    d = 2
    H_0 = MPO()
    H_0.init_Ising(L, d, J, g)
    # Define the initial state
    state = MPS(L, state='zeros')

    # Define the noise model
    # gamma_relaxation = noise_params[0]
    # gamma_dephasing = noise_params[1]
    noise_model = NoiseModel(['relaxation', 'dephasing'], [gamma_rel, gamma_deph])

    sample_timesteps = True
    # N = 10
    threshold = 1e-6
    max_bond_dim = 4
    order = 2
    measurements = [Observable('x', site) for site in range(L)]  + [Observable('y', site) for site in range(L)] + [Observable('z', site) for site in range(L)]

    sim_params = PhysicsSimParams(measurements, T, dt, sample_timesteps, N, max_bond_dim, threshold, order)
    Simulator.run(state, H_0, sim_params, noise_model)

    tjm_exp_vals = []
    for observable in sim_params.observables:
        tjm_exp_vals.append(observable.results)
        # print(f"Observable at site {observable.site}: {observable.results}")
    # print(tjm_exp_vals)


    return t, tjm_exp_vals

#%%
sim_params = SimulationParameters()

#%%

def loss_function(sim_params, qt_exp_vals):
    """
    Calculates the squared distance between corresponding entries of QuTiP and TJM expectation values.

    Args:
        noise_params (list): Noise parameters for the TJM simulation.
        qt_exp_vals (list of arrays): QuTiP expectation values for each site.

    Returns:
        float: The total squared loss.
    """
    
    # Run the TJM simulation with the given noise parameters

    start_time = time.time()
    tjm_exp_vals, _ = tjm(sim_params)  
    end_time = time.time()
    tjm_time = end_time - start_time
    print(f"TJM time -> {tjm_time:.4f}")
    
    # Initialize loss
    loss = 0.0
    
    # Ensure both lists have the same structure
    if len(qt_exp_vals) != len(tjm_exp_vals):
        raise ValueError("Mismatch in the number of sites between qt_exp_vals and tjm_exp_vals.")

    # Compute squared distance for each site
    for qt_vals, tjm_vals in zip(qt_exp_vals, tjm_exp_vals):
        loss += np.sum((np.array(qt_vals) - np.array(tjm_vals)) ** 2)
    
    return loss, tjm_exp_vals





#%%

sim_params = SimulationParameters()


#%%

## Run both simulations with the same set of parameters
t, qt_exp_vals=qutip_traj(sim_params)

t_traj, tjm_exp_vals=tjm(sim_params)

#%%


# Initialize loss
loss = 0.0

# Ensure both lists have the same structure
if len(qt_exp_vals) != len(tjm_exp_vals):
    raise ValueError("Mismatch in the number of sites between qt_exp_vals and tjm_exp_vals.")

# Compute squared distance for each site
i=0
for qt_vals, tjm_vals in zip(qt_exp_vals, tjm_exp_vals):
    print(i,np.array(qt_vals).shape,np.array(qt_vals).shape,((np.array(qt_vals) - np.array(tjm_vals)) ** 2).shape)
    loss += np.sum((np.array(qt_vals) - np.array(tjm_vals)) ** 2)
    i+=1


#%%
np.array(zip(qt_exp_vals, tjm_exp_vals)).shape

#%%
    # L = 5
    # T = 5
    # dt = 0.1
    # J = 1
    # g = 0.5
    # gamma = 0.1



    # sample_timesteps = True
    # N = 1000
    # threshold = 1e-6
    # max_bond_dim = 4
    # order = 2
    # measurements = [Observable('x', site) for site in range(L)] # + [Observable('y', site) for site in range(L)] + [Observable('z', site) for site in range(L)]

    # sim_params = PhysicsSimParams(measurements, T, dt, sample_timesteps, N, max_bond_dim, threshold, order)




    # t = np.arange(0, sim_params.T + sim_params.dt, sim_params.dt)

    # # Define Pauli matrices
    # sx = qt.sigmax()
    # sy = qt.sigmay()
    # sz = qt.sigmaz()

    # # Construct the Ising Hamiltonian
    # H = 0
    # for i in range(L-1):
    #     H += -J * qt.tensor([sz if n==i or n==i+1 else qt.qeye(2) for n in range(L)])
    # for i in range(L):
    #     H += -g * qt.tensor([sx if n==i else qt.qeye(2) for n in range(L)])

    # # Construct collapse operators
    # c_ops = []

    # # Relaxation operators
    # for i in range(L):
    #     c_ops.append(np.sqrt(gamma) * qt.tensor([qt.destroy(2) if n==i else qt.qeye(2) for n in range(L)]))

    # # Dephasing operators
    # for i in range(L):
    #     c_ops.append(np.sqrt(gamma) * qt.tensor([sz if n==i else qt.qeye(2) for n in range(L)]))

    # # Initial state
    # psi0 = qt.tensor([qt.basis(2, 0) for _ in range(L)])

    # # Define measurement operators
    # sx_list = [qt.tensor([sx if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]

    # # Exact Lindblad solution
    # result_lindblad = qt.mesolve(H, psi0, t, c_ops, sx_list, progress_bar=True)
    # qutip_results = []
    # for site in range(len(sx_list)):
    #     qutip_results.append(result_lindblad.expect[site])

    # H_0 = MPO()
    # H_0.init_Ising(L, 2, J, g)

    # # Define the initial state
    # state = MPS(L, state='zeros')

    # # Define the noise model
    # noise_model = NoiseModel(['relaxation', 'dephasing'], [gamma, gamma])

    # Simulator.run(state, H_0, sim_params, noise_model)

    # tjm_results = []
    # for observable in sim_params.observables:
    #     tjm_results.append(observable.results)




    




    # Plot results
plt.figure(figsize=(10,8))
for i in range(len(tjm_results)):
    plt.plot(t, qutip_results[i], label=f'exp val qutip obs {i}')
    plt.plot(t, tjm_results[i], label=f'exp val tjm obs {i}')
    plt.plot(t, qutip_results[i]-tjm_results[i], label = f'observable {i}')
plt.xlabel('times')
plt.ylabel('expectation value')
plt.legend()
plt.show()
# %%
