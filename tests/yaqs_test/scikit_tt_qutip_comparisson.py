#%%
import numpy as np
from mqt.yaqs.noise_char.propagation import *
import matplotlib.pyplot as plt
import re

import qutip as qt

import scikit_tt.tensor_train as tt
from scikit_tt.tensor_train import TT
import scikit_tt.solvers.ode as ode
import scikit_tt

from mqt.yaqs.core.libraries.gate_library import X, Y, Z, Destroy, Create

from mqt.yaqs.core.data_structures.networks import MPO, MPS
from mqt.yaqs.core.data_structures.noise_model import NoiseModel
from mqt.yaqs.core.data_structures.simulation_parameters import Observable,AnalogSimParams

#%%

class SimulationParameters:
    T: float = 1
    dt: float = 0.1
    J: float = 1
    g: float = 0.5

    threshold: float = 1e-6
    max_bond_dim: int = 8
    order: int = 2

    # For scikit_tt
    N:int = 100
    rank: int= 8

    observables = ['x','y','z']



    scikit_tt_solver: dict = {"solver": 'tdvp1', "method": 'krylov', "dimension": 5}


    def __init__(self, L : int, gamma_rel : list | float, gamma_deph : list | float):

        self.L = L

        self.set_gammas(gamma_rel, gamma_deph)

        

    def set_gammas(self, gamma_rel : list | float, gamma_deph : list | float):

        if isinstance(gamma_rel, list) and len(gamma_rel) != self.L:
            raise ValueError("gamma_rel must be a list of length L.")
        if isinstance(gamma_deph, list) and len(gamma_deph) != self.L:
            raise ValueError("gamma_deph must be a list of length L.")

        if isinstance(gamma_rel, float): 
            self.gamma_rel = [gamma_rel] * self.L
        else:
            self.gamma_rel = list(gamma_rel)

        if isinstance(gamma_deph, float):
            self.gamma_deph = [gamma_deph] * self.L
        else:
            self.gamma_deph = list(gamma_deph)

    def set_solver(self, solver: str = 'tdvp1', local_solver: str = 'krylov_5'):
        if solver not in ('tdvp1', 'tdvp2'):
            raise ValueError("solver can be only 'tdvp1' or 'tdvp2'")
        self.scikit_tt_solver["solver"] = solver

        if not re.match(r'^krylov_\d+$', local_solver) and not local_solver == 'exact':
            raise ValueError("local_solver must match the pattern 'krylov_<number>' or be 'exact'")
        
        if local_solver == 'exact':
            self.scikit_tt_solver["method"] = local_solver

        if local_solver.startswith('krylov_'):
            self.scikit_tt_solver["method"] = 'krylov'
            self.scikit_tt_solver["dimension"] = int(local_solver.split('_')[-1])
            if self.scikit_tt_solver["dimension"] < 1:
                raise ValueError("local_solver must be a positive integer when using 'krylov_<number>' format")


    
def qutip_traj(sim_params_class: SimulationParameters):

    T = sim_params_class.T
    dt = sim_params_class.dt
    L = sim_params_class.L
    J = sim_params_class.J
    g = sim_params_class.g
    gamma_rel = sim_params_class.gamma_rel
    gamma_deph = sim_params_class.gamma_deph


    t = np.arange(0, T + dt, dt) 

    n_t = len(t)

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
        c_ops.append(np.sqrt(gamma_rel[i]) * qt.tensor([sx if n==i else qt.qeye(2) for n in range(L)]))
        gammas.append(gamma_rel)

    # # Dephasing operators
    # for i in range(L):
    #     c_ops.append(np.sqrt(gamma_deph[i]) * qt.tensor([sz if n==i else qt.qeye(2) for n in range(L)]))
    #     gammas.append(gamma_deph)

    #c_ops = [rel0, rel1, rel2,... rel(L-1), deph0, deph1,..., deph(L-1)]

    # Initial state
    psi0 = qt.tensor([qt.basis(2, 0) for _ in range(L)])



    # Create obs_list based on the observables in sim_params_class.observables
    obs_list = []


    for obs_type in sim_params_class.observables:
        if obs_type.lower() == 'x':
            # For each site, create the measurement operator for 'x'
            obs_list.extend([qt.tensor([sx if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)])
        elif obs_type.lower() == 'y':
            obs_list.extend([qt.tensor([sy if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)])
        elif obs_type.lower() == 'z':
            obs_list.extend([qt.tensor([sz if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)])




    new_obs_list = obs_list 

    n_obs= len(obs_list)

    # Exact Lindblad solution
    result_lindblad = qt.mesolve(H, psi0, t, c_ops, new_obs_list, progress_bar=True)

    exp_vals = []
    for i in range(len(new_obs_list)):
        exp_vals.append(result_lindblad.expect[i])


    # return t, original_exp_vals, d_On_d_gk, A_kn_exp_vals
    return t, np.array(exp_vals).T




def scikit_tt_traj(sim_params_class: SimulationParameters):


    T = sim_params_class.T
    dt = sim_params_class.dt
    L = sim_params_class.L
    J = sim_params_class.J
    g = sim_params_class.g
    gamma_rel = sim_params_class.gamma_rel
    gamma_deph = sim_params_class.gamma_deph

    rank = sim_params_class.rank
    N = sim_params_class.N

    scikit_tt_solver = sim_params_class.scikit_tt_solver


    t = np.arange(0, T + dt, dt) 
    timesteps=len(t)-1


    # Parameters
    X = np.array([[0, 1], [1, 0]])
    Y = np.array([[0, -1j], [1j, 0]])
    Z = np.array([[1, 0], [0, -1]])
    I = np.eye(2)
    L_1 = np.array([[0, 1], [0, 0]])
    L_2 = np.array([[1, 0], [0, -1]])
    O_list = [X, Y, Z]


    # A_nk = construct_Ank(O_list, L_list)

    
    cores = [None] * L
    cores[0] = tt.build_core([[-g * X, - J * Z, I]])
    for i in range(1, L - 1):
        cores[i] = tt.build_core([[I, 0, 0], [Z, 0, 0], [-g * X, - J * Z, I]])
    cores[-1] = tt.build_core([I, Z, -g*X])
    hamiltonian = TT(cores)# jump operators and parameters

    jump_operator_list = [[X] for _ in range(L)]
    jump_parameter_list = [[gamma_rel[i]] for i in range(L)]


    obs_list=[]

    for pauli in O_list:
       for j in range(L):
           obs= tt.eye(dims=[2]*L)
           obs.cores[j]=np.zeros([1,2,2,1], dtype=complex)
           obs.cores[j][0,:,:,0]=pauli
           obs_list.append(obs)

    


    n_obs_total = len(obs_list)


    exp_vals = np.zeros([n_obs_total,timesteps+1])


    print("Hamiltonian", hamiltonian.cores)
    print("Jump operators", jump_operator_list)
    print("Jump parameters", jump_parameter_list)
    print("Solver", scikit_tt_solver)

    for k in range(N):
        initial_state = tt.unit([2] * L, [0] * L)
        for i in range(rank - 1):
            initial_state += tt.unit([2] * L, [0] * L)
        initial_state = initial_state.ortho()
        initial_state = (1 / initial_state.norm()) * initial_state

        
        # for j in range(n_obs_total):
        #    exp_vals[j,0] += initial_state.transpose(conjugate=True)@obs_list[j]@initial_state
        
        # trajectory = ode.tjm(hamiltonian, jump_operator_list, jump_parameter_list, initial_state, dt, timesteps, solver=scikit_tt_solver)

        # print(len(trajectory), timesteps)

        # for i in range(timesteps):
        #     for j in range(n_obs_total):                
        #         exp_vals[j,i+1] += trajectory[i+1].transpose(conjugate=True)@obs_list[j]@trajectory[i+1]

        



        for i in range(timesteps):
            initial_state = ode.tjm(hamiltonian, jump_operator_list, jump_parameter_list, initial_state, dt, 1, solver=scikit_tt_solver)[-1]

            for j in range(n_obs_total):                
                exp_vals[j,i+1] += initial_state.transpose(conjugate=True)@obs_list[j]@initial_state


    exp_vals = (1/N)*exp_vals
    

    return t, np.array(exp_vals).T







def tjm_traj(sim_params_class: SimulationParameters):
    """
    Simulates the time evolution of an open quantum system using the Lindblad master equation with TJM.
    This function constructs the system Hamiltonian and collapse operators for a spin chain with relaxation and dephasing noise,
    initializes the system state, and computes the expectation values of specified observables and their derivatives with respect
    to the noise parameters over time.
    Parameters
    ----------
    sim_params_class : SimulationParameters
        An instance of SimulationParameters containing simulation parameters:
            - T (float): Total simulation time.
            - dt (float): Time step.
            - L (int): Number of sites (spins) in the chain.
            - J (float): Ising coupling strength.
            - g (float): Transverse field strength.
            - gamma_rel (array-like): Relaxation rates for each site.
            - gamma_deph (array-like): Dephasing rates for each site.
            - observables (list of str): List of observables to measure ('x', 'y', 'z').
    Returns
    -------
    t : numpy.ndarray
        Array of time points at which the system was evolved.
    original_exp_vals : numpy.ndarray
        Expectation values of the specified observables at each site and time, shape (n_obs_site, L, n_t).
    d_On_d_gk : numpy.ndarray
        Derivatives of the observables with respect to the noise parameters, shape (n_jump_site, n_obs_site, L, n_t).
    avg_min_max_traj_time : list
        Placeholder list [None, None, None] for compatibility with other interfaces.
    Notes
    -----
    - The function uses QuTiP for quantum object and solver operations.
    - The system is initialized in the ground state |0>^{⊗L}.
    - The Hamiltonian is an Ising model with a transverse field.
    - Collapse operators are constructed for both relaxation and dephasing noise.
    - The function computes both the expectation values of observables and their derivatives with respect to noise parameters
      using the Lindblad master equation.
    """


    T = sim_params_class.T
    dt = sim_params_class.dt
    L = sim_params_class.L
    J = sim_params_class.J
    g = sim_params_class.g
    gamma_rel = sim_params_class.gamma_rel
    gamma_deph = sim_params_class.gamma_deph
    N=sim_params_class.N


    threshold = sim_params_class.threshold
    max_bond_dim = sim_params_class.max_bond_dim
    order = sim_params_class.order



    t = np.arange(0, T + dt, dt) 
    n_t = len(t)


    # Define the system Hamiltonian
    H_0 = MPO()

    H_0.init_ising(L, J, g)
    # Define the initial state
    state = MPS(L, state='zeros')

    # Define the noise model
    # gamma_relaxation = noise_params[0]
    # gamma_dephasing = noise_params[1]


    noise_model = NoiseModel([{"name": "pauli_x", "sites": [i], "strength": gamma_rel[i]} for i in range(L)])


    sample_timesteps = True

    
    obs_list = [Observable(X(), site) for site in range(L)]  + [Observable(Y(), site) for site in range(L)] + [Observable(Z(), site) for site in range(L)]


    n_obs = len(obs_list)


    

    new_obs_list = obs_list




    sim_params = AnalogSimParams(observables=new_obs_list, elapsed_time=T, dt=dt, num_traj=N, max_bond_dim=max_bond_dim, threshold=threshold, order=order, sample_timesteps=True)
    simulator.run(state, H_0, sim_params, noise_model)

    exp_vals = []
    for observable in sim_params.observables:
        exp_vals.append(observable.results)




    # Separate original and new expectation values from result_lindblad.
    n_obs = len(obs_list)  # number of measurement operators (should be L * n_types)
    original_exp_vals = exp_vals[:n_obs]



    original_exp_vals = np.array(original_exp_vals).reshape(n_obs, n_t)


    return t, original_exp_vals.T







    

#%%






if __name__=="__main__":

    L=2
    g_rel=0.1
    g_deph=0.
    ntraj=1000
    threshold=1e-6

    local_solver="krylov_5"

    sim_params = SimulationParameters(L,g_rel,g_deph)
    sim_params.N = ntraj
    sim_params.T = 1
    sim_params.dt = 0.1
    sim_params.order = 1
    sim_params.max_bond_dim = 8
    sim_params.rank = 8
    sim_params.threshold = threshold


    sim_params.set_solver("tdvp1",local_solver)


    scikit_time, scikit_ref_traj=scikit_tt_traj(sim_params)


    # qutip_time, qutip_ref_traj = qutip_traj(sim_params)


    # yaqs_time, yaqs_ref_traj = tjm_traj(sim_params)


#%%
# %matplotlib qt
col=0
plt.plot(scikit_time, scikit_ref_traj[:,col],"o", label="SciKit-TT")
plt.plot(qutip_time, qutip_ref_traj[:,col],"x", label="QuTiP")
plt.plot(yaqs_time, yaqs_ref_traj[:,col], label="YAQS")

plt.legend()
plt.show()


#%%



# #%%
# %matplotlib qt



# # %%
# yaqs_ref_traj.shape
# # %%
