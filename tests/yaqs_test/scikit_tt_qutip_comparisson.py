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

#%%

class SimulationParameters:
    T: float = 5
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
    jump_parameter_list = [[np.sqrt(gamma_rel[i])] for i in range(L)]


    obs_list=[]

    for pauli in O_list:
       for j in range(L):
           obs= tt.eye(dims=[2]*L)
           obs.cores[j]=np.zeros([1,2,2,1], dtype=complex)
           obs.cores[j][0,:,:,0]=pauli
           obs_list.append(obs)

    


    n_obs_total = len(obs_list)


    exp_vals = np.zeros([n_obs_total,timesteps+1])



    for k in range(N):
        initial_state = tt.unit([2] * L, [0] * L)
        for i in range(rank - 1):
            initial_state += tt.unit([2] * L, [0] * L)
        initial_state = initial_state.ortho()
        initial_state = (1 / initial_state.norm()) * initial_state

        
        for j in range(n_obs_total):
           exp_vals[j,0] += initial_state.transpose(conjugate=True)@obs_list[j]@initial_state
        
        trajectory = ode.tjm(hamiltonian, jump_operator_list, jump_parameter_list, initial_state, dt, timesteps, solver=scikit_tt_solver)

        for i in range(timesteps):
            for j in range(n_obs_total):                
                exp_vals[j,i+1] += trajectory[i].transpose(conjugate=True)@obs_list[j]@trajectory[i]


        # for i in range(timesteps):
        #     initial_state = ode.tjm(hamiltonian, jump_operator_list, jump_parameter_list, initial_state, dt, 1, solver=scikit_tt_solver)[-1]

        #     for j in range(n_obs_total):                
        #         exp_vals[j,i+1] += initial_state.transpose(conjugate=True)@obs_list[j]@initial_state


    exp_vals = (1/N)*exp_vals
    

    return t, np.array(exp_vals).T



    

#%%






if __name__=="__main__":

    L=2
    g_rel=0.1
    g_deph=0.
    ntraj=400
    threshold=1e-6

    local_solver="krylov_5"

    sim_params = SimulationParameters(L,g_rel,g_deph)
    sim_params.N = ntraj
    sim_params.T = 5
    sim_params.threshold = threshold


    sim_params.set_solver("tdvp1",local_solver)


    scikit_time, scikit_ref_traj=scikit_tt_traj(sim_params)


    qutip_time, qutip_ref_traj = qutip_traj(sim_params)




#%%
%matplotlib qt

col=0
plt.plot(scikit_time, scikit_ref_traj[:,col], label="SciKit-TT")
plt.plot(qutip_time, qutip_ref_traj[:,col], label="QuTiP")
plt.legend()
plt.show()


# %%
