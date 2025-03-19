
#%%
import numpy as np
import scikit_tt.tensor_train as tt
from scikit_tt.tensor_train import TT
import scikit_tt.solvers.ode as ode
import scikit_tt
import qutip as qt
import matplotlib.pyplot as plt
import random






#%%
# Parameters
N = L = 4  # number of sites
J = 1.0  # Ising coupling strength
h = g = 0.5  # transverse field strength
operator_site = 1
gamma_dephasing = 0.1 # 1 / 1.0  # dephasing rate (1/T2star)
gamma_relaxation = 0.1 #1 / 1.0  # relaxation rate (1/T1)# Define Pauli matrices
sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()# Construct the Ising Hamiltonian
H = 0
for i in range(N-1):
    H += -J * qt.tensor([sz if n==i or n==i+1 else qt.qeye(2) for n in range(N)])
for i in range(N):
    H += -h * qt.tensor([sx if n==i else qt.qeye(2) for n in range(N)])# Construct collapse operators
c_ops = []# Dephasing operators
for i in range(N):
    c_ops.append(np.sqrt(gamma_dephasing) * qt.tensor([sz if n==i else qt.qeye(2) for n in range(N)]))# Relaxation operators
for i in range(N):
    c_ops.append(np.sqrt(gamma_relaxation) * qt.tensor([qt.destroy(2) if n==i else qt.qeye(2) for n in range(N)]))# Initial state (all spins up)
psi0 = qt.tensor([qt.basis(2, 0) for _ in range(N)])# # Time vector
#%%


#%%
T = 1
timesteps = 20
t = np.linspace(0, T, timesteps+1)
dt=t[1]-t[0]
print(t)
# Define Z measurement operator for the fifth qubit

obs = qt.tensor([sy if n==operator_site else qt.qeye(2) for n in range(N)])#Exact Lindblad solution
result_lindblad = qt.mesolve(H, psi0, t, c_ops, [obs], progress_bar=True)
print(result_lindblad.expect[0][-1])


#%%
'''start of scikit initialization'''# chain length

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]])

paulis = [X, Y, Z]


I = np.eye(2)
g = 0.5
J = 1
cores = [None] * L
cores[0] = tt.build_core([[-g * X, - J * Z, I]])
for i in range(1, L - 1):
    cores[i] = tt.build_core([[I, 0, 0], [Z, 0, 0], [-g * X, - J * Z, I]])
cores[-1] = tt.build_core([I, Z, -g*X])
hamiltonian = TT(cores)# jump operators and parameters
L_1 = np.array([[0, 1], [0, 0]])
L_2 = np.array([[1, 0], [0, -1]])
jump_operator_list = [[L_1, L_2] for _ in range(L)]
jump_parameter_list = [[np.sqrt(0.1), np.sqrt(0.1)] for _ in range(L)]


#%%



#%%
# initial state
rank = 6

num_trajectories = 100
exp_vals = np.zeros(timesteps+1,dtype=complex)

for k in range(num_trajectories):
    initial_state = tt.unit([2] * L, [0] * L)
    for i in range(rank - 1):
        initial_state += tt.unit([2] * L, [0] * L)
    initial_state = initial_state.ortho()
    initial_state = (1 / initial_state.norm()) * initial_state
    #print(initial_state.cores)
    # time step, number of steps, operator_site and max rank
    

    max_rank = rank



    # observable
    observable = tt.eye(dims=[2]*L)
    observable.cores[operator_site] = np.zeros([1,2,2,1], dtype=complex)
    observable.cores[operator_site][0,:,:,0]=Y

    

    exp_vals[0] += initial_state.transpose(conjugate=True)@observable@initial_state
    for i in range(timesteps):
        initial_state = ode.tjm(hamiltonian, jump_operator_list, jump_parameter_list, initial_state, dt, 1)[-1]
        #initial_state = ode.tdvp(hamiltonian, initial_state, time_step, 1)[-1]
        exp_vals[i+1] += initial_state.transpose(conjugate=True)@observable@initial_state

exp_vals = (1/num_trajectories)*exp_vals

#%%
%matplotlib qt
plt.plot(t, exp_vals,'o-', label='scikit_tt')
plt.plot(t, result_lindblad.expect[0], label='qutip')
plt.legend()
plt.show()


##%




# %%

import numpy as np
import scikit_tt.tensor_train as tt
from scikit_tt.tensor_train import TT

N = [2,3,4,5]


#%%

M_mat = np.random.rand(np.prod(N),np.prod(N))
print('M as matrix:', M_mat.shape)
print('-----')
M = M_mat.reshape(N+N)
print('M as tensor:', M.shape)

#%%

M_tt1 = TT(M)
print('-----')
print('M in TT format with full ranks:')
print(M_tt1)


#%%
M_tt2 = TT(M, max_rank=30)
print('-----')
print('M in TT format with reduced ranks:')
print(M_tt2)

#%%
print('-----')
print('errors:')
print(np.linalg.norm(M-M_tt1.full()))
print(np.linalg.norm(M-M_tt2.full()))
print((M_tt1-M_tt2).norm())
print('-----')

#%%
error_list = []
for i in range(1,50):
    M_tt2 = TT(M, max_rank=i)
    error_list.append([i,np.linalg.norm(M-M_tt2.full())])
# %%
plt.plot(np.array(error_list)[:,0],np.array(error_list)[:,1])
# %%



qnot=np.array([[0,1],[1,0]])
hadamard=1/np.sqrt(2)*np.array([[1,1],[1,-1]])
zero=np.array([1,0])
one=np.array([0,1])
superposition=1/np.sqrt(2)*np.array([1,1])

hadamard@superposition

# %%
t = np.linspace(0, 1, 20+1)
# %%
t
# %%
len(t)
# %%
