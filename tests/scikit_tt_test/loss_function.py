
#%%
import numpy as np
import scikit_tt.tensor_train as tt
from scikit_tt.tensor_train import TT
import scikit_tt.solvers.ode as ode
import scikit_tt
import qutip as qt
import matplotlib.pyplot as plt
import random

%matplotlib qt


#%%
# Parameters
N = L = 5  # number of sites
J = 1.0  # Ising coupling strength
h = g = 0.5  # transverse field strength
operator_site = 3
gamma_dephasing = 0.1 # 1 / 1.0  # dephasing rate (1/T2star)
gamma_relaxation = 0.1 #1 / 1.0  # relaxation rate (1/T1)# Define Pauli matrices
sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()# Construct the Ising Hamiltonian

qt_paulis = [sx, sy, sz]
qt_obs=qt.tensor([np.random.choice(qt_paulis) for i in range(L)])


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
T = 2
timesteps = 20
t = np.linspace(0, T, timesteps+1)
print(t)
# Define Z measurement operator for the fifth qubit

sz_fifth = qt.tensor([sx if n==operator_site else qt.qeye(2) for n in range(N)])#Exact Lindblad solution
result_lindblad = qt.mesolve(H, psi0, t, c_ops, [qt_obs], progress_bar=True)


Eh_traj = result_lindblad.expect[0]
    



#%%
'''start of scikit initialization'''# chain length

X = np.array([[0, 1], [1, 0]])
Y = np.array([[0, -1j], [1j, 0]])
Z = np.array([[1, 0], [0, -1]])

I = np.eye(2)
g = 0.5
J = 1
cores = [None] * L
cores[0] = tt.build_core([[-g * X, - J * Z, I]])
for i in range(1, L - 1):
    cores[i] = tt.build_core([[I, 0, 0], [Z, 0, 0], [-g * X, - J * Z, I]])
cores[-1] = tt.build_core([I, Z, -g*X])
hamiltonian = TT(cores)
# jump operators and parameters
L_1 = np.array([[0, 1], [0, 0]])
L_2 = np.array([[1, 0], [0, -1]])
jump_operator_list = [[L_1, L_2] for _ in range(L)]


dims=qt_obs.dims
tens_obs=qt_obs.full().reshape(dims[0]+dims[1])
tt_obs=TT(tens_obs,threshold=1e-12)



#%%
# initial state
rank = 6

num_trajectories = 500

def loss_function(gamma_1,gamma_2):
    jump_parameter_list = [[np.sqrt(gamma_1), np.sqrt(gamma_2)] for _ in range(L)]

    exp_vals = np.zeros(timesteps+1)


    for k in range(num_trajectories):
        initial_state = tt.unit([2] * L, [0] * L)
        for i in range(rank - 1):
            initial_state += tt.unit([2] * L, [0] * L)
        initial_state = initial_state.ortho()
        initial_state = (1 / initial_state.norm()) * initial_state
        #print(initial_state.cores)
        # time step, number of steps, operator_site and max rank
        time_step = 0.1

        max_rank = rank



        # observable
        observable = tt.eye(dims=[2]*L)
        observable.cores[operator_site]=tt.build_core([X])


        exp_vals[0] += initial_state.transpose(conjugate=True)@tt_obs@initial_state
        for i in range(timesteps):
            initial_state = ode.tjm(hamiltonian, jump_operator_list, jump_parameter_list, initial_state, time_step, 1)[-1]
            #initial_state = ode.tdvp(hamiltonian, initial_state, time_step, 1)[-1]
            exp_vals[i+1] += initial_state.transpose(conjugate=True)@tt_obs@initial_state

    exp_vals = (1/num_trajectories)*exp_vals

    loss = np.linalg.norm(exp_vals-Eh_traj)

    return exp_vals, loss






def gradient_loss_function(loss,delta,gamma_1,gamma_2):
    # delta = 1e-3
    grad_gamma_1 = (loss_function(gamma_1+delta,gamma_2)[1]-loss)/delta
    grad_gamma_2 = (loss_function(gamma_1,gamma_2+delta)[1]-loss)/delta

    return grad_gamma_1, grad_gamma_2



#%%

exp_vals, loss = loss_function(0.1,0.1)

delta_list = [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6,1e-7,1e-8]

grad_list = []

for delta in delta_list:
    grad_gamma_1, grad_gamma_2 = gradient_loss_function(loss,delta,0.1,0.1)
    grad_list.append([grad_gamma_1,grad_gamma_2])
    
grad_list = np.array(grad_list)


#%%
plt.plot(grad_list[:,1])




#%%



print(loss,grad_gamma_1,grad_gamma_2)
plt.plot(exp_vals)
plt.plot(result_lindblad.expect[0])
plt.legend(['scikit_tt', 'qutip'])


#%%

gamma1_list = np.linspace(0.01,0.2,20)

loss_list = []

exp_vals_list = []

for gamma_1 in gamma1_list:
    loss = loss_function(gamma_1,0.1)[1]
    exp_vals = loss_function(gamma_1,0.1)[0]
    loss_list.append(loss)
    exp_vals_list.append(exp_vals)

#%%
plt.plot(gamma1_list,loss_list,'o-')
plt.title("Loss vs gamma1. N_traj = 100")



#%%
plt.plot(Eh_traj,label="Exact")

for i in range(7,13):
    plt.plot(exp_vals_list[i],label=f"gamma1 = {gamma1_list[i]}")

plt.title(" Observable. N_traj = 100")
plt.legend()

# # %%

# qt_paulis = [sx, sy, sz]
# qt_obs=qt.tensor([np.random.choice(qt_paulis) for i in range(L)])

# # tt_obs=TT(qt_obs.data)

# # %%

# # %%
# dims=qt_obs.dims

# tens_obs=qt_obs.full().reshape(dims[0]+dims[1])

# tt_obs=TT(tens_obs)

# # %%

# tt_obs

# #%%
# hamiltonian

# # %%
# N=[2,3,4,5]
# Mat=np.random.rand(np.prod(N),np.prod(N))

# print(Mat.shape)

# Mat=Mat.reshape(N+N)

# print(Mat.shape)

# # %%

# %%
