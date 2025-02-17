#%%
import matplotlib.pyplot as plt
import numpy as np
import qutip as qt
import random
import time
#%%
import yaqs
#%%
from yaqs.general.data_structures.MPO import MPO
from yaqs.general.data_structures.MPS import MPS
from yaqs.general.data_structures.noise_model import NoiseModel
from yaqs.general.data_structures.simulation_parameters import Observable, PhysicsSimParams
from yaqs.physics.methods import TJM

#%%

'''TODO: 

Maybe tjm should not be called inside loss function.
Check Learning rate optimization and epsilon in gradient approximation
and try to learn parameters with qutip as comparison

Try KL divergence as loss function
'''


'''run code via:
PYTHONPATH=$(pwd)/src python3 examples/4_Nois
e_characterization.py '''



# ######### QuTip Exact Solver ############
# Time vector
t = np.arange(0, 0.1 + 0.1, 0.1) # has to be np.arange(0, sim_params.T+sim_params.dt, sim_params.dt)python /Users/maximilianfrohlich/Desktop/test_yaqs.py

# Define Pauli matrices
sx = qt.sigmax()
sy = qt.sigmay()
sz = qt.sigmaz()

L = 10
J = 1
g = 0.5

gamma = 0.1

# Construct the Ising Hamiltonian
H = 0
for i in range(L-1):
    H += -J * qt.tensor([sz if n==i or n==i+1 else qt.qeye(2) for n in range(L)])
for i in range(L):
    H += -g * qt.tensor([sx if n==i else qt.qeye(2) for n in range(L)])

# Construct collapse operators
c_ops = []

# Relaxation operators
for i in range(L):
    c_ops.append(np.sqrt(gamma) * qt.tensor([qt.destroy(2) if n==i else qt.qeye(2) for n in range(L)]))

# Dephasing operators
for i in range(L):
    c_ops.append(np.sqrt(gamma) * qt.tensor([sz if n==i else qt.qeye(2) for n in range(L)]))

# Initial state
psi0 = qt.tensor([qt.basis(2, 0) for _ in range(L)])

# Define measurement operators
sx_list = [qt.tensor([sx if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]
sy_list = [qt.tensor([sy if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]
sz_list = [qt.tensor([sz if n == i else qt.qeye(2) for n in range(L)]) for i in range(L)]

obs_list = sx_list + sy_list + sz_list

# Exact Lindblad solution
result_lindblad = qt.mesolve(H, psi0, t, c_ops, obs_list, progress_bar=True)
qt_exp_vals = []
for site in range(len(obs_list)):
    qt_exp_vals.append(result_lindblad.expect[site])



def tjm(noise_params):

    # Define the system Hamiltonian
    L = 10
    d = 2
    J = 1
    g = 0.5
    H_0 = MPO()
    H_0.init_Ising(L, d, J, g)
    # Define the initial state
    state = MPS(L, state='zeros')

    # Define the noise model
    gamma_relaxation = noise_params[0]
    gamma_dephasing = noise_params[1]
    noise_model = NoiseModel(['relaxation', 'dephasing'], [gamma_relaxation, gamma_dephasing])

    T = 0.1
    dt = 0.1
    sample_timesteps = True
    N = 1000
    threshold = 1e-6
    max_bond_dim = 4
    order = 2
    measurements = [Observable('x', site) for site in range(L)] + [Observable('y', site) for site in range(L)] + [Observable('z', site) for site in range(L)]

    sim_params = PhysicsSimParams(measurements, T, dt, sample_timesteps, N, max_bond_dim, threshold, order)
    TJM.run(state, H_0, sim_params, noise_model)

    tjm_exp_vals = []
    for observable in sim_params.observables:
        tjm_exp_vals.append(observable.results)
        # print(f"Observable at site {observable.site}: {observable.results}")
    # print(tjm_exp_vals)


    return tjm_exp_vals, sim_params






def loss_function(noise_params, qt_exp_vals):
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
    tjm_exp_vals, _ = tjm(noise_params)  
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




def compute_gradient(base_loss, loss_function, noise_params, qt_exp_vals, epsilon=1e-2):
    """
    Approximates the gradient of the loss function with respect to noise_params using central differences.

    Args:
        loss_function: Function to compute the loss.
        noise_params: Current noise parameters.
        qt_exp_vals: QuTiP expectation values.
        epsilon: Small value for finite difference approximation.

    Returns:
        grad: Gradient vector with respect to noise_params.
    """
    grad = np.zeros_like(noise_params)
    for i in range(len(noise_params)):
        # Perturb parameter i positively and negatively
        params_up = np.copy(noise_params)
        
        params_up[i] += epsilon
        
        
        # Compute loss for perturbed parameters
        loss_up,_ = loss_function(params_up, qt_exp_vals)
        
        
        # Approximate gradient
        grad[i] = (loss_up - base_loss) / (epsilon)
    return grad

def gradient_descent(qt_exp_vals, init_noise_params, learning_rate=0.1, epochs=500):
    """
    Implements stochastic gradient descent to optimize noise parameters.

    Args:
        qt_exp_vals: QuTiP expectation values.
        init_noise_params: Initial noise parameters.
        learning_rate: Step size for updates.
        epochs: Number of iterations.

    Returns:
        optimized_params: Optimized noise parameters.
        loss_history: List of loss values during training.
    """
    # Initialize noise parameters
    noise_params = np.copy(init_noise_params)
    loss_history = []

    for epoch in range(epochs):
        # Compute the loss
        start_time = time.time()
        # Compute the loss
        loss, _ = loss_function(noise_params, qt_exp_vals)
        # End timing
        end_time = time.time()

        # Record the loss calculation time
        loss_time = end_time - start_time
        print(f"Epoch {epoch + 1}/{epochs}, Loss Calculation Time: {loss_time:.4f} seconds")

        
        loss_history.append(loss)

        start_time = time.time()

        # Compute the gradient
        grad = compute_gradient(loss, loss_function, noise_params, qt_exp_vals)

        end_time = time.time()

        gradient_time = end_time - start_time
        print(f"Epoch {epoch +1}/ {epochs}, Gradient Calculation Time: {gradient_time:.4f} seconds")

        # Update parameters
        noise_params -= learning_rate * grad

        noise_params = [max(p, 0) for p in noise_params]

        # Print progress
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss}, Params: {noise_params}")

    return noise_params, loss_history



def adam_optimizer_update(noise_params, grad, m, v, t, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    Performs a single Adam update for the given parameters.

    Args:
        noise_params (list): Current noise parameters.
        grad (array): Gradient of the loss with respect to noise parameters.
        m (array): Exponential moving average of the gradients (first moment).
        v (array): Exponential moving average of the squared gradients (second moment).
        t (int): Current timestep (epoch).
        learning_rate (float): Learning rate for Adam.
        beta1 (float): Decay rate for the first moment.
        beta2 (float): Decay rate for the second moment.
        epsilon (float): Small constant to avoid division by zero.

    Returns:
        updated_params (list): Updated noise parameters.
        m (array): Updated first moment estimate.
        v (array): Updated second moment estimate.
    """
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    updated_params = noise_params - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)
    updated_params = [max(p, 0) for p in updated_params]  # Ensure parameters stay non-negative
    return updated_params, m, v

def adam_optimized_gradient_descent(qt_exp_vals, init_noise_params, learning_rate=0.01, epochs=100):
    """
    Implements Adam optimizer for minimizing the loss.

    Args:
        qt_exp_vals (list): QuTiP expectation values.
        init_noise_params (list): Initial noise parameters.
        learning_rate (float): Step size for updates.
        epochs (int): Number of iterations.

    Returns:
        optimized_params (list): Optimized noise parameters.
        loss_history (list): List of loss values during optimization.
    """
    # Initialize noise parameters
    noise_params = np.copy(init_noise_params)
    m, v = np.zeros_like(noise_params), np.zeros_like(noise_params)  # Initialize moments
    loss_history = []

    for epoch in range(1, epochs + 1):
        # Compute the loss
        loss, _ = loss_function(noise_params, qt_exp_vals)
        loss_history.append(loss)

        # Compute the gradient
        grad = compute_gradient(loss, loss_function, noise_params, qt_exp_vals)

        # Update parameters using Adam
        noise_params, m, v = adam_optimizer_update(noise_params, grad, m, v, epoch, learning_rate)

        # Print progress
        print(f"Epoch {epoch}/{epochs}, Loss: {loss:.4f}, Params: {noise_params}")

    return noise_params, loss_history





if __name__ == "__main__":
    init_noise_params = [0.2, 0.4]  # Initial guesses for gamma_relaxation and gamma_dephasing
    
    # Run Adam optimization
    optimized_params, loss_history = adam_optimized_gradient_descent(qt_exp_vals, init_noise_params, learning_rate=0.01, epochs=100)

    print(f"Optimized Parameters: {optimized_params}")

    # # Plot the loss history
    # plt.plot(loss_history)
    # plt.xlabel("Epochs")
    # plt.ylabel("Loss")
    # plt.title("Loss History During Adam Optimization")
    # plt.show()
        # Plot the loss history with a logarithmic scale
    plt.figure(figsize=(8, 5))
    plt.plot(loss_history, label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss History During Adam Optimization (Log Scale)")
    plt.yscale("log")  # Set the y-axis to logarithmic scale
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)  # Improve visibility
    plt.legend()
    plt.show()











