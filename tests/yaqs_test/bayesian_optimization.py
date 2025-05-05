
#%%
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

from mqt.yaqs.noise_char.optimization import *
from mqt.yaqs.noise_char.propagation import *
from scipy.interpolate import RegularGridInterpolator

#%%
def loss_function(sim_params, ref_traj, traj_der, loss_std, dJ_d_gr_std, dJ_d_gd_std):
    """
    Compute the loss function and its gradients for the given simulation parameters.
    Parameters:
    sim_params (dict): Dictionary containing the simulation parameters.
    ref_traj (list): List of reference trajectories for comparison.
    traj_der (function): Function that runs the simulation and returns the time, 
                         expected values trajectory, and derivatives of the observables 
                         with respect to the noise parameters.
    Returns:
    tuple: A tuple containing:
        - loss (float): The computed loss value.
        - exp_vals_traj (list): The expected values trajectory from the TJM simulation.
        - gradients (numpy.ndarray): Array containing the gradients of the loss with respect 
                                     to gamma_relaxation and gamma_dephasing.
    """
    
    
    # Run the TJM simulation with the given noise parameters

    start_time = time.time()
    # t, exp_vals_traj, d_On_d_gk, A_kn_exp_vals = traj_der(sim_params)  
    t, exp_vals_traj, d_On_d_gk = traj_der(sim_params)  
   
    end_time = time.time()
    tjm_time = end_time - start_time
    # print(f"TJM time -> {tjm_time:.4f}")
    
   
    
    # Ensure both lists have the same structure
    if np.shape(ref_traj) != np.shape(exp_vals_traj):
        raise ValueError("Mismatch in the number of sites between qt_exp_vals and tjm_exp_vals.")


    n_jump_site, n_obs_site, L, nt = np.shape(d_On_d_gk)


    # Initialize loss
    loss = 0.0

    dJ_d_gr = 0
    dJ_d_gd = 0



    for i in range(n_obs_site):
        for j in range(L):
            for k in range(nt):

                loss += (exp_vals_traj[i,j,k] - ref_traj[i,j,k])**2

                # I have to add all the derivatives with respect to the same gamma_relaxation and gamma_dephasing
                dJ_d_gr += 2*(exp_vals_traj[i,j,k] - ref_traj[i,j,k]) * d_On_d_gk[0,i,j,k]

                dJ_d_gd += 2*(exp_vals_traj[i,j,k] - ref_traj[i,j,k]) * d_On_d_gk[1,i,j,k]


    if loss_std == 0:
        error=np.random.normal(loc=0.0, scale=0)
    else:  
        error=np.random.normal(loc=0.0, scale=loss_std((sim_params.gamma_rel, sim_params.gamma_deph))[0])


    if loss + error > 0:
        loss += error

    else:
        loss = loss - error

    if dJ_d_gr_std == 0:
        dJ_d_gr += np.random.normal(loc=0.0, scale=0)
    else:
        dJ_d_gr += np.random.normal(loc=0.0, scale=dJ_d_gr_std((sim_params.gamma_rel, sim_params.gamma_deph))[0])

    if dJ_d_gd_std == 0:
        dJ_d_gd += np.random.normal(loc=0.0, scale=0)
    else:
        dJ_d_gd += np.random.normal(loc=0.0, scale=dJ_d_gd_std((sim_params.gamma_rel, sim_params.gamma_deph))[0])


    return loss, exp_vals_traj, np.array([dJ_d_gr])

#%%




#%%
N= 500
num_samples=50
file_list=["loss.txt","gamma_rel.txt","gamma_deph.txt"]



for file in file_list:

    folder=f"noise_char/N_{N}/samples_50/"

    full_file=folder+file
    data=np.genfromtxt(full_file)


    std_data = np.array([[data[i,0],data[i,1], np.std(data[i,2:])] for i in range(len(data))])

    # Create a callable function by interpolating std_data
    x = std_data[:, 0]
    y = std_data[:, 1]
    z = std_data[:, 2]

    # Ensure the data is sorted for interpolation
    sorted_indices = np.lexsort((y, x))
    x_sorted = x[sorted_indices]
    y_sorted = y[sorted_indices]
    z_sorted = z[sorted_indices]

    # Reshape the data for RegularGridInterpolator
    unique_x = np.unique(x_sorted)
    unique_y = np.unique(y_sorted)
    z_grid = z_sorted.reshape(len(unique_x), len(unique_y))

    # Create the interpolator
    if file=="loss.txt":
        interpolator_loss = RegularGridInterpolator((unique_x, unique_y), z_grid)
    if file=="gamma_rel.txt":
        interpolator_gr = RegularGridInterpolator((unique_x, unique_y), z_grid)
    if file=="gamma_deph.txt":
        interpolator_gd = RegularGridInterpolator((unique_x, unique_y), z_grid)




#%%
file="loss.txt"
folder=f"noise_char/N_{N}/samples_50/"

full_file=folder+file
data=np.genfromtxt(full_file)



# Filter rows where the third column is equal to a specific number
gamma_name_list=["gamma_rel", "gamma_deph"]
gamma= 0.1  # Replace with the desired number
gamma_name="gamma_deph"

index=gamma_name_list.index(gamma_name)

data_1 = data[data[:, index] == gamma]


#%%
plt.figure(figsize=(8, 6))
plt.plot(data_1[:, 1 - index ], data_1[:,2:], 'o', markersize=5)
# x = np.linspace(min(data_1[:, 1 - index]), max(data_1[:, 1 - index]), 1000)
# y= interpolator_gr((x, gamma))
# plt.plot(x, y, label='Interpolated Loss')
# plt.xlabel(gamma_name_list[1 -index])
# %%
samples=data_1[0,2:]
print(data_1[0,:2])
kde = gaussian_kde(samples, bw_method=1)

x = np.linspace(min(samples), max(samples), 1000)
plt.plot(x, kde(x))
plt.title("Smoothed KDE")
plt.show()


# %%

gamma_list=[0.01, 0.05,0.1,0.15,0.2]


sim_params = SimulationParameters()

t, qt_ref_traj, d_On_d_gk=qutip_traj(sim_params)

#%%
data_loss=np.zeros([len(gamma_list),num_samples])

for i, g in enumerate(gamma_list):
    for j in range(num_samples):
    
        sim_params.gamma_rel = g
        sim_params.gamma_deph = gamma

        # Call the loss function with the random parameters
        loss, exp_vals_traj, grad=loss_function(sim_params, qt_ref_traj, qutip_traj, loss_std=0, dJ_d_gr_std=0, dJ_d_gd_std=0)
        data_loss[i,j]=loss

 #%%


samples=data_loss[0]

kde = gaussian_kde(samples, bw_method=1)

x = np.linspace(min(samples), max(samples), 1000)
plt.plot(x, kde(x))
plt.title("Smoothed KDE")
plt.show()

# %%
plt.figure(figsize=(8, 6))
plt.plot(data_1[:, 1 - index ], data_1[:,2:], 'o', markersize=5)

# %%
plt.figure(figsize=(8, 6))
plt.plot(gamma_list, data_loss, 'o', markersize=5)
# %%




















# %%


### For Bayesian Optimization
import torch
import botorch
import gpytorch
import matplotlib.pyplot as plt
from botorch.acquisition import UpperConfidenceBound
from botorch.optim import optimize_acqf
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood
from gpytorch.models import ExactGP
from gpytorch.means import ConstantMean
from gpytorch.kernels import ScaleKernel, RBFKernel
from gpytorch.likelihoods import FixedNoiseGaussianLikelihood
from gpytorch.distributions import MultivariateNormal
from botorch.models.model import Model
from botorch.posteriors import GPyTorchPosterior
from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.settings import debug
from gpytorch.likelihoods import GaussianLikelihood
from botorch.models.gp_regression import SingleTaskGP
from botorch.models.transforms.outcome import Standardize
from botorch.acquisition.objective import ScalarizedPosteriorTransform

# GP model for noisy observations
class NoisyDerivativeGPModel(ExactGP):
    def __init__(self, train_x, train_y, noise, likelihood):
        self.num_outputs = 1
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(RBFKernel(ard_num_dims=train_x.shape[-1]))
        self.likelihood.noise = noise  # fixed noise per point

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)

    def transform_inputs(self, *args, **kwargs):
        if args:
            return args[0]
        elif "X" in kwargs:
            return kwargs["X"]
        else:
            raise ValueError("transform_inputs called without inputs")


class GPModelWithDerivatives(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModelWithDerivatives, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGrad()
        self.base_kernel = gpytorch.kernels.RBFKernelGrad()
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)
        self.num_outputs = likelihood.num_tasks

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
    
    def posterior(self, X, output_indices=None, observation_noise=False, **kwargs):
        """
        Compute the posterior distribution at the given input points X.
        """
        self.eval()  # Ensure the model is in evaluation mode
        with torch.no_grad():
            mvn = self(X)  # Compute the MultitaskMultivariateNormal
        return GPyTorchPosterior(mvn)

    def transform_inputs(self, *args, **kwargs):
        if args:
            return args[0]
        elif "X" in kwargs:
            return kwargs["X"]
        else:
            raise ValueError("transform_inputs called without inputs")


class WrappedNoisyDerivativeModel(Model):
    def __init__(self, gp_model):
        super().__init__()
        self.gp = gp_model
        self._dtype = gp_model.train_inputs[0].dtype
        self._device = gp_model.train_inputs[0].device

    def posterior(self, X, output_indices=None, observation_noise=False, **kwargs):
        self.gp.eval()
        mvn = self.gp(X)  # allow gradients to flow!
        return GPyTorchPosterior(mvn)

    def condition_on_observations(self, X, Y, **kwargs):
        raise NotImplementedError("Not needed for this use case.")

    def transform_inputs(self, *args, **kwargs):
        return self.gp.transform_inputs(*args, **kwargs)

    @property
    def num_outputs(self):  # ✅ here’s the fix
        return 1


#%%

sim_params_copy = SimulationParameters()
t, qt_ref_traj, d_On_d_gk=qutip_traj(sim_params_copy)
ref_traj = qt_ref_traj
traj_der = qutip_traj
bounds_list = [(0.01,0.2)]
acquisition="UCB"
n_init=5
max_iterations=200
tolerance=1e-8
beta=0.1
num_restarts=2
raw_samples=5
file_name=" "
device="cpu"
loss_std=interpolator_loss
dJ_d_gr_std=interpolator_gr
dJ_d_gd_std=interpolator_gd


#%%

sim_params = copy.deepcopy(sim_params_copy)


d = len(bounds_list)  # Number of parameters to optimize (gamma_rel and gamma_deph)

loss_history = []

x_history = []

dL_dx_history = []

if os.path.exists(file_name) and file_name != " ":
    os.remove(file_name)

if file_name != " ":
    with open(file_name, 'w') as file:
        file.write('#  Iter    Loss    Log10(Loss)    x \n')

# Config
bounds = torch.tensor(bounds_list, device=device, dtype=torch.double).T  # Transpose to match the shape [[lower1, lower2, ...], [upper1, upper2, ...]]

# Initial data
X_train = torch.empty(n_init, len(bounds_list), device=device, dtype=torch.double, requires_grad=True)
X_new = torch.empty_like(X_train)  # Create a new tensor to avoid in-place operations
for i, (lower, upper) in enumerate(bounds_list):
    X_new[:, i] = torch.rand(n_init, device=device, dtype=torch.double) * (upper - lower) + lower  # Scale and shift random values to the bounds
X_train = X_new  # Assign the new tensor to X
X_train.requires_grad_()
Y_vals = []
grad_vals = []

for i in range(n_init):
    sim_params.gamma_rel = X_train[i].detach().cpu().numpy()
    loss, _, dJ_dg = loss_function(sim_params, ref_traj, traj_der, loss_std=loss_std, dJ_d_gr_std=dJ_d_gr_std, dJ_d_gd_std=dJ_d_gd_std)
    Y_vals.append([-loss])
    grad_vals.append(dJ_dg)

    if file_name != " ":
        with open(file_name, 'a') as file:
            file.write('    '.join(map(str, [i, loss, np.log10(loss), sim_params.gamma_rel, sim_params.gamma_deph])) + '\n')

    loss_history.append(loss)
    x_history.append(X_train[i].detach().cpu().numpy())
    dL_dx_history.append(dJ_dg)


Y_tensor= torch.tensor(Y_vals, dtype=torch.double, device=device,requires_grad=True)  # shape: (n_init, 1)
grad_tensor = torch.tensor(grad_vals, dtype=torch.double, device=device,requires_grad=True)   

Y_train=torch.cat([Y_tensor,grad_tensor],dim=1).requires_grad_(True)

#%%
for iteration in range(n_init,max_iterations+n_init):
    print(f"Iteration {iteration}")
   

    #GP model with unknown noise

    likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=d+1)
    model = GPModelWithDerivatives(X_train, Y_train, likelihood)
    mll = ExactMarginalLogLikelihood(likelihood, model)

    model.train()
    likelihood.train()


    with debug(True):
        fit_gpytorch_mll(mll)
    model.eval()

    # Acquisition
    if acquisition == "UCB":
        # Define weights to select the first output
        weights = torch.tensor([1.0, 0.0])  # Only consider the first output
        posterior_transform = ScalarizedPosteriorTransform(weights=weights)
        acq_func = UpperConfidenceBound(model, beta=beta, posterior_transform=posterior_transform)

    # elif acquisition == "qNEI":
    #     acq_func = qNoisyExpectedImprovement(model, X_baseline=X_joint)
    
    # elif acquisition == "qLNEI":
    #     acq_func = qLogNoisyExpectedImprovement(model, X_baseline=X_joint)

    # else:
    #     raise ValueError(f"Unknown acquisition function: {acquisition}. Valid options are 'UCB', 'qNEI', and 'qLNEI'.")

    candidate, _ = optimize_acqf(
        acq_function=acq_func,
        bounds=bounds,
        q=1,
        num_restarts=num_restarts,
        raw_samples=raw_samples,
    )

    # Sample at new point
    candidate.requires_grad_()

    sim_params.gamma_rel = candidate.detach().cpu().numpy()[0]
    loss, _, dJ_dg = loss_function(sim_params, ref_traj, traj_der, loss_std=loss_std, dJ_d_gr_std=dJ_d_gr_std, dJ_d_gd_std=dJ_d_gd_std)

    if file_name != " ":
        with open(file_name, 'a') as file:
            file.write('    '.join(map(str, [iteration, loss, np.log10(loss), sim_params.gamma_rel, sim_params.gamma_deph])) + '\n')


    loss_history.append(loss)
    x_history.append(candidate.detach().cpu().numpy()[0])
    dL_dx_history.append(dJ_dg)


    X_train.append(candidate)
    Y_train.append(torch.tensor([[-loss] + dJ_dg], dtype=torch.double, device=device))

    if loss < tolerance:
        print(f"Converged after {iteration + 1} iterations.")
        break



# %%
print("X_train requires_grad:", X_train.requires_grad)
print("Y_train requires_grad:", Y_train.requires_grad)
# %%
print("Model's training inputs require_grad:", acq_func.model.train_inputs[0].requires_grad)
# %%
