
#%%
import numpy as np
import matplotlib.pyplot as plt

import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood


from botorch.acquisition import LogExpectedImprovement,UpperConfidenceBound

import gpytorch
from botorch.optim import optimize_acqf

#%%

def loss_function(x):
    f= (x - 0.5) ** 2 
    grad = 2 * (x - 0.5)
    return f, grad


xplot = np.linspace(0, 1, 100)
yplot = loss_function(xplot)[0]
#%%
def plot_model(mean, std_dev, xplot, yplot, train_X, train_Y, iter=0, output_dim=0):

    plt.figure(figsize=(8, 6))
    plt.plot(xplot, yplot, label='True Loss Function', color='blue')
    plt.plot(test_x.numpy(), mean, label='GP Mean', color='orange')
    plt.fill_between(
        test_x.numpy().flatten(),
        mean - 1 * std_dev,
        mean + 1 * std_dev,
        color='orange',
        alpha=0.2,
        label='Confidence Interval (±σ)'
    )
    plt.scatter(train_X.numpy(), train_Y[:,output_dim].numpy(), color='red', label='Training Data')
    # plt.axhline(y=0.5, color='green', linestyle='--', label='y=0.5')
    plt.legend()
    plt.title('Gaussian Process Model with Uncertainty')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.savefig(f'plot_gp_with_der/gp_model_{iter}.png')
    plt.close()


#%%
class transform:
    def __init__(self, displacement, range):

        self.displacement = displacement
        self.range = range

    def displace(self, x):
            return x - self.displacement

    def scale(self, x):
        return x / self.range
    
    def transform(self, x):
        x = self.displace(x)
        x = self.scale(x)
        return x
    
    def undisplace(self, x):
        return x + self.displacement
    
    def unscale(self, x):
        return x * self.range

    def untransform(self, x):
        x = self.unscale(x)
        x = self.undisplace(x)
        return x
    


class normalize(transform):

    def __init__(self, x):

        if isinstance(x, torch.Tensor):
            self.displacement = x.min(axis=0).values
            self.range = x.max(axis=0).values - x.min(axis=0).values

        elif isinstance(x, np.ndarray):
            self.displacement = x.min(axis=0)
            self.range = x.max(axis=0) - x.min(axis=0)
        

class standardize(transform):
    def __init__(self, x):

        self.displacement = x.mean(axis=0)
        self.range = x.std(axis=0)


#%%
from typing import Optional
from torch import Tensor
from gpytorch.models import ExactGP
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.distributions import MultivariateNormal
from gpytorch.kernels import RBFKernel, ScaleKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.means import ConstantMean
from botorch.acquisition.objective import ScalarizedPosteriorTransform

class SimpleCustomGP(ExactGP, GPyTorchModel):

    _num_outputs = 1  # to inform GPyTorchModel API

    def __init__(self, train_X, train_Y, l, train_Yvar: Optional[Tensor] = None, input_transform=None, outcome_transform=None):
        
        if outcome_transform is not None:
            train_Y, _ = outcome_transform(train_Y)

        super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
        self.mean_module = ConstantMean()
        self.covar_module = ScaleKernel(
            base_kernel=RBFKernel(ard_num_dims=train_X.shape[-1]),
        )
        self.covar_module.base_kernel.lengthscale = l

        self.input_transform = input_transform # Normalize input to [0, 1]^d
        self.outcome_transform = outcome_transform  # Standardize output to zero mean and unit variance

        self.to(train_X)  # make sure we're on the right device/dtype

    def forward(self, x):
        if self.input_transform is not None:
            x = self.input_transform(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)



class GPModelWithDerivatives(ExactGP, GPyTorchModel):
    def __init__(self, train_x, train_y, likelihood, input_transform=None, outcome_transform=None):
        super(GPModelWithDerivatives, self).__init__(train_x, train_y, likelihood)
        self.likelihood = likelihood
        self.mean_module = gpytorch.means.ConstantMeanGrad()
        self.base_kernel = gpytorch.kernels.RBFKernelGrad()
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)
        self._num_outputs = train_y.shape[-1] 

        # self.input_transform = input_transform # Normalize input to [0, 1]^d
        # self.outcome_transform = outcome_transform  # Standardize output to zero mean and unit variance

        # if self.input_transform is not None:
        #     self.train_inputs = (self.input_transform(train_x),)

        # if self.outcome_transform is not None:
        #     self.train_targets = self.outcome_transform(train_y)[0]

    @property
    def num_outputs(self):
        return self._num_outputs  # Return the stored number of outputs


    def forward(self, x):
        # if self.input_transform is not None:
        #     x = self.input_transform(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


    # def postprocess_transform(self, Y):
    #     # Optional: apply inverse standardization for interpretability
    #     return self.outcome_transform.untransform(Y)



#%% 

n_init = 2
n_iter = 10
d=1  # number of dimensions
m=2 # number of outputs
bounds_list = [[0,1]]

num_restarts = 5
raw_samples = 20

acq_name="UCB"
beta=1
noise=1e-0

l=0.2
gp_name="GPWithDerivatives"
#%%

loss_history = []
x_history = []

bounds = torch.tensor(bounds_list,dtype=torch.double).T

X_train = torch.rand(n_init, d, dtype=torch.double)

Y_train = torch.zeros(n_init, m, dtype=torch.double)

for i in range(n_init):
    loss,grad = loss_function(X_train[i].numpy())

    Y_train[i,0]= torch.tensor(loss, dtype=torch.double)
    if m > 1:
        Y_train[i,1]= torch.tensor(grad, dtype=torch.double)


    x_history.append(X_train[i].numpy())
    loss_history.append(loss)




#%%
# Plot the model with the standard deviation

for i in range(n_iter):

    train_Yvar = torch.full_like(Y_train, noise)

    norm = normalize(X_train)
    stand = standardize(Y_train)

    X_transform = norm.transform(X_train)
    Y_transform = stand.transform(Y_train)



    if gp_name == "SimpleCustomGP":
        gp = SimpleCustomGP(
            train_X=X_transform,
            train_Y=Y_transform,
            l=torch.tensor(l),
            # train_Yvar=train_Yvar, 
            # input_transform=Normalize(d=d),
            # outcome_transform=Standardize(m=m),
        ).to(X_transform)

    if gp_name == "SingleTaskGP":
        # Use the SingleTaskGP model from BoTorch
        kernel = ScaleKernel(RBFKernel(ard_num_dims=d))
        # kernel.base_kernel.lengthscale = torch.tensor(l)

        gp = SingleTaskGP(
            train_X=X_transform,
            train_Y=Y_transform,
            train_Yvar=train_Yvar,
            # # covar_module=kernel,
            # input_transform=Normalize(d=d),
            # outcome_transform=Standardize(m=m),
        ).to(X_transform)


    if gp_name == "GPWithDerivatives":
        gp = GPModelWithDerivatives(
            train_x=X_transform,
            train_y=Y_transform,
            likelihood=gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=m),
            # input_transform=Normalize(d=d),
            # outcome_transform=Standardize(m=m)
        ).to(X_transform)


    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)


    ### Computing mean and std deviation
    output_dim=0
    with torch.no_grad():
        test_x = torch.tensor(xplot, dtype=torch.double).unsqueeze(-1)
        posterior = gp.posterior(norm.transform(test_x))
        mean = stand.untransform(posterior.mean).numpy()[:,output_dim].flatten()
        std_dev = stand.unscale(posterior.variance.sqrt()).numpy()[:,output_dim].flatten()

    

    plot_model(mean, std_dev, xplot, yplot, X_train, Y_train, iter=i)

    scal_transf = ScalarizedPosteriorTransform(weights=torch.tensor([1.0, 0.0], dtype=torch.double))

    if acq_name == "LEI":
        acqf = LogExpectedImprovement(model=gp, best_f=Y_train.min(), posterior_transform=scal_transf, maximize=False)
    
    if acq_name == "UCB":
        acqf = UpperConfidenceBound(model=gp, beta=beta, posterior_transform=scal_transf, maximize=False)



    candidate, acq_value = optimize_acqf(
        acqf, bounds=bounds, q=1, num_restarts=num_restarts, raw_samples=raw_samples,
    )



    X_new = norm.untransform(candidate)
    loss, grad=loss_function(X_new.numpy()[0])

    x_history.append(X_new.numpy()[0])
    loss_history.append(loss)

                       
    Y_new=torch.tensor(np.array([loss,grad]), dtype=torch.double).T               

    X_train = torch.cat([X_train, X_new], dim=0)
    Y_train = torch.cat([Y_train, Y_new], dim=0)

    

plt.plot(np.array(x_history)[:,0], '-', label='history')
plt.axhline(y=0.5, color='r', linestyle='--', label='y=0.5')
plt.legend()
plt.savefig('plot_gp_with_der/x_history.png')
plt.close()

#%%
Y_train.shape
# %%
X_train.shape
# %%
xplot.shape
# %%
yplot.shape
# %%
with torch.no_grad():
    test_x = torch.tensor(xplot, dtype=torch.double).unsqueeze(-1)
    posterior = gp.posterior(test_x)
    mean = posterior.mean.numpy().flatten()
# %%
posterior.mean.numpy().shape
# %%
X_train.shape
# %%
torch.tensor([loss,grad], dtype=torch.double).T.shape
# %%
gp = GPModelWithDerivatives(
            train_x=X_train,
            train_y=Y_train,
            likelihood=gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=1 + d),
        ).to(X_train)
# %%
output_dim=0
with torch.no_grad():
        test_x = torch.tensor(xplot, dtype=torch.double).unsqueeze(-1)
        posterior = gp.posterior(test_x)
        mean = stand.untransform(posterior.mean).numpy()[:,output_dim].flatten()
        std_dev = stand.untransform(posterior.variance.sqrt()).numpy()[:,output_dim].flatten()
# %%
max(std_dev)
# %%
plt.figure(figsize=(8, 6))
plt.plot(xplot, yplot, label='True Loss Function', color='blue')
plt.plot(test_x.numpy(), mean, label='GP Mean', color='orange')
plt.fill_between(
    test_x.numpy().flatten(),
    mean - 1 * std_dev,
    mean + 1 * std_dev,
    color='orange',
    alpha=0.2,
    label='Confidence Interval (±σ)'
)
plt.scatter(X_train.numpy(), Y_train[:,output_dim].numpy(), color='red', label='Training Data')
# plt.axhline(y=0.5, color='green', linestyle='--', label='y=0.5')
plt.legend()
plt.title('Gaussian Process Model with Uncertainty')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.savefig(f'plot_gp_with_der/gp_model_{iter}.png')


# %%
plot_model(gp, xplot, yplot, iter=1000)

# %%
norm=Normalize(d=d)
stand=Standardize(m=m)

X_transform = norm(X_train)
Y_transform = stand(Y_train)


# %%
posterior.mean.shape
# %%
norm.untransform(candidate)

# %%
torch.tensor(np.array([loss,grad]), dtype=torch.double).T
# %%
X_new
# %%
plt.plot(X_train.numpy(), Y_train[:,0].numpy(), 'o', label='Training Data')
# %%
X_transform
# %%
test_x

# %%
