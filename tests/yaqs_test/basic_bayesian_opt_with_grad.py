
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
    f= (x + 0.5) ** 2 
    grad = 2 * (x + 0.5)
    return f, grad



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

    def __init__(self, x_input):

        x=np.array(x_input)

        self.displacement = x.min(axis=0)
        self.range = x.max(axis=0) - x.min(axis=0)

        zero_range_mask = self.range == 0
        self.range[zero_range_mask] = np.abs(x).max(axis=0)[zero_range_mask]
        

class standardize(transform):

    def __init__(self, x_input):

        x=np.array(x_input)

        self.displacement = x.mean(axis=0)
        self.range = x.std(axis=0)

        zero_range_mask = self.range == 0
        self.range[zero_range_mask] = np.abs(x).max(axis=0)[zero_range_mask]





#%%


1/np.append(Y_train[:, 0].numpy().std(axis=0), np.zeros(2))
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
from gpytorch.likelihoods import MultitaskGaussianLikelihood

# class SimpleCustomGP(GPyTorchModel,ExactGP):

#     _num_outputs = 1  # to inform GPyTorchModel API

#     def __init__(self, train_X, train_Y, l, train_Yvar: Optional[Tensor] = None, input_transform=None, outcome_transform=None):
        
#         if outcome_transform is not None:
#             train_Y, _ = outcome_transform(train_Y)

#         super().__init__(train_X, train_Y.squeeze(-1), GaussianLikelihood())
#         self.mean_module = ConstantMean()
#         self.covar_module = ScaleKernel(
#             base_kernel=RBFKernel(ard_num_dims=train_X.shape[-1]),
#         )
#         self.covar_module.base_kernel.lengthscale = l

#         self.input_transform = input_transform # Normalize input to [0, 1]^d
#         self.outcome_transform = outcome_transform  # Standardize output to zero mean and unit variance

#         self.to(train_X)  # make sure we're on the right device/dtype

#     def forward(self, x):
#         if self.input_transform is not None:
#             x = self.input_transform(x)
#         mean_x = self.mean_module(x)
#         covar_x = self.covar_module(x)
#         return MultivariateNormal(mean_x, covar_x)



class GPModelWithDerivatives(ExactGP, GPyTorchModel):
    def __init__(self, train_X, train_Y):
        d = train_X.shape[-1]
        likelihood = MultitaskGaussianLikelihood(num_tasks=1 + d)
        super().__init__(train_X, train_Y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGrad()
        self.base_kernel = gpytorch.kernels.RBFKernelGrad(ard_num_dims=d)
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

    @property
    def num_outputs(self):
        return self._num_outputs  # Return the stored number of outputs


    def forward(self, x):
        # if self.input_transform is not None:
        #     x = self.input_transform(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)



#%% 

n_init = 50
n_iter = 1
d=1  # number of dimensions
m=2 # number of outputs
bounds_list = [[-2,2]]

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

xplot = np.linspace(bounds_list[0][0], bounds_list[0][1], 100)
yplot = loss_function(xplot)[0]


bounds = torch.tensor(bounds_list,dtype=torch.double).T

X_train = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n_init, d, dtype=torch.double)


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




    gp = GPModelWithDerivatives(
        train_x=X_train,
        train_y=Y_train,
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
        mean = posterior.mean.numpy()[:,output_dim].flatten()
        std_dev = posterior.variance.sqrt().numpy()[:,output_dim].flatten()

    

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


# %%
plt.plot(xplot, yplot, label='True Loss Function', color='blue')
plt.plot(X_train.numpy(), Y_train[:,0].numpy(), 'o', label='Training Data')
plt.plot(X_transform.numpy(), Y_transform[:,0].numpy(), 'o', label='Training Data Transformed')

# test_x = torch.tensor(xplot, dtype=torch.double).unsqueeze(-1)
# posterior = gp.posterior(test_x)
# mean = posterior.mean.detach().numpy()[:,output_dim]
# # std_dev = stand.unscale(posterior.variance.sqrt().detach()).numpy()[:,output_dim]

plt.plot(test_x, mean,'o', label='GP Mean', color='orange')
plt.legend()
# %%

gp.posterior(test_x).mean.shape
# %%
gp.posterior(test_x).mean.numpy()[:,output_dim].flatten()
# %%
gp.posterior(norm.transform(test_x)).mean.detach().numpy()[:,output_dim]
# %%

# %%
gp.eval()
likelihood=gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=m)
likelihood.eval()
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    preds = likelihood(gp(norm.transform(test_x)))
    mean = preds.mean
    var = preds.variance
# %%
plt.plot(norm.transform(test_x), mean[:,0])
# %%
class derivative_transform(transform):

    def __init__(self, x_input, y_input):

        x=np.array(x_input)
        y=np.array(y_input)

        d=x.shape[1]

        self.x_displacement = x.min(axis=0)
        self.x_range = x.max(axis=0) - x.min(axis=0)

        self.x_range[self.x_range == 0] = np.abs(x).max(axis=0)[self.x_range == 0]


        self.y_scale = y[:,0].std(axis=0)

        if self.y_scale == 0:
            self.y_scale = np.abs(y[:,0]).max(axis=0)

        self.y_displacement = np.append(y[:,0].mean(axis=0),np.zeros(d))
        self.y_range = np.append(self.y_scale, self.y_scale/self.x_range)



    def displace(self, x, displacement):
        return x - displacement

    def scale(self, x, range):
        return x / range
    
    def undisplace(self, x, displacement):
        return x + displacement
    
    def unscale(self, x, range):
        return x * range
    
    def transform_x(self, x):
        x = self.displace(x, self.x_displacement)
        x = self.scale(x, self.x_range)
        return x

    def transform_y(self, y):
        y = self.displace(y, self.y_displacement)
        y = self.scale(y, self.y_range)
        return y

    def untransform_x(self, x):
        x = self.unscale(x, self.x_range)
        x = self.undisplace(x, self.x_displacement)
        return x
    
    def untransform_y(self, y):
        y = self.unscale(y, self.y_range)
        y = self.undisplace(y, self.y_displacement)
        return y
    

    def transform(self, x, y):
        x = self.transform_x(x)
        y = self.transform_y(y)
        return x, y

    def untransform(self, x, y):
        x = self.untransform_x(x)
        y = self.untransform_y(y)
        return x, y

    def unscale_std(self, y):
        y = self.unscale(y, self.y_range)
        return y
# %%

def loss_function(x):
    f= (x + 0.5) ** 2 - 1
    grad = 2 * (x + 0.5)
    return f, grad


n_init = 50
bounds_list = [[-2,2]]


xplot = np.linspace(bounds_list[0][0], bounds_list[0][1], 100)
xplot_transform = np.linspace(0, 1, 100)

yplot = np.array(loss_function(xplot)).T


bounds = torch.tensor(bounds_list,dtype=torch.double).T

X_train = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n_init, d, dtype=torch.double)


Y_train = torch.zeros(n_init, m, dtype=torch.double)

for i in range(n_init):
    loss,grad = loss_function(X_train[i].numpy())

    Y_train[i,0]= torch.tensor(loss, dtype=torch.double)
    if m > 1:
        Y_train[i,1]= torch.tensor(grad, dtype=torch.double)




trans=derivative_transform(X_train, Y_train)

X_transform, Y_transform = trans.transform(X_train, Y_train)

# X_train = X_transform
# Y_train = Y_transform


gp = GPModelWithDerivatives(
    train_X=X_transform,
    train_Y=Y_transform,
    # likelihood=gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=m),
    # input_transform=Normalize(d=d),
    # outcome_transform=Standardize(m=m)
)

# gp = SingleTaskGP(
#             train_X=X_transform,
#             train_Y=Y_transform,
#             # train_Yvar=train_Yvar,
#             # covar_module=kernel,
#             # input_transform=Normalize(d=d),
#             # outcome_transform=Standardize(m=m),
#         )

mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
fit_gpytorch_mll(mll)


### Computing mean and std deviation

with torch.no_grad():
    test_x = torch.tensor(xplot_transform, dtype=torch.double).unsqueeze(-1)

    posterior = gp.posterior(test_x)
    mean = posterior.mean.numpy()
    std_dev = posterior.variance.sqrt().numpy()




#%%
output=0
plt.plot(xplot, yplot[:,output], label='True Loss Function', color='blue')
plt.plot(X_train.numpy(), Y_train[:,output].numpy(), 'o', label='Training Data', markersize=10)
plt.plot(X_transform.numpy(), Y_transform[:,output].numpy(), 'o', label='Training Data Transformed')
plt.plot(trans.untransform_x(X_transform).numpy(), trans.untransform_y(Y_transform)[:,output].numpy(), 'o', label='Training Data UN-Transformed')

# test_x = torch.tensor(xplot, dtype=torch.double).unsqueeze(-1)
# posterior = gp.posterior(test_x)
# mean = posterior.mean.detach().numpy()[:,output_dim]
# # std_dev = stand.unscale(posterior.variance.sqrt().detach()).numpy()[:,output_dim]

plt.plot(test_x.numpy(), mean[:,output],'--', label='GP Mean', color='orange')
plt.legend()
# %%


# %%
