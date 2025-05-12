
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
x_opt=0.1
c=0.5
std=0.001
m_std=300
der_std=0.01
m_der_std=200

def loss_function(x, x_opt=x_opt, c=c, std=std, m_std=m_std, der_std=der_std, m_der_std=m_der_std):

    std_f = std * (1 + m_std * (x - x_opt) ** 2)
    std_grad = der_std * (1 + m_der_std * (x - x_opt) ** 2)

    f= c*(x - x_opt) ** 2 + std_f*np.random.randn(*x.shape)
    grad = 2 * c * (x - x_opt) + std_grad*np.random.randn(*x.shape)
    return f, grad, std_f, std_grad


#%%
output=0
xplot = np.linspace(0, 0.2, 100)

# Compute loss_function 100 times and calculate mean and std
num_samples = 400
y_samples = np.zeros((num_samples, len(xplot)))

for i in range(num_samples):
    y_samples[i, :] = loss_function(xplot)[output]

y_mean = y_samples.mean(axis=0)
y_std = y_samples.std(axis=0)

# Plot mean and standard deviation
plt.plot(xplot, y_mean, label='Mean of Loss Function', color='green')
plt.fill_between(
    xplot,
    y_mean - y_std,
    y_mean + y_std,
    color='green',
    alpha=0.2,
    label='Standard Deviation (±σ)'
)
plt.legend()
plt.show()
#%%
print(min(y_std),max(y_std))


#%%
def plot_model(mean, std_dev, xplot, yplot, train_X, train_Y, iter=0, output_dim=0):

    plt.figure(figsize=(8, 6))
    plt.plot(xplot, yplot, label='True Loss Function', color='blue')
    plt.plot(xplot, mean, label='GP Mean', color='orange')
    plt.fill_between(
        xplot,
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

    def scale_std(self, y):
        y = self.scale(y, self.y_range)
        return y

    def unscale_std(self, y):
        y = self.unscale(y, self.y_range)
        return y
    


#%%
from gpytorch.models import ExactGP
from botorch.models.gpytorch import GPyTorchModel
from botorch.acquisition.objective import ScalarizedPosteriorTransform
from gpytorch.likelihoods import MultitaskGaussianLikelihood



class GPModelWithDerivatives(ExactGP, GPyTorchModel):
    def __init__(self, train_X, train_Y, train_Yvar=None):
        d = train_X.shape[-1]
        # likelihood = GaussianLikelihood()
        likelihood = MultitaskGaussianLikelihood(num_tasks=1 + d)
        super().__init__(train_X, train_Y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGrad()
        self.base_kernel = gpytorch.kernels.RBFKernelGrad(ard_num_dims=d)
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

        self.train_Yvar = train_Yvar

        # self.base_kernel.lengthscale = lengthscale

    @property
    def num_outputs(self):
        return self._num_outputs  # Return the stored number of outputs


    def forward(self, x):

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        l_shape = len(mean_x.shape)

        n_samp=mean_x.shape[-2]

        flat_noise = torch.tensor(np.tile(self.train_Yvar.mean(axis=0), (1, n_samp))[0])

        noise_mat = torch.diag(flat_noise)

        if l_shape == 3: 
            n_batch = mean_x.shape[0]
            noise_mat = noise_mat.unsqueeze(0).expand(n_batch, -1, -1)

        # print("Noise shape: ", noise_mat.shape, "Covar shape: ", covar_x.shape, "Mean shape: ", mean_x.shape)

        covar_x = covar_x + noise_mat
        
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)



#%% 

n_init = 4
n_iter = 10
d=1  # number of dimensions
m=2 # number of outputs
bounds_list = [[0,0.3]]

num_restarts = 10
raw_samples = 20

acq_name="LEI"
beta=20
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

Y_std = torch.zeros(n_init, m, dtype=torch.double)

for i in range(n_init):
    loss, grad, std_loss, std_grad = loss_function(X_train[i].numpy())

    Y_train[i,0]= torch.tensor(loss, dtype=torch.double)
    if m > 1:
        Y_train[i,1]= torch.tensor(grad, dtype=torch.double)

    Y_std[i,0]= torch.tensor(std_loss, dtype=torch.double)
    if m > 1:
        Y_std[i,1]= torch.tensor(std_grad, dtype=torch.double)


    x_history.append(X_train[i].numpy())
    loss_history.append(loss)




#%%
# Plot the model with the standard deviation

for i in range(n_iter):

    trans=derivative_transform(X_train, Y_train)

    X_transform, Y_transform = trans.transform(X_train, Y_train)
    Y_std_transform = trans.scale_std(Y_std)

    Y_var_transform = Y_std_transform ** 2


    gp = GPModelWithDerivatives(
        train_X=X_transform,
        train_Y=Y_transform,
        train_Yvar=Y_var_transform,
        # lengthscale=l,
        # likelihood=gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=m),
        # input_transform=Normalize(d=d),
        # outcome_transform=Standardize(m=m)
    ).to(X_transform)


    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)


    ### Computing mean and std deviation
    output_dim=0
    with torch.no_grad():
        test_x_trans = trans.transform_x(torch.tensor(xplot, dtype=torch.double).unsqueeze(-1))
        posterior = gp.posterior(test_x_trans)
        mean = trans.untransform_y(posterior.mean).numpy()[:,output_dim]
        std_dev = trans.unscale_std(posterior.variance.sqrt()).numpy()[:,output_dim]

    

    plot_model(mean, std_dev, xplot, yplot, X_train, Y_train, iter=i)

    scal_transf = ScalarizedPosteriorTransform(weights=torch.tensor([1.0, 0.0], dtype=torch.double))

    if acq_name == "LEI":
        acqf = LogExpectedImprovement(model=gp, best_f=Y_transform.min(), posterior_transform=scal_transf, maximize=False)
    
    if acq_name == "UCB":
        acqf = UpperConfidenceBound(model=gp, beta=beta, posterior_transform=scal_transf, maximize=False)



    candidate, acq_value = optimize_acqf(
        acqf, bounds=bounds, q=1, num_restarts=num_restarts, raw_samples=raw_samples,
    )

    # Plot the acquisition function
    acq_values = acqf(test_x_trans.unsqueeze(-2)).detach().numpy()
    plt.figure(figsize=(8, 6))
    plt.plot(xplot, acq_values, label='Acquisition Function', color='purple')
    plt.axvline(x=trans.untransform_x(candidate).numpy(), color='red', linestyle='--', label='Next Candidate')
    plt.title('Acquisition Function')
    plt.xlabel('x')
    plt.ylabel('Acquisition Value')
    plt.legend()
    plt.savefig(f'plot_gp_with_der/acquisition_function_{i}.png')
    plt.close()


    X_new = trans.untransform_x(candidate)
    loss, grad, std_loss, std_grad =loss_function(X_new.numpy()[0])

    x_history.append(X_new.numpy()[0])
    loss_history.append(loss)

                       
    Y_new=torch.tensor(np.array([loss,grad]), dtype=torch.double).T       

    Y_std_new=torch.tensor(np.array([std_loss,std_grad]), dtype=torch.double).T          

    X_train = torch.cat([X_train, X_new], dim=0)
    Y_train = torch.cat([Y_train, Y_new], dim=0)
    Y_std = torch.cat([Y_std, Y_std_new], dim=0)


    
plt.plot(np.array(x_history)[:,0], '-', label='history')
plt.axhline(y=x_opt, color='r', linestyle='--')
plt.legend()
plt.savefig('plot_gp_with_der/x_history.png')



# %%
x_history


# %%

# %%
from gpytorch.likelihoods import Likelihood
class GPModelWithDerivatives(ExactGP, GPyTorchModel):
    def __init__(self, train_X, train_Y, train_Yvar=None):
        d = train_X.shape[-1]
        # likelihood = GaussianLikelihood()
        likelihood = MultitaskGaussianLikelihood(num_tasks=1 + d)
        super().__init__(train_X, train_Y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGrad()
        self.base_kernel = gpytorch.kernels.RBFKernelGrad(ard_num_dims=d)
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

        self.train_Yvar = train_Yvar

        # self.base_kernel.lengthscale = lengthscale

    @property
    def num_outputs(self):
        return self._num_outputs  # Return the stored number of outputs


    def forward(self, x):
        # if self.input_transform is not None:
        #     x = self.input_transform(x)
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        # l_shape=len(covar_x.shape)

        n_samp=mean_x.shape[-2]

        # if l_shape == 3:
        #     batch_size=covar_x.shape[-2]

        flat_noise=torch.tensor(np.tile(self.train_Yvar.mean(axis=0), (1, n_samp))[0])

        covar_x = covar_x.add_diag(flat_noise)
        
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)


#%%


n_init = 10
bounds_list = [[0,0.2]]


xplot = np.linspace(bounds_list[0][0], bounds_list[0][1], 100)
xplot_transform = np.linspace(0, 1, 100)

yplot = np.array(loss_function(xplot)).T


bounds = torch.tensor(bounds_list,dtype=torch.double).T

X_train = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n_init, d, dtype=torch.double)


Y_train = torch.zeros(n_init, m, dtype=torch.double)

Y_std = torch.zeros(n_init, m, dtype=torch.double)

for i in range(n_init):
    loss, grad, std_loss, std_grad = loss_function(X_train[i].numpy())

    Y_train[i,0]= torch.tensor(loss, dtype=torch.double)
    if m > 1:
        Y_train[i,1]= torch.tensor(grad, dtype=torch.double)

    Y_std[i,0]= torch.tensor(std_loss, dtype=torch.double)
    if m > 1:
        Y_std[i,1]= torch.tensor(std_grad, dtype=torch.double)




trans=derivative_transform(X_train, Y_train)

X_transform, Y_transform = trans.transform(X_train, Y_train)
Y_std_transform = trans.scale_std(Y_std)

Y_var_transform = Y_std_transform ** 2
print(type(Y_var_transform))


# X_train = X_transform
# Y_train = Y_transform


gp = GPModelWithDerivatives(
    train_X=X_transform,
    train_Y=Y_transform,
    train_Yvar=Y_var_transform,
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
    test_x = torch.tensor(xplot, dtype=torch.double).unsqueeze(-1)

    posterior = gp.posterior(trans.transform_x(test_x))
    mean = trans.untransform_y(posterior.mean).numpy()
    std_dev = trans.unscale_std(posterior.variance.sqrt()).numpy()




#%%
output=1
plt.plot(xplot, yplot[:,output], label='True Loss Function', color='blue')
plt.plot(X_train.numpy(), Y_train[:,output].numpy(), 'o', label='Training Data', markersize=10)
# plt.plot(X_transform.numpy(), Y_transform[:,output].numpy(), 'o', label='Training Data Transformed')
plt.plot(trans.untransform_x(X_transform).numpy(), trans.untransform_y(Y_transform)[:,output].numpy(), 'o', label='Training Data UN-Transformed')

plt.plot(test_x.numpy(), mean[:,output],'--', label='GP Mean', color='orange')
plt.fill_between(
        xplot,
        mean[:,output] - 1 * std_dev[:,output],
        mean[:,output] + 1 * std_dev[:,output],
        color='orange',
        alpha=0.2,
        label='Confidence Interval (±σ)'
    )

plt.legend()
plt.show()
# %%
std_dev.mean(axis=0)

# %%
np.tile(Y_std.mean(axis=0), (1, 10))[0].shape

# %%
noise_mat = torch.tensor(np.array([[1,0],[0,4]]))
noise_mat.unsqueeze(0).expand(3, -1, -1).shape
# %%
test_x_trans.unsqueeze(-2).shape
# %%
