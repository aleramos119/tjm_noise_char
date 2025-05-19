
#%%
import numpy as np
import matplotlib.pyplot as plt

import torch
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood


from botorch.acquisition import LogExpectedImprovement,UpperConfidenceBound

import gpytorch
from botorch.optim import optimize_acqf


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
from gpytorch.likelihoods import MultitaskGaussianLikelihood, GaussianLikelihood
import time
from gpytorch.kernels import ScaleKernel, RBFKernelGrad, AdditiveKernel, RBFKernel


class GPModel(ExactGP, GPyTorchModel):
    def __init__(self, train_X, train_Y, train_Yvar=None):
        d = train_X.shape[-1]
        # likelihood = GaussianLikelihood()
        likelihood = GaussianLikelihood(num_tasks=1)
        super().__init__(train_X, train_Y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        # self.base_kernel = RBFKernel(ard_num_dims=d)

        self.base_kernel = AdditiveKernel(
            *[RBFKernel(active_dims=[i]) for i in range(d)]
        )

        # kernels = [RBFKernelGrad(active_dims=[i]) for i in range(d)]
        # self.base_kernel = kernels[0]
        # for k in kernels[1:]:
        #     self.base_kernel += k

        self.covar_module = ScaleKernel(self.base_kernel)

        self.train_Yvar = train_Yvar

        # self.base_kernel.lengthscale = lengthscale

    @property
    def num_outputs(self):
        return self._num_outputs  # Return the stored number of outputs


    def forward(self, x):

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        l_shape = len(mean_x.shape)

        # print("Shape",mean_x.shape )

        n_samp=mean_x.shape[-1]

        flat_noise = torch.tensor(np.tile(self.train_Yvar.mean(axis=0), (1, n_samp))[0])

        noise_mat = torch.diag(flat_noise)

        if l_shape == 2: 
            n_batch = mean_x.shape[0]
            noise_mat = noise_mat.unsqueeze(0).expand(n_batch, -1, -1)

        # print("Noise shape: ", noise_mat.shape, "Covar shape: ", covar_x.shape, "Mean shape: ", mean_x.shape)

        # print("covar_x.shape:", covar_x.shape)
        # print("noise_mat.shape:", noise_mat.shape)

        covar_x = covar_x + noise_mat
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


#%%

acq_name="UCB"

file_name=f"gp_nograd_time_test/{acq_name}_time_error_vs_d.txt"


with open(file_name, "w") as file:
    file.write("#d  min_error              avg_error                max_error                n_iter            time_first_iter(sec)\n")


#%%
    
for d_for in range(1,100):

    d=d_for

    x_opt=0.01+(0.4-0.01)*np.random.rand(d) ##The center of the 
    c=0.5+(4-0.5)*np.random.rand(d)
    std=0.001
    m_std=300
    der_std=0.01+(0.1-0.01)*np.random.rand(d)
    m_der_std=200

    def loss_function(x, x_opt=x_opt, c=c, std=std, m_std=m_std, der_std=der_std, m_der_std=m_der_std):

        std_f = std * (1 + m_std * np.sum((x - x_opt) ** 2))
        std_grad =  der_std + m_der_std * der_std * (x - x_opt) ** 2

        f= np.sum(c*(x - x_opt) ** 2) + std_f*np.random.randn()
        grad = 2 * c * (x - x_opt) + std_grad*np.random.randn(*x.shape)
        return f, grad, std_f, std_grad


    n_init = 30
    iter_max = 300 
    m=d+1 # number of outputs
    bounds_list = [[-0.1, 0.5] for _ in range(d)]

    num_restarts = 20
    raw_samples = 40


    beta=20
    noise=1e-0



    loss_history = []
    x_history = []

    diff_list = []
    time_list =[]

    bounds = torch.tensor(bounds_list,dtype=torch.double).T

    X_train = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(n_init, d, dtype=torch.double)

    Y_train = torch.zeros(n_init, m, dtype=torch.double)

    Y_std = torch.zeros(n_init, m, dtype=torch.double)

    for i in range(n_init):
        loss, grad, std_loss, std_grad = loss_function(X_train[i].numpy())

        Y_train[i,0]= torch.tensor(loss, dtype=torch.double)
        if m > 1:
            Y_train[i,1:]= torch.tensor(grad, dtype=torch.double)

        Y_std[i,0]= torch.tensor(std_loss, dtype=torch.double)
        if m > 1:
            Y_std[i,1:]= torch.tensor(std_grad, dtype=torch.double)


        x_history.append(X_train[i].numpy())
        loss_history.append(loss)



    n_convergence=0
    max_n_convergence=20
    threshhold = 1e-3

    for i in range(iter_max):


        trans=derivative_transform(X_train, Y_train)

        X_transform, Y_transform = trans.transform(X_train, Y_train)
        Y_std_transform = trans.scale_std(Y_std)

        Y_var_transform = Y_std_transform ** 2

        if i == 0:
            start_time = time.time()


        gp = GPModel(
            train_X=X_transform,
            train_Y=Y_transform[:,0],
            train_Yvar=Y_var_transform[:,0]
        ).to(X_transform)


        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)


        scal_transf = ScalarizedPosteriorTransform(weights=torch.tensor([1.0], dtype=torch.double))

        if acq_name == "LEI":
            acqf = LogExpectedImprovement(model=gp, best_f=Y_transform.min(), posterior_transform=scal_transf, maximize=False)
        
        if acq_name == "UCB":
            acqf = UpperConfidenceBound(model=gp, beta=beta, posterior_transform=scal_transf, maximize=False)

        bounds_trans=trans.transform_x(bounds)

        candidate, acq_value = optimize_acqf(
            acqf, bounds=bounds_trans, q=1, num_restarts=num_restarts, raw_samples=raw_samples,
        )


        if i == 0:
            end_time = time.time()

        X_new = trans.untransform_x(candidate)
        loss, grad, std_loss, std_grad =loss_function(X_new.numpy()[0])

        x_history.append(X_new.numpy()[0])
        loss_history.append(loss)
            
        Y_new=torch.tensor(np.array([np.append(loss,grad)]), dtype=torch.double)      

        Y_std_new=torch.tensor(np.array([np.append(std_loss,std_grad)]), dtype=torch.double)      

        X_train = torch.cat([X_train, X_new], dim=0)
        Y_train = torch.cat([Y_train, Y_new], dim=0)
        Y_std = torch.cat([Y_std, Y_std_new], dim=0)

        # print(f"Error: {max(abs(X_new[0]-X_old[0]))}")

        if max(abs(X_new[0]-x_opt)) < threshhold:
            # print(f"Cnverved!! Iter:{i} N_convergence: {n_convergence}")
            n_convergence+=1
        else:
            n_convergence=0

        if n_convergence==max_n_convergence:
            break



    sec_first_iter = (end_time - start_time)

    max_diff=max(abs(x_history[-1] - x_opt))

    min_diff=min(abs(x_history[-1] - x_opt))

    avg_diff=np.mean(abs(x_history[-1] - x_opt))

    with open(file_name, "a") as file:
        file.write(f"{d}    {min_diff}   {avg_diff}   {max_diff}   {i}    {sec_first_iter}\n")

# %%
Y_transform[:,0].shape
# %%
