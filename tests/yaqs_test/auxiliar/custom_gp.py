from gpytorch.models import ExactGP
from botorch.models.gpytorch import GPyTorchModel
from gpytorch.likelihoods import MultitaskGaussianLikelihood, GaussianLikelihood

from gpytorch.kernels import ScaleKernel, RBFKernel, RBFKernelGrad, AdditiveKernel
import gpytorch

import numpy as np

import torch

class GPModelWithDerivatives(ExactGP, GPyTorchModel):
    def __init__(self, train_X, train_Y, train_Yvar=None):
        d = train_X.shape[-1]

        self._num_outputs = 1 + d  # Store the number of outputs (1 for the function value and d for the derivatives)

        likelihood = MultitaskGaussianLikelihood(num_tasks=1 + d)
        super().__init__(train_X, train_Y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGrad()

        self.base_kernel = RBFKernelGrad(ard_num_dims=d)


        self.covar_module = ScaleKernel(self.base_kernel)

        self.train_Yvar = train_Yvar


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

        covar_x = covar_x + noise_mat
        
        return gpytorch.distributions.MultitaskMultivariateNormal(mean_x, covar_x)
 



class GPModel(ExactGP, GPyTorchModel):
    def __init__(self, train_X, train_Y, train_Yvar=None):

        d = train_X.shape[-1]

        self._num_outputs = 1  # Store the number of outputs    

        likelihood = GaussianLikelihood(num_tasks=1)
        super().__init__(train_X, train_Y, likelihood)

        self.mean_module = gpytorch.means.ConstantMean()

        self.base_kernel = AdditiveKernel(
            *[RBFKernel(active_dims=[i]) for i in range(d)]
        )

        self.covar_module = ScaleKernel(self.base_kernel)

        self.train_Yvar = train_Yvar


    @property
    def num_outputs(self):
        return self._num_outputs  # Return the stored number of outputs


    def forward(self, x):

        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        l_shape = len(mean_x.shape)

        n_samp=mean_x.shape[-1]

        flat_noise = torch.tensor(np.tile(self.train_Yvar.mean(axis=0), (1, n_samp))[0])

        noise_mat = torch.diag(flat_noise)

        if l_shape == 2: 
            n_batch = mean_x.shape[0]
            noise_mat = noise_mat.unsqueeze(0).expand(n_batch, -1, -1)


        covar_x = covar_x + noise_mat
        
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)