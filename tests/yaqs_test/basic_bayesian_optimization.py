
#%%
import numpy as np
import matplotlib.pyplot as plt

import torch
from botorch.models import SingleTaskGP
from botorch.models.transforms import Normalize, Standardize
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood


from botorch.acquisition import LogExpectedImprovement


from botorch.optim import optimize_acqf

#%%

def loss_function(x):
    f= (x - 0.5) ** 2 
    grad = 2 * (x - 0.5)
    return f, grad


xplot = np.linspace(0, 1, 100)
yplot = loss_function(xplot)[0]

#%% 

n_init = 5
n_iter = 100
d=1
m=1
bounds_list = [[0,1]]

num_restarts = 5
raw_samples = 20


#%%

loss_history = []
x_history = []

bounds = torch.tensor(bounds_list,dtype=torch.double).T

X_train = torch.rand(n_init, d, dtype=torch.double)

Y_train = torch.zeros(n_init, m, dtype=torch.double)

for i in range(n_init):
    loss,grad = loss_function(X_train[i].numpy())

    Y_train[i,0]= torch.tensor(-loss, dtype=torch.double)

    x_history.append(X_train[i].numpy())
    loss_history.append(loss)


#%%
for i in range(n_iter):
    gp = SingleTaskGP(
        train_X=X_train,
        train_Y=Y_train,
        input_transform=Normalize(d=d),
        outcome_transform=Standardize(m=m),
    ).to(X_train)

    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)



    logNEI = LogExpectedImprovement(model=gp, best_f=Y_train.max())


    candidate, acq_value = optimize_acqf(
        logNEI, bounds=bounds, q=1, num_restarts=num_restarts, raw_samples=raw_samples,
    )


    X_train = torch.cat([X_train, candidate], dim=0)
    loss, grad=loss_function(X_train[-1].numpy())
    Y_train = torch.cat([Y_train, torch.tensor([-loss], dtype=torch.double)], dim=0)

    x_history.append(X_train[-1].numpy())
    loss_history.append(loss)



#%%
Y_train
# %%
plt.plot(xplot, yplot, label='Loss function')
plt.plot(X_train.numpy(), Y_train.numpy(), 'o', label='Initial data')

# %%
plt.plot(np.array(x_history)[:,0], 'o-', label='history')
# %%
np.array(x_history)[:,0]
# %%
np.array(x_history)[:,0]
# %%
np.array(loss_history)[:,0]
# %%
loss_history
# %%
