
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
def plot_model(gp, xplot, yplot, iter=0):
    with torch.no_grad():
        test_x = torch.tensor(xplot, dtype=torch.double).unsqueeze(-1)
        posterior = gp.posterior(test_x)
        mean = posterior.mean.numpy().flatten()
        std_dev = posterior.variance.sqrt().numpy().flatten()

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
    plt.scatter(X_train.numpy(), Y_train.numpy(), color='red', label='Training Data')
    # plt.axhline(y=0.5, color='green', linestyle='--', label='y=0.5')
    plt.legend()
    plt.title('Gaussian Process Model with Uncertainty')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.savefig(f'plot/gp_model_{iter}.png')



#%% 

n_init = 1
n_iter = 10
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

    Y_train[i,0]= torch.tensor(loss, dtype=torch.double)

    x_history.append(X_train[i].numpy())
    loss_history.append(loss)


#%%
# Plot the model with the standard deviation

for i in range(n_iter):
    gp = SingleTaskGP(
        train_X=X_train,
        train_Y=Y_train,
        input_transform=Normalize(d=d),
        outcome_transform=Standardize(m=m),
    ).to(X_train)

    gp.likelihood.noise_covar.noise = torch.tensor(0.0, dtype=torch.double)

    mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
    fit_gpytorch_mll(mll)


    plot_model(gp, xplot, yplot, iter=i)

    logNEI = LogExpectedImprovement(model=gp, best_f=Y_train.min(), maximize=False)


    candidate, acq_value = optimize_acqf(
        logNEI, bounds=bounds, q=1, num_restarts=num_restarts, raw_samples=raw_samples,
    )


    X_train = torch.cat([X_train, candidate], dim=0)
    loss, grad=loss_function(X_train[-1].numpy())
    Y_train = torch.cat([Y_train, torch.tensor([loss], dtype=torch.double)], dim=0)

    x_history.append(X_train[-1].numpy())
    loss_history.append(loss)



#%%
Y_train
# %%
plt.plot(xplot, yplot, label='Loss function')
plt.plot(X_train.numpy(), Y_train.numpy(), 'o', label='Initial data')

# %%
plt.plot(np.array(x_history)[:,0], '-', label='history')
plt.axhline(y=0.5, color='r', linestyle='--', label='y=0.5')
plt.legend()
plt.savefig('plot/x_history.png')
# %%
with torch.no_grad():
    posterior = gp.posterior(X_train)
    print("Standard deviation at training points:", posterior.variance.sqrt().numpy())
# %%
print("Noise level:", gp.likelihood.noise_covar.noise.item())

# %%
