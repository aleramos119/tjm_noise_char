
#%%
import numpy as np
import matplotlib.pyplot as plt

import torch
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood


from botorch.acquisition import LogExpectedImprovement,UpperConfidenceBound

from botorch.optim import optimize_acqf


import time

from botorch.acquisition.objective import ScalarizedPosteriorTransform

from auxiliar.transform import derivative_transform
from auxiliar.custom_gp import GPModel, GPModelWithDerivatives
import sys
import os

args = sys.argv[1:]

model_name = args[0]

acq_name = args[1]

d_max= int(args[2])

#%%

folder = f"results/{model_name}_{acq_name}_test/"

if not os.path.exists(folder):
    os.makedirs(folder)

file_name=folder+"time_error_vs_d.txt"


print_to_file = True
if print_to_file:
    with open(file_name, "w") as file:
        file.write("#d  min_error              avg_error                max_error                n_iter            time_first_iter(sec)\n")




#%%
    
for d_for in range(1,d_max+1):

    d=d_for

    print(f"Running for d={d}....")

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


    x_opt_file_name=folder+f"x_opt_values_{d}.txt"

    x_history_file_name=folder+f"x_history_{d}.txt"

    with open(x_opt_file_name, "w") as file:
        file.write("# " + "  ".join([f"x{i+1}_opt" for i in range(d)]) + "\n")
        np.savetxt(file, x_opt, newline=" ", fmt="%.6f")
        file.write("\n")

    with open(x_history_file_name, "w") as file:
        file.write("#iter  loss  " + "  ".join([f"x{i+1}" for i in range(d)]) + "\n")


    n_init = 10*d
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

        with open(x_history_file_name, "a") as file:
            file.write(f"{i}    {loss}  " + "  ".join([f"{X_train[i,j].item():.6f}" for j in range(d)]) + "\n")



    n_convergence=0
    max_n_convergence=20
    threshhold = 1e-2

    for i in range(iter_max):


        trans=derivative_transform(X_train, Y_train)

        X_transform, Y_transform = trans.transform(X_train, Y_train)
        Y_std_transform = trans.scale_std(Y_std)

        Y_var_transform = Y_std_transform ** 2

        if i == 0:
            start_time = time.time()

        if model_name == "GPModel":

            gp = GPModel(
                train_X=X_transform,
                train_Y=Y_transform[:,0],
                train_Yvar=Y_var_transform[:,0]
            ).to(X_transform)

            scal_transf = ScalarizedPosteriorTransform(weights=torch.tensor([1.0], dtype=torch.double))



        elif model_name == "GPModelWithDerivatives":
            gp = GPModelWithDerivatives(
                train_X=X_transform,
                train_Y=Y_transform,
                train_Yvar=Y_var_transform
            ).to(X_transform)

            scal_transf = ScalarizedPosteriorTransform(weights=torch.tensor([1.0] + [0.0]*d, dtype=torch.double))

        else:
            raise ValueError("model_name must be either 'GPModel' or 'GPModelWithDerivatives'")



        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fit_gpytorch_mll(mll)


        if acq_name == "LEI":
            acqf = LogExpectedImprovement(model=gp, best_f=Y_transform.min(), posterior_transform=scal_transf, maximize=False)
        
        elif acq_name == "UCB":
            acqf = UpperConfidenceBound(model=gp, beta=beta, posterior_transform=scal_transf, maximize=False)
        
        else:
            raise ValueError("acq_name must be either 'LEI' or 'UCB'")
        

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
    
        with open(x_history_file_name, "a") as file:
            file.write(f"{i+n_init}    {loss}  " + "  ".join([f"{X_train[i,j].item():.6f}" for j in range(d)]) + "\n")


        Y_new=torch.tensor(np.array([np.append(loss,grad)]), dtype=torch.double)      

        Y_std_new=torch.tensor(np.array([np.append(std_loss,std_grad)]), dtype=torch.double)      

        X_train = torch.cat([X_train, X_new], dim=0)
        Y_train = torch.cat([Y_train, Y_new], dim=0)
        Y_std = torch.cat([Y_std, Y_std_new], dim=0)


        if max(abs(X_new[0]-x_opt)) < threshhold:
            n_convergence+=1
        else:
            n_convergence=0

        if n_convergence==max_n_convergence:
            break



    sec_first_iter = (end_time - start_time)

    max_diff=max(abs(x_history[-1] - x_opt))

    min_diff=min(abs(x_history[-1] - x_opt))

    avg_diff=np.mean(abs(x_history[-1] - x_opt))

    if print_to_file:
        with open(file_name, "a") as file:
            file.write(f"{d}    {min_diff}   {avg_diff}   {max_diff}   {i}    {sec_first_iter}\n")


    
    print(f"Finish for d={d}!!!")


# %%
