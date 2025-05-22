
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
from auxiliar.plot import plot_model
import sys
import os

# args = sys.argv[1:]

args=["GPModelWithDerivatives", "UCB", 3]

model_name = args[0]

acq_name = args[1]

d_max= int(args[2])




#%%

folder = f"results/sweap_{model_name}_{acq_name}_test/"

if not os.path.exists(folder):
    os.makedirs(folder)

file_name=folder+"time_error_vs_d.txt"


print_to_file = True
if print_to_file:
    with open(file_name, "w") as file:
        file.write("#d  min_error              avg_error                max_error                n_iter          \n")





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

    if print_to_file:
        with open(x_opt_file_name, "w") as file:
            file.write("# " + "  ".join([f"x{i+1}_opt" for i in range(d)]) + "\n")
            np.savetxt(file, x_opt, newline=" ", fmt="%.6f")
            file.write("\n")


    if print_to_file:

        with open(x_history_file_name, "w") as file:
            file.write("#iter  loss  " + "  ".join([f"x{i+1}" for i in range(d)]) + "\n")


    n_init = 10*d
    iter_max = 10
    inner_iter_max = 40
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



    outer_n_convergence=0
    outer_max_n_convergence=2

    inner_n_convergence=0
    inner_max_n_convergence=5


    threshhold = 1e-2




    f=0 ## Number of function evaluations


    X_train = bounds[0] + (bounds[1] - bounds[0]) * torch.rand(1, d, dtype=torch.double)
    Y_train = torch.zeros(1, m, dtype=torch.double)
    Y_std = torch.zeros(1, m, dtype=torch.double)



    loss, grad, std_loss, std_grad = loss_function(X_train[-1].numpy())
    f+=1

    with open(x_history_file_name, "a") as file:
        file.write(f"{f}    {loss}  " + "  ".join([f"{X_train[-1,j].item():.6f}" for j in range(d)]) + "\n")




    Y_train[0,0]= torch.tensor(loss, dtype=torch.double)
    if m > 1:
        Y_train[0,1:]= torch.tensor(grad, dtype=torch.double)

    Y_std[0,0]= torch.tensor(std_loss, dtype=torch.double)
    if m > 1:
        Y_std[0,1:]= torch.tensor(std_grad, dtype=torch.double)


    x_history.append(X_train.numpy()[-1].copy())
    loss_history.append(loss)


    X_old=X_train[-1].clone()
    


    for j in range(iter_max): ## Goes through the full optimization steps
        for k in range(d): ## Goes through all dimensions

            k_init = 5
            X_k_train = torch.tensor(np.linspace(bounds[0, k], bounds[1, k], k_init), dtype=torch.double).unsqueeze(-1)

            Y_k_train = torch.zeros(k_init, 2, dtype=torch.double)

            Y_k_std = torch.zeros(k_init, 2, dtype=torch.double)

            for i in range(k_init):

                X_train[-1, k] = X_k_train[i, 0]

                loss, grad, std_loss, std_grad = loss_function(X_train[-1].numpy())
                f+=1

                Y_k_train[i]= torch.tensor(np.array(np.append(loss,grad[k])), dtype=torch.double)

                Y_k_std[i]= torch.tensor(np.array(np.append(std_loss,std_grad[k])), dtype=torch.double)     

                x_history.append(X_train[-1].numpy().copy())
                loss_history.append(loss)

                if print_to_file:
                    with open(x_history_file_name, "a") as file:
                        file.write(f"{f}    {loss}  " + "  ".join([f"{X_train[-1,j].item():.6f}" for j in range(d)]) + "\n")
            

            X_k_old = X_k_train[-1].clone()

            for i in range(inner_iter_max):

                trans=derivative_transform(X_k_train, Y_k_train)

                X_transform, Y_transform = trans.transform(X_k_train, Y_k_train)
                Y_std_transform = trans.scale_std(Y_k_std)

                Y_var_transform = Y_std_transform ** 2


                if model_name == "GPModelWithDerivatives":

                    gp = GPModelWithDerivatives(
                        train_X=X_transform,
                        train_Y=Y_transform,
                        train_Yvar=Y_var_transform
                    ).to(X_transform)

                    scal_transf = ScalarizedPosteriorTransform(weights=torch.tensor([1.0] + [0.0], dtype=torch.double))

                else:
                    raise ValueError("model_name must be 'GPModelWithDerivatives'")
                
            
                mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
                fit_gpytorch_mll(mll)


                if acq_name == "LEI":
                    acqf = LogExpectedImprovement(model=gp, best_f=Y_transform.min(), posterior_transform=scal_transf, maximize=False)
                
                elif acq_name == "UCB":
                    acqf = UpperConfidenceBound(model=gp, beta=beta, posterior_transform=scal_transf, maximize=False)
                
                else:
                    raise ValueError("acq_name must be either 'LEI' or 'UCB'")



                bounds_trans=trans.transform_x(bounds[:, [k]])

                candidate, acq_value = optimize_acqf(
                    acqf, bounds=bounds_trans, q=1, num_restarts=num_restarts, raw_samples=raw_samples,
                )

                

                X_k_new = trans.untransform_x(candidate)
                X_train[-1, k] = X_k_new[0, 0]

                # print(k,trans.untransform_x(candidate),X_train )

                loss, grad, std_loss, std_grad =loss_function(X_train[-1].numpy())
                f+=1



                x_history.append(X_train.numpy()[-1].copy())
                loss_history.append(loss)



                if print_to_file:
                    with open(x_history_file_name, "a") as file:
                        file.write(f"{f}    {loss}  " + "  ".join([f"{X_train[-1,j].item():.6f}" for j in range(d)]) + "\n")


                Y_k_new=torch.tensor(np.array([np.append(loss,grad[k])]), dtype=torch.double)      

                Y_k_std_new=torch.tensor(np.array([np.append(std_loss,std_grad[k])]), dtype=torch.double)      

                X_k_train = torch.cat([X_k_train, X_k_new], dim=0)
                Y_k_train = torch.cat([Y_k_train, Y_k_new], dim=0)
                Y_k_std = torch.cat([Y_k_std, Y_k_std_new], dim=0)


                if max(abs(X_k_new[0]-X_k_old[0])) < threshhold:
                    inner_n_convergence+=1
                else:
                    inner_n_convergence=0

                if inner_n_convergence==inner_max_n_convergence:
                    break

                X_k_old = X_k_new.clone()


        if max(abs(X_train[-1]-X_old)) < threshhold:
            outer_n_convergence+=1
        else:
            outer_n_convergence=0

        if outer_n_convergence==outer_max_n_convergence:
            break

        X_old = X_train[-1].clone()





    max_diff=max(abs(x_history[-1] - x_opt))

    min_diff=min(abs(x_history[-1] - x_opt))

    avg_diff=np.mean(abs(x_history[-1] - x_opt))

    if print_to_file:
        with open(file_name, "a") as file:
            file.write(f"{d}    {min_diff}   {avg_diff}   {max_diff}   {f} \n")



    print(f"Finish for d={d}!!!")

#%%


# Plot x_history and x_opt
x_history_np = np.array(x_history)

plt.figure(figsize=(10, 6))
for i in range(d):
    plt.plot(x_history_np[:, i],'o-', label=f"x{i+1} history")
    plt.axhline(y=x_opt[i], color=f"C{i}", linestyle="--", label=f"x{i+1}_opt")

plt.xlabel("Iteration")
plt.ylabel("x values")
plt.title("Convergence of x_history to x_opt")
plt.legend()
plt.grid()
plt.show()





# %%
xplot=np.linspace(bounds[0, 1], bounds[1, 1], 100)
yplot=np.array([loss_function(np.array([X_train[0,0],x]))[0] for x in xplot])

output_dim=0

with torch.no_grad():
        test_x_trans = trans.transform_x(torch.tensor(xplot, dtype=torch.double).unsqueeze(-1))
        posterior = gp.posterior(test_x_trans)
        mean = trans.untransform_y(posterior.mean).numpy()[:,output_dim]
        std_dev = trans.unscale_std(posterior.variance.sqrt()).numpy()[:,output_dim]



plot_model(mean, std_dev, xplot, yplot, X_k_train, Y_k_train, iter=0, output_dim=output_dim)
# %%

X_train
# %%
