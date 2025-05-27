
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

args=["ADAM", 4]

opt_name = args[0]

d_max= int(args[1])


#%%
def ADAM_gradient_descent(f, x_copy, print_to_file = False, x_history_file_name=" ", avg_len = 100, alpha=0.01, max_iterations=200, threshhold = 1e-4, max_n_convergence = 50, tolerance=1e-8, beta1 = 0.5, beta2 = 0.999, epsilon = 1e-8):

    x=x_copy.copy()  # Make a copy of the input x to avoid modifying the original

    loss_history = []
    x_history = []
    x_avg_history = []
    update_history = []

    d = len(x)

    if print_to_file:

        with open(x_history_file_name, "w") as file:
            file.write("#iter  loss  " + "  ".join([f"x{i+1}" for i in range(d)]) + "\n")


    # Adam hyperparameters and initialization (NEW)
    m = np.array([0.0]*d)  # First moment (for [gamma_rel, gamma_deph])
    v = np.array([0.0]*d)  # Second moment (for [gamma_rel, gamma_deph])

    x_avg_old = x.copy()  # Store the old x for convergence check
    x_avg_history.append(x_avg_old.copy())

    n_convergence = 0  # Counter for convergence checks

    for i in range(max_iterations):
        # Calculate loss and gradients (unchanged)
        loss, grad, std_loss, std_grad = f(x)

        loss_history.append(loss)
        x_history.append(x.copy())
        

        if len(x_history) < avg_len:
            x_avg = np.mean(x_history, axis=0)
        else:
            x_avg = np.mean(x_history[avg_len-1:], axis=0)

        x_avg_history.append(x_avg.copy())



        if print_to_file:
            with open(x_history_file_name, "a") as file:
                file.write(f"{i}    {loss}  " + "  ".join([f"{x_avg[j]:.6f}" for j in range(d)]) + "\n")


        if abs(loss) < tolerance:
            print(f"Loss converged after {i} iterations. Loss={loss}, tolerance={tolerance}")
            break

    
        # Adam update steps (NEW)
        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad ** 2)

        beta1_t = beta1 ** (i + 1)
        beta2_t = beta2 ** (i + 1)


        m_hat = m / (1 - beta1_t)
        v_hat = v / (1 - beta2_t)





        update = alpha * m_hat / (np.sqrt(v_hat) + epsilon)

        update_history.append(update.copy())

        # Update simulation parameters with Adam update (NEW)
        x -= update    

        if max(abs(x_avg-x_avg_old)) < threshhold:
            n_convergence+=1
        else:
            n_convergence=0

        if n_convergence==max_n_convergence:
            print(f"Converged after {i} iterations.")
            break

        x_avg_old = x_avg.copy()  # Update old x for next iteration




    return loss_history, x_history, x_avg_history, update_history


def line_search(f, x, loss, p, grad, alpha=1.0, beta=0.8, c=1e-4):
    """
    Backtracking line search to find a suitable step size.
    """
    loss_history = []
    x_history = []
    grad_history = []

    # loss, grad, _, _ = f(x)
    # loss_history.append(loss)
    # x_history.append(x.copy())
    # grad_history.append(grad.copy())

    # print(f"Line search: x={x}, alpha={alpha}")

    iter=0
    while True and iter < 50:
        x_new = x + alpha * p
        loss_new, grad_new, _, _ = f(x_new)

        loss_history.append(loss_new)
        x_history.append(x_new.copy())
        grad_history.append(grad_new.copy())

        # print(f"Line search: x_new={x_new}, alpha={alpha}")


        if loss_new <= loss + c * alpha * grad.dot(p):
            break
        alpha *= beta
        iter+=1

    return alpha, loss_history, x_history, grad_history



def Secant_Penalized_BFGS(f, x_copy, print_to_file = False, x_history_file_name=" ", avg_len = 100, alpha=0.01, max_iterations=200, threshhold = 1e-4, max_n_convergence = 50, tolerance=1e-8, Ns=10e8, N0=10e-10):


    x = x_copy.copy()  # Make a copy of the input x to avoid modifying the original


    loss_history = []
    x_history = []
    x_avg_history = []
    update_history = []

    d = len(x)


    if print_to_file:

        with open(x_history_file_name, "w") as file:
            file.write("#iter  loss  " + "  ".join([f"x{i+1}" for i in range(d)]) + "\n")


    # Initial inverse Hessian approximation
    H_inv = np.eye(d)

    I = np.eye(d)


    x_old = x.copy()  # Store the old x for convergence check

    # Calculate first loss and gradients
    loss, grad_old, std_loss, std_grad = f(x_old)



    loss_history.append(loss)
    x_history.append(x_old.copy())

    

    if len(x_history) < avg_len:
        x_avg = np.mean(x_history, axis=0)
    else:
        x_avg = np.mean(x_history[-avg_len:], axis=0)

    x_avg_history.append(x_avg.copy())

    x_avg_old = x_avg.copy()  # Store the old x for convergence check
    n_convergence = 0  # Counter for convergence checks

    if print_to_file:
        with open(x_history_file_name, "a") as file:
            file.write(f"{0}    {loss}  " + "  ".join([f"{x_avg[j]:.6f}" for j in range(d)]) + "\n")



    for i in range(max_iterations-1):

        # Compute search direction
        p = -H_inv.dot(grad_old)


        
        alpha, line_loss_history, line_x_history, line_grad_history = line_search(f, x_old, loss, p, grad_old)

        loss= line_loss_history[-1]
        x_new = line_x_history[-1].copy()
        grad_new = line_grad_history[-1].copy()


        loss_history.extend(line_loss_history)
        x_history.extend(line_x_history)

        # update= alpha * p


        # # Update parameters
        # x_new = x_old + update


        # # Update simulation parameters
        # loss, grad_new, _, _ = f(x_new)


        # loss_history.append(loss)
        # x_history.append(x_new.copy())

        # update_history.append(update.copy())
        

        # for k in range(1,len(x_history)+1):
        #     if k < avg_len:
        #         x_avg = np.mean(x_history[:k], axis=0)
        #     else:
        #         x_avg = np.mean(x_history[k-avg_len:k], axis=0)

        #     x_avg_history.append(x_avg.copy())

  


        if print_to_file:
            with open(x_history_file_name, "a") as file:
                file.write(f"{i+1}    {loss}  " + "  ".join([f"{x_avg[j]:.6f}" for j in range(d)]) + "\n")
        
        # print(f"Iteration {i+1}: x_new={x_new}, alpha={alpha}")

        # if abs(loss) < tolerance:
        #     print(f"Loss Converged after {i} iterations.x_new={x_new}, loss={loss_history}, tolerance={tolerance}")
        #     break


        # Compute differences
        s = alpha * p
        y = grad_new - grad_old

        
        prod=y.dot(s)

        if prod != 0:
            
            # Update inverse Hessian approximation using BFGS formula

            rho = 1.0 / prod
            H_inv = (I - rho * np.outer(s, y)).dot(H_inv).dot(I - rho * np.outer(y, s)) + rho * np.outer(s, s)

        # # beta=(Ns/np.linalg.norm(std_grad))*np.linalg.norm(s) + N0
        # beta=1e+10

        # if prod > -1/beta:

        #     gamma=1.0/(prod+1/beta)

        #     omega=1.0/(prod+2/beta)

        #     H_inv = (I - omega * np.outer(s, y)).dot(H_inv).dot(I - omega * np.outer(y, s))    +     omega * (gamma/omega  + (gamma-omega)*y.dot(H_inv.dot(y))) * np.outer(s, s)

        #     # H_inv = np.eye(d)



        x_old = x_new.copy()  # Update old x for next iteration
        grad_old = grad_new.copy()  # Update old gradient for next iteration


        if len(x_history) < avg_len:
            x_avg = np.mean(x_history, axis=0)
        else:
            x_avg = np.mean(x_history[-avg_len:], axis=0)


        if max(abs(x_avg-x_avg_old)) < threshhold:
            n_convergence+=1
            print(f"Convergence check: {n_convergence} iterations with max change {max(abs(x_avg-x_avg_old))} < {threshhold}")
        else:
            n_convergence=0

        if n_convergence==max_n_convergence:
            print(f"Converged after {i} iterations.")
            break

        x_avg_old = x_avg.copy()  # Update old x for next iteration



    x_avg_history=[]

    for k in range(1, len(x_history) + 1):
        if k < avg_len:
            x_avg = np.mean(x_history[:k], axis=0)
        else:
            x_avg = np.mean(x_history[k - avg_len:k], axis=0)

        x_avg_history.append(x_avg.copy())


    return loss_history, x_history, x_avg_history, update_history


#%%

folder = f"results/{opt_name}_test/"

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

    # std=0
    # m_std=0
    # der_std=0*np.random.rand(d)
    # m_der_std=0

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



    loss_history = []
    x_history = []

    diff_list = []
    time_list =[]

    x0= 0.01+(0.4-0.01)*np.random.rand(d)
    
    if opt_name == "ADAM":
        loss_history, x_history, x_avg_history, update_history = ADAM_gradient_descent(loss_function, x0,  beta1 = 0.5, beta2 = 0.999, avg_len = 50, print_to_file = print_to_file, x_history_file_name=x_history_file_name, threshhold=5e-4, alpha=0.05, max_iterations=1000)
    # bfgs_loss_history, bfgs_x_history, bfgs_x_avg_history, bfgs_update_history = Secant_Penalized_BFGS(loss_function, x0, print_to_file = print_to_file, x_history_file_name=" ", avg_len = 100, alpha=0.4, max_iterations=100, threshhold = 1e-3, max_n_convergence = 50, tolerance=1e-8, Ns=10e8, N0=10e-10)


    max_diff=max(abs(x_avg_history[-1] - x_opt))

    min_diff=min(abs(x_avg_history[-1] - x_opt))

    avg_diff=np.mean(abs(x_avg_history[-1] - x_opt))


    f=len(loss_history)

    if print_to_file:
        with open(file_name, "a") as file:
            file.write(f"{d}    {min_diff}   {avg_diff}   {max_diff}   {f} \n")



    print(f"Finish for d={d}!!!")

#%%


# Plot x_history and x_opt


# x_history_np = np.array([ np.mean(x_history[i-200:i], axis=0) for i in range(200,len(x_history))])
# x_history_np = np.array([ np.mean(update_history[i-100:i], axis=0) for i in range(100,len(x_history))])



# x_history_np = np.array(x_avg_history)
# # x_history_np_2 = np.array(bfgs_x_history)



# plt.figure(figsize=(10, 6))
# for i in range(d):
#     plt.plot(x_history_np[:, i],'-', color=f"C{i}", label=f"adam x{i+1} history")
#     # plt.plot(x_history_np_2[:, i],'x-', color=f"C{i}", label=f"bfgs x{i+1} history")
#     plt.axhline(y=x_opt[i], color=f"C{i}", linestyle="--", label=f"x{i+1}_opt")



# # plt.xlabel("Iteration")
# # plt.ylabel("x values")
# # plt.title("Convergence of x_history to x_opt")
# # plt.legend()
# # plt.grid()
# # plt.show()



# # #%%
# # bfgs_loss_history
# # %%
# plt.plot([np.log10(x) for x in loss_history], label='BFGS Loss History')
# # %%
# # bfgs_x_history
# # # %%
# # bfgs_x_avg_history
# # # %%
# # [1,2,3,4][:1]
# # # %%

# # %%
# len(x_avg_history)
# %%
