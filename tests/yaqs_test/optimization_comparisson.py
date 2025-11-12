 #%%
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from auxiliar.nelder_mead import nelder_mead_opt
from auxiliar.cma import cma_opt
from pathlib import Path

from scipy.optimize import minimize_scalar

import torch

from auxiliar.bayesian_optimization import bayesian_opt
from auxiliar.differential_evolution import differential_evolution_opt

import sys
#%%



class InterpolatedFunction:

    def __init__(self, f, d, std =5, path = "", avg_tol = 1e-6, n_conv=20):
        
        self.f=f
        self.d = d
        self.std=std

        self.n_eval = 0

        self.print_to_file=True

        self.work_dir: Path = Path(path)

        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.history_file_name = self.work_dir / "loss_x_history.txt"
        self.history_avg_file_name = self.work_dir / "loss_x_history_avg.txt"


        self.n_avg=100


        self.x_history = []

        self.x_avg_history = []

        self.diff_avg_history = []


        self.converged = False

        self.avg_tol = avg_tol

        self.n_conv = n_conv


        




    def write_to_file(self, file_name: Path, f: float, x: np.ndarray) -> None:

        if self.print_to_file:
            if not file_name.exists():
                with file_name.open("w", encoding="utf-8") as file:
                    file.write(
                        "# iter  loss  "
                        + "  ".join([f"x{i + 1}" for i in range(self.d)])
                        + "    "
                        + "\n"
                    )

            with file_name.open("a", encoding="utf-8") as file:
                file.write(
                    f"{self.n_eval}    {f}  "
                    + "  ".join([f"{x[j]:.6f}" for j in range(self.d)])
                    + "    "
                    + "\n"
                ) 

    def compute_avg(self) -> None:
        """Computes the average of the parameter history and appends it to the average history.

        If the length of `x_history` is less than or equal to `n_avg`, computes the mean over the entire `x_history`.
        Otherwise, computes the mean over the entries in `x_history` starting from index `n_avg`.
        The computed average is appended to `x_avg_history`.


        """
        if len(self.x_history) <= self.n_avg:
            x_avg = np.mean(self.x_history, axis=0)
        else:
            x_avg = np.mean(self.x_history[self.n_avg :], axis=0)

        self.x_avg_history.append(x_avg.copy())

    def compute_diff_avg(self) -> None:
        """Computes the maximum absolute difference between the last two entries in `x_avg_history`.

        This method is intended to track the change in the average values stored in `x_avg_history`
        over successive iterations.
        """
        if len(self.x_avg_history) > 1:
            diff: float = np.max(np.abs(self.x_avg_history[-1] - self.x_avg_history[-2]))
            self.diff_avg_history.append(diff)

    def check_convergence(self):

        if len(self.diff_avg_history) > self.n_conv and all(
            diff < self.avg_tol for diff in f.diff_avg_history[-self.n_conv:]
        ):
            
            self.converged = True

    
    def post_process(self, x: np.ndarray, f: float) -> None:
   
        self.n_eval += 1
        self.x_history.append(x)


        self.compute_avg()
        self.compute_diff_avg()


        self.write_to_file(self.history_file_name, f, self.x_history[-1])
        self.write_to_file(self.history_avg_file_name, f, self.x_avg_history[-1])


        self.check_convergence()



    def __call__(self, x):

        noise = abs(np.random.normal(0, self.std))


        if np.any(x > 1) or np.any(x < 0):
            f = 1e10 

        else:
            f=np.sum(self.f(x)) + noise

        
        self.post_process(x, f)
        
        return f
    

def plot_gamma_optimization(folder: str, file , gammas, error, best_y) -> None:
    """Plot the optimization history of gamma parameters from a given folder.

    Parameters
    ----------
    folder : str
        The folder containing the optimization data files.
    """
    x_avg_file = folder + file+".txt"

    data = np.genfromtxt(x_avg_file, skip_header=1, ndmin=2)

    d = len(gammas)

    for i in range(d):
        plt.plot(data[:, 0], data[:, 2 + i], label=f"$\\gamma_{{{i+1}}}$")
        plt.axhline(gammas[i], color=plt.gca().lines[-1].get_color(), linestyle='--', linewidth=2)


    plt.text(
    0.95, 0.95, f"rel error: {error/gammas[0]:.3e}\nmin loss: {best_y:.3e}",
    transform=plt.gca().transAxes,
    fontsize=10,
    verticalalignment='top',
    horizontalalignment='right',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7)
)

    plt.xlabel("Iterations")
    plt.ylabel(r"$\gamma$")
    plt.legend()
    plt.title("Gamma Parameter Optimization History")
    plt.savefig(folder + "gamma_"+file+".pdf")
    plt.close()

    max_diff=max(abs(np.mean(data[10:,2:2+d],axis=0)-gammas))

    plt.plot(data[:,0], data[:,1], label="Loss")
    plt.title("Loss Optimization History")

    plt.legend()
    plt.savefig(folder + "loss.pdf")
    plt.close()



#%%
if __name__ == '__main__':


    d_list = [i+1 for i in range(1,25)]

    std_list = [0, 0.02, 0.04]


    # method_list = ["bo_ucb", "bo_ei", "bo_pi", "cma", "diff_evol"]

    # method_list = ["diff_evol"]


    # method_list = ["diff_evol"]



    data_dir="test/loss_scan/gamma_ref_0.01/N_4000/T_6/obs_Z/noise_X/"

    loss_list=np.genfromtxt(data_dir+"loss_list.txt")
    gamma_list=np.genfromtxt(data_dir+"gamma_list.txt")


    f1d= interp1d(
    gamma_list,
    loss_list,
    kind='cubic',   # options: 'linear', 'cubic', 'quadratic', 'nearest'
    fill_value='extrapolate'  # allow values outside x range
    )


    x_dense=np.linspace(gamma_list[0],gamma_list[-1],100000)
    y_dense=f1d(x_dense)

    x_min = x_dense[np.argmin(y_dense)]
    y_min = np.min(y_dense)


    # for method in method_list:

    method=sys.argv[1]

    x_lim=0.1



    for d in d_list:
        for std in std_list:

            print(f"Method: {method}, d: {d}, std: {std}")


            gammas = np.array([x_min]*d)

            x_low=np.array([0]*d)

            x_up=np.array([x_lim]*d)



            work_dir=f"test/optimization_comparisson/opt_{method}/std_{std}/d_{d}/"

            work_dir_path = Path(work_dir)

            work_dir_path.mkdir(parents=True, exist_ok=True)



            for file in work_dir_path.iterdir():
                if file.is_file():
                    file.unlink()  # delete the file



            np.savetxt(work_dir + "/x_min.txt", gammas, header="##", fmt="%.6f")




            f=InterpolatedFunction(f1d, d, std=std, path = work_dir, avg_tol = 1e-5, n_conv=20)


            

            x0=np.random.uniform(0, x_lim, d)

            if method == "nelder_mead":
                best_x, best_y=nelder_mead_opt(f,x0, x_low, x_up, max_iter=300, step=0.01)

            if method == "cma":
                best_x, best_y=cma_opt(f, x0, x_low, x_up, 0.01, popsize=4, max_iter=300)
            


            if method == "bo_ucb" or method == "bo_ei" or method == "bo_pi":


                acq=method.rsplit("_", 1)[-1].upper()

                if std==0:
                    std = 10**(-2.8)


                best_x, best_y, X_train, Y_train = bayesian_opt(
                        f=f,
                        x_low=x_low,
                        x_up=x_up,
                        n_init=3*d,
                        n_iter=300,
                        std=std,
                        beta=100.0,
                        acq_name=acq,  # <-- Try "EI", "PI", or "UCB"
                )

            if method == "diff_evol":

                best_x, best_y, history=differential_evolution_opt(
                    f,
                    x_low,
                    x_up,
                    pop_size=5*d,
                    F=0.7,
                    Cr=0.7,
                    max_iter=300,
                    tol=1e-5,
                    workers=1,
                    noise_averaging=1,
                    seed=None,
                    verbose=True,
                )

            error = np.max(np.abs(best_x - gammas))

            np.savetxt(work_dir + "/error.txt", [error], header="##", fmt="%.6f")
            np.savetxt(work_dir + "/rel_error.txt", [error/gammas[0]], header="##", fmt="%.6f")



            plot_gamma_optimization(work_dir, "loss_x_history", gammas, error, best_y)
            plot_gamma_optimization(work_dir, "loss_x_history_avg", gammas, error, best_y)





    #%%

#     x_samples = np.random.uniform(0, 1, (1000, 2))  # shape (N, 2)

#     # Evaluate f at each point
#     z = np.array([f(x) for x in x_samples])

#     # Extract coordinates
#     x = x_samples[:, 0]
#     y = x_samples[:, 1]

#     # Create the contour plot
#     plt.figure(figsize=(6, 5))
#     contour = plt.tricontourf(x, y, z, levels=30, cmap='viridis')
#     plt.colorbar(contour, label='f(x)')
#     plt.title("2D Contour Plot from f([x, y])")
#     plt.xlabel("x")
#     plt.ylabel("y")
#     plt.show()


#     y0=0.1
#     gammas=np.linspace(-0.5,1.5,200)
#     x_test=np.array([[g, y0] for g in gammas])

#     f_test=np.array([f(x) for x in x_test])

#     plt.plot(gammas, f_test,'-', label=f"y0 = {y0}")
#     plt.legend()

# # %%
# gammas.shape
# # %%

#%%
# d=10
# bounds = [(0,1)]*d
# # %%
# np.array(bounds).shape
# %%
# # %%
# %matplotlib qt

# loss_list=np.genfromtxt(data_dir+"loss_list.txt")
# gamma_list=np.genfromtxt(data_dir+"gamma_list.txt")

# f1d= interp1d(
#     gamma_list,
#     loss_list,
#     kind='quadratic',   # options: 'linear', 'cubic', 'quadratic', 'nearest'  # allow values outside x range
#     )

# x=np.linspace(gamma_list[0],gamma_list[-1],100000)
# y=f1d(x)

# x_min = x[np.argmin(y)]
# y_min = np.min(y)


# print(x_min)

# plt.plot(gamma_list, loss_list, 'x')

# plt.plot(x_dense,y_dense)

# # %%
# best_x

# # %%
# best_y
# # %%
