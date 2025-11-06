 #%%
from scipy.interpolate import interp1d
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import UnivariateSpline
from auxiliar.nelder_mead import nelder_mead_opt
from auxiliar.cma import cma_opt
from pathlib import Path

from scipy.optimize import minimize_scalar

#%%



class InterpolatedFunction:

    def __init__(self, f, d, std =5, path = ""):
        
        self.f=f
        self.d = d
        self.std=std

        self.n_eval = 0

        self.print_to_file=True

        self.work_dir: Path = Path(path)

        self.work_dir.mkdir(parents=True, exist_ok=True)

        self.history_file_name = self.work_dir / "loss_x_history.txt"
        self.history_avg_file_name = self.work_dir / "loss_x_history_avg.txt"



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

    def __call__(self, x):

        self.n_eval += 1

        noise = abs(np.random.normal(0, self.std))


        if np.any(x > 1) or np.any(x < 0):
            f = 1e10 

        else:
            f=np.sum(self.f(x)) + noise

        self.write_to_file(self.history_file_name, f, x)
        
        return f
    

def plot_gamma_optimization(folder: str, gammas) -> None:
    """Plot the optimization history of gamma parameters from a given folder.

    Parameters
    ----------
    folder : str
        The folder containing the optimization data files.
    """
    x_avg_file = folder + "loss_x_history.txt"

    data = np.genfromtxt(x_avg_file, skip_header=1, ndmin=2)

    d = len(gammas)

    for i in range(d):
        plt.plot(data[:, 0], data[:, 2 + i], label=f"$\\gamma_{{{i+1}}}$")
        plt.axhline(gammas[i], color=plt.gca().lines[-1].get_color(), linestyle='--', linewidth=2)

    plt.xlabel("Iterations")
    plt.ylabel(r"$\gamma$")
    plt.legend()
    plt.title("Gamma Parameter Optimization History")
    plt.savefig(folder + "gamma.pdf")
    plt.close()

    max_diff=max(abs(np.mean(data[10:,2:2+d],axis=0)-gammas))

    plt.plot(data[:,0], data[:,1], label="Loss")
    plt.title("Loss Optimization History")

    plt.legend()
    plt.savefig(folder + "loss.pdf")
    plt.close()



#%%
if __name__ == '__main__':


    d_list = [2*(i+1) for i in range(25)]

    std_list = [0, 0.2]

    method_list = ["nelder_mead", "cma"]


    data_dir="test/gamma_scan_T_4/"

    loss_list=np.genfromtxt(data_dir+"loss_list.txt")
    gamma_list=np.genfromtxt(data_dir+"gamma_list.txt")


    f1d= interp1d(
    gamma_list,
    loss_list,
    kind='cubic',   # options: 'linear', 'cubic', 'quadratic', 'nearest'
    fill_value='extrapolate'  # allow values outside x range
    )


    for method in method_list:

        for d in d_list:
            for std in std_list:

                print(d)

                gammas = [0.368159153509982]*d

                work_dir=f"test/opt_{method}/std_{std}/d_{d}/"

                work_dir_path = Path(work_dir)

                work_dir_path.mkdir(parents=True, exist_ok=True)



                for file in work_dir_path.iterdir():
                    if file.is_file():
                        file.unlink()  # delete the file

                


                f=InterpolatedFunction(f1d, d, std=std, path = work_dir)

                x0=np.random.rand(d)  

                if method == "nelder_mead":
                    nelder_mead_opt(f,x0, max_iter=500, step=0.3)

                if method == "cma":
                    cma_opt(f, x0, 0.2, popsize=4)

                plot_gamma_optimization(work_dir, gammas)




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
