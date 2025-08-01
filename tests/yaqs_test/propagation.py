#%%
import numpy as np
from mqt.yaqs.noise_char.propagation import *
import matplotlib.pyplot as plt

from auxiliar.write import *

import sys
import time
import pandas as pd

#%%
import psutil
import os
import threading
from datetime import datetime



os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"




stop_event = threading.Event()


def main_code(folder, ntraj, L, order , threshold, method, solver, req_cpus):
    



    sim_params = SimulationParameters(L,0.1,0.1)
    sim_params.N = ntraj
    sim_params.T = 5
    sim_params.order = order
    sim_params.threshold = threshold
    sim_params.req_cpus = req_cpus


    sim_params.set_solver("tdvp1",solver)


    start_time = time.time()

    if method == "scikit_tt":
        print("Using SciKit-TT method")
        t, ref_traj, d_On_d_gk, avg_min_max_traj_time, results=scikit_tt_traj(sim_params)
        print("SciKit-TT method completed")

    elif method == "tjm":
        t, ref_traj, d_On_d_gk, avg_min_max_traj_time = tjm_traj(sim_params)


    end_time = time.time()


    write_ref_traj(t, ref_traj, f"{folder}/ref_traj.txt")

    n_obs_site, L, n_t = ref_traj.shape

    for i in range(ntraj):
        write_ref_traj(t, results[i][0].reshape(n_obs_site, L, n_t), f"{folder}/res_traj_{i}.txt")

    
    avg_traj = np.sum([res[0] for res in results], axis=0)/ntraj

    write_ref_traj(t, avg_traj.reshape(n_obs_site, L, n_t), f"{folder}/avg_traj.txt")

    print("ref_traj saved!!!")

    duration = end_time - start_time
    with open(f"{folder}/time_sec.txt", "w") as f:
        f.write(f"#sim_time    avg_traj_time     min_traj_time     max_traj_time  \n")
        f.write(f"#{duration}    {avg_min_max_traj_time[0]}     {avg_min_max_traj_time[1]}       {avg_min_max_traj_time[2]}  \n")


    print("time saved!!!")

    return t, ref_traj, d_On_d_gk, results
    

#%%






if __name__=="__main__":
    # args = sys.argv[1:]

    args = ["test/propagation/", "100", "4", "1", "1e-4", "scikit_tt", "exact", "4"]

    folder = args[0]

    ntraj = int(args[1])

    L = int(args[2])

    order= int(args[3])


    threshold = float(args[4])

    method = args[5]

    solver = args[6]


    allocated_cpus = int(args[7])


    print("Inputs:")
    print(f"  folder: {folder}")
    print(f"  ntraj: {ntraj}")
    print(f"  L: {L}")
    print(f"  order: {order}")
    print(f"  threshold: {threshold}")
    print(f"  method: {method}")
    print(f"  solver: {solver}")
    print(f"  req_cpus: {allocated_cpus}")


    pid = os.getpid()



    log_file = folder+"/self_memory_log.csv"

    # Start memory logging in a background thread
    logger_thread = threading.Thread(target=log_memory, args=(pid, log_file, 10, stop_event), daemon=True)
    logger_thread.start()

    # Run your main code


    req_cpus=allocated_cpus-1

    t, ref_traj, d_On_d_gk, results=main_code(folder, ntraj, L, order , threshold, method, solver, req_cpus)

    stop_event.set()

    logger_thread.join()



    # Wait briefly to ensure logger finishes last write
    time.sleep(1)




#%%

# # for i in range(100):
# #     plt.plot(t, results[i][0][1])



# average=np.sum([res[0]/100 for res in results], axis=0)


# col=2
# plt.plot(t, ref_traj[0, col], label="ref_traj")
# plt.plot(t, average[col], label="average")



# #%%

# average=np.sum([res[0][1]/100 for res in results])
# average.shape




# #%%
# np.sum([results[0][0][1], results[1][0][1]], axis=0)


# #%%
# cpu_list=[10, 18, 34]

# sites=100

# ntraj=512

# precision_list=[[1,"1e-4"],[2,"1e-6"]]

# precision=precision_list[0]


# job="memory"



# for precision in precision_list:


#     data_list=[]


#     for cpu in cpu_list:

#         folder=f"results/cpu_traj_scan/order_{precision[0]}/threshold_{precision[1]}/{sites}_sites/{cpu}_cpus/{ntraj}_traj"

#         if job=="time":

#             time_file= f"{folder}/time_sec.txt"
#             data_list.append(np.loadtxt(time_file)/60/60)
#             plt.ylabel("Time (hours)")


#         if job=="memory":

#             mem_file= f"{folder}/sstat_log.csv"
#             data_list.append(max(pd.read_csv(mem_file).values[:,1]))

#             plt.ylabel("Max Memory (GB)")


#     plt.plot(cpu_list, data_list,'-o', label=f"order_{precision[0]}/threshold_{precision[1]}")
# plt.title(f"{sites}_sites  {ntraj}_Ntraj")
# plt.xlabel("Number of CPUs")
# plt.legend()
# #%%
# precision=precision_list[1]
# sites=100
# cpu=34
# ntraj=512
# plt.plot(pd.read_csv(f"results/cpu_traj_scan/order_{precision[0]}/threshold_{precision[1]}/{sites}_sites/{cpu}_cpus/{ntraj}_traj/self_memory_log.csv").values[:,1])
# plt.ylim(22,23)
# plt.ylabel("Memory Usage (GB)")
# plt.xlabel("Time (seconds)")
#%%
# plt.plot(cpu_list, mem_list,'-o')
# plt.xlabel("Number of CPUs")
# plt.ylabel("Memory Usage (MB)")


# # %%
# import matplotlib.pyplot as plt

# ncpus=32

# csv_file=f"/home/aramos/Dokumente/Work/Simulation of Open Quantum Systems/tjm_noise_char/tests/yaqs_test/results/cpu_traj_scan/4_sites/{ncpus}_cpus/4096_traj/cpu_usage.csv"

# cpu_usage_df = pd.read_csv(csv_file)

# # Pivot the data to have CPUs as columns and timestamps as rows
# cpu_usage_pivot = cpu_usage_df.pivot(index='timestamp', columns='cpu', values='usr')

# # Plot the CPU usage for each CPU
# plt.figure(figsize=(12, 6))
# for cpu in cpu_usage_pivot.columns:
#     plt.plot(np.array(cpu_usage_pivot[cpu]), label=f"CPU {cpu}")

# plt.xlabel("Time (seconds)")
# plt.ylabel("CPU Usage (%)")
# plt.title(f"CPU Usage Over Time for {ncpus} CPUs")
# plt.legend(loc="upper left", ncol=2)
# # plt.grid()
# # plt.tight_layout()
# plt.show()
# # %%
# np.array(cpu_usage_pivot[cpu]).shape
# # %%

# mem_usage = pd.read_csv(f"/home/aramos/Dokumente/Work/Simulation of Open Quantum Systems/tjm_noise_char/tests/yaqs_test/results/cpu_traj_scan/4_sites/{ncpus}_cpus/4096_traj/mem_usage.csv")

# plt.plot(np.array(mem_usage/1024/1024))


# # %%
# mem_usage
# # %%
# # %%



# #%%
# sim_params = SimulationParameters()
# t, ref_traj, d_On_d_gk=tjm_traj(sim_params)
# # %%
# sci_kit_t, sci_kit_ref_traj, sci_kit_d_On_d_gk=scikit_tt_traj(sim_params)

# # %%




# plt.plot(t, d_On_d_gk[1, 1, 3,:], label="TJM")
# plt.plot(sci_kit_t, np.array(sci_kit_d_On_d_gk)[1, 1, 3,:], label="SciKit-TT")

# plt.legend()

# # plt.plot(t, ref_traj[:, 0, 0], label="TJM")
# # %%
