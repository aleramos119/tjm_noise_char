#%%
import numpy as np
from mqt.yaqs.noise_char.propagation import *
import matplotlib.pyplot as plt

import sys
import time
import pandas as pd

#%%
import psutil
import os
import threading
from datetime import datetime


stop_event = threading.Event()
def log_memory(pid, log_file, interval, stop_event=stop_event):
    process = psutil.Process(pid)
    with open(log_file, "w") as f:
        f.write("timestamp,ram_GB\n")
    try:
        while stop_event.is_set() == False:
            # Get memory usage of the main process
            total_mem_bytes = process.memory_info().rss

            # Add memory usage of all child processes (recursively)
            for child in process.children(recursive=True):
                try:
                    total_mem_bytes += child.memory_info().rss
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass  # child might have exited

            mem_gb = total_mem_bytes / 1024 / 1024 / 1024  # Convert to GB
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            with open(log_file, "a") as f:
                f.write(f"{timestamp},{mem_gb:.2f}\n")

            time.sleep(interval)
    except Exception as e:
        pass  # Silently ignore exceptions when stopping


def main_code(folder, ntraj, L, order , threshold, method, solver):
    



    sim_params = SimulationParameters(L,0.1,0.1)
    sim_params.N = ntraj
    sim_params.T = 5
    sim_params.order = order
    sim_params.threshold = threshold


    sim_params.set_solver("tdvp1",solver)


    start_time = time.time()

    if method == "scikit_tt":
        print("Using SciKit-TT method")
        t, qt_ref_traj, d_On_d_gk=scikit_tt_traj(sim_params)
        print("SciKit-TT method completed")

    elif method == "tjm":
        t, qt_ref_traj, d_On_d_gk = tjm_traj(sim_params)


    end_time = time.time()


    qt_ref_traj_reshaped = qt_ref_traj.reshape(-1, *qt_ref_traj.shape[2:]).T

    np.savetxt(f"{folder}/qt_ref_traj.txt", qt_ref_traj_reshaped )

    print("ref_traj saved!!!")

    duration = end_time - start_time
    with open(f"{folder}/time_sec.txt", "w") as f:
        f.write(f"{duration}\n")

    print("time saved!!!")
    

#%%






if __name__=="__main__":
    # args = sys.argv[1:]

    args = ["test/propagation/", "100", "3", "1", "1e-4", "tjm", "exact"]

    folder = args[0]

    ntraj = int(args[1])

    L = int(args[2])

    order= int(args[3])


    threshold = float(args[4])

    method = args[5]

    solver = args[6]


    pid = os.getpid()


    log_file = folder+"/self_memory_log.csv"

    # Start memory logging in a background thread
    logger_thread = threading.Thread(target=log_memory, args=(pid, log_file, 1,stop_event), daemon=True)
    logger_thread.start()

    # Run your main code


    main_code(folder, ntraj, L, order , threshold, method, solver)

    stop_event.set()

    logger_thread.join()



    # Wait briefly to ensure logger finishes last write
    time.sleep(1)


















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
