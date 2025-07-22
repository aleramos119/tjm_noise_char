
#%%
import numpy as np


def write_ref_traj(t, ref_traj, file_name):
    """
    Write the reference trajectory to a file.
    
    Parameters:
    - t: Time array.
    - ref_traj: Reference trajectory data.
    - file_name: Name of the output file.
    """

    n_obs_site, L, n_t = ref_traj.shape


    ref_traj_reshaped = ref_traj.reshape(-1, ref_traj.shape[-1])

    ref_traj_with_t=np.concatenate([np.array([t]), ref_traj_reshaped], axis=0)




    ## Saving reference trajectory and gammas
    header =   "t  " +  "  ".join([obs+str(i)   for obs in ["x","y","z"][:n_obs_site] for i in range(L) ])

    np.savetxt(file_name, ref_traj_with_t.T, header=header, fmt='%.6f')




def log_memory(pid, log_file, interval, stop_event):
    """
    Logs memory usage (in GB) of parent and child processes individually,
    plus the total RAM used by all of them.

    Output CSV columns: timestamp,pid,name,ram_GB,type
    Where type is "parent", "child", or "total".
    """
    import psutil
    from datetime import datetime
    import time

    parent = psutil.Process(pid)

    with open(log_file, "w") as f:
        f.write("timestamp,pid,name,ram_GB,type\n")

    try:
        while not stop_event.is_set():
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            entries = []
            total_rss = 0

            # Log parent
            try:
                parent_rss = parent.memory_info().rss
                total_rss += parent_rss
                entries.append((parent.pid, parent.name(), parent_rss, "parent"))
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

            # Log children
            try:
                for child in parent.children(recursive=True):
                    try:
                        child_rss = child.memory_info().rss
                        total_rss += child_rss
                        entries.append((child.pid, child.name(), child_rss, "child"))
                    except (psutil.NoSuchProcess, psutil.AccessDenied):
                        continue
            except psutil.Error:
                pass

            # Add total as a virtual process line
            entries.append(("NA", "ALL_PROCESSES", total_rss, "total"))

            # Write to file
            with open(log_file, "a") as f:
                for pid, name, rss_bytes, label in entries:
                    ram_gb = rss_bytes / 1024 / 1024 / 1024
                    f.write(f"{timestamp},{pid},{name},{ram_gb:.3f},{label}\n")

            time.sleep(interval)

    except Exception:
        pass  # silent fail is safer for daemonized threads
