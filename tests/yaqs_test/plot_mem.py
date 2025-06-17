
#%%
import matplotlib.pyplot as plt
import numpy as np
import os
#%%
import re
from pathlib import Path

def parse_slurm_report(file_path):
    data = {}

    with open(file_path, 'r') as f:
        content = f.read()

    # Define regex patterns
    patterns = {
        'job_id': r'Job ID:\s*(\d+)',
        'cluster': r'Cluster:\s*(\S+)',
        'user_group': r'User/Group:\s*(\S+)',
        'state': r'State:\s*([A-Z]+)',
        'exit_code': r'exit code (\d+)',
        'nodes': r'Nodes:\s*(\d+)',
        'cores_per_node': r'Cores per node:\s*(\d+)',
        'cpu_utilized': r'CPU Utilized:\s*([0-9\-:]+)',
        'cpu_efficiency': r'CPU Efficiency:\s*([\d\.]+)% of ([0-9\-:]+)',
        'wall_time': r'Job Wall-clock time:\s*([0-9:]+)',
        'memory_utilized': r'Memory Utilized:\s*([\d\.]+) GB',
        'memory_efficiency': r'Memory Efficiency:\s*([\d\.]+)% of ([\d\.]+) GB \(([\d\.]+) GB/core\)'
    }

    # Apply regex
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            if key == 'cpu_efficiency':
                data['cpu_efficiency_percent'] = float(match.group(1))
                data['cpu_efficiency_reference'] = match.group(2)
            elif key == 'memory_efficiency':
                data['memory_efficiency_percent'] = float(match.group(1))
                data['memory_reference_gb'] = float(match.group(2))
                data['memory_per_core_gb'] = float(match.group(3))
            elif key in ['job_id', 'nodes', 'cores_per_node', 'exit_code']:
                data[key] = int(match.group(1))
            elif key in ['cpu_utilized', 'wall_time']:
                data[key] = match.group(1)  # Keep as raw string
            elif key == 'memory_utilized':
                data[key] = float(match.group(1))
            else:
                data[key] = match.group(1)

    return data


#%%

folder="results/cpu_traj_scan/"


L_list=[4,20,40,80,100]
cpu_list=[16]
ntraj_list=[4096]




plt.rcParams.update({'axes.linewidth': 1.2})
plt.rcParams.update({'font.size': 12, 'lines.linewidth': 2, 'lines.markersize': 6})



for cpu in cpu_list:



    par_list=[]
    mem_list=[]

    data_name='memory_utilized'

    for L in L_list:

        seff_file=folder + f"{L}_sites/{cpu}_cpus/4096_traj/seff_output"

        if os.path.exists(seff_file):

            data = parse_slurm_report(seff_file)

            print(data.keys())

            if data_name in data:

                print(data_name)

                par_list.append(L)
                mem_list.append(data[data_name])



    plt.plot(par_list, np.array(mem_list), marker='o', label=f'cpu_{cpu}')

plt.xlabel(r"N")
plt.ylabel(data_name)
plt.legend()



# %%
data_name in data
# %%
data_name

# %%
