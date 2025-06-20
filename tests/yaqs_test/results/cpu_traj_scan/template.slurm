#!/bin/bash
#SBATCH --job-name=%JOB_NAME% # Job name
#SBATCH --time=24:00:00 # Specify a Time limit in the format days-hrs:min:sec. Use sinfo to see node time limits
#SBATCH --chdir=/home/htc/aramos/simulation_of_open_quantum_systems/tjm_noise_char/tests/yaqs_test  # Navigate to the working directory where your script lies
#SBATCH --output=/home/htc/aramos/simulation_of_open_quantum_systems/tjm_noise_char/tests/yaqs_test/results/cpu_traj_scan/%JOB_DIR%/%j.log     # Standard output and error log
#SBATCH --partition=high-mem  # Specify the desired partition, e.g. gpu or big. Note that small is the default partition.
#SBATCH --cpus-per-task=%NCPUS%

##SBATCH --exclude=htc-cmp[101-148,501-532] # exclude nodes. Your job will run on nodes not in the list.

#echo 'Getting node information'
#date;hostname;id;pwd



# Capture the start time
start_time=$(date +%s)
echo "Job started on $(hostname) at $(date)"

# Log SLURM environment variables
echo "SLURM Job name: $SLURM_JOB_NAME"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node(s): $SLURM_NODELIST"
echo "Number of tasks: $SLURM_NTASKS"
echo "Allocated memory: $SLURM_MEM_PER_NODE MB"




echo 'Activating virtual environment'
source /opt/conda/etc/profile.d/conda.sh

conda activate yaqs

echo 'Enabling Internet Access'
export https_proxy=http://squid.zib.de:3128
export http_proxy=http://squid.zib.de:3128

echo 'Running python script'
CPU_LIST=$(taskset -cp $$ | awk -F: '{print $2}' | tr -d ' ')



echo 'Running python script'
CPU_LIST=$(taskset -cp $$ | awk -F: '{print $2}' | tr -d ' ')

echo "CPU_LIST:  $CPU_LIST "

LC_NUMERIC=C mpstat -P $CPU_LIST 1 300 | awk '
BEGIN {
  print "timestamp,cpu,usr"
}
$2 ~ /^[0-9]+$/ {
  cpu = $2
  usr = $3
  print $1 "," cpu "," usr
}' > results/cpu_traj_scan/%JOB_DIR%/cpu_usage.csv &

MPSTAT_PID=$!




# Start sstat logging in the background
{
  echo "timestamp,MaxRSS"
  while true; do
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    #maxrss=$(sstat -j ${SLURM_JOB_ID}.batch --format=MaxRSS --noheader 2>/dev/null | awk '{print $1}')
    maxrss_kb=$(sstat -j ${SLURM_JOB_ID}.batch --format=MaxRSS --noheader 2>/dev/null | awk '{print $1}')
    maxrss_gb=$(awk -v kb="$maxrss_kb" 'BEGIN { printf "%.2f", kb / 1024 / 1024 }')
    if [ -z "$maxrss_gb" ]; then
      break
    fi
    echo "$timestamp,$maxrss_gb"
    sleep 1
  done
} > results/cpu_traj_scan/%JOB_DIR%/sstat_log.csv &
SSTAT_PID=$!



python propagation.py results/cpu_traj_scan/%JOB_DIR% %NTRAJ% %L% %ORDER% %THRESHOLD%

# Stop background loggers after script ends
kill $MPSTAT_PID
kill $SSTAT_PID






# Capture the end time
end_time=$(date +%s)

# Compute and log the duration
duration=$((end_time - start_time))
days=$((duration / 86400))
hours=$(((duration % 86400) / 3600))
minutes=$(((duration % 3600) / 60))
seconds=$((duration % 60))

echo "Job completed at $(date)"
echo "Total duration: $days days, $hours hours, $minutes minutes, $seconds seconds"




#seff $SLURM_JOB_ID >  results/cpu_traj_scan/%JOB_DIR%/mem_usage.csv 

