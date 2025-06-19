#!/bin/bash

# Define your two parameter sets
cpus=(10 18 34)
#cpus=(8)
trajectories=(512)

L=100 

order=1

threshold="1e-4"

# Paths
template="template.slurm"


# Loop over all combinations
for ncpus in "${cpus[@]}"; do
    for ntraj in "${trajectories[@]}"; do


        job_name="${ncpus}_cpus/${ntraj}_traj"

        job_dir="order_${order}/threshold_${threshold}/${L}_sites/${job_name}"

        mkdir -p "${job_dir}"

        # Generate unique output filename
        output_script="$job_dir/run.slurm"

        # Replace placeholders in the template
        sed -e "s|%THRESHOLD%|${threshold}|g" -e "s|%ORDER%|${order}|g" -e "s|%L%|${L}|g"  -e "s|%NCPUS%|${ncpus}|g" -e "s|%NTRAJ%|${ntraj}|g" -e "s|%JOB_NAME%|${job_name}|g" -e "s|%JOB_DIR%|${job_dir}|g" "$template" > "$output_script"

        # Optional: submit the job
        sbatch "$output_script"

        echo "Generated $output_script"
    done
done

