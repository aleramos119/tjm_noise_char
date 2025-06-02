#!/bin/bash

# Define your two parameter sets
cpus=(32 64 128 256)
trajectories=(4096)

# Paths
template="template.slurm"


# Loop over all combinations
for ncpus in "${cpus[@]}"; do
    for ntraj in "${trajectories[@]}"; do


        job_name="${ncpus}_cpus_${ntraj}_traj"

        job_dir="cpu_traj_scan/${job_name}"

        mkdir -p "${job_dir}"

        # Generate unique output filename
        output_script="$job_dir/run.slurm"

        # Replace placeholders in the template
        sed -e "s|%NCPUS%|${ncpus}|g" -e "s|%NTRAJ%|${ntraj}|g" -e "s|%JOB_NAME%|${job_name}|g" -e "s|%JOB_DIR%|${job_dir}|g" "$template" > "$output_script"

        # Optional: submit the job
        # sbatch "$output_script"

        echo "Generated $output_script"
    done
done

