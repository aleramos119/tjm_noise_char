#!/bin/bash

# Define your two parameter sets
cpus=(8 16 32 64)
<<<<<<< Updated upstream
=======
#cpus=(8)
>>>>>>> Stashed changes
trajectories=(4096)

L=80


# Paths
template="template.slurm"


# Loop over all combinations
for ncpus in "${cpus[@]}"; do
    for ntraj in "${trajectories[@]}"; do


        job_name="${ncpus}_cpus/${ntraj}_traj"

<<<<<<< Updated upstream
        job_dir="4_sites/${job_name}"
=======
        job_dir="${L}_sites/${job_name}"
>>>>>>> Stashed changes

        mkdir -p "${job_dir}"

        # Generate unique output filename
        output_script="$job_dir/run.slurm"

        # Replace placeholders in the template
        sed  -e "s|%L%|${L}|g"  -e "s|%NCPUS%|${ncpus}|g" -e "s|%NTRAJ%|${ntraj}|g" -e "s|%JOB_NAME%|${job_name}|g" -e "s|%JOB_DIR%|${job_dir}|g" "$template" > "$output_script"

        # Optional: submit the job
        sbatch "$output_script"

        echo "Generated $output_script"
    done
done

