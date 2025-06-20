#!/bin/bash

# Define your two parameter sets
<<<<<<< HEAD
cpus=(32)

trajectories=(512 1024)

#L_list=(10 20 40 80)

L_list=(80 100)
=======
cpus=(16)

trajectories=(128 256 512 1024)

L_list=(10 20 40 80)
>>>>>>> tjm_noise_char/main

L_list=(80 100)

d=2

# Paths
template="template.slurm"


# Loop over all combinations
for ncpus in "${cpus[@]}"; do
    for ntraj in "${trajectories[@]}"; do
	for L in "${L_list[@]}"; do

        	job_name="d_${d}_ntraj0_8192/L_${L}/ntraj_${ntraj}"

        	job_dir="results/optimization/${job_name}"

        	mkdir -p "${job_name}"

        	# Generate unique output filename
        	output_script="$job_name/run.slurm"

        	# Replace placeholders in the template
        	sed  -e "s|%L%|${L}|g"  -e "s|%NCPUS%|${ncpus}|g" -e "s|%NTRAJ%|${ntraj}|g" -e "s|%JOB_NAME%|${job_name}|g" -e "s|%JOB_DIR%|${job_dir}|g" "$template" > "$output_script"

        	# Optional: submit the job
        	sbatch "$output_script"

        	echo "Generated $output_script"

	done
    done
done

