#!/bin/bash

# Define your two parameter sets
cpus=(17)

trajectories=(512)

#L_list=(10 20 40 80)

L_list=(5)

d="2L"

# Paths
template="template.slurm"


RESTART="False" 
ORDER=1 
THRESHOLD="1e-4"




# Loop over all combinations
for ncpus in "${cpus[@]}"; do
    for ntraj in "${trajectories[@]}"; do
	for L in "${L_list[@]}"; do

        	job_name="d_${d}/L_${L}/ntraj_${ntraj}"

        	job_dir="results/optimization/${job_name}"

        	mkdir -p "${job_name}"

        	# Generate unique output filename
        	output_script="$job_name/run.slurm"

        	# Replace placeholders in the template
        	sed -e "s|%D%|${d}|g" -e "s|%RESTART%|${RESTART}|g" -e "s|%ORDER%|${ORDER}|g" -e "s|%THRESHOLD%|${THRESHOLD}|g"  -e "s|%L%|${L}|g"  -e "s|%NCPUS%|${ncpus}|g" -e "s|%NTRAJ%|${ntraj}|g" -e "s|%JOB_NAME%|${job_name}|g" -e "s|%JOB_DIR%|${job_dir}|g" "$template" > "$output_script"

        	# Optional: submit the job
        	sbatch "$output_script"

        	echo "Generated $output_script"

	done
    done
done

