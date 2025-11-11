#!/bin/bash

<<<<<<< HEAD

partition="big"

# Define your two parameter sets
cpus=(96)

trajectories=(256 512)

#L_list=(10 20 40 80)

L_list=(120 240) 

#L_list=(15 30)

d="2"
=======
# Define your two parameter sets
cpus=(96)

trajectories=(512 1024)

#L_list=(10 20 40 80)

L_list=(80 100)

d="2L"
>>>>>>> 49f37b7d0e8f7b0d79baf042b688e97b1c38ef6b

# Paths
template="template.slurm"


<<<<<<< HEAD
RESTART="True" 
ORDER=1 
THRESHOLD="1e-4"

max_bond_dim=8


method="tjm"
solver="exact"

ntraj_0=2048
gamma="random"
gamma_0="random"

=======
RESTART="False" 
ORDER=1 
THRESHOLD="1e-4"


method="scikit_tt"
solver="krylov_5"
>>>>>>> 49f37b7d0e8f7b0d79baf042b688e97b1c38ef6b

# Loop over all combinations
for ncpus in "${cpus[@]}"; do
    for ntraj in "${trajectories[@]}"; do
	for L in "${L_list[@]}"; do

<<<<<<< HEAD
        	job_name="method_${method}_${solver}_opt_script_test/max_bond_dim_${max_bond_dim}/d_${d}/gamma_${gamma}/gamma_0_${gamma_0}/L_${L}/ntraj_${ntraj}"
=======
        	job_name="d_${d}/L_${L}/ntraj_${ntraj}"
>>>>>>> 49f37b7d0e8f7b0d79baf042b688e97b1c38ef6b

        	job_dir="results/optimization/${job_name}"

        	mkdir -p "${job_name}"

        	# Generate unique output filename
        	output_script="$job_name/run.slurm"

        	# Replace placeholders in the template
<<<<<<< HEAD
        	sed -e "s|%NTRAJ0%|${ntraj_0}|g" -e "s|%PARTITION%|${partition}|g" -e "s|%GAMMA0%|${gamma_0}|g" -e "s|%GAMMA%|${gamma}|g"  -e "s|%MAX_BOND_DIM%|${max_bond_dim}|g" -e "s|%SOLVER%|${solver}|g" -e "s|%METHOD%|${method}|g"  -e "s|%D%|${d}|g" -e "s|%RESTART%|${RESTART}|g" -e "s|%ORDER%|${ORDER}|g" -e "s|%THRESHOLD%|${THRESHOLD}|g"  -e "s|%L%|${L}|g"  -e "s|%NCPUS%|${ncpus}|g" -e "s|%NTRAJ%|${ntraj}|g" -e "s|%JOB_NAME%|${job_name}|g" -e "s|%JOB_DIR%|${job_dir}|g" "$template" > "$output_script"

		cp ../../optimization_test.py $job_name
=======
        	sed -e "s|%SOLVER%|${solver}|g" -e "s|%METHOD%|${method}|g"  -e "s|%D%|${d}|g" -e "s|%RESTART%|${RESTART}|g" -e "s|%ORDER%|${ORDER}|g" -e "s|%THRESHOLD%|${THRESHOLD}|g"  -e "s|%L%|${L}|g"  -e "s|%NCPUS%|${ncpus}|g" -e "s|%NTRAJ%|${ntraj}|g" -e "s|%JOB_NAME%|${job_name}|g" -e "s|%JOB_DIR%|${job_dir}|g" "$template" > "$output_script"

>>>>>>> 49f37b7d0e8f7b0d79baf042b688e97b1c38ef6b
        	# Optional: submit the job
        	sbatch "$output_script"

        	echo "Generated $output_script"

	done
    done
done

