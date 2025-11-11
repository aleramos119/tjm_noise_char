#!/bin/bash

# Define your two parameter sets
<<<<<<< HEAD
cpus=(10)
#cpus=(8)
trajectories=(256 512 1024 2048 4096)
#trajectories=(1)
L=5

order=2

threshold="1e-6"
=======
cpus=(3 34)
#cpus=(8)
trajectories=(512)

L=100 

order=1

threshold="1e-4"
>>>>>>> 49f37b7d0e8f7b0d79baf042b688e97b1c38ef6b

# Paths
template="template.slurm"


method="scikit_tt"

<<<<<<< HEAD
solver="exact"

#g_rel_list=(0.05 0.1 0.2 0.4 0.6 0.8)
#g_deph_list=(0.05 0.1 0.2 0.4 0.6 0.8)


g_rel_list=(0.1)
g_deph_list=(0.1)



# Loop over all combinations
#
for g_rel in "${g_rel_list[@]}"; do
	for g_deph in "${g_deph_list[@]}"; do
=======
solver="krylov_5"

# Loop over all combinations
>>>>>>> 49f37b7d0e8f7b0d79baf042b688e97b1c38ef6b
for ncpus in "${cpus[@]}"; do
    for ntraj in "${trajectories[@]}"; do


<<<<<<< HEAD
        job_name="${ncpus}_cpus/${ntraj}_traj/gamma_rel_${g_rel}/gamma_deph_${g_deph}"
=======
        job_name="${ncpus}_cpus/${ntraj}_traj"
>>>>>>> 49f37b7d0e8f7b0d79baf042b688e97b1c38ef6b

        job_dir="method_${method}/solver_${solver}/order_${order}/threshold_${threshold}/${L}_sites/${job_name}"

        mkdir -p "${job_dir}"

        # Generate unique output filename
        output_script="$job_dir/run.slurm"

        # Replace placeholders in the template
<<<<<<< HEAD
        sed -e "s|%GAMMA_DEPH%|${g_deph}|g" -e "s|%GAMMA_REL%|${g_rel}|g" -e "s|%SOLVER%|${solver}|g" -e "s|%METHOD%|${method}|g" -e "s|%THRESHOLD%|${threshold}|g" -e "s|%ORDER%|${order}|g" -e "s|%L%|${L}|g"  -e "s|%NCPUS%|${ncpus}|g" -e "s|%NTRAJ%|${ntraj}|g" -e "s|%JOB_NAME%|${job_name}|g" -e "s|%JOB_DIR%|${job_dir}|g" "$template" > "$output_script"
=======
        sed -e "s|%SOLVER%|${solver}|g" -e "s|%METHOD%|${method}|g" -e "s|%THRESHOLD%|${threshold}|g" -e "s|%ORDER%|${order}|g" -e "s|%L%|${L}|g"  -e "s|%NCPUS%|${ncpus}|g" -e "s|%NTRAJ%|${ntraj}|g" -e "s|%JOB_NAME%|${job_name}|g" -e "s|%JOB_DIR%|${job_dir}|g" "$template" > "$output_script"
>>>>>>> 49f37b7d0e8f7b0d79baf042b688e97b1c38ef6b

        # Optional: submit the job
        sbatch "$output_script"

        echo "Generated $output_script"
    done
done

<<<<<<< HEAD


done

done

=======
>>>>>>> 49f37b7d0e8f7b0d79baf042b688e97b1c38ef6b
