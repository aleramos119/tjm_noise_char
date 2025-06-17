

(
for i in {1..300}; do
  timestamp=$(date +%s)
  mem_bytes=$(cat /sys/fs/cgroup/memory/slurm/uid_$(id -u)/job_$1/memory.usage_in_bytes)
  echo "$timestamp,$mem_bytes" >> mem_usage.csv
  sleep 1
done
) &

