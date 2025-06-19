#!/bin/bash

# Go through all files ending with .log in subdirectories
find . -type f -name "*.log" | while read logfile; do
    # Extract the JOB_ID from the filename (assumes filename is JOB_ID.log)
    filename=$(basename "$logfile")
    job_id="${filename%.log}"

    # Run `seff` and write output to a file called `seff_output` in the same directory
    logdir=$(dirname "$logfile")
    seff "$job_id" > "$logdir/seff_output"

    echo "Processed JOB_ID $job_id -> $logdir/seff_output"
done
