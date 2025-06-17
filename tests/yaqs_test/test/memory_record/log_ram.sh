#!/bin/bash

PID=$1                # Pass the PID as the first argument
OUTFILE=$2            # Pass output filename as the second argument

echo "timestamp,ram_MB" > "$OUTFILE"  # CSV header

while kill -0 "$PID" 2>/dev/null; do
    RAM_KB=$(ps -p "$PID" -o rss=)
    RAM_MB=$(awk "BEGIN {printf \"%.2f\", $RAM_KB/1024}")
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    echo "$TIMESTAMP,$RAM_MB" >> "$OUTFILE"
    sleep 1
done

echo "Process $PID finished. Logging stopped."

