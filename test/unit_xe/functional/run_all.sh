#!/bin/bash

# Get the directory where the current script is located
base_dir="$(dirname "$0")"

# Find all files named run_online.sh and execute them
find "$base_dir" -type f -name "run_online.sh" | while read -r script; do
    subdir="$(dirname "$script")"
    log_file="log.txt"
    echo "Running $script with arguments: $@"
    if (cd "$subdir" && time ./run_online.sh "$@" > "$log_file" 2>&1); then
        echo "Execution of $script succeeded."
        # Count occurrences of "Test Pass" and "Test Fail" in the log file and echo the results
        echo "Test Pass count: $(grep -c "Test Pass" "$log_file")"
        echo "Test Fail count: $(grep -c "Test Fail" "$log_file")"
    else
        echo "Execution of $script failed. Check $subdir/$log_file for details."
    fi
done
