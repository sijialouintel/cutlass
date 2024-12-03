#!/bin/bash

# Get the directory where the current script is located
base_dir="$(dirname "$0")"

# Find all files named run_online.sh and execute them
find "$base_dir" -type f -name "run_online.sh" | while read -r script; do
    subdir="$(dirname "$script")"
    log_file="log.txt"
    echo "Running $script with arguments: $@"
    if (cd "$subdir" && ./run_online.sh "$@" > "$log_file" 2> /dev/null); then
        echo "Execution of $script succeeded."
    else
        echo "Execution of $script failed. Check $subdir/$log_file for details."
    fi
done
