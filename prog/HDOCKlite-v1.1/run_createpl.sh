#!/bin/bash

# Define the save directory
save_dir="/home/lwang/models/gromacs-docker/BMC_struct/EP20_PduA_docked"

# Check if the save directory exists, if not, create it
if [ ! -d "$save_dir" ]; then
    mkdir -p "$save_dir"
fi

# Define an array of Hdock.out files to be processed
file_list=(
    "PduP20_PduA_diHex.out"
    "PduL20_PduA_diHex.out"
    "PduD20_PduA_diHex.out"
    # Add more files as needed
)

# Loop through each 'Hdock.out' file in the file_list array
for hdock_file in "${file_list[@]}"; do
    # Create a subfolder in save_dir based on the file name
    subfolder="${save_dir}/$(basename "$hdock_file" .out)"
    mkdir -p "$subfolder"

    if [ -f "$hdock_file" ]; then
        echo "Processing file: $hdock_file"

        # Run the creapl command
        ./createpl "$hdock_file" top10.pdb -nmax 10 -complex -models -rmsd 5

        # Move the generated pdb files to the subfolder
        for i in {1..10}; do
            model_file="model_${i}.pdb"
            if [ -f "$model_file" ]; then
                mv "$model_file" "$subfolder"
            fi
        done
    else
        echo "File $hdock_file not found."
    fi
done

echo "All specified files have been processed and moved to their respective subfolders."
