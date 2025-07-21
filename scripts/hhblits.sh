#!/bin/bash

# Ensure the output directory exists
mkdir -p hhm_files

# Loop over all .fasta files in the ./fasta directory
for file in ./fasta_files/*.fasta; do
    # Extract base filename without extension
    BASE=$(basename "$file" .fasta)

    if [ -f "./hhm_data/${BASE}.hhm" ]; then
        echo "Skipping $file"
        continue
    fi
    
    # Run hhblits
    hhblits -i "$file" -ohhm "./fasta_files/${BASE}.hhm" -d ./databases/UniRef30_2020_06

    # Check if hhblits was successful
    if [ -f "./fasta_files/${BASE}.hhm" ]; then
        # Move the .hhm file to the hhm_files directory
        mv "./fasta_files/${BASE}.hhm" hhm_files/
    else
        echo "hhblits failed for $file"
    fi
done
