#!/bin/bash

# Step 1: Create a new Conda environment with Python 3.9
echo "Creating a new Conda environment named 'HDXRank' with Python 3.9..."
conda create -n HDXRank python=3.9 -y

# Step 2: Activate the environment
echo "Activating the 'HDXRank' environment..."
CONDA_BASE=$(conda info --base)
if [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    source "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate HDXRank
else
    echo "conda.sh not found. Make sure conda is installed and initialized."
    exit 1
fi

# Step 3: Install PyTorch with CUDA
#minimum version requirements: torch>=1.8.0
echo "Installing PyTorch with CUDA..."
conda install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y

# Step 4: Install TorchDrug
echo "Installing TorchDrug..."
conda install torchdrug -c milagraph -c conda-forge -c pytorch -c pyg -y

# Step 5: Install scikit-learn, scipy, and biotite
echo "Installing scikit-learn, scipy, and biotite..."
conda install scikit-learn scipy biotite -c conda-forge -y

# Step 6: Install additional packages with pip
echo "Installing Biopython, Openpyxl, and pdb2sql using pip..."
pip install biopython==1.83 openpyxl pdb2sql