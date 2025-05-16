#!/bin/bash

# Step 1: Create a new Conda environment with Python 3.9
echo "Creating a new Conda environment named 'HDXRank' with Python 3.9..."
conda update -n base -c conda-forge conda
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
pip install torch==2.0.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Step 4: Install TorchDrug
echo "Installing TorchDrug..."
pip install numpy==1.26
pip install torch-scatter torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+cu118.html
pip install torchdrug

# Step 5: Install scikit-learn, scipy, and biotite
echo "Installing scikit-learn, scipy, and biotite..."
pip install scikit-learn biotite

# Step 6: Install additional packages with pip
echo "Installing Biopython, Openpyxl, and pdb2sql using pip..."
pip install biopython==1.83 openpyxl pdb2sql