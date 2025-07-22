# HDXRank [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.15426072.svg)](https://doi.org/10.5281/zenodo.15426072)

**HDXRank is an deep learning pipeline that applies HDX-MS (Hydrogen-Deuterium Exchange Mass Spectrometry) restraints to rank protein-protein complex predictions.**

## Overview

<img src="figures/HDXRank_overview.jpg" style="width:100%;">

HDXRank addresses the challenge of selecting accurate protein complex models by integrating experimental HDX-MS data with graph-based deep learning. The method uses HDX restraints to evaluate how well predicted complex structures align with experimental binding interface data, providing a robust framework for complex model ranking with improved prediction accuracy.

## Key Features

- **HDX-MS data integration** for experimental restraints
- **Support for multiple input sources** (docking predictions, AlphaFold models)
- **Flexible and extensible framework** for incorporating new experimental data

## Installation

HDXRank requires Python with CUDA 11.8 support. We provide both Docker and Conda installation options.

### Prerequisites
- [Docker](https://docs.docker.com/get-docker/) (recommended) or Conda
- CUDA 11.8 compatible GPU (for model training/prediction)

### Quick Start with Docker (Recommended)

1. **Clone the repository:**
```bash
git clone https://github.com/SuperChrisW/HDXRank.git
cd HDXRank
```

2. **Run with Docker:**
```bash
docker pull superchrisw/hdxrank:latest
docker run -it --rm -v $(pwd):/job/code superchrisw/hdxrank:latest /bin/bash
cd /job/code
python main.py --help
```

### Alternative: Conda Installation

```bash
chmod +x ./install.sh
./install.sh
conda activate HDXRank
python main.py --help
```

## Required Input Files

HDXRank requires four main types of input files:

1. **Protein Structure Files (`.pdb`)** - Complex structure predictions to be ranked + apo structures
2. **Multiple Sequence Alignments (`.hhm`)** - Generated using HHblits against UniRef30
3. **HDX-MS Data (`.xlsx`)** - Experimental HDX data with specific column format
4. **Configuration File (`.yaml`)** - Pipeline settings and parameters

### Preparing MSA Files

HDXRank requires `.hhm` format multiple sequence alignments generated using HHblits:

#### Install HHblits
```bash
conda create -n hhblits -y
conda activate hhblits
conda install hhsuite -c conda-forge -c bioconda -y
```

#### Download UniRef30 Database
```bash
mkdir -p databases
cd databases
wget http://wwwuser.gwdg.de/~compbiol/uniclust/2020_06/UniRef30_2020_06_hhsuite.tar.gz
tar -xvfz UniRef30_2020_06_hhsuite.tar.gz
rm UniRef30_2020_06_hhsuite.tar.gz
cd ..
```

#### Generate `.hhm` Files
```bash
bash ./scripts/hhblits.sh
```
This processes all `.fasta` files in `/HDXRank/fasta_files/` and saves `.hhm` files to `/HDXRank/hhm_files/`

### HDX-MS Data Format

Your Excel file should contain the following columns:
- `protein` - Protein identifier
- `state` - Experimental state (apo/complex)
- `start` - Peptide start position
- `end` - Peptide end position
- `sequence` - Peptide sequence
- `log_t` - Log exchange time
- `RFU` - Relative fractional uptake

## Usage

### Configuration Setup

HDXRank uses YAML configuration files to define all pipeline parameters. See `configs/config.template.yaml` for a complete template.

#### Key Configuration Sections:

**GeneralParameters:** File paths and execution mode 

**TaskParameters:** Control protein embedding and graph construction 

**PredictionParameters:** Model prediction settings 

**ScorerParameters:** Scoring and ranking settings 


### Running HDXRank

#### Basic Usage
```bash
python main.py --config path/to/config.yaml
```

### Output Files

Results are saved to the specified output directory:
- `HDX_scores.csv` - Ranked structures with HDXRank scores
- `predictions/` - Raw RFU predictions for each structure
- `results/scores/` - Detailed scoring analysis and plots

## Example Data

Download example datasets and configurations:

```bash
# HDX-MS dataset for training/validation
wget -O dataset.zip https://zenodo.org/records/15426072/files/dataset.zip?download=1
unzip dataset.zip

# Example structures and configurations
wget -O example.zip https://zenodo.org/records/15426072/files/example.zip?download=1
unzip example.zip

rm dataset.zip example.zip
```

## Model Training

### Preparing Training Data

1. **Add new HDX-MS files** to `dataset/HDX_files/`
2. **Update the dataset record** in `dataset/250110_HDXRank_dataset.xlsx`
3. **Generate embeddings and graphs:**
   ```bash
   python main.py --config ./configs/config_retrain_HDXRank.yaml
   ```

### Training the Model

```bash
python ./hdxrank/HDXRank_train.py --config ./configs/config_retrain_HDXRank.yaml
```

## Citation

If you use HDXRank in your research, please cite:

```bibtex
@article{Wang2025HDXRank,
  author    = {Liyao Wang, Andrejs Tucš, Songting Ding, Koji Tsuda and Adnan Sljoka},
  title     = {HDXRank: A Deep Learning Framework for Ranking Protein Complex Predictions With Hydrogen–Deuterium Exchange Data},
  journal   = {Journal of Chemical Theory and Computation},
  year      = {2025},
  doi       = {10.1021/acs.jctc.5c00175}
}
```

## Support

For questions, bug reports, or feature requests, please open an issue on GitHub


