# HDXRank [![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14625492.svg)](https://doi.org/10.5281/zenodo.14625492)
**HDXRank is an open-source pipeline to apply HDX-MS restraints to protein-protein complex prediction ranking.**

## Method overview:
<img src="figures/HDXRank_overview.jpg" style="width:100%;">
Integrating sparse experimental data into protein complex modeling workflows can significantly improve the accuracy and reliability of predicted models. Despite the valuable insights that hydrogen-deuterium exchange (HDX) data provide about protein binding interfaces, there is currently no standard protocol for incorporating this information into the complex model selection process. Inspired by advances in graph-based deep learning for protein representation, we utilize it as a backbone for a flexible scoring framework in protein-protein complex model ranking based on their alignment with experimental HDX profiles. It offers a robust, HDX-informed selection protocol with improved prediction accuracy.

## Installation:
Assume a Linux environment with CUDA dependencies, otherwise please use Google colab
clone the repository and Use the `install.sh` file to create a Conda environment with all necessary dependencies:
```
git clone https://github.com/SuperChrisW/HDXRank.git
cd HDXRank
chmod +x ./install.sh
./install.sh
conda activate HDXRank
```

## Preparation
* Obtain HDX-MS dataset and examples from Zenodo: 10.5281/zenodo.14625492, unzip and move to the HDXRank root directory.
* Install Hhblits to get MSA file, temporarily refers to AI-HDX document(https://github.com/Environmentalpublichealth/AI-HDX/blob/main/Documentations/MSA_embedding.md)

## Getting Started
HDXRank requires three input files:

1. **Protein structure file** (`.pdb`)  
2. **MSA file** (`.hhm`)  
3. **HDX-MS file** (`.xlsx`)  

Additionally, HDXRank uses a settings file (`.xml`) to control the pipeline.
We offers examples for ranking docking and AF predictions in folder `example`, as shown in our paper.
Users can run HDXRank with any `.xml` file under subfolder in folder `example` to get HDX predictions, such as:
```bash
python main.py -input ./example/1UGH_docking/BatchTable_1UGH.xml
```

### Workflow:

1. **Protein embedding**: HDXRank extracts embeddings from `.pdb` and `.hhm` files.  
2. **Protein graph construction**: Constructs a protein graph from the `.pdb` file.
3. **Peptide graph splitting**: Splits the protein graph into peptide graphs based on the provided HDX-MS `.xlsx` file.

### Execution:
With all input files prepared, run the following command to start the pipeline:
```bash
python main.py -input input.xml
```

## Merge data and Retrain the model:
HDXRank model was trained upon a curated HDX-MS dataset collected from public database PRIDE and MassIVE, up to March 2024. New HDX-MS data can be merged with the current dataset and used to re-train our model.

To merge the newly collected data to dataset:

1. **copy HDX-MS file into `dataset/HDX_files/`**: the table should contain columns `protein` `state` `start` `end` `sequence` `log_t` `RFU`.
2. **update record file `dataset/250110_HDXRank_dataset.xlsx`**: record all `protein+state` pairs and corresponding structures.
3. **run HDXRank to generate embedding and peptide graphs**:
```bash
python main.py -input ./dataset/BatchTable_setting.xml
```

To re-train the HDXRank model:
```bash
python HDXRank_train.py -input ./dataset/BatchTable_setting.xml -save ./Model
```

## Citing HDXRank
please cite the following paper: 
https://doi.org/10.1021/acs.jctc.5c00175
