import os
import re
import math
import warnings
import yaml
import logging
from dataclasses import dataclass, field
from collections import defaultdict
from itertools import groupby
from matplotlib import pyplot as plt

import pandas as pd
import numpy as np
import torch

from Bio.PDB import Selection
from BioWrappers import *
from GVP_strucFeats import *

# Suppress warnings globally
warnings.filterwarnings("ignore")

# --- Configuration Utilities ---

def config_process(config_file):
    """
    Parses a YAML configuration file, processes all necessary parameters,
    and attaches apo_states and complex_states directly into the config dict.

    Args:
        config_file (str): Path to the YAML configuration file.

    Returns:
        dict: The processed configuration dictionary with all settings,
              including 'apo_states' and 'complex_states' as keys.
    """
    # Load YAML
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Extract scorer and task settings
    general_settings = config.get('GeneralParameters', {})
    scorer_settings = config.get('ScorerParameters', {})
    task_settings = config.get('TaskParameters', {})
    prediction_settings = config.get('PredictionParameters', {})

    # Path joining
    root_dir = general_settings.get("RootDir", ".")
    if "GeneralParameters" in config:
        for key in ["pepGraphDir", "EmbeddingDir", "PDBDir", "HDXDir", "hhmDir"]:
            if key in general_settings and general_settings[key]:
                general_settings[key] = os.path.join(root_dir, general_settings[key])
    if "PredictionParameters" in config:
        for key in ["ModelDir", "PredDir"]:
            if key in prediction_settings and prediction_settings[key]:
                prediction_settings[key] = os.path.join(root_dir, prediction_settings[key])

    # Post-process PepRange for GraphParameters
    if "TaskParameters" in config:
        pep_range = task_settings.get("PepRange") # pepRange e.g. A:300-600,B:300-600,C:300-600, convert to dict {chainID: [start, end]}
        if isinstance(pep_range, str) and ':' in pep_range:
            pep_range_dict = {}
            # Split by comma to get each chain's range
            for chain_range in pep_range.split(','):
                chain_range = chain_range.strip()
                if ':' in chain_range:
                    # Split into chain ID and range
                    chain_id, range_str = chain_range.split(':')
                    chain_id = chain_id.strip()
                    # Split range into start and end
                    if '-' in range_str:
                        start, end = map(int, range_str.split('-'))
                        pep_range_dict[chain_id] = [start, end]
            task_settings["PepRange"] = pep_range_dict
        elif pep_range is None:
            task_settings["PepRange"] = {}

    # Parse protein states
    apo_states = []
    complex_states = []
    for complex_info in task_settings.get('HDX_Chains', []):
        apo = complex_info.get('apo_state', {})
        complex = complex_info.get('complex_state', {})
        ref_structure = complex_info.get('ref_structure', None)

        apo_states.append((
            apo.get('protein'),
            apo.get('state'),
            apo.get('correction_value'),
            apo.get('hhm_prefix'),
            apo.get('chainID'),
            ref_structure
        ))

        complex_states.append((
            complex.get('protein'),
            complex.get('state'),
            complex.get('correction_value'),
            complex.get('hhm_prefix'),
            complex.get('chainID'),
            ref_structure
        ))

    if not apo_states or not complex_states:
        raise ValueError("Apo and complex states must be defined in the configuration.")
    if not scorer_settings:
        raise ValueError("Scorer settings must be defined in the configuration.")

    # Store states directly into config
    config["apo_states"] = apo_states
    config["complex_states"] = complex_states
    config["structure_list"] = [f for f in os.listdir(general_settings.get("PDBDir", None)) if f.endswith('.pdb')]
    config.get("TaskParameters", {}).update(task_settings) # update task settings
    config.get("ScorerParameters", {}).update(scorer_settings) # update scorer settings
    config.get("GeneralParameters", {}).update(general_settings) # update general settings
    return config

# TODO: modify parse_xlsx_task to return a list of tasks
def parse_xlsx_task(tasks):
    """
    Parse an Excel task file for batch/train mode.
    """
    task_fpath = os.path.join(tasks["GeneralParameters"]["RootDir"], f"{tasks['GeneralParameters']['TaskFile']}.xlsx")
    if not os.path.exists(task_fpath):
        raise FileNotFoundError(f"Missing task file in {task_fpath}")

    df = pd.read_excel(task_fpath, sheet_name="Sheet1")
    df = df.dropna(subset=['structure_file'])
    df = df[df['complex_state'] != 'ligand complex']

    apo_identifier = list(df['structure_file'].astype(str).unique())
    protein, state, chain_identifier = [], [], []
    correction = []
    database_id = []
    protein_chains = []
    complex_state = []
    structure_list = []
    embedding_fname = []

    for temp_apo in apo_identifier:
        temp_df = df[(df['structure_file'] == temp_apo)].dropna(subset=['protein', 'state', 'chain_identifier', 'correction_value', 'protein_chain', 'complex_state'])
        if temp_df.empty:
            raise ValueError(f"No data found for structure {temp_apo} in the task file.")

        temp_protein = temp_df['protein'].astype(str).to_list()
        temp_state = temp_df['state'].astype(str).to_list()
        temp_chain = temp_df['chain_identifier'].astype(str).to_list()
        temp_correction = temp_df['correction_value'].astype(int).to_list()
        temp_protein_chains = temp_df['protein_chain'].astype(str).to_list()
        temp_complex_state = temp_df['complex_state'].astype(str).to_list()

        structure_list.append(temp_apo)
        embedding_fname.append([temp_apo.upper()] * len(temp_protein_chains))
        protein.append(temp_protein)
        state.append(temp_state)
        chain_identifier.append(temp_chain)
        correction.append(temp_correction)
        database_id.extend(temp_df['database_id'].astype(str).unique())
        protein_chains.append(temp_protein_chains)
        complex_state.append(temp_complex_state[0])

    keys = [
        database_id, protein, state, structure_list, chain_identifier,
        correction, protein_chains, complex_state, embedding_fname
    ]
    return keys

def parse_task(input_file):
    """
    Parse a YAML configuration file for graph construction tasks.

    Args:
        input_file (str): Path to the YAML input file.

    Returns:
        list: Keys for creating graph tasks.
        dict: Parsed tasks.
    """
    tasks = config_process(input_file)
    keys = None
    if "Mode" in tasks["GeneralParameters"]:
        mode = tasks["GeneralParameters"]["Mode"]
        if mode.lower() in ['batch', 'train']:
            keys = parse_xlsx_task(tasks)
        elif mode.lower() == 'single':
            hdx_filename = tasks["GeneralParameters"]["HDX_File"]
            apo_states = tasks["apo_states"]
            complex_states = tasks["complex_states"]
            pdb_files = tasks['structure_list']
            protein_chain_hhms = {item[-2]:f"{item[-3]}_{item[-2]}" for item in tasks['apo_states']}
            protein_chains = protein_chain_hhms.keys()
            tasks['TaskParameters']['protein_chain_hhms'] = protein_chain_hhms
            tasks['TaskParameters']['is_complex'] = ['protein complex'] * len(pdb_files) # 1: protein complex, 0: single

            if not pdb_files:
                raise FileNotFoundError(f"No PDB files found in {tasks['GeneralParameters']['PDBDir']}")
            else:
                logging.info(f"Found {len(pdb_files)} PDB files")
        else:
            raise ValueError("Invalid mode specified in YAML file.")

    logging.info(f"Parsed Tasks: {len(keys[0]) if keys else 0}")
    logging.info("General settings:")
    for key, value in tasks.get("GeneralParameters", {}).items():
        logging.info(f"  {key}: {value}")
    return tasks

# --- Chemical Data and Feature Utilities ---

class ChemData:
    def __init__(self):
        self.NAATOKENS = 20 + 1 + 10 + 10 + 1  # 20 AAs, 1 UNK res, 8 NAs+2UN NAs, 10 atoms +1 UNK atom
        self.UNKINDEX = 20  # residue unknown

        # Bond types
        self.num2btype = [0, 1, 2, 3, 4, 5, 6, 7]

        self.NATYPES = ['DA', 'DC', 'DG', 'DT', 'DU', 'A', 'T', 'C', 'G', 'U']
        self.STDAAS = [
            'ALA', 'ARG', 'ASN', 'ASP', 'CYS', 'GLN', 'GLU', 'GLY', 'HIS', 'ILE',
            'LEU', 'LYS', 'MET', 'PHE', 'PRO', 'SER', 'THR', 'TRP', 'TYR', 'VAL'
        ]

        self.three_to_one = {
            'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
            'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
            'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
            'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
        }
        self.one_to_three = {v: k for k, v in self.three_to_one.items()}

        self.residue_charge = {
            'CYS': -0.64, 'HIS': -0.29, 'ASN': -1.22, 'GLN': -1.22, 'SER': -0.80,
            'THR': -0.80, 'TYR': -0.80, 'TRP': -0.79, 'ALA': -0.37, 'PHE': -0.37,
            'GLY': -0.37, 'ILE': -0.37, 'VAL': -0.37, 'MET': -0.37, 'PRO': 0.0,
            'LEU': -0.37, 'GLU': -1.37, 'ASP': -1.37, 'LYS': -0.36, 'ARG': -1.65
        }
        self.residue_polarity = {
            'CYS': 'polar', 'HIS': 'polar', 'ASN': 'polar', 'GLN': 'polar',
            'SER': 'polar', 'THR': 'polar', 'TYR': 'polar', 'TRP': 'polar',
            'ALA': 'apolar', 'PHE': 'apolar', 'GLY': 'apolar', 'ILE': 'apolar',
            'VAL': 'apolar', 'MET': 'apolar', 'PRO': 'apolar', 'LEU': 'apolar',
            'GLU': 'neg_charged', 'ASP': 'neg_charged', 'LYS': 'neg_charged', 'ARG': 'pos_charged'
        }
        self.polarity_encoding = {'apolar': 0, 'polar': 1, 'neg_charged': 2, 'pos_charged': 3}
        self.ss_list = ['H', 'B', 'E', 'G', 'I', 'T', 'S', 'P', '-']
        self.AA_array = {
            "A": [-0.591, -1.302, -0.733,  1.570, -0.146],
            "C": [-1.343,  0.465, -0.862, -1.020, -0.255],
            "D": [ 1.050,  0.302, -3.656, -0.259, -3.242],
            "E": [ 1.357, -1.453,  1.477,  0.113, -0.837],
            "F": [-1.006, -0.590,  1.891, -0.397,  0.412],
            "G": [-0.384,  1.652,  1.330,  1.045,  2.064],
            "H": [ 0.336, -0.417, -1.673, -1.474, -0.078],
            "I": [-1.239, -0.547,  2.131,  0.393,  0.816],
            "K": [ 1.831, -0.561,  0.533, -0.277,  1.648],
            "L": [-1.019, -0.987, -1.505,  1.266, -0.912],
            "M": [-0.663, -1.524,  2.219, -1.005,  1.212],
            "N": [ 0.945,  0.828,  1.299, -0.169,  0.933],
            "P": [ 0.189,  2.081, -1.628,  0.421, -1.392],
            "Q": [ 0.931, -0.179, -3.005, -0.503, -1.853],
            "R": [ 1.538, -0.055,  1.502,  0.440,  2.897],
            "S": [-0.228,  1.399, -4.760,  0.670, -2.647],
            "T": [-0.032,  0.326,  2.213,  0.908,  1.313],
            "V": [-1.337, -0.279, -0.544,  1.242, -1.262],
            "W": [-0.595,  0.009,  0.672, -2.128, -0.184],
            "Y": [ 0.260,  0.830,  3.097, -0.838,  1.512]
        }# HDMD data adopted from Atchley et al. (2005): https://www.pnas.org/doi/epdf/10.1073/pnas.0408677102

chemdata = ChemData()

# --- Data Classes ---

@dataclass
class RawInputData:
    msa: torch.Tensor = field(default_factory=torch.Tensor)
    res_HDMD: torch.Tensor = field(default_factory=torch.Tensor)
    res_polarity: torch.Tensor = field(default_factory=torch.Tensor)
    res_charge: torch.Tensor = field(default_factory=torch.Tensor)
    SASA: torch.Tensor = field(default_factory=torch.Tensor)
    hse: torch.Tensor = field(default_factory=torch.Tensor)
    dihedrals: torch.Tensor = field(default_factory=torch.Tensor)
    orientations: torch.Tensor = field(default_factory=torch.Tensor)
    seq_data: dict = field(default_factory=dict)  # [id]: {'token_type', 'coord'}
    type_label: str = ''

    def construct_embedding(self):
        embedding = torch.cat((
            self.msa, self.res_HDMD, self.res_polarity, self.res_charge,
            self.SASA, self.hse, self.dihedrals, self.orientations
        ), dim=1)
        return embedding

    def merge(self, data):
        self.seq_data = {**self.seq_data, **data.seq_data}
        return self

# --- Feature Extraction Functions ---

def load_embedding(fpath):
    if not os.path.isfile(fpath):
        logging.error(f"Missing embedding file: {fpath}")
        return False
    embedding_dict = torch.load(fpath)
    protein_embedding = embedding_dict['embedding']
    chain_label = np.array(embedding_dict['chain_label'])
    return chain_label, protein_embedding

def get_seq_polarity(seq):
    encode_index = [
        chemdata.polarity_encoding[chemdata.residue_polarity[chemdata.one_to_three[res]]]
        for res in seq
    ]
    polarity_mtx = np.zeros((len(seq), 4))
    for i, idx in enumerate(encode_index):
        polarity_mtx[i, idx] = 1
    return polarity_mtx

def parse_hhm(hhm_file, pepRange=None):
    # adapted from AI-HDX: https://github.com/Environmentalpublichealth/AI-HDX
    hhm_mtx = []
    with open(hhm_file) as f:
        for i in f:
            if i.startswith("HMM"):
                break
        for _ in range(2):
            f.readline()
        lines = f.read().split("\n")
        sequence = ""
        for idx in range(0, int((len(lines) - 2) / 3) + 1):
            first_line = lines[idx * 3].replace("*", "99999")
            next_line = lines[idx * 3 + 1].replace("*", "99999")
            content1 = first_line.strip().split()
            content2 = next_line.strip().split()
            if content1[0] == '//':
                break
            elif content1[0] == '-':
                continue
            sequence += str(content1[0])
            hhm_val1 = [10 / (1 + math.exp(-1 * int(val1) / 2000)) for val1 in content1[2:-1]]
            hhm_val2 = [10 / (1 + math.exp(-1 * int(val2) / 2000)) for val2 in content2]
            hhm_val = hhm_val1 + hhm_val2
            hhm_mtx.append(hhm_val)
    return np.array(hhm_mtx), sequence

# --- Protein, Ligand, and Nucleic Acid Loading ---

def load_protein(hhm_file, pdb_file, chain_id, pepRange = None):
    """
    Generate embedding file from the pre-computed HMM file and rigidity file, PDB structure
    processing the protein chain featurization
    """
    model = get_bio_model(pdb_file, pepRange)
    residue_data = {}
    residue_coord = []
    res_seq = []
    residue_list = Selection.unfold_entities(model, 'R')
    logging.info(f"residue numbers:{len(residue_list)}")
    for res in residue_list:
        if res.get_parent().get_id() != chain_id:
            continue
        res_id = res.get_id()[1]
        res_name = res.get_resname()
        if res_name in chemdata.STDAAS:
            res_seq.append(chemdata.three_to_one[res_name])
        else:
            logging.error(f'Non-standard AA found: {res_name}')
        try:
            N_coord = list(res['N'].get_coord() if 'N' in res else [0, 0, 0])
            Ca_coord = list(res['CA'].get_coord() if 'CA' in res else [0, 0, 0])
            C_coord = list(res['C'].get_coord() if 'C' in res else [0, 0, 0])
        except KeyError:
            logging.error(f'KeyError at residue {res_id} {res_name} in chain {chain_id}')
        res_coord = [N_coord, Ca_coord, C_coord]
        residue_coord.append(res_coord)
        residue_data[res_id] = {
            'token_type': res_name,
            'coord': Ca_coord,
        }

    max_len = len(res_seq)
    # sequence-based features
    res_charge = np.array([
        chemdata.residue_charge[chemdata.one_to_three[res]] for res in res_seq
    ]).reshape(-1, 1)
    res_polarity = get_seq_polarity(res_seq)
    HDMD = np.array([chemdata.AA_array[res] for res in res_seq]).reshape(-1, 5)

    # physical-based features
    hse_dict = get_hse(model, chain_id)
    corrected_hse_mtx = np.zeros((max_len, 3))
    for i, res_j in enumerate(residue_data.keys()):
        res_j = str(res_j)
        if (chain_id, res_j) in hse_dict.keys():
            corrected_hse_mtx[i, :] = list(hse_dict[(chain_id, res_j)])
    SASA = biotite_SASA(pdb_file, chain_id)[:max_len]

    # MSA-based features
    hhm_mtx, hhm_seq = parse_hhm(hhm_file) # hhm file is chain-wise
    logging.debug(f"hhm_mtx shape: {hhm_mtx.shape}, hhm_seq length: {len(hhm_seq)}")
    # Align sequences and get masks, 1 if the residue is different, 0 if the residue is aligned
    res_seq = ''.join(res_seq)
    _, hhm_mask = max_aligned_seq(res_seq, hhm_seq)
    hhm_mtx = hhm_mtx[~hhm_mask, :]
    hhm_seq = "".join(hhm_seq[i] for i in range(len(hhm_seq)) if not hhm_mask[i])
    logging.debug(f"hhm_mtx shape: {hhm_mtx.shape}, hhm_seq length: {len(hhm_seq)}")

    # structura based features: dihedral angels and orientations
    GVP_feats = ProteinGraphDataset(data_list=[])
    dihedrals = GVP_feats._dihedrals(torch.as_tensor(residue_coord))
    orientations = GVP_feats._orientations(torch.as_tensor(residue_coord))

    if not hhm_seq == res_seq:
        print("hhm_sequenece:", hhm_seq)
        print("dssp_sequenece:", res_seq)
        raise ValueError('Sequence mismatch between HMM and DSSP')

    return RawInputData(
        msa=torch.tensor(hhm_mtx, dtype=torch.float32),
        res_HDMD=torch.tensor(HDMD, dtype=torch.float32),
        res_polarity=torch.tensor(res_polarity, dtype=torch.float32),
        res_charge=torch.tensor(res_charge, dtype=torch.float32),
        SASA=torch.tensor(SASA, dtype=torch.float32).reshape(-1, 1),
        hse=torch.tensor(corrected_hse_mtx, dtype=torch.float32),
        dihedrals=torch.tensor(dihedrals, dtype=torch.float32),
        orientations=torch.tensor(orientations, dtype=torch.float32),
        seq_data=residue_data,
        type_label='protein'
    )

def load_sm(pdb_file, preset_chain_id):
    atom_data = {}
    structure = get_bio_model(pdb_file)
    with open(pdb_file, 'r') as f:
        data = f.read().strip().split('\n')
        for line in data:
            if line[:6] == 'HETATM':
                chain_id = line[21]
                if chain_id != preset_chain_id:
                    continue
                LG_name = line[17:20].replace(' ', '')
                atom_id = int(line[6:11].strip())
                atom_type = line[12:16].strip()
                element_symbol_regex = ''.join(re.findall('[A-Za-z]', atom_type)).upper()
                token_type = chemdata.aa2num[element_symbol_regex] if element_symbol_regex in chemdata.aa2num else chemdata.aa2num['ATM']
                res_id = line[22:26]
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                atom_data[atom_id] = {
                    'token_type': token_type,
                    'coord': [x, y, z]
                }
        if len(atom_data.keys()) == 0:
            print('No small molecule found in the PDB file')
            return None
    return RawInputData(
        msa=None,
        res_HDMD=None,
        res_polarity=None,
        res_charge=None,
        SASA=None,
        hse=None,
        dihedrals=None,
        orientations=None,
        seq_data=atom_data,
        type_label='ligand'
    )

def load_nucleic_acid(pdb_file, chain_id):
    na_data = {}
    model = get_bio_model(pdb_file)
    for chain in model.get_chains():
        if chain.get_id() != chain_id:
            continue
        for res in chain:
            na_id = res.get_id()[1]
            na_name = res.get_resname().strip()
            if na_name not in chemdata.NATYPES:
                na_name = 'UNK'
            else:
                if na_name in ['DA', 'A', 'RA']:
                    na_name = 'A'
                elif na_name in ['DC', 'C', 'RC']:
                    na_name = 'C'
                elif na_name in ['DG', 'G', 'RG']:
                    na_name = 'G'
                elif na_name in ['DT', 'T']:
                    na_name = 'T'
                elif na_name in ['RU', 'U']:
                    na_name = 'U'
            atom_coord = [atom.get_coord() for atom in res.get_atoms()]
            na_coord = np.mean(atom_coord, axis=0)
            na_data[na_id] = {
                'token_type': chemdata.aa2num[na_name],
                'coord': na_coord,
            }
    return RawInputData(
        msa=None,
        res_HDMD=None,
        res_polarity=None,
        res_charge=None,
        SASA=None,
        hse=None,
        dihedrals=None,
        orientations=None,
        seq_data=na_data,
        type_label='NA'
    )

# --- PDB Parsing and Chain Utilities ---

class Chain:
    def __init__(self):
        self.atoms = []
        self.sequence_type = None

    def add_atom(self, atom_index, residue_index, residue_type, atom_type, coordinates):
        atom_info = {
            'atom_index': atom_index,
            'residue_index': residue_index,
            'residue_type': residue_type,
            'atom_type': atom_type,
            'coordinates': coordinates
        }
        self.atoms.append(atom_info)

    def get_atoms(self):
        return self.atoms

    def get_residues(self):
        atoms = self.get_atoms()
        key = lambda x: x['residue_index']
        residues = [list(group) for key, group in groupby(atoms, key)]
        return residues

# --- HDXRank Scorer  ---

pred_labels = ['Batch','Y_Pred_short','Y_Pred_middle','Y_Pred_long','Chain','Range']

class Scorer:
    @staticmethod
    def max_min_scale(data):
        return (data - np.min(data)) / (np.max(data) - np.min(data))

    @staticmethod
    def get_hdx_epitopes(true_diff):
        x_labels = list(true_diff.keys())
        x_pos = np.arange(len(x_labels))
        diff = np.array(list(true_diff.values()))
        mean_diff = diff[diff < 0].mean()
        hdx_epitope_id = np.where(diff < mean_diff)[0]
        hdx_epitope_pep = [x_labels[i] for i in hdx_epitope_id]
        return hdx_epitope_id, hdx_epitope_pep
        
    @staticmethod
    def get_weighted_uptake(HDX_df, protein, state, cluster_id, correction, timepoints=[1.35, 2.85]):
        """
        Calculate weighted and unweighted HDX uptake for a specified protein/state/cluster.
        
        Args:
            HDX_df (pd.DataFrame): DataFrame with HDX-MS data, containing columns ['state', 'protein', 'log_t', 'exposure', 'start', 'end', 'RFU'].
            protein (str): Protein identifier.
            state (str): State identifier.
            cluster_id (int): Cluster index (0, 1, or 2).
            timepoints (list-like): List or array with at least two float log time boundaries.
            correction (int): Integer correction to be added to peptide start/end indices.

        Returns:
            weighted_uptake (np.ndarray): Mean RFU for each peptide in the cluster, normalized to [0,1].
            x_label (list of str): Peptide label with correction, e.g. "5-13".
            unweighted_RFU (dict): {exposure_time: {peptide_label: RFU}}
        """
        # Filter for relevant protein and state
        HDX_df['protein'] = HDX_df['protein'].apply(lambda x: x.strip().replace(' ', ''))
        HDX_df['state'] = HDX_df['state'].apply(lambda x: x.strip().replace(' ', ''))
        temp = HDX_df.loc[(HDX_df['state'] == state) & (HDX_df['protein'] == protein)]
        logging.debug(f'Processing {len(temp)} rows for protein: {protein}, state: {state}, cluster_id: {cluster_id}')

        if not isinstance(timepoints, (list, tuple, np.ndarray)) or len(timepoints) < 2:
            raise ValueError('timepoints must be a list/array of at least 2 elements')
        if cluster_id == 0:
            temp = temp[temp['log_t'] < timepoints[0]]
        elif cluster_id == 1:
            temp = temp[(temp['log_t'] < timepoints[1]) & (temp['log_t'] >= timepoints[0])]
        elif cluster_id == 2:
            temp = temp[temp['log_t'] >= timepoints[1]]
        else:
            raise ValueError(f'Invalid cluster_id: {cluster_id} (must be 0, 1, or 2)')

        # Precompute peptide labels with correction
        temp = temp.copy()
        temp['peptide_label'] = (temp['start'] + correction).astype(str) + '-' + (temp['end'] + correction).astype(str)
        temp = temp.sort_values(by='peptide_label')

        unweighted_RFU_df = temp.pivot(index='peptide_label', columns='exposure', values='RFU')
        # Convert to dict-of-dict: {exposure: {peptide_label: RFU}}
        unweighted_RFU = {exposure: unweighted_RFU_df[exposure].dropna().to_dict() for exposure in unweighted_RFU_df.columns}

        grouped = temp.groupby(['start', 'end'], sort=False)
        weighted_uptake = grouped['RFU'].mean().to_numpy() / 100
        x_label = [f'{start+correction}-{end+correction}' for start, end in grouped.groups.keys()]

        return weighted_uptake, temp['peptide_label'].to_numpy(), unweighted_RFU

    @staticmethod
    def get_true_diff(HDX_fpath, apo_states, complex_states, cluster_id, timepoints):
        """
        Calculate true HDX difference and exposure-wise difference matrices for common peptides.
        
        Args:
            HDX_fpath (str): Path to HDX-MS data Excel file.
            apo_states (tuple/list): (protein, state, correction) for apo.
            complex_states (tuple/list): (protein, state, correction) for complex.
            cluster_id (int): Cluster index to use.
            timepoints (list): Time boundary list for clusters.

        Returns:
            true_diff (dict): {peptide_label: RFU difference (complex - apo)}
            diff_mtx (dict): {peptide_label: {exposure_time: RFU difference}}
        """

        HDX_df = pd.read_excel(HDX_fpath)
        HDX_df['protein'].apply(lambda x: x.strip().replace('_', ' '))
        HDX_df['state'].apply(lambda x: x.strip().replace('_', ' '))
        
        def get_uptake(states):
            protein, state, correction, _, _, _ = states
            protein = protein.strip().replace(' ', '')
            state = state.strip().replace(' ', '')
            uptake, labels, mtx = Scorer.get_weighted_uptake(HDX_df, protein, state, cluster_id, correction, timepoints)
            return dict(zip(labels, uptake)), mtx

        true_apo, apo_mtx = get_uptake(apo_states)
        true_complex, complex_mtx = get_uptake(complex_states)

        # Find common peptide labels
        common_keys = sorted(set(true_apo) & set(true_complex), key=lambda x: sum(map(int, x.split('-'))))

        # Compute differences
        true_diff = {k: true_complex[k] - true_apo[k] for k in common_keys}
        diff_mtx = {
            k: {t: complex_mtx[t][k] - apo_mtx[t][k]
                for t in apo_mtx
                if k in apo_mtx[t] and k in complex_mtx.get(t, {})}
            for k in common_keys
        }

        return true_diff, diff_mtx

    @staticmethod
    def parse_predictions(pred_dir, suffix=''):
        """
        Parse predictions from a directory and return a DataFrame with scores.
        
        Args:
            pred_dir (str): Directory containing prediction files.

        Returns:
            pd.DataFrame: DataFrame with columns ['peptide', 'score'].
        """
        pred_files = [f for f in os.listdir(pred_dir) if f.endswith('.csv')]

        if suffix:
            pred_files = [f for f in pred_files if f.startswith(suffix)]
        all_scores = []

        for file in pred_files:
            df = pd.read_csv(os.path.join(pred_dir, file))
            all_scores.append(df[pred_labels])

        return pd.concat(all_scores, ignore_index=True)

    @staticmethod
    def root_mean_square_error(y_true, y_pred, error_limit=1):
        return np.mean(((y_true - y_pred) / error_limit) ** 2)

    @staticmethod
    def prepare_data(pred_df_dict, complex_batch, apo_states, hdx_true_diffs, hdx_epitope_peps=None, pred_cluster=1):
        truth = []
        pred = []
        pred_cluster_dict = {0: 'Y_Pred_short', 1: 'Y_Pred_middle', 2: 'Y_Pred_long'}

        if complex_batch not in pred_df_dict:
            return None, None

        if hdx_epitope_peps is None:
            hdx_epitope_peps = [list(hdx_dict.keys()) for hdx_dict in hdx_true_diffs]

        complex_df = pred_df_dict[complex_batch]
        complex_pep_set = set(complex_df['Range'].unique())  # Convert to Set for O(1) lookup
        complex_df = complex_df.groupby('Range', as_index=False)[pred_cluster_dict[pred_cluster]].mean()

        for apo_state, epitope_peps, hdx_dict in zip(apo_states, hdx_epitope_peps, hdx_true_diffs):
            apo_batch = apo_state[-1] # ref_structure
            if apo_batch not in pred_df_dict:
                continue

            apo_df = pred_df_dict[apo_batch]  # Pre-fetched DataFrame
            apo_df = apo_df.groupby('Range', as_index=False)[pred_cluster_dict[pred_cluster]].mean()
            
            apo_pep_set = set(apo_df['Range'].unique())  # Convert to Set for O(1) lookup

            common_peptides = [pep for pep in epitope_peps if pep in apo_pep_set and pep in complex_pep_set]

            if not common_peptides:
                continue

            # Ensure both DataFrames extract values in the same order using .reindex()
            apo_values = apo_df.set_index('Range').reindex(common_peptides)[pred_cluster_dict[pred_cluster]].values
            complex_values = complex_df.set_index('Range').reindex(common_peptides)[pred_cluster_dict[pred_cluster]].values

            # Subtract values
            pred_diffs = complex_values - apo_values
            true_diffs = np.array([hdx_dict[pep] for pep in common_peptides])

            truth.extend(true_diffs)
            pred.extend(pred_diffs)

        return np.array(truth), np.array(pred)
    
    @staticmethod
    def plot_hdx(true_diff, diff_mtx, fpath, size=(10, 6)):
        plt.figure(figsize=size)
        x_labels = list(true_diff.keys())
        x_positions = np.arange(len(x_labels))  # numerical positions for x-axis

        diff = np.array(list(true_diff.values()))
        diff_neg = diff[diff<0]
        mean_diff = np.mean(diff_neg)
        hdx_epitope_id = np.where(diff<mean_diff)[0]
        hdx_epitope_pep = [x_labels[i] for i in hdx_epitope_id]

        plt.xticks(x_positions, x_labels, rotation=90)  # apply labels with rotation for clarity
        all_times = set()
        for diffs in diff_mtx.values():
            all_times.update(diffs.keys())
        sorted_times = sorted(all_times, key=lambda x: float(x))
        for time in sorted_times:
            time_values = [diff_mtx[label].get(time, 0) / 100 for label in x_labels]
            plt.plot(x_positions, time_values, label=f'time_{time}', linestyle='--', alpha=1)
        plt.plot(x_positions, list(true_diff.values()), label='True diff', color='k', marker='o', linestyle='-', linewidth=1, markersize=4)
        plt.ylabel('RFU difference')
        plt.axhline(y=mean_diff, color='r', linestyle='--', label='Epitope cutoff')
        plt.legend()
        plt.savefig(f"{fpath}", bbox_inches='tight', dpi=300)
