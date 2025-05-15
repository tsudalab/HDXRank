import os
import re
import math
import pandas as pd
import numpy as np
import torch
from BioWrappers import *
from dataclasses import dataclass, field
from Bio.PDB import Selection

from itertools import groupby
from collections import defaultdict
from GVP_strucFeats import *

import xml.etree.ElementTree as ET
import warnings
warnings.filterwarnings("ignore")

import logging

def XML_process(input_file):
    tree = ET.parse(input_file)
    root = tree.getroot()

    tasks = {
        "GeneralParameters": {},
        "EmbeddingParameters": {},
        "TaskParameters": {},
        "GraphParameters": {},
        "PredictionParameters": {}
    }

    for section in root:
        section_name = section.tag
        if section_name in tasks:
            for param in section:
                tasks[section_name][param.tag] = param.text.strip()

    # Post-process specific parameters for consistency
    #tasks["GeneralParameters"]["pepGraphDir"] = [os.path.join(tasks["GeneralParameters"]["RootDir"], dir) for dir in tasks["GeneralParameters"]["pepGraphDir"].split(',')]
    tasks["GeneralParameters"]["pepGraphDir"] = os.path.join(tasks["GeneralParameters"]["RootDir"], tasks["GeneralParameters"]["pepGraphDir"])
    tasks["GeneralParameters"]["EmbeddingDir"] = os.path.join(tasks["GeneralParameters"]["RootDir"], tasks["GeneralParameters"]["EmbeddingDir"])
    tasks["GeneralParameters"]["PDBDir"] = os.path.join(tasks["GeneralParameters"]["RootDir"], tasks["GeneralParameters"]["PDBDir"])
    tasks["GeneralParameters"]["HDXDir"] = os.path.join(tasks["GeneralParameters"]["RootDir"], tasks["GeneralParameters"]["HDXDir"])
    tasks["GeneralParameters"]["hhmDir"] = os.path.join(tasks["GeneralParameters"]["RootDir"], tasks["GeneralParameters"]["hhmDir"])

    tasks["EmbeddingParameters"]["NAChains"] = None if tasks["EmbeddingParameters"]["NAChains"] == 'None' else list(tasks["EmbeddingParameters"]["NAChains"])
    tasks["EmbeddingParameters"]["SMChains"] = None if tasks["EmbeddingParameters"]["SMChains"] == 'None' else list(tasks["EmbeddingParameters"]["SMChains"])

    if "SeedNum" in tasks["TaskParameters"]:
        tasks["TaskParameters"]["SeedNum"] = int(tasks["TaskParameters"]["SeedNum"])
    if "Correction" in tasks["TaskParameters"]:
        tasks["TaskParameters"]["Correction"] = [int(x) for x in tasks["TaskParameters"]["Correction"].split(',')]

    tasks["GraphParameters"]["RadiusMax"] = float(tasks["GraphParameters"]["RadiusMax"])
    tasks["GraphParameters"]["SeqMin"] = int(tasks["GraphParameters"]["SeqMin"])
    tasks["GraphParameters"]["MaxLen"] = int(tasks["GraphParameters"]["MaxLen"])
    if tasks["GraphParameters"]["PepRange"] == 'None':
        tasks["GraphParameters"]["PepRange"] = None
    else:
        PepRange = tasks["GraphParameters"]["PepRange"].split(',')
        input_range = []
        for subrange in PepRange:
            subrange = subrange.strip()
            i, j = subrange.split('-')[0].strip(), subrange.split('-')[1].strip()
            input_range.append((int(i), int(j)))
        tasks["GraphParameters"]["PepRange"] = input_range
    
    tasks["PredictionParameters"]["CudaID"] = int(tasks["PredictionParameters"]["CudaID"])
    tasks["PredictionParameters"]["ModelList"] = tasks["PredictionParameters"]["ModelList"].split(',')
    tasks["PredictionParameters"]["BatchSize"] = int(tasks["PredictionParameters"]["BatchSize"])
    tasks["PredictionParameters"]["ModelDir"] = os.path.join(tasks["GeneralParameters"]["RootDir"], tasks["PredictionParameters"]["ModelDir"])
    tasks["PredictionParameters"]["PredDir"] = os.path.join(tasks["GeneralParameters"]["RootDir"], tasks["PredictionParameters"]["PredDir"])
    return tasks

def parse_task(input_file):
    """
    Parse an XML file for graph construction tasks.

    Args:
        input_file (str): Path to the XML input file.

    Returns:
        list: Keys for creating graph tasks.
        dict: Parsed tasks.
    """
    tasks = XML_process(input_file)

    if "Mode" in tasks["GeneralParameters"]:
        if tasks["GeneralParameters"]["Mode"] in ["BatchAF", "BatchDock", "Single"]:
            chain_num = len(tasks["TaskParameters"]["ChainToConstruct"])
            if tasks["GeneralParameters"]["Mode"] == "BatchAF":
                seed_num = tasks["TaskParameters"]["SeedNum"]
                model_num = 5 # each round AlphaFold gives 5 predictions 
                copy_num = seed_num * model_num

                tasks["EmbeddingParameters"]['StructureList'] = [
                    f"{tasks['EmbeddingParameters']['StructureList']}_seed{i}_model_{j}".upper()
                    for i in range(1, seed_num+1) 
                    for j in range(5)
                ]
                tasks["TaskParameters"]['EmbeddingToUse'] = [
                    [
                        f"{tasks['TaskParameters']['EmbeddingToUse']}_seed{i}_model_{j}".upper()
                        for u in range(chain_num)
                    ]
                    for i in range(1, seed_num+1)
                    for j in range(5)
                ]
            elif tasks["GeneralParameters"]["Mode"] == "BatchDock":
                copy_num = int(tasks["TaskParameters"]["DockingModelNum"])
                tasks["EmbeddingParameters"]["StructureList"] = [f"MODEL_{i}_REVISED" for i in range(1, copy_num+1)]
                tasks["TaskParameters"]["EmbeddingToUse"] = [[f"MODEL_{j}_REVISED" for i in range(chain_num)] for j in range(copy_num)]
            elif tasks["GeneralParameters"]["Mode"] == "Single":
                copy_num = 1
                tasks["EmbeddingParameters"]["StructureList"] = [tasks["EmbeddingParameters"]["StructureList"]]
                tasks["TaskParameters"]["EmbeddingToUse"] = [[tasks["TaskParameters"]["EmbeddingToUse"] for i in range(chain_num)]]

            entries=tasks["EmbeddingParameters"]["hhmToUse"].strip('/').split('/')
            tasks["EmbeddingParameters"]["hhmToUse"] = {entry.split(':')[0]: entry.split(':')[1] for entry in entries}
            tasks["EmbeddingParameters"]["ProteinChains"] = ",".join(list(tasks["EmbeddingParameters"]["ProteinChains"]))
            tasks["EmbeddingParameters"]["ProteinChains"] = [[tasks["EmbeddingParameters"]["ProteinChains"] for i in range(chain_num)]] * copy_num

            tasks["TaskParameters"]["DatabaseID"] = [tasks["TaskParameters"]["DatabaseID"]] * copy_num
            tasks["TaskParameters"]["Protein"] = [[tasks["TaskParameters"]["Protein"] for i in range(chain_num)]] * copy_num
            tasks["TaskParameters"]["State"] = [[tasks["TaskParameters"]["State"] for i in range(chain_num)]] * copy_num
            tasks["TaskParameters"]["ChainToConstruct"] = [list(tasks["TaskParameters"]["ChainToConstruct"])] * copy_num
            tasks["TaskParameters"]["Correction"] = [list(tasks["TaskParameters"]["Correction"])] * copy_num
            tasks["TaskParameters"]["ComplexState"] = [tasks["TaskParameters"]["ComplexState"]] * copy_num

            keys = [
                tasks["TaskParameters"]["DatabaseID"],
                tasks["TaskParameters"]["Protein"],
                tasks["TaskParameters"]["State"],
                tasks["EmbeddingParameters"]["StructureList"],
                tasks["TaskParameters"]["ChainToConstruct"],
                tasks["TaskParameters"]["Correction"],
                tasks["EmbeddingParameters"]["ProteinChains"],
                tasks["TaskParameters"]["ComplexState"],
                tasks["TaskParameters"]["EmbeddingToUse"]
            ]
        elif tasks["GeneralParameters"]["Mode"] == 'BatchTable':
            keys = parse_xlsx_task(tasks)
        else:
            raise ValueError("Invalid mode specified in XML file.")
    logging.info(f"Parsed Tasks: {len(keys[0])}")
    logging.info(f"General settings:")
    for key, value in tasks["GeneralParameters"].items():
        logging.info(f"  {key}: {value}")
    return keys, tasks

def parse_xlsx_task(tasks):
    task_fpath = os.path.join(tasks["GeneralParameters"]["RootDir"], f"{tasks['GeneralParameters']['TaskFile']}.xlsx")
    if os.path.exists(task_fpath):
        df = pd.read_excel(task_fpath, sheet_name="Sheet1")
    else:
        raise FileNotFoundError(f"Missing task file in {task_fpath}")
    df = df.dropna(subset=['structure_file'])
    df = df[df['complex_state'] != 'ligand complex']

    apo_identifier = list(df['structure_file'].astype(str).unique())
    protein, state, chain_identifier = [], [], []
    correction = []
    database_id = []
    protein_chains=  []
    complex_state = []
    structure_list = []
    embedding_fname = []

    #process HDX data by apo_identifier (pdb structures)
    for i, temp_apo in enumerate(apo_identifier):
        if temp_apo.split(":")[0] != 'MODEL':
            temp_df = df[(df['structure_file'] == temp_apo)]
            temp_protein = temp_df['protein'].astype(str).to_list()
            temp_state = temp_df['state'].astype(str).to_list()
            temp_chain = temp_df['chain_identifier'].astype(str).to_list()
            temp_correction = temp_df['correction_value'].astype(int).to_list()
            temp_protein_chains= temp_df['protein_chain'].astype(str).to_list()
            temp_complex_state = temp_df['complex_state'].astype(str).to_list()

            structure_list.append(temp_apo)
            embedding_fname.append([temp_apo.upper()]*len(temp_protein_chains))
            protein.append(temp_protein)
            state.append(temp_state)
            chain_identifier.append(temp_chain)
            correction.append(temp_correction)
            database_id.extend(temp_df['database_id'].astype(str).unique())
            protein_chains.append(temp_protein_chains)
            complex_state.append(temp_complex_state[0])
        else:
            #BatchTable mode for docking models
            N_model = int(tasks["TaskParameters"]["DockingModelNum"])
            model_list = [f'MODEL_{i}_NATIVE' for i in range(1, N_model+1)]
            apo_models = temp_apo.split(":")[1:] # suppose the format is MODEL:apo1:apo2: ...
            temp_df = df[df['structure_file'].isin(apo_models)]

            temp_protein = [temp_df['protein'].astype(str).to_list()] * N_model
            temp_state = [temp_df['state'].astype(str).to_list()] * N_model
            temp_chain = [temp_df['chain_identifier'].astype(str).to_list()] * N_model
            temp_correction = [temp_df['correction_value'].astype(int).to_list()] * N_model
            temp_complex_state = ['protein complex'] * N_model
            temp_database_id = [temp_df['database_id'].astype(str).to_list()[0]] * N_model

            temp_protein_chains= [temp_df['protein_chain'].astype(str).to_list()] * N_model

            protein.extend(temp_protein)
            state.extend(temp_state)
            chain_identifier.extend(temp_chain)
            correction.extend(temp_correction)
            database_id.extend(temp_database_id)
            protein_chains.extend(temp_protein_chains)
            complex_state.extend(temp_complex_state)
            embedding_fname.extend([apo_models] * N_model)

            structure_list = structure_list + model_list
    keys = [database_id, protein, state, structure_list, chain_identifier, correction, protein_chains, complex_state, embedding_fname]
    return keys

class ChemData():
    def __init__(self):
        self.NAATOKENS = 20+1+10+10+1 # 20 AAs, 1 UNK res, 8 NAs+2UN NAs, 10 atoms +1 UNK atom
        self.UNKINDEX = 20  # residue unknown

        #bond types
        self.num2btype = [0,1,2,3,4,5,6,7] # UNK, SINGLE, DOUBLE, TRIPLE, AROMATIC, 
                                            # PEPTIDE/NA BACKBONE, PROTEIN-LIGAND (PEPTIDE), OTHER

        self.NATYPES = ['DA','DC','DG','DT', 'DU', 'A', 'T', 'C', 'G', 'U']
        self.STDAAS = ['ALA','ARG','ASN','ASP','CYS',
            'GLN','GLU','GLY','HIS','ILE',
            'LEU','LYS','MET','PHE','PRO',
            'SER','THR','TRP','TYR','VAL',]
        
        self.three_to_one = {
                'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
                'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
                'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
            }
        self.one_to_three = {v: k for k, v in self.three_to_one.items()}

        self.num2aa=[
            'ALA','ARG','ASN','ASP','CYS',
            'GLN','GLU','GLY','HIS','ILE',
            'LEU','LYS','MET','PHE','PRO',
            'SER','THR','TRP','TYR','VAL',
            'UNK',
            'A','C','G','T', 'U',
            'Br', 'F', 'Cl','I',
            'C', 'N', 'O', 'S', 'P',
            'ZN', 'MG', 'NA', 'CA', 'K', 'FE',
            'ATM'
        ]
        self.num2aa = [item.upper() for item in self.num2aa]
        self.aa2num= {x:i for i,x in enumerate(self.num2aa)}
        self.aa2num['MEN'] = 20

        # Mapping 3 letter AA to 1 letter AA (e.g. ALA to A)
        self.one_letter = ["A", "R", "N", "D", "C", \
                            "Q", "E", "G", "H", "I", \
                            "L", "K", "M", "F", "P", \
                            "S", "T", "W", "Y", "V", "?", "-"]

        self.n_non_protein = len(self.num2aa) - len(self.one_letter)

        self.aa_321 = {a:b for a,b in zip(self.num2aa,self.one_letter+['a']*self.n_non_protein)}

        self.frame_priority2atom = [
            "F",  "Cl", "Br", "I",  "O",  "S",  "Se", "Te", "N",  "P",  "As", "Sb", 
            "C",  "Si", "Sn", "Pb", "B",  "Al", "Zn", "Hg", "Cu", "Au", "Ni", "Pd", 
            "Pt", "Co", "Rh", "Ir", "Pr", "Fe", "Ru", "Os", "Mn", "Re", "Cr", "Mo", 
            "W",  "V",  "U",  "Tb", "Y",  "Be", "Mg", "Ca", "Li", "K",  "ATM"]

        # these atomic numbers are incorrect, but keeping for fold&dock3 and correcting it 
        # in util.writepdb() during output.
        self.atom_num= [
            9,    17,   35,   53,   8,    16,   34,   52,   7,    15,   33,   51, 
            6,    14,   32,   50,   82,   5,    13,   30,   80,   29,   79,   28, 
            46,   78,   27,   45,   77,   26,   44,   76,   25,   75,   24,   42, 
            23,   74,   92,   65,   39,   4,    12,   20,   3,    19,   0] # in same order as frame priority

        self.atom2frame_priority = {x:i for i,x in enumerate(self.frame_priority2atom)}
        self.atomnum2atomtype = dict(zip(self.atom_num, self.frame_priority2atom))
        self.atomtype2atomnum = {v:k for k,v in self.atomnum2atomtype.items()}
                
        self.residue_charge = {'CYS': -0.64, 'HIS': -0.29, 'ASN': -1.22, 'GLN': -1.22, 'SER': -0.80, 'THR': -0.80, 'TYR': -0.80,
                                'TRP': -0.79, 'ALA': -0.37, 'PHE': -0.37, 'GLY': -0.37, 'ILE': -0.37, 'VAL': -0.37, 'MET': -0.37,
                                'PRO': 0.0, 'LEU': -0.37, 'GLU': -1.37, 'ASP': -1.37, 'LYS': -0.36, 'ARG': -1.65}

        self.residue_polarity = {'CYS': 'polar', 'HIS': 'polar', 'ASN': 'polar', 'GLN': 'polar', 'SER': 'polar', 'THR': 'polar', 'TYR': 'polar', 'TRP': 'polar',
                                    'ALA': 'apolar', 'PHE': 'apolar', 'GLY': 'apolar', 'ILE': 'apolar', 'VAL': 'apolar', 'MET': 'apolar', 'PRO': 'apolar', 'LEU': 'apolar',
                                    'GLU': 'neg_charged', 'ASP': 'neg_charged', 'LYS': 'neg_charged', 'ARG': 'pos_charged'}

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
        }## adopted from Atchley et al. (2005): https://www.pnas.org/doi/epdf/10.1073/pnas.0408677102
chemdata = ChemData()

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
        # construct embedding
        embedding = torch.cat((self.msa, self.res_HDMD, self.res_polarity, self.res_charge, self.SASA, self.hse, self.dihedrals, self.orientations), dim=1)
        ########### Dims:          30            5              4                   1            1         3            6                6          = 56 ###########
        return embedding
    
    def merge(self, data):
        self.seq_data = {**self.seq_data, **data.seq_data}
        return self

def load_embedding(fpath):
    if not os.path.isfile(fpath):
        logging.error(f"Missing embedding file: {fpath}")
        return False
    embedding_dict = torch.load(fpath)
    protein_embedding = embedding_dict['embedding']
    chain_label = np.array(embedding_dict['chain_label'])
    return chain_label, protein_embedding

def get_seq_polarity(seq):
    encode_index = [chemdata.polarity_encoding[chemdata.residue_polarity[chemdata.one_to_three[res]]] for res in seq]
    polarity_mtx = np.zeros((len(seq), 4))
    for i, idx in enumerate(encode_index):
        polarity_mtx[i, idx] = 1
    return polarity_mtx

def parse_hhm(hhm_file):
    # from AI-HDX: https://github.com/Environmentalpublichealth/AI-HDX
    hhm_mtx = []
    with open(hhm_file) as f:
        for i in f:
            if i.startswith("HMM"):
                break
        # start from the lines with HMM values
        for i in range(2):
            f.readline()
        lines = f.read().split("\n")
        # print(len(lines)) ## The file consist of three lines for each AA, first line is the HMM number against each AA,
        ## second line is the 10 conversion values, and the last line is empty. Group the three lines into one AA representative.
        sequence = ""
        for idx in range(0,int((len(lines)-2)/3)+1):
            first_line = lines[idx*3].replace("*","99999") # The * symbol is like NA, so here we assigned a big number to it
            next_line = lines[idx*3+1].replace("*","99999")
            content1 = first_line.strip().split()
            content2 = next_line.strip().split()
            if content1[0]=='//':
                break
            elif content1[0]=='-':
                continue
            sequence += str(content1[0])
            hhm_val1 = [10/(1 + math.exp(-1 * int(val1)/2000)) for val1 in content1[2:-1]]
            hhm_val2 = [10/(1 + math.exp(-1 * int(val2)/2000)) for val2 in content2]
            hhm_val = hhm_val1 + hhm_val2
            hhm_mtx.append(hhm_val)

    return np.array(hhm_mtx), sequence

# loading protine, NA, molecule ligand
def load_protein(hhm_file, pdb_file, chain_id):
    '''
    Generate embedding file from the pre-computed HMM file and rigidity file, PDB strucutre
    processing the protein chain featurization
    '''
    ### convert the list in dictionary to array
    model = get_bio_model(pdb_file)
    residue_data = {}
    residue_coord = []
    res_seq = []
    residue_list = Selection.unfold_entities(model, 'R')
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
            'coord': Ca_coord,  # used for appending heteroatom embedding to near residue
        }

    max_len = len(res_seq)

    # sequence-based features
    res_charge = np.array([chemdata.residue_charge[chemdata.one_to_three[res]] for res in res_seq]).reshape(-1, 1)
    res_polarity = get_seq_polarity(res_seq)
    HDMD = np.array([chemdata.AA_array[res] for res in res_seq]).reshape(-1, 5)

    # physical-based features
    hse_dict = get_hse(model, chain_id)
    SASA = biotite_SASA(pdb_file, chain_id)[:max_len]

    # MSA-based features
    hhm_mtx, hhm_seq = parse_hhm(hhm_file) # hhm file is chain-wise

    # structura based features: dihedral angels and orientations
    GVP_feats = ProteinGraphDataset(data_list=[])
    dihedrals = GVP_feats._dihedrals(torch.as_tensor(residue_coord))
    orientations = GVP_feats._orientations(torch.as_tensor(residue_coord))

    # check the sequence match among feat.
    res_seq = ''.join(res_seq)
    if hhm_seq == res_seq:
        pass
    elif hhm_seq[:-1] == res_seq:
        hhm_mtx = hhm_mtx[:-1]
    else:
        print("hhm_sequenece:", hhm_seq)
        print("dssp_sequenece:", res_seq)
        raise ValueError('Sequence mismatch between HMM and DSSP')

    corrected_hse_mtx = np.zeros((max_len, 3)) #hse feature doestn't influence the prediction according to SHAP analysis, can be removed
    for i, res_j in enumerate(residue_data.keys()):
        res_j = str(res_j)
        if (chain_id, res_j) in hse_dict.keys():
            corrected_hse_mtx[i, :] = list(hse_dict[(chain_id, res_j)])

    '''print('protein length:', len(residue_data.keys()))
    print('SASA length:', SASA.shape)
    print('HSE length:', corrected_hse_mtx.shape)
    print('HDMD length:', HDMD.shape)
    print('res_charge length:', res_charge.shape)
    print('res_polarity length:', res_polarity.shape)
    print('hhm length:', hhm_mtx.shape)
    print('dihedrals length:', dihedrals.shape)
    print('orientations length:', orientations.shape)'''

    return RawInputData(
        msa = torch.tensor(hhm_mtx, dtype = torch.float32),
        res_HDMD = torch.tensor(HDMD, dtype = torch.float32),
        res_polarity = torch.tensor(res_polarity, dtype = torch.float32),
        res_charge = torch.tensor(res_charge, dtype = torch.float32),
        SASA = torch.tensor(SASA, dtype = torch.float32).reshape(-1, 1),
        hse = torch.tensor(corrected_hse_mtx, dtype = torch.float32),
        dihedrals = torch.tensor(dihedrals, dtype = torch.float32),
        orientations = torch.tensor(orientations, dtype = torch.float32),
        seq_data = residue_data,
        type_label = 'protein'
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
                LG_name = line[17:20].replace(' ','') # resn is residue name, remove spaces
                atom_id = int(line[6:11].strip())
                atom_type = line[12:16].strip() # only allows for one character symbol for atom type
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
        msa = None,
        res_HDMD = None,
        res_polarity = None,
        res_charge = None,
        SASA = None,
        hse = None,
        dihedrals = None,
        orientations = None,
        seq_data = atom_data,
        type_label = 'ligand'
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
        msa = None,
        res_HDMD = None,
        res_polarity = None,
        res_charge = None,
        SASA = None,
        hse = None,
        dihedrals = None,
        orientations = None,
        seq_data = na_data,
        type_label = 'NA'
    )

# Find contact residue pairs
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
        # group atoms by residue
        atoms = self.get_atoms()
        key = lambda x: x['residue_index']
        #atoms = sorted(atoms, key=key)
        residues = [list(group) for key, group in groupby(atoms, key)]
        return residues

def read_PDB(key, PDB_path):
    if not os.path.isfile(PDB_path):
        print("cannot find the file", key)
        return None
    chains = defaultdict(Chain)

    with open(PDB_path, 'r') as f:
        data = f.read().strip().split('\n') 
        for line in data:
            if line[:4] == 'ATOM':
                n_res = int(line[23:26].strip())
                n_atom = int(line[6:11].strip())
                res_type = line[17:20].strip()
                atom_type = line[12:16].strip()
                chain = line[21].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                chain_id = line[21].strip()
                if atom_type == 'CA':
                    chains[chain_id].add_atom(n_atom, n_res, res_type, atom_type, [x, y, z])
    return chains