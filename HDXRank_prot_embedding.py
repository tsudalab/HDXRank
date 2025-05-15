"""
2025/1/8
Author: WANG Liyao
Paper: HDXRank: A Deep Learning Framework for Ranking Protein complex predictions with Hydrogen Deuterium Exchange Data
Note: 
Take the HHBlits output hhm format, PDB structure as input to encode peptide sequence and structure information into embedding.
The Heteroatoms (NA and SM) encoding are also supported and merged with protein sequence,
but it is not used in the current HDXRank model.
"""
import os
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from Bio import PDB
from Bio.PDB import NeighborSearch, Selection
import warnings
from BioWrappers import get_bio_model
from HDXRank_utilis import load_protein, load_nucleic_acid, load_sm, RawInputData, parse_task

# Logging setup
import logging

def find_contact_res(HETATM_input, res_list, cutoff):
    """
    Find the contact residues of the heteroatoms (NA or SM).

    Args:
        HETATM_input: Input data for heteroatoms.
        res_list: List of residues.
        cutoff: Distance cutoff for identifying contacts.

    Returns:
        np.ndarray: Contact matrix.
    """
    entity_index_map = {tuple(entity['coord']): entity['token_type'] for entity in HETATM_input.seq_data.values()}
    res_index_map = {res: index for index, res in enumerate(res_list)}
    contact_mtx = np.full((len(res_list), len(entity_index_map.keys())), 20)  # 20 for 'UNK' unknown type

    atom_list = Selection.unfold_entities(res_list, 'A')
    ns = NeighborSearch(atom_list)
    for entity_idx, entity_coord in enumerate(entity_index_map.keys()):
        for nearby_res in ns.search(entity_coord, cutoff, level='R'):
            res_idx = res_index_map[nearby_res]
            contact_mtx[res_idx, entity_idx] = entity_index_map[entity_coord]

    return contact_mtx

def merge_inputs(inputs_list):
    """
    Merge input data from different chains.

    Args:
        inputs_list (list): List of RawInputData objects.

    Returns:
        RawInputData: Merged input data.
    """
    if not inputs_list:
        return RawInputData()
    elif len(inputs_list) == 1:
        return inputs_list[0]
    else:
        running_input = inputs_list[0]
        for i in range(1, len(inputs_list)):
            running_input = running_input.merge(inputs_list[i])
        return running_input

def embed_protein(structure_dir, hhm_dir, save_dir, pdb_fname, protein_chain_hhms, NA_chain=[], SM_chain=[]):
    """
    Embed protein sequence and structure into feature matrices.

    Args:
        structure_dir (str): Directory containing PDB structures.
        hhm_dir (str): Directory containing HMM files.
        save_dir (str): Directory to save embeddings.
        pdb_fname (str): Name of the PDB file.
        protein_chain_hhms (dict): Mapping of chain IDs to HMM files.
        NA_chain (list): Nucleic acid chains.
        SM_chain (list): Small molecule chains.

    Returns:
        None
    """
    logging.info(f'Processing: {pdb_fname}')

    save_file = os.path.join(save_dir, f'{pdb_fname}.pt')
    if os.path.isfile(save_file):
        logging.info(f'File already exists: {pdb_fname}')
        return

    pdb_file = os.path.join(structure_dir, f'{pdb_fname}.pdb')
    if not os.path.isfile(pdb_file):
        logging.warning(f'File does not exist: {pdb_file}')
        return

    structure = get_bio_model(pdb_file)
    chains = list(structure.get_chains())

    protein_inputs, NA_inputs, SM_inputs = [], [], []

    for chain in chains:
        chain_id = chain.get_id()
        hhm_file = os.path.join(hhm_dir, f'{protein_chain_hhms.get(chain_id, "")}_{chain_id}.hhm')
        if chain_id in protein_chain_hhms:
            if os.path.isfile(hhm_file):
                protein_inputs.append(load_protein(hhm_file, pdb_file, chain_id))
            else:
                logging.error(f'Missing HMM for chain {chain_id}: {hhm_file}')
        elif chain_id in NA_chain:
            NA_inputs.append(load_nucleic_acid(pdb_file, chain_id))
        elif chain_id in SM_chain:
            SM_inputs.append(load_sm(pdb_file, chain_id))

    if len(protein_inputs) >0:
        embedding = [protein.construct_embedding() for protein in protein_inputs]
        embed_mtx = torch.cat(embedding, dim=0)

        chain_list = [chain for chain in structure.get_chains() if chain.id in protein_chain_hhms]
        res_list = Selection.unfold_entities(chain_list, 'R')
    else:
        return False

    ### block the following code as the Heteroatom encoding is not used in the current HDXRank model ###
    '''
    res_idx_list = []
    for res in res_list:
        name = res.get_resname()
        res_idx_list.append(chemdata.aa2num[name] if name in chemdata.aa2num else 20)
    res_idx_list = np.array(res_idx_list).reshape(-1, 1)

    contact_ensemble = []
    if len(NA_inputs) != 0 or len(SM_inputs) != 0:
        for inputs in [NA_inputs, SM_inputs]:
            merged_HETinput = merge_inputs(inputs)
            if len(merged_HETinput.seq_data.keys()) == 0:
                continue
            contact_mtx = find_contact_res(merged_HETinput, res_list, cutoff = 5.0)  # [#res, #entity of NA/SM] where elements are type encoding
            contact_ensemble.append(contact_mtx)

    contact_ensemble.insert(0, res_idx_list)
    contact_mtx = np.concatenate(contact_ensemble, axis=1) if len(contact_ensemble) > 1 else contact_ensemble[0]

    contact_tensor = torch.tensor(contact_mtx, dtype=torch.long).flatten()
    encoded_tensor = F.one_hot(contact_tensor, num_classes=len(chemdata.num2aa))

    encoded_tensor = encoded_tensor.view(contact_mtx.shape[0], -1, encoded_tensor.shape[1])
    encoded_tensor = torch.sum(encoded_tensor, dim=1)
    encoded_tensor = torch.log(encoded_tensor + 1) # apply log(e) to elements

    protein_embedding = torch.cat((embed_mtx, encoded_tensor), dim=1)
    print('protein_embedding:', protein_embedding.shape)
    '''

    res_idx = [res.id[1] for res in res_list]
    res_name = [res.get_resname() for res in res_list]
    chain_label = [res.get_parent().id for res in res_list]

    data_to_save = {
        'res_idx': res_idx,
        'res_name': res_name,
        'chain_label': chain_label,
        'embedding': embed_mtx
    }
    torch.save(data_to_save, save_file)

def BatchTable_embedding(tasks):
    """
    Generate embeddings for a set of proteins according to xlsx table.

    Args:
        tasks (dict): Parsed XML task dictionary.

    Returns:
        None
    """
    warnings.filterwarnings("ignore")

    hhm_dir = tasks['GeneralParameters']['hhmDir']
    save_dir = tasks['GeneralParameters']['EmbeddingDir']
    structure_dir = tasks['GeneralParameters']['PDBDir']

    df = pd.read_excel(os.path.join(tasks["GeneralParameters"]["RootDir"], f"{tasks['GeneralParameters']['TaskFile']}.xlsx"), 
                       sheet_name='Sheet1')
    df = df.dropna(subset=['structure_file']).drop_duplicates(subset=['structure_file'])
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    for _, row in df.iterrows():
        file_string = str(row['structure_file']).upper().split('.')[0]
        # pdb_fnames and protein_chain should have same orders
        pdb_fnames = file_string.split(':')
        protein_chain = row['protein_chain'].split(',')

        if pdb_fnames[0] != 'MODEL':
            protein_chain_hhms = {chain: pdb_fnames[0] for chain in protein_chain}
            embed_protein(structure_dir, hhm_dir, save_dir, pdb_fnames[0], protein_chain_hhms, NA_chain=[], SM_chain=[])
        else:
            N_model = int(tasks['TaskParameters']['DockingModelNum'])
            protein_chain_hhms = {chain: pdb_fnames[j+1] for j, chain in enumerate(protein_chain)}
            for i in range(1, N_model+1):
                embed_protein(structure_dir, hhm_dir, save_dir, f'MODEL_{i}_NATIVE', protein_chain_hhms, NA_chain=[], SM_chain=[])

def XML_embedding(tasks):
    structure_dir = tasks['GeneralParameters']['PDBDir']
    hhm_dir = tasks['GeneralParameters']['hhmDir']
    save_dir = tasks['GeneralParameters']['EmbeddingDir']
    pdb_fname = tasks['EmbeddingParameters']['StructureList']
    protein_chain_hhms = tasks['EmbeddingParameters']['hhmToUse']

    os.makedirs(save_dir, exist_ok=True)

    if tasks['GeneralParameters']['Mode'] == 'Single':
        embed_protein(
            structure_dir=structure_dir,
            hhm_dir=hhm_dir,
            save_dir=save_dir,
            pdb_fname=pdb_fname[0],
            protein_chain_hhms=protein_chain_hhms,
            NA_chain=[],
            SM_chain=[]
        )
    elif tasks['GeneralParameters']['Mode'] in ['BatchAF', 'BatchDock']:
        for pdb in pdb_fname:
            embed_protein(
                structure_dir=structure_dir,
                hhm_dir=hhm_dir,
                save_dir=save_dir,
                pdb_fname=pdb,
                protein_chain_hhms=protein_chain_hhms,
                NA_chain=[],
                SM_chain=[]
            )

if __name__ == "__main__":
    # Logging setup
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()]
    )

    import argparse
    parser = argparse.ArgumentParser(description='Generate protein embeddings.')
    parser.add_argument('-input', type=str, required=True, help='XML task file path')
    args = parser.parse_args()

    _, tasks = parse_task(args.input)
    if tasks['GeneralParameters']['Mode'] == 'BatchTable':
        BatchTable_embedding(tasks = tasks)
    else:
        XML_embedding(tasks = tasks)