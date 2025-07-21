import numpy as np
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.HSExposure import HSExposureCA
from Bio import Align
import biotite.structure as struc
import biotite.structure.io as strucio

def get_bio_model(pdbfile, pepRange = None):
    """Get the model

    Args:
        pdbfile (str): pdbfile
        pepRange (str): peptide range

    Returns:
        [type]: Bio object
    """
    parser = PDBParser(QUIET = True)
    structure = parser.get_structure('_tmp', pdbfile)

    if pepRange is not None:
        # Create a new structure with only residues in the specified range
        for chain in structure[0]:
            chain_id = chain.get_id()
            if chain_id in pepRange:
                start, end = pepRange[chain_id]
                # Get list of residues to remove (those outside the range)
                residues_to_remove = []
                for residue in chain:
                    res_id = residue.get_id()[1]  # Get residue number
                    if res_id < start or res_id > end:
                        residues_to_remove.append(residue.get_id())
                # Remove residues outside range
                for res_id in residues_to_remove:
                    chain.detach_child(res_id)
                # if not in pepRange, keep all residues
    return structure[0]

def get_hse(model, chain='A'):
    """Get the hydrogen surface exposure

    Args:
        model (bio model): model of the strucrture

    Returns:
        dict: hse data
    """

    hse = HSExposureCA(model)
    data = {}
    hse_mtx = []
    index_range = []
    for k in list(hse.keys()):
        if not k[0] == chain:
            continue
        new_key = (k[0], k[1][1])
        index_range.append(k[1][1])

        x = hse[k]
        if x[2] is None:
            x = list(x)
            x[2] = 0.0
            x = tuple(x)

        data[new_key] = x
        hse_mtx.append(list(x))
    return data

def biotite_SASA(pdb_file, chain_id, HDXparser = None):
    '''
    biotite.structure.sasa(array, probe_radius=1.4, atom_filter=None, ignore_ions=True, point_number=1000, point_distr='Fibonacci', vdw_radii='ProtOr')
    '''
    model = struc.io.load_structure(pdb_file)
    chain_mask = (model.chain_id == chain_id)

    atom_sasa = struc.sasa(model, atom_filter = chain_mask, vdw_radii="Single")
    res_sasa = struc.apply_residue_wise(model, atom_sasa, np.sum)
    return res_sasa[~np.isnan(res_sasa)]

# sequence alignment
def seq_align(seq1, seq2):
    aligner = Align.PairwiseAligner()
    aligner.match = 5
    aligner.mismatch = 0
    aligner.open_gap_score = -4
    aligner.extend_gap_score = -0.5
    aln = aligner.align(seq1, seq2)[0]
    return aln

def max_aligned_seq(seq1, seq2):
    aln = seq_align(seq1, seq2)

    aa1_id, aa2_id = 0, 0
    seq1_mask = np.zeros(len(seq1), dtype=bool)
    seq2_mask = np.zeros(len(seq2), dtype=bool)
    for aa1, aa2 in zip(aln[0], aln[1]):
        if aa1 != '-' and aa2 != '-':
            if aa1 != aa2:
                seq1_mask[aa1_id] = True
                seq2_mask[aa2_id] = True
            aa1_id += 1
            aa2_id += 1
        elif aa1 == '-':
            seq2_mask[aa2_id] = True
            aa2_id += 1
        elif aa2 == '-':
            seq1_mask[aa1_id] = True
            aa1_id += 1

    return seq1_mask, seq2_mask
