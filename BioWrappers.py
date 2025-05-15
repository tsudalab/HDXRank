import numpy as np
from Bio.PDB.PDBParser import PDBParser
from Bio.PDB.HSExposure import HSExposureCA
import biotite.structure as struc
import biotite.structure.io as strucio

def get_bio_model(pdbfile):
    """Get the model

    Args:
        pdbfile (str): pdbfile

    Returns:
        [type]: Bio object
    """
    parser = PDBParser(QUIET = True)
    structure = parser.get_structure('_tmp', pdbfile)
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
