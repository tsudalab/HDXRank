# run DockQ to evaluate predicted structures
## ========= input csv format ============
# columns: model_path, native_path, model_chains, native_chains
# eg.
# 'model_path': './pred/alphafold_relaxed_rank_001_alphafold2_multimer_v3_model_2_seed_000.pdb'
# 'native_path': './truth/wt.pdb'
# 'model_chains': 'AB' (the chain order should match between model and native)
# 'native_chains': 'BC'

# merge chains: enable --merge_chains and use comma ',' to separate chain groups like 'BC,A' and 'CD,E'
## ========= input csv format ============

import os
import argparse
import pandas as pd
import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
from DockQ.DockQ import *
from DockQ.constants import FNAT_THRESHOLD, INTERFACE_THRESHOLD, BACKBONE_ATOMS

def calc_fnat(aligned_mdl, aligned_nat):   
    ref_res_distances = get_residue_distances(aligned_nat[0], aligned_nat[1], "ref")
    nat_total = np.nonzero(np.asarray(ref_res_distances) < FNAT_THRESHOLD ** 2)[0].shape[0]
    if nat_total == 0:
        # if the native has no interface between the two chain groups
        # nothing to do here
        return None

    sample_res_distances = get_residue_distances(aligned_mdl[0], aligned_mdl[1], "sample")

    assert (
        sample_res_distances.shape == ref_res_distances.shape
    ), f"Native and model have incompatible sizes ({sample_res_distances.shape} != {ref_res_distances.shape})"

    nat_correct, nonnat_count, _, model_total = get_fnat_stats(
        sample_res_distances, ref_res_distances, threshold=FNAT_THRESHOLD
    )
    # avoids divide by 0 errors
    fnat = nat_total and nat_correct / nat_total or 0
    fnonnat = model_total and nonnat_count / model_total or 0
    return fnat, fnonnat, sample_res_distances, ref_res_distances, nat_total, nat_correct, nonnat_count, _, model_total

def calc_irms(aligned_mdl, aligned_nat, ref_res_distances):
    interacting_pairs = get_interacting_pairs(
        # working with squared thresholds to avoid using sqrt
        ref_res_distances,
        threshold=INTERFACE_THRESHOLD ** 2,
    )

    sample_interface_atoms1, ref_interface_atoms1 = subset_atoms(
        aligned_mdl[0],
        aligned_nat[0],
        atom_types=BACKBONE_ATOMS,
        residue_subset=interacting_pairs[0],
    )
    sample_interface_atoms2, ref_interface_atoms2 = subset_atoms(
        aligned_mdl[1],
        aligned_nat[1],
        atom_types=BACKBONE_ATOMS,
        residue_subset=interacting_pairs[1],
    )

    sample_interface_atoms = np.asarray(
        sample_interface_atoms1 + sample_interface_atoms2
    )
    ref_interface_atoms = np.asarray(ref_interface_atoms1 + ref_interface_atoms2)

    super_imposer = SVDSuperimposer()
    super_imposer.set(sample_interface_atoms, ref_interface_atoms)
    super_imposer.run()
    irms = super_imposer.get_rms()
    return irms

def calc_lrms(aligned_mdl, aligned_nat):
    # assign which chains constitute the receptor, ligand
    receptor_chains = ((aligned_nat[0], aligned_mdl[0]))
    ligand_chains = ((aligned_nat[1], aligned_mdl[1]))

    class1, class2 = (("receptor", "ligand"))

    receptor_atoms_native, receptor_atoms_sample = subset_atoms(
        receptor_chains[0],
        receptor_chains[1],
        atom_types=BACKBONE_ATOMS,
        what="receptor",
    )
    ligand_atoms_native, ligand_atoms_sample = subset_atoms(
        ligand_chains[0], ligand_chains[1], atom_types=BACKBONE_ATOMS, what="ligand"
    )
    # Set to align on receptor
    super_imposer = SVDSuperimposer()
    super_imposer.set(
        np.asarray(receptor_atoms_native), np.asarray(receptor_atoms_sample)
    )
    super_imposer.run()

    rot, tran = super_imposer.get_rotran()
    rotated_sample_atoms = np.dot(np.asarray(ligand_atoms_sample), rot) + tran

    lrms = super_imposer._rms(
        np.asarray(ligand_atoms_native), rotated_sample_atoms
    )  # using the private _rms function which does not superimpose
    return lrms

def calc_NCAA_DockQ(model_structure, native_structure, model_chainIDs, native_chainIDs):
    model_rec_chainID, model_lig_chainID = model_chainIDs[0], model_chainIDs[1]
    native_rec_chainID, native_lig_chainID = native_chainIDs[0], native_chainIDs[1]

    model_lig_chain = model_structure[model_lig_chainID]
    native_lig_chain = native_structure[native_lig_chainID]

    # align receptor chain, assuming ligand chain has same length and matched order
    aln = align_chains(
        model_structure[model_rec_chainID],
        native_structure[native_rec_chainID],
        use_numbering=False,
    )
    alignment = format_alignment(aln)
    aligned_mdl_rec, aligned_nat_rec = get_aligned_residues(
        model_structure[model_rec_chainID], native_structure[native_rec_chainID], tuple(alignment.values())
    )

    # calculate fnat
    mdl_chains = [aligned_mdl_rec, model_lig_chain]
    nat_chains = [aligned_nat_rec, native_lig_chain]
    fnat, fnonnat, sample_res_distances, ref_res_distances, \
        nat_total, nat_correct, nonnat_count, _, model_total = calc_fnat(mdl_chains, nat_chains)
    irms = calc_irms(mdl_chains, nat_chains, ref_res_distances)
    lrms = calc_lrms(mdl_chains, nat_chains)

    info = {}
    F1 = f1(nat_correct, nonnat_count, nat_total)
    info["DockQ"] = dockq_formula(fnat, irms, lrms)

    info["F1"] = F1
    info["iRMSD"] = irms
    info["LRMSD"] = lrms
    info["fnat"] = fnat
    info["nat_correct"] = nat_correct
    info["nat_total"] = nat_total

    info["fnonnat"] = fnonnat
    info["nonnat_count"] = nonnat_count
    info["model_total"] = model_total
    info["clashes"] = np.nonzero(
        np.asarray(sample_res_distances) < CLASH_THRESHOLD ** 2
    )[0].shape[0]
    info["len1"] = len(native_structure[native_rec_chainID])
    info["len2"] = len(native_structure[native_lig_chainID])
    info["is_het"] = True

    return info

def merge_chains(model, chains_to_merge):
    for chain in chains_to_merge[1:]:
        for i, res in enumerate(model[chain]):
            res.id = (chain, res.id[1], res.id[2])
            model[chains_to_merge[0]].add(res)
        model.detach_child(chain)
    model[chains_to_merge[0]].id = "".join(chains_to_merge)
    return model

def parse_args():
    parser = argparse.ArgumentParser(description="Run DockQ evaluation on model-native structure pairs from a CSV file.")
    parser.add_argument('--input', type=str, required=True, help='CSV file with columns: model_path, native_path, model_chains, native_chains')
    parser.add_argument('--save', type=str, required=True, help='Output CSV file to save DockQ results')
    parser.add_argument('--merge_chain', type=bool, default=False, help='Merge the other chains into one group for evaluating multiple interfaces of the leading chain as whole')
    parser.add_argument('--ncaa', type=bool, default=False, help='native ligand chain contains non-canonical AAs, the backbone atoms must match with predicted backbone, ligand chain must be the last chain')
    return parser.parse_args()

def main():
    args = parse_args()
    if not os.path.exists(args.input):
        raise FileNotFoundError(f'Cannot find {args.input}')

    df = pd.read_csv(args.input)
    dockq_results = []
    
    for i, row in df.iterrows():
        model = row['model_path']
        native = row['native_path']
        model = model+'.pdb' if not model.endswith('.pdb') else model
        native = native +'.pdb' if not native.endswith('.pdb') else native
        print(f'processing {model}')

        model_chains = row['model_chains'].strip()
        native_chains = row['native_chains'].strip()
        assert len(model_chains) == len(native_chains)

        model_ = load_PDB(model) if not args.ncaa else load_PDB(model, small_molecule=True) # set small_molecule=true to parse HETATM records
        native_ = load_PDB(native) if not args.ncaa else load_PDB(native, small_molecule=True)

        # merge chains
        if len(model_chains) > 3 and args.merge_chain: # two chains: A,B -> 3 letters
            mapping = {}
            for mdl_chain_set, nat_chain_set in zip(model_chains.split(','), native_chains.split(',')):
                if len(mdl_chain_set) >= 2:
                    if len(mdl_chain_set) == len(nat_chain_set):
                        model_ = merge_chains(model_, mdl_chain_set)
                        native_ = merge_chains(native_, nat_chain_set)
                    else:
                        raise ValueError(f"model chain doesn't match with native chain:{mdl_chain_set} : {nat_chain_set}")
                mapping.update({nat_chain_set:mdl_chain_set})
        else:
            model_chains = model_chains.replace(",", "")
            native_chains = native_chains.replace(",", "")
            mapping = {nc:mc for mc,nc in zip(model_chains, native_chains)}

        # run DockQ
        try:
            if args.ncaa:
                result = {model: calc_NCAA_DockQ(model_, native_, list(model_chains), list(native_chains))}
            else:
                result = run_on_all_native_interfaces(model_, native_, chain_map=mapping, no_align=False)[0]
        except Exception as e:
            print(f"ERROR: {e}")
            continue
            
        # save results
        for key, dockq_dict in result.items():
            # Create a copy of the dockq_dict to avoid modifying the original
            result_row = dockq_dict.copy()
            result_row.update({
                'model': model,
                'native': native,
                'row_index': i,
                'mapping(native:model)': native_chains+ ':'+model_chains
            })
            dockq_results.append(result_row)

    # Convert results to DataFrame
    if dockq_results:
        dockq_df = pd.DataFrame(dockq_results)
        # Reorder columns to put metadata first
        dockq_df['pdb'] = dockq_df['model'].str.split('/').str[-1].str.split('_').str[0]
        metadata_cols = ['pdb', 'model', 'native', 'row_index']
        other_cols = [col for col in dockq_df.columns if col not in metadata_cols]
        dockq_df = dockq_df[metadata_cols + other_cols]
    else:
        # Create empty DataFrame with expected columns if no results
        dockq_df = pd.DataFrame(columns=['model', 'native', 'row_index'])

    os.makedirs(os.path.dirname(args.save), exist_ok=True)
    save_path = os.path.join(args.save, os.path.basename(args.input).replace('.csv', '_result.csv'))
    dockq_df.to_csv(save_path, index=False)
    print(f'DockQ results saved to {save_path}')

    return dockq_df

if __name__ == '__main__':
    dockq_df = main()

    if dockq_df is not None and not dockq_df.empty:
        # Filter valid entries with non-null DockQ
        valid_df = dockq_df[dockq_df['DockQ'].notna()]
        total = len(valid_df)

        if total > 0:
            high_quality = (valid_df['DockQ'] >= 0.8).sum()
            medium_quality = ((valid_df['DockQ'] >= 0.49) & (valid_df['DockQ'] < 0.8)).sum()
            acceptable_quality = ((valid_df['DockQ'] >= 0.23) & (valid_df['DockQ'] < 0.49)).sum()
            incorrect = (valid_df['DockQ'] < 0.23).sum()

            avg_dockq = valid_df['DockQ'].mean()

            print("\n====== DockQ Evaluation Summary ======")
            print(f"Total models evaluated: {total}")
            print(f"Average DockQ score: {avg_dockq:.3f}")
            print(f"High-quality (DockQ ≥ 0.80): {high_quality} ({high_quality / total * 100:.2f}%)")
            print(f"Medium-quality (0.49 ≤ DockQ < 0.80): {medium_quality} ({medium_quality / total * 100:.2f}%)")
            print(f"Acceptable (0.23 ≤ DockQ < 0.49): {acceptable_quality} ({acceptable_quality / total * 100:.2f}%)")
            print(f"Incorrect (DockQ < 0.23): {incorrect} ({incorrect / total * 100:.2f}%)")
        else:
            print("\nNo valid DockQ entries found for statistical analysis.")
    else:
        print("\nDockQ DataFrame is empty or None.")


