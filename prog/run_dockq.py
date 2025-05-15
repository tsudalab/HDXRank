# run DockQ to evaluate the quality of Hdock decoys
import os
import sys
from DockQ.DockQ import load_PDB, run_on_all_native_interfaces
import pickle
import numpy as np

def merge_chains(model, chains_to_merge):
    for chain in chains_to_merge[1:]:
        for i, res in enumerate(model[chain]):
            res.id = (chain, res.id[1], res.id[2])
            model[chains_to_merge[0]].add(res)
        model.detach_child(chain)
    model[chains_to_merge[0]].id = "".join(chains_to_merge)
    return model

root_dir = '/home/lwang/models/HDX_LSTM/data/hdock/structure/'
protein_name = '8F7A_1016'
#N_decoys = 1000
N_decoys = len(os.listdir(f'{root_dir}/{protein_name}'))
hdock_dir = f'{root_dir}/{protein_name}/eval'
model_list = [f'MODEL_{i}_REVISED.pdb' for i in range(1, N_decoys+1)]
mapping = {"A": "A", "B": "B"}

dockq_results = []
native = load_PDB(f'/home/lwang/models/HDX_LSTM/data/hdock/structure/{protein_name[:4]}/{protein_name[:4]}.pdb')
#native = merge_chains(native, ["A", "B"])
for model in model_list:
    if not model.endswith('.pdb'):
        continue
    model_fpath = f'{hdock_dir}/{model}'
    model = load_PDB(model_fpath)
    #model = merge_chains(model, ["A", "B"])
    result = run_on_all_native_interfaces(model, native, chain_map=mapping)
    dockq_results.extend(list(result[0].values()))

if not os.path.exists(f'{hdock_dir}'):
    os.makedirs(f'{hdock_dir}')
with open(f'{hdock_dir}/{protein_name}_dockq.pkl', 'wb') as f:
    pickle.dump(dockq_results, f)