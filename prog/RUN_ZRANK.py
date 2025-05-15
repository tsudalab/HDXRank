# run zrank to output the zrank score
import os
import shutil

def run_zrank(zrank_dir, pdb_list, protein_name, save_dir):
    os.environ['PATH'] += os.pathsep + zrank_dir
    os.chdir(zrank_dir)

    command = f'./zrank {pdb_list}'
    os.system(command)

    src_file = os.path.join(zrank_dir, f'zrank_{protein_name}.txt.zr.out')
    dst_file = os.path.join(save_dir, f'zrank_{protein_name}.txt.zr.out')
    shutil.move(src_file, dst_file)

def run_zrank2(zrank_dir, pdb_list, protein_name, save_dir):
    os.environ['PATH'] += os.pathsep + zrank_dir
    os.chdir(zrank_dir)

    command = f'./zrank -R {pdb_list}'
    os.system(command)

    src_file = os.path.join(zrank_dir, f'zrank2_{protein_name}.txt.zr.out')
    dst_file = os.path.join(save_dir, f'zrank2_{protein_name}.txt.zr.out')
    shutil.move(src_file, dst_file)

def clean_pdb(fpath, lig_chain='B'):
    with open(fpath, 'r') as f, open(fpath.replace('.pdb', '_clean.pdb'), 'w') as out:
        ligand_found = False
        for line in f:
            if line.startswith('ATOM') and line[21] == lig_chain and not ligand_found:
                out.write('TER\n')
                ligand_found = True
            out.write(line)

zrank_dir = '/home/lwang/models/zrank_linux'
zrank2_dir = '/home/lwang/models/zrank2_linux'
root_dir = f'/home/lwang/models/HDX_LSTM/data/Cov19_icoHu23/HDX_top100'

spike_protein = 'Wuhan'
proteins = [f'icohu23_seed3_model0',
            f'icohu23_seed3_model1',
            f'icohu23_seed3_model3',
            f'icohu23_seed5_model1',
            f'icohu23_seed8_model0']
proteins = [f'{spike_protein}_{protein}' for protein in proteins]
lig_chain = 'D' # separate receptor chains A/B/C and ligand chains D/E 

for protein in proteins:
    pdb_dir = f'{root_dir}/{protein}/hbplus'
    save_dir = f'{root_dir}/{protein}/eval'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    with open(f'{zrank_dir}/zrank_{protein}.txt', 'w') as f1, open(f'{zrank2_dir}/zrank2_{protein}.txt', 'w') as f2:
        for file in os.listdir(pdb_dir):
            if file.endswith('_Hplus.pdb'):
                clean_pdb(f'{pdb_dir}/{file}', lig_chain)
                dst_file = file.replace('.pdb', '_clean.pdb')
                f1.write(f'{pdb_dir}/{dst_file}\n')
                f2.write(f'{pdb_dir}/{dst_file}\n')

    run_zrank(zrank_dir, f'{zrank_dir}/zrank_{protein}.txt', protein, save_dir)
    run_zrank2(zrank2_dir, f'{zrank2_dir}/zrank2_{protein}.txt', protein, save_dir)