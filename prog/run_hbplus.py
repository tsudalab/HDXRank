# add H atoms to HDX top 100 structures
# add hydrogens to pdb files
import os
import shutil

def hbplus(hbplus_dir, pdb_dir, pdb_fname_list, save_dir):
    os.environ['PATH'] += os.pathsep + hbplus_dir
    os.chdir(hbplus_dir)
    count = 0
    for pdb in pdb_fname_list:
        if not pdb.endswith('.pdb'):
            continue
        filename = pdb.split('.')[0]
        src_file = os.path.join(pdb_dir, f'{filename}.pdb')
        command = f'hbplus -O {src_file}'
        os.system(command)

        src_file = os.path.join(hbplus_dir, f'{filename}.h')
        dst_file = os.path.join(save_dir, f'{filename}_Hplus.pdb')
        shutil.move(src_file, dst_file)

        '''src_file = os.path.join(hbplus_dir, f'{filename}.hb2')
        dst_file = os.path.join(save_dir, f'{filename}.hb2')
        shutil.move(src_file, dst_file) '''
        count+=1
    print(f'pdb file dir: {pdb_dir}')
    print(f'{count} pdb files are saved in {save_dir}')

hbplus_dir = '/home/lwang/models/hbplus'
HDXtop_dir = f'/home/lwang/models/HDX_LSTM/data/Cov19_icoHu23/HDX_top100'
/home/lwang/models/HDX_LSTM/data/Cov19_icoHu23/AF_50/pdb/fold_1203_wuhan_icohu104_seed5_model_2.pdb
spike_protein = 'Wuhan'
proteins = [f'icohu23_seed3_model0',
            f'icohu23_seed3_model1',
            f'icohu23_seed3_model3',
            f'icohu23_seed5_model1',
            f'icohu23_seed8_model0']
proteins = [f'{spike_protein}_{protein}' for protein in proteins]

for protein in proteins:
    pdb_dir = f'{HDXtop_dir}/{protein}'

    save_dir = f'{pdb_dir}/hbplus'
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    hbplus(hbplus_dir, pdb_dir, os.listdir(pdb_dir), save_dir)