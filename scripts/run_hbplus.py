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