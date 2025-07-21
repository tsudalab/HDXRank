"""
2025/1/8
Author: WANG Liyao
Paper: HDXRank: A Deep Learning Framework for Ranking Protein complex predictions with Hydrogen Deuterium Exchange Data
Note: 
Construct torchdrug protein graph dataset
"""

import os
import numpy as np
import pandas as pd
import networkx as nx
from HDXRank_utils import load_embedding

import torch
from torch.utils.data import Dataset
from torch_cluster import knn_graph, radius_graph
from torchdrug import data
#from torch_geometric.data import HeteroData
#from GVP_dataset import modified_GVPdataset

from Bio.PDB.Polypeptide import protein_letters_3to1
from Bio import BiopythonWarning
import warnings
warnings.filterwarnings("ignore", category=BiopythonWarning)

from sklearn.preprocessing import MinMaxScaler
min_max_scaler = MinMaxScaler()

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

class atoms(object):
    _registry = {}
    def __init__(self, i, name, res_cluster):
        self.__class__._registry[i] = self
        self.i = i
        self.name = name
        self.res_cluster = res_cluster
    
    @classmethod
    def get_atom(cls, atom_index):
        if not isinstance(atom_index, int):
            atom_index = int(atom_index)
        return cls._registry.get(atom_index, None)  # Returns None if atom_index is not found

class res(object):
    _registry = []
    def __init__(self, i, name, chain_id, position):
        self._registry.append(self)
        self.clusters = []
        self.energy = 0
        self.i = i
        self.name = name
        self.chain_id = chain_id
        self.position = position
    
    @classmethod
    def get_res(cls, res_index, chain_id):
        if not isinstance(res_index, int):
            res_index = int(res_index)
        for res in cls._registry:
            if res.i == res_index and res.chain_id == chain_id:
                return res
        return None

class pep(object):
    _registry = []
    #protein_embedding = {}
    def __init__(self, i, pep_sequence, chain, start_pos, end_pos, hdx_value):
        self._registry.append(self)
        self.i = i
        self.sequence = pep_sequence
        self.chain = chain
        self.start = start_pos
        self.end = end_pos
        self.clusters = []
        self.position = None
        self.node_embedding = None
        self.seq_embedding = None
        self.hdx_value = hdx_value

def parse_PDB(PDB_path, protein_chains=['A'], atom_type_list=['N','CA','C', 'O'], size_limit=None, pepRange=None):
    """Parse a PDB file and return the residue information.

    Args:
        PDB_path (str): Path to the PDB file.
        protein_chain (list): List of chains to be considered.
        atom_type_list (list): List of atom types to be considered.
        size_limit (dict): Dictionary of size limits for each chain.
        pepRange (list): List of peptide ranges to be considered.

    Returns:
        list: A list of residue information.
    """
    chain_letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    if not os.path.isfile(PDB_path):
        logging.error(f"Error: File not found. {PDB_path}")
        return None
    
    chain_data = {chain: {'nodes': [], 
                          'backbone_coord': np.zeros((len(atom_type_list), 3), dtype=np.float32), 
                          'bfactor': 0,
                          'last_res': 0, 
                          'last_res_info': None} 
                          for chain in protein_chains}
    
    atom_count = 0
    with open(PDB_path, 'r') as f:
        for line in f:
            if line.startswith('ATOM'):
                n_res = int(line[22:26].strip())
                atom_type = line[12:16].strip()
                x = float(line[30:38].strip())
                y = float(line[38:46].strip())
                z = float(line[46:54].strip())
                chain_id = line[21].strip()
                bfactor = float(line[60:66].strip())
                res_name = line[17:20].strip()
                
                if chain_id not in protein_chains:
                    logger.debug(f"Skipping chain {chain_id}")
                    continue 
                
                # filter out residues outside of the peptide range
                if pepRange is not None and chain_id in pepRange.keys():
                    if n_res < pepRange[chain_id][0] or n_res > pepRange[chain_id][1]:
                        logger.debug(f"Skipping residue {n_res} in chain {chain_id}")
                        continue
                
                chain_info = chain_data[chain_id]
                
                if n_res != chain_info['last_res']:
                    if chain_info['last_res'] != 0:  # Not the first residue
                        if size_limit is not None and len(chain_info['nodes']) >= size_limit[chain_id]:
                            continue
                        chain_info['nodes'].append(
                            {
                                'res_name': chain_info['last_res_info'][0], 
                                'residue_coord': chain_info['backbone_coord'].copy(),
                                'residue_id': chain_info['last_res_info'][1],
                                'chain_id': chain_letters.index(chain_info['last_res_info'][2]),
                                'bfactor': chain_info['bfactor']/atom_count
                            }
                        )
                        chain_info['backbone_coord'] = np.zeros((len(atom_type_list), 3), dtype=np.float32)
                        chain_info['bfactor'] = 0
                        atom_count = 0
                    
                    chain_info['last_res'] = n_res
                    chain_info['last_res_info'] = [res_name, n_res, chain_id]

                if atom_type.strip() in atom_type_list:
                    idx = atom_type_list.index(atom_type.strip())
                    chain_info['backbone_coord'][idx] = np.array([x, y, z])
                chain_info['bfactor'] += bfactor ### sum up b-factor/pLDDT score for all atoms in residue
                atom_count += 1
    
    # Ensure that the last residue of each chain is added
    for chain_id, chain_info in chain_data.items():
        if chain_info['last_res_info'] is not None:
            if size_limit is not None and len(chain_info['nodes']) >= size_limit[chain_id]:
                continue
            chain_info['nodes'].append(
                {
                    'res_name': chain_info['last_res_info'][0], 
                    'residue_coord': chain_info['backbone_coord'].copy(),
                    'residue_id': chain_info['last_res_info'][1],
                    'chain_id': chain_letters.index(chain_info['last_res_info'][2]),
                    'bfactor': chain_info['bfactor']/atom_count
                }
            )
        else:
            continue
            #raise ValueError(f"{chain_id} is not found in the PDB file.")
    
    nodes = []
    idx = 0
    for chain_info in chain_data.values():
        for node_info in chain_info['nodes']:
            nodes.append((idx, node_info))
            idx += 1

    return nodes

def read_HDX_table(HDX_df, protein_args, mode='single'): 
    '''parse the HDX table
    Args:
        HDX_df: pandas dataframe, HDX table
        protein_args:
            proteins: list of str, protein names
            states: list of str, state names
            correction: list of int, correction value for the residue index
            hhm_prefix: str, hhm prefix
            chainID: list of str, protein chains
            ref_structure: str, reference structure filename
    '''
    pep._registry = [] # reset the peptide registry
    npep = 0
    res_chainsplit = {}
    protein_chains = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    timepoints = [1.35, 2.85] #predefined log_t boundaries according to k-mean clustering results
    HDX_df['protein'].apply(lambda x: x.strip().replace(' ', ''))  # Clean protein names
    HDX_df['state'].apply(lambda x: x.strip().replace(' ', ''))  # Clean state names

    for residue in res._registry:
        if residue.chain_id not in res_chainsplit.keys():
            res_chainsplit[residue.chain_id] = []
        res_chainsplit[residue.chain_id].append(residue)
    ## seperately process states
    for index, (protein, state, correction, hhm_prefix, chain_id, ref_structure) in enumerate(protein_args):
        protein = protein.strip().replace(' ', '')
        state = state.strip().replace(' ', '')
        temp_HDX_df = HDX_df[(HDX_df['state']==state) & (HDX_df['protein']==protein)]
        temp_HDX_df = temp_HDX_df.sort_values(by=['start', 'end'], ascending=[True, True])
        chain_index = protein_chains.index(chain_id)
        if chain_index not in res_chainsplit.keys():
            continue

        logger.debug(f"HDX table after filtering by protein and state: {temp_HDX_df.shape}")

        clusters = [temp_HDX_df[temp_HDX_df['log_t'] < timepoints[0]],
                    temp_HDX_df[(temp_HDX_df['log_t'] >= timepoints[0]) & (temp_HDX_df['log_t'] < timepoints[1])],
                    temp_HDX_df[temp_HDX_df['log_t'] >= timepoints[1]]]

        HDX_grouped = temp_HDX_df.groupby(['start', 'end']) 

        for name, group in HDX_grouped:
            mean_rfu = [-1,-1,-1]

            sequence = group['sequence'].iloc[0].strip() if not group['sequence'].empty else ''
            start_pos = int(name[0])+correction
            end_pos = int(name[1])+correction
            
            for i, cluster_df in enumerate(clusters):
                cluster_group = cluster_df[(cluster_df['start']==name[0]) & (cluster_df['end']==name[1])]
                if not cluster_group.empty:
                    mean_rfu[i] = cluster_group['RFU'].mean()/100
                else:
                    continue

            # to train the model, all three mean_rfu should be non-negative
            # for prediction, at least one mean_rfu should be non-negative
            if mode == 'train':
                if any(x == -1 for x in mean_rfu):
                    logger.debug(f"[train Mode]: mean_rfu is -1 for {protein} {state} {chain_id} {name}")
                    continue
            elif mode in ['batch', 'single']:
                if all(x == -1 for x in mean_rfu):
                    logger.debug(f"[predict Mode]: mean_rfu is -1 for {protein} {state} {chain_id} {name}")
                    continue

            res_seq = [res for res in res_chainsplit[chain_index] if start_pos <= res.i <= end_pos]
            pdb_seq = ''.join([protein_letters_3to1[res.name] for res in res_seq])

            if pdb_seq != sequence: # residue lack/mutation/mismatch in pdb will be filtered out
                logging.info(f"sequence mismatch: chain: {chain_id}, pdb_Seq: {pdb_seq}, HDX_seq:, {sequence}")
                continue
            else:
                peptide = pep(npep, sequence, chain_index, start_pos, end_pos, mean_rfu)
                for residue in res_seq:
                    residue.clusters.append(npep)
                    peptide.clusters.append(residue.i)
                npep += 1
    if npep == 0:
        logging.info("no peptide found")
        return False
    else:
        return True

def attach_node_attributes(G, embedding_file):
    """
    Attach node attributes to the graph from the embedding file.

    Args:
        G (networkx.Graph): The graph.
        embedding_file (str or torch.Tensor): The embedding file path or tensor.

    Returns:
        networkx.Graph: Graph with updated node attributes.
    """
    if isinstance(embedding_file, str):
        embedding_data = torch.load(embedding_file)
        protein_embedding = min_max_scaler.fit_transform(embedding_data['embedding'].detach().numpy())
    else:
        protein_embedding = embedding_file
    
    if protein_embedding.shape[0] == G.number_of_nodes():
        for node, embedding in zip(G.nodes(), protein_embedding):
            G.nodes[node]['x'] = torch.as_tensor(embedding)
        return G
    else:
        logging.error('embedding shape does not match with the graph nodes')
        return None

def ResGraph(pdb_file, protEmbed_dict, protein_chains = ['A'], pepRange=None): 
    ''' create networkx graphs
        Args:
            pdb_file: str, path to the pdb file
            protEmbed_dict: dict, protein embedding dictionary
            protein_chains: list of str, protein chains

        Returns:
            networkx.Graph: Graph with updated node attributes.
    '''

    size_limit = {key:value.shape[0] for key, value in protEmbed_dict.items()}
    nodes = parse_PDB(pdb_file, protein_chains=protein_chains, size_limit=size_limit, pepRange=pepRange)

    G = nx.MultiGraph()
    G.add_nodes_from(nodes)
    G.graph['name'] = pdb_file

    embedding = torch.tensor([])
    for key in protEmbed_dict.keys():
        if embedding.nelement() == 0:
            embedding = protEmbed_dict[key]
        else:
            embedding = torch.cat([embedding, protEmbed_dict[key]], dim=0)
            
    embedding = min_max_scaler.fit_transform(embedding.detach().numpy())
    G = attach_node_attributes(G, embedding)
    if G is None:
        return None
    for node in nodes:
        res(node[1]['residue_id'], node[1]['res_name'], node[1]['chain_id'], node[1]['residue_coord'])
    return G

def add_edges(G, edge_types, coord = None, max_distance=8.0, min_seq_sep=3):
    """
    Add edges to the graph based on specified types.

    Args:
        G (networkx.Graph): The graph.
        edge_types (list): Types of edges to add.
        coord (torch.Tensor or None): Coordinates for radius/knn edges.
        max_distance (float): Maximum distance for radius edges.
        min_seq_sep (int): Minimum sequence separation for edges.

    Returns:
        networkx.Graph: Graph with edges added.
    """
    batch = torch.tensor([0] * len(G.nodes))
    if coord is not None:
        node_coord = torch.as_tensor(coord, dtype=torch.float32)
    else:
        node_coord = np.array([G.nodes[i]['residue_coord'] for i in G.nodes])
        node_coord = torch.tensor(node_coord, dtype = torch.float32)

    # add radius edges
    if 'radius_edge' in edge_types:
        edge_index = radius_graph(node_coord, r=max_distance, batch=batch, loop=False)
        edge_list = edge_index.t().tolist()
        for u, v in edge_list:
            if u < v:
                G.add_edge(u, v, edge_type='radius_edge')
        logger.debug(f"add radius edges: {G.number_of_edges()}")

    # add knn edges
    if 'knn_edge' in edge_types:
        edge_index = knn_graph(node_coord, k=10, batch=batch, loop=False)
        edge_list = edge_index.t().tolist()
        for u, v in edge_list:
            if u < v:
                G.add_edge(u, v, edge_type='knn_edge')
        logger.debug(f"add knn edges: {G.number_of_edges()}")

    # remove edges with sequential distance smaller than min_seq_sep
    if 'remove_edge' in edge_types:
        edges_to_remove = []
        for u, v in G.edges():
            distance = abs(u - v)
            if distance < min_seq_sep:
                if (u,v) not in edges_to_remove:
                    edges_to_remove.append((u, v))
        G.remove_edges_from(edges_to_remove)
        logger.debug(f"after removing edges: {G.number_of_edges()}")

    # add sequential edges
    if 'sequential_edge' in edge_types:
        i2res_id = {(data['chain_id'], data['residue_id']): node for node, data in G.nodes(data=True)}
        for node in G.nodes:
            res_id = G.nodes[node]['residue_id']
            chain = G.nodes[node]['chain_id']
            if (chain,res_id+1) in i2res_id.keys():
                G.add_edge(node, node+1, edge_type='forward_1_edge')
            if (chain,res_id+2) in i2res_id.keys():
                G.add_edge(node, node+2, edge_type='forward_2_edge')
            if (chain,res_id-1) in i2res_id.keys():
                G.add_edge(node, node-1, edge_type='backward_1_edge')
            if (chain,res_id-2) in i2res_id.keys():
                G.add_edge(node, node-2, edge_type='backward_2_edge')  
            G.add_edge(node, node, edge_type='self_edge')
        logger.debug(f"add sequential edges: {G.number_of_edges()}")

    logging.info(G)
    return G

def networkx_to_tgG(G): # convert to torchdrug protein graph
    node_position = torch.as_tensor(np.array([G.nodes[node]['residue_coord'] for node in G.nodes()]), dtype=torch.float32)
    num_atom = G.number_of_nodes()
    atom_type = ['CA'] * num_atom
    atom_type = torch.as_tensor([data.Protein.atom_name2id.get(atom, -1) for atom in atom_type])

    residue_type = []
    residue_feature = []
    residue_id = []
    id_map = {}
    for i, (node, attrs) in enumerate(G.nodes(data=True)):
        if node not in id_map.keys():
            id_map[node] = i
        residue_type.append(data.Protein.residue2id.get(attrs['res_name'], 0))
        residue_feature.append(attrs['x'])
        residue_id.append(attrs['residue_id'])
    residue_type = torch.tensor(residue_type, dtype=torch.long)
    residue_feature = torch.stack(residue_feature)
    atom2residue = torch.as_tensor([id_map[node] for node in G.nodes()], dtype=torch.long)

    edge_list = []
    bond_type = []
    edge_type_list = ['knn_edge', 'radius_edge', 'self_edge', 'forward_1_edge', 'forward_2_edge', 'backward_1_edge', 'backward_2_edge']

    for u, v, attrs in G.edges(data=True):
        edge_type = attrs['edge_type']
        u = id_map[u]
        v = id_map[v]
        if edge_type in edge_type_list:
            edge_list.append([u, v, edge_type_list.index(edge_type)])
            bond_type.append(edge_type_list.index(edge_type))

    edge_list = torch.tensor(edge_list, dtype=torch.long)
    bond_type = torch.tensor(bond_type, dtype=torch.long).unsqueeze(-1)


    protein = data.Protein(edge_list, atom_type, bond_type, view='residue', residue_number=residue_id,
                           node_position=node_position, atom2residue=atom2residue,residue_feature=residue_feature, 
                           residue_type=residue_type, num_relation = len(edge_type_list))

    with protein.graph():
        protein.y = torch.as_tensor(G.graph['y'], dtype=torch.float32)
        protein.range = torch.as_tensor(G.graph['range'], dtype=torch.float32)
        protein.chain = torch.as_tensor(G.graph['chain'], dtype=torch.int64)
        protein.is_complex = torch.as_tensor(G.graph['is_complex'], dtype=torch.int64)
    return protein

#FIXME: embedding will be sliced according to pepRange, fit graph accordingly
def create_graph(pdb_fpath, embedding_dir, embedding_fname, pepRange, **kwargs):
    '''
    assemble embedding files and protein structure files into a graph
    Args:
        pdb_fpath: str, path to the pdb file
        embedding_dir: str, path to the embedding directory
        embedding_fname: str, embedding file name
        **kwargs: additional parameters
    Returns:
        networkx.Graph: protein graphs with nodes, edges and attributes
    '''

    protEmbed_dict = {}
    embedding_fpaths=[]
    chain_labels = np.array([])
    protein_embeddings = torch.tensor([], dtype=torch.float32)

    if kwargs['embedding_type'] == 'manual':
        if isinstance(embedding_fname, str):
            embedding_fpaths.append(os.path.join(embedding_dir, f"{embedding_fname}.pt"))
        else:
            embedding_fpaths.extend([os.path.join(embedding_dir, f"{fname}.pt") for fname in embedding_fname])

        for sub_fpath in embedding_fpaths:
            chain_label, protein_embedding = load_embedding(sub_fpath)
            chain_labels = np.concatenate([chain_labels, chain_label], axis=0)
            protein_embeddings = (
                torch.cat([protein_embeddings, protein_embedding])
                if protein_embeddings.numel() > 0 else protein_embedding
            )

        for chain in set(chain_labels):
            mask = (chain_labels == chain)
            protEmbed_dict[chain] = protein_embeddings[mask]

    #elif  add other types of embedding here 
    else:
        raise ValueError("Unknown embedding type:", kwargs['embedding_type'])

    G = ResGraph(pdb_fpath, protEmbed_dict, protein_chains=list(protEmbed_dict.keys()), pepRange=pepRange)
    if G is None:
        return None
    G.graph['protein_chains'] = list(protEmbed_dict.keys())

    residue_coord = np.array([G.nodes[node]['residue_coord'] for node in G.nodes()])
    coords = torch.as_tensor(residue_coord, dtype=torch.float32)
    coords = coords[:,1,:]
    edge_to_add = ['radius_edge', 'knn_edge', 'sequential_edge', 'remove_edge']
    G = add_edges(G, edge_to_add, coord=coords, max_distance=kwargs['max_distance'], min_seq_sep=kwargs['min_seq_sep'])
    return G

def load_pep(HDX_fpath, protein_args, max_len =30, pep_range = None, mode = 'single'):
    '''
    Pre-process the HDX table and load peptides into the pep.registry

    Args:
        HDX_fpath: str, path to the HDX table file
        protein_args:
            proteins: list of str, protein names
            states: list of str, state names
            correction: list of int, correction value for the residue index
            hhm_prefix: str, hhm prefix
            chains: list of str, protein chains
            ref_structure: str, reference structure filename
        max_len: int, maximum length of the peptide
        pep_range: chain-wise peptide range
    '''
    HDX_df = pd.read_excel(HDX_fpath, sheet_name='Sheet1')
    ### -------- filter HDX table ------ ####
    HDX_df['state'] = HDX_df['state'].str.replace(' ', '')
    HDX_df['protein'] = HDX_df['protein'].str.replace(' ', '')
    fn = lambda x: x.replace(' ', '').strip()
    protein = [fn(p[0]) for p in protein_args]
    state = [fn(p[1]) for p in protein_args]
    corrections = [p[2] for p in protein_args]
    chains = [p[4] for p in protein_args]

    # filter the HDX table
    HDX_df = HDX_df[(HDX_df['state'].isin(state)) & (HDX_df['protein'].isin(protein))]
    HDX_df = HDX_df[HDX_df['sequence'].str.len() < max_len]
    
    if pep_range:
        filtered_df = pd.DataFrame()
        for p,s,c, id in zip(protein, state, corrections, chains):
            temp_df = HDX_df[(HDX_df['protein'] == p) & (HDX_df['state'] == s)]
            start_id, end_id = pep_range.get(id, (None, None))
            if start_id is None or end_id is None:
                filtered_df = pd.concat([filtered_df, temp_df])
                continue

            start_id += c
            end_id += c
            temp_df = temp_df[(temp_df['start'] >= start_id) & (temp_df['end'] <= end_id)]
            filtered_df = pd.concat([filtered_df, temp_df])
        HDX_df = filtered_df.drop_duplicates().reset_index(drop=True)
    HDX_df = HDX_df.sort_values(by=['start', 'end'], ascending=[True, True])
    
    if not read_HDX_table(HDX_df, protein_args, mode): # read HDX table file
        return None
    logging.info(f"Read in {len(pep._registry)} peptides.")

def find_neigbhors(G, nodes, hop=1):
    '''
    Find the neigbhors of the nodes in the graph
    
    Args:
        G (networkx.Graph): The graph.
        nodes (list): List of nodes.
        hop (int): Number of hops.

    Returns:
        list: List of neigbhor nodes.
    '''
    neigbhor_node = []
    for node in nodes:
        neigbhors = list(G.neighbors(node))
        neigbhor_node.extend(neigbhors)
    if hop > 1:
        neigbhor_node = find_neigbhors(neigbhor_node, hop-1)
    return neigbhor_node

def split_graph(G, max_len = 30, complex_state_id = 0, graph_type = 'GearNet', plddt_filter = None):
    '''
    Split the protein graph into peptide graphs

    Args:
        G (networkx.Graph): The protein graph.
        max_len (int): Maximum length of the peptide.
        complex_state_id (int): Identifier for the complex state.
        graph_type (str): Type of the graph.

    Returns:
        list: List of peptide graphs.
    '''
    i2res_id = {(data['chain_id'], data['residue_id']): node for node, data in G.nodes(data=True)}
    graph_ensemble = []
    logging.info('Spliting peptide graph...')
    for peptide in pep._registry:           
        chain = peptide.chain
        node_ids = [i2res_id[(chain, res_id)] for res_id in peptide.clusters if (chain, res_id) in i2res_id]

        min_bfactor = min([G.nodes[node]['bfactor'] for node in node_ids]) ## b-factor here is pLDDT score from AlphaFold
        if plddt_filter is not None: #and (peptide.hdx_value < 60): ## filter out low HDX regions (stable regions) with low pLDDT score (inaccurate structure predictions)
            if min_bfactor < plddt_filter:
                continue

        if len(node_ids) == 0:
            continue

        neigbhor_node = find_neigbhors(G, node_ids, hop=1)
        node_ids.extend(neigbhor_node)
        node_ids = set(node_ids)

        subG = G.subgraph(node_ids).copy()
        if graph_type == 'GearNet':      
            subG.graph['y'] = peptide.hdx_value
            subG.graph['range'] = (peptide.start, peptide.end)
            subG.graph['chain'] = peptide.chain
            subG.graph['is_complex'] = complex_state_id
            data = networkx_to_tgG(subG)
            graph_ensemble.append(data)
        else:
            logging.error(f"Unknown graph type: {graph_type}")
            raise ValueError(f"Unknown graph type: {graph_type}")
    logging.info(f'Done.')
    return graph_ensemble

'''
def networkx_to_HeteroG(subG): # convert to pytorch geometric HeteroData
    data = HeteroData()
    id_map = {}
    for i, (node, node_attr) in enumerate(subG.nodes(data=True)):
        # Assuming node feature vector 'x' is already a tensor or can be converted as such
        #data['residue'].x = torch.cat([data['residue'].x, node_attr['x'].unsqueeze(0)], dim=0) if 'x' in data['residue'] else node_attr['x'].unsqueeze(0)
        if node not in id_map.keys():
            id_map[node] = i
    data['residue'].num_nodes = len(id_map.keys())

    edge_index_dict = {}
    for u, v, edge_attr in subG.edges(data=True):
        edge_type = edge_attr['edge_type']
        edge_label = ('residue', edge_type, 'residue')
        edge_index = [id_map[u], id_map[v]]
        if edge_label not in edge_index_dict.keys():
            edge_index_dict[edge_label] = [edge_index]
        else:
            edge_index_dict[edge_label].append(edge_index)
    for edge_label, edge_index in edge_index_dict.items():
        edge_index = torch.as_tensor(edge_index, dtype=torch.long).t().contiguous()
        data[edge_label].edge_index = edge_index

    if 'y' in subG.graph:
        data['residue'].y = torch.as_tensor([subG.graph['y']], dtype=torch.float32)
    if 'range' in subG.graph:
        data['residue'].range = torch.as_tensor([subG.graph['range']], dtype=torch.float32)
    if 'chain' in subG.graph:
        data['residue'].chain = torch.as_tensor([subG.graph['chain']], dtype=torch.int64)
    if 'is_complex' in subG.graph:
        data['residue'].is_complex = torch.as_tensor([subG.graph['is_complex']], dtype=torch.int64)
    return data
'''
def seq_embedding(G, max_len=30, complex_state_id = 0):
    embedding_ensemble = []
    for peptide in pep._registry:
        chain = peptide.chain
        node_ids = [node for node in G.nodes() if G.nodes[node]['chain_id'] == chain]

        seq_embedding = [G.nodes[node]['x'] for node in node_ids]
        seq_embedding = torch.stack(seq_embedding, dim=0)
        pad_needed = max_len - len(node_ids)
        seq_embedding = torch.nn.functional.pad(seq_embedding, (0, 0, 0, pad_needed), 'constant', 0)

        data = {
            'y': peptide.hdx_value,
            'range': (peptide.start, peptide.end),
            'chain': peptide.chain,
            'is_complex': complex_state_id,
            'x': seq_embedding
        }
        embedding_ensemble.append(data)
    return embedding_ensemble

# block of code for GVP
'''
def split_PyG_graph(G, complex_state_id = 0):
    i2res_id = {(data['chain_id'], data['residue_id']): node for node, data in G.nodes(data=True)}
    graph_ensemble = []
    for peptide in pep._registry:           
        chain = peptide.chain
        node_ids = [i2res_id[(chain, res_id)] for res_id in peptide.clusters if (chain, res_id) in i2res_id]
        if len(node_ids) == 0:
            continue

        neigbhor_node = find_neigbhors(G, node_ids, hop=1)
        node_ids.extend(neigbhor_node)
        node_ids = set(node_ids)

        subG = G.subgraph(node_ids).copy()
        
        residue_coord = np.array([subG.nodes[node]['residue_coord'] for node in subG.nodes()])
        coords = torch.as_tensor(residue_coord, dtype=torch.float32)
        node_s = torch.stack([subG.nodes[node]['node_s'] for node in subG.nodes()], dim=0)
        node_v = torch.stack([subG.nodes[node]['node_v'] for node in subG.nodes()], dim=0)

        edge_types = set(attr_dict['edge_type'] for _,_,attr_dict in subG.edges(data=True))
        edge_s_dict = {('residue',etype,'residue'):[] for etype in edge_types}
        edge_v_dict = {('residue',etype,'residue'):[] for etype in edge_types}
        for u, v, k in subG.edges(keys=True):
            etype = subG.edges[(u,v,k)]['edge_type']
            edge_s_dict[('residue',etype,'residue')].append(subG.edges[(u,v,k)]['edge_s'])
            edge_v_dict[('residue',etype,'residue')].append(subG.edges[(u,v,k)]['edge_v'])
        for etype in edge_types:
            edge_s_dict[('residue',etype,'residue')] = torch.stack(edge_s_dict[('residue',etype,'residue')], dim=0)
            edge_v_dict[('residue',etype,'residue')] = torch.stack(edge_v_dict[('residue',etype,'residue')], dim=0)
        mask = torch.isfinite(coords.sum(dim=(1,2)))
        
        subG.graph['y'] = peptide.hdx_value
        subG.graph['range'] = (peptide.start, peptide.end)
        subG.graph['chain'] = peptide.chain
        subG.graph['is_complex'] = complex_state_id
        Hetero_data = networkx_to_HeteroG(subG)
        Hetero_data['residue'].node_s = {'residue':node_s}
        Hetero_data['residue'].node_v = {'residue':node_v}
        Hetero_data['residue'].edge_s = edge_s_dict
        Hetero_data['residue'].edge_v = edge_v_dict
        Hetero_data['residue'].mask = mask
        graph_ensemble.append(Hetero_data)
    return graph_ensemble
'''

class pepGraph(Dataset):
    def __init__(self, files, tasks, **kwargs):
        """
        Initialize the dataset.

        Args:
            keys (list): List of task keys.
            root_dir (str): Root directory for data.
            embedding_dir (str): Directory for embeddings.
            pdb_dir (str): Directory for PDB files.
            save_dir (str): Directory to save the processed data.
            **kwargs: Additional parameters.
        """
        self.keys = files
        self.root_dir = tasks["GeneralParameters"]["RootDir"]
        self.save_dir = tasks["GeneralParameters"]["pepGraphDir"]
        self.pdb_dir = tasks["GeneralParameters"]["PDBDir"]        
        self.hdx_dir = tasks["GeneralParameters"]["HDXDir"]
        self.embedding_dir = tasks["GeneralParameters"]["EmbeddingDir"]
        self.tasks = tasks

        self.params = {
            'min_seq_sep': tasks["GraphParameters"]["SeqMin"],
            'max_distance': tasks["GraphParameters"]["RadiusMax"],
            'graph_type': tasks["GraphParameters"]["GraphType"],
            'embedding_type': tasks["GraphParameters"]["EmbeddingType"],
            'max_len': tasks["GraphParameters"]["MaxLen"],
            'pep_range': tasks["TaskParameters"]["PepRange"],
            'mode': tasks["GeneralParameters"]["Mode"],
            **kwargs
        }
        self.complex_state_dict = {'apo':0, 'single':0, 'protein complex':1, 'ligand complex':2, 'dna complex':3, 'rna complex': 4}

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        '''
        return the graph for the index-th protein complex
        keys format: [database_id, protein, state, pdb_fname, chain_identifier, correction, protein_chains, complex_state, embedding_fname]
        '''
        atoms._registry = {}
        res._registry = []
        pep._registry = []

        pdb_fname = self.keys[index].rstrip('.pdb')
        database_id = self.tasks["GeneralParameters"]["HDX_File"]
        pdb_fpath = os.path.join(self.pdb_dir, f'{pdb_fname}.pdb')
        target_HDX_fpath = os.path.join(self.hdx_dir, f'{database_id}.xlsx')
        is_complex = self.tasks['TaskParameters']['is_complex'][index]
        logging.info(f'Processing task: {database_id} {pdb_fname}')

        if os.path.isfile(os.path.join(self.save_dir, f'{pdb_fname}.pt')):
            logging.info(f'File already exists: {pdb_fname}.pt')
            return None
        if not os.path.isfile(target_HDX_fpath):
            logging.error(f'Missing HDX table: {target_HDX_fpath}')
            return None 
        if not os.path.isfile(pdb_fpath):
            logging.error(f'Missing PDB file: {pdb_fpath}')
            return None

        # residue graph generation #
        logging.info('Creating graph...')
        #TODO: revise create_graph for peptide range format change
        G = create_graph(pdb_fpath, self.embedding_dir, embedding_fname=pdb_fname, embedding_type = self.params['embedding_type'], 
                        max_distance = self.params['max_distance'], min_seq_sep = self.params['min_seq_sep'], pepRange=self.params['pep_range'])
        if G is None:
            logging.error("Cannot create graph")
            return None

        load_pep(target_HDX_fpath, self.tasks['apo_states'], 
                max_len = self.params['max_len'], pep_range = self.params['pep_range'], mode = self.params['mode'])

        #plddt filter can be applied here
        #plddt_filter = None if not pdb_fname.startswith('FOLD') else 90 # only filter out low res-pLDDT regions for AlphaFold models
        if self.params['graph_type'] == 'GearNet': # for GearNet
            graph_ensemble = split_graph(G, max_len = self.params['max_len'], complex_state_id = self.complex_state_dict[is_complex],
                                        graph_type = self.params['graph_type'], plddt_filter = None)
        else:
            logging.error(f"Unknown graph type: {self.params['graph_type']}")
            return None
 
        label = f'{pdb_fname}'.upper()
        return graph_ensemble, label