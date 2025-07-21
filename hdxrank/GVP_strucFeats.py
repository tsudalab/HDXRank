"""
Note:
Codes adopted from GVP structural features (https://github.com/drorlab/gvp/blob/master/src/models.py)
"""
import numpy as np
import torch, math
import torch.utils.data as data
import torch.nn.functional as F
import torch_cluster

def _normalize(tensor, dim=-1):
    '''
    Normalizes a `torch.Tensor` along dimension `dim` without `nan`s.
    '''
    return torch.nan_to_num(
        torch.div(tensor, torch.norm(tensor, dim=dim, keepdim=True)))


def _rbf(D, D_min=0., D_max=20., D_count=16, device='cpu'):
    '''
    From https://github.com/jingraham/neurips19-graph-protein-design
    
    Returns an RBF embedding of `torch.Tensor` `D` along a new axis=-1.
    That is, if `D` has shape [...dims], then the returned tensor will have
    shape [...dims, D_count].
    '''
    D_mu = torch.linspace(D_min, D_max, D_count, device=device)
    D_mu = D_mu.view([1, -1])
    D_sigma = (D_max - D_min) / D_count
    D_expand = torch.unsqueeze(D, -1)

    RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)
    return RBF

class ProteinGraphDataset(data.Dataset):
    '''
    A map-syle `torch.utils.data.Dataset` which transforms JSON/dictionary-style
    protein structures into featurized protein graphs as described in the 
    manuscript.
    
    Returned graphs are of type `torch_geometric.data.Data` with attributes
    -x          alpha carbon coordinates, shape [n_nodes, 3]
    -seq        sequence converted to int tensor according to `self.letter_to_num`, shape [n_nodes]
    -name       name of the protein structure, string
    -node_s     node scalar features, shape [n_nodes, 6] 
    -node_v     node vector features, shape [n_nodes, 3, 3]
    -edge_s     edge scalar features, shape [n_edges, 32]
    -edge_v     edge scalar features, shape [n_edges, 1, 3]
    -edge_index edge indices, shape [2, n_edges]
    -mask       node mask, `False` for nodes with missing data that are excluded from message passing
    
    Portions from https://github.com/jingraham/neurips19-graph-protein-design.
    
    :param data_list: JSON/dictionary-style protein dataset as described in README.md.
    :param num_positional_embeddings: number of positional embeddings
    :param top_k: number of edges to draw per node (as destination node)
    :param device: if "cuda", will do preprocessing on the GPU
    '''
    def __init__(self, data_list, 
                 num_positional_embeddings=16,
                 top_k=30, num_rbf=16, device="cpu"):
        
        super(ProteinGraphDataset, self).__init__()
        
        self.data_list = data_list
        self.top_k = top_k
        self.num_rbf = num_rbf
        self.num_positional_embeddings = num_positional_embeddings
        self.device = device
        self.node_counts = [len(e['seq']) for e in data_list]
        
        self.letter_to_num = {'C': 4, 'D': 3, 'S': 15, 'Q': 5, 'K': 11, 'I': 9,
                       'P': 14, 'T': 16, 'F': 13, 'A': 0, 'G': 7, 'H': 8,
                       'E': 6, 'L': 10, 'R': 1, 'W': 17, 'V': 19, 
                       'N': 2, 'Y': 18, 'M': 12}
        self.num_to_letter = {v:k for k, v in self.letter_to_num.items()}
        
    def __len__(self): return len(self.data_list)
    
    def __getitem__(self, i): return self._featurize_as_graph(self.data_list[i])
                                
    def _dihedrals(self, X, eps=1e-7):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        
        X = torch.reshape(X[:, :3], [3*X.shape[0], 3])
        dX = X[1:] - X[:-1]
        U = _normalize(dX, dim=-1)
        u_2 = U[:-2]
        u_1 = U[1:-1]
        u_0 = U[2:]

        # Backbone normals
        n_2 = _normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = _normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = torch.sum(n_2 * n_1, -1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign(torch.sum(u_2 * n_1, -1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, [1, 2]) 
        D = torch.reshape(D, [-1, 3])
        # Lift angle representations to the circle
        D_features = torch.cat([torch.cos(D), torch.sin(D)], 1)
        return D_features
    
    
    def _positional_embeddings(self, edge_index, 
                               num_embeddings=None,
                               period_range=[2, 1000]):
        # From https://github.com/jingraham/neurips19-graph-protein-design
        num_embeddings = num_embeddings or self.num_positional_embeddings
        d = edge_index[0] - edge_index[1]
     
        frequency = torch.exp(
            torch.arange(0, num_embeddings, 2, dtype=torch.float32, device=self.device)
            * -(np.log(10000.0) / num_embeddings)
        )
        angles = d.unsqueeze(-1) * frequency
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E

    def _orientations(self, X):
        # From https://github.com/drorlab/gvp
        X = X[:,1] # take CA atom coordinates

        forward = _normalize(X[1:] - X[:-1])
        backward = _normalize(X[:-1] - X[1:])
        forward = F.pad(forward, [0, 0, 0, 1])
        backward = F.pad(backward, [0, 0, 1, 0])

        return torch.cat([forward, backward], -1)

    def _sidechains(self, X):
        n, origin, c = X[:, 0], X[:, 1], X[:, 2]
        c, n = _normalize(c - origin), _normalize(n - origin)
        bisector = _normalize(c + n)
        perp = _normalize(torch.cross(c, n))
        vec = -bisector * math.sqrt(1 / 3) - perp * math.sqrt(2 / 3)
        return vec 