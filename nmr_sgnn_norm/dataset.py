"""Graph dataset for the NMR shift prediction SGNN (structure graph neural network).

Loads a pre-processed nmrshiftdb2 graph file (``.npz``) that stores every
molecule of the dataset in a *flattened* form: node/edge attributes of all
molecules are concatenated into single arrays and per-molecule sizes are kept in
``n_node`` / ``n_edge``. ``__getitem__`` slices that flat storage back into one
DGL graph plus the matching node-level labels.

The dataset yields one sample per *molecule* (not per atom); node-level shift
labels are padded/masked via the ``mask`` array so that the loss only sees
annotated atoms.
"""

import numpy as np
import torch
from dgl.convert import graph
import os

class GraphDataset():
    """Map-style dataset producing (DGLGraph, sizes, shifts, mask) per molecule.

    Parameters
    ----------
    target : str
        Either ``'13C'`` or ``'1H'``; selects the .npz file and, for protons,
        switches the label handling to multi-shift (per-hydrogen) mode.
    graph_representation : str
        Either ``'sparsified'`` or ``'fully_connected'``; part of the .npz name.
    """

    def __init__(self, target, graph_representation):

        self.target = target
        self.graph_representation = graph_representation
        self.split = None
        self.load()

    def load(self):
        """Read the .npz file once and keep the flattened arrays in memory."""
        # Path to the new location in the models/sgnn directory
        # Use the correct path to the data files
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        npy_path = os.path.join(project_root, 'models', 'sgnn', 'data', f'nmrshiftdb2_graph_{self.graph_representation}_{self.target}.npz')
        [mol_dict] = np.load(npy_path, allow_pickle=True)['data']

        self.n_node = mol_dict['n_node']
        self.n_edge = mol_dict['n_edge']
        self.node_attr = mol_dict['node_attr']
        self.edge_attr = mol_dict['edge_attr']
        self.src = mol_dict['src']
        self.dst = mol_dict['dst']
                
        self.shift = mol_dict['shift']
        self.mask = mol_dict['mask']
        self.smi = mol_dict['smi']

        # Offsets into the flat node/edge arrays: molecule i owns the slice
        # [csum[i], csum[i+1]).  Prepending a 0 makes the slice bounds direct.
        self.n_csum = np.concatenate([[0], np.cumsum(self.n_node)])
        self.e_csum = np.concatenate([[0], np.cumsum(self.n_edge)])
        

    def __getitem__(self, idx):
        """Rebuild molecule ``idx`` as a DGL graph and return its labels.

        Returns
        -------
        tuple
            ``(g, n_node, numHshifts, shift_test, shift, mask)`` where ``g`` is
            the molecule graph, ``n_node`` its node count, ``shift`` the per-node
            target used for training, ``mask`` the per-node "has an annotation"
            flag, and ``numHshifts`` / ``shift_test`` the number of annotated
            hydrogens and their raw shifts (only meaningful for target ``'1H'``).
        """
        g = graph((self.src[self.e_csum[idx]:self.e_csum[idx+1]], self.dst[self.e_csum[idx]:self.e_csum[idx+1]]), num_nodes = self.n_node[idx])
        g.ndata['node_attr'] = torch.from_numpy(self.node_attr[self.n_csum[idx]:self.n_csum[idx+1]]).float()
        g.edata['edge_attr'] = torch.from_numpy(self.edge_attr[self.e_csum[idx]:self.e_csum[idx+1]]).float()

        n_node = self.n_node[idx:idx+1].astype(int)
        # For 13C these two stay as zeros / the plain per-atom shifts;
        # the '1H' branch below overwrites them.
        numHshifts = np.zeros(n_node)
        shift = self.shift[self.n_csum[idx]:self.n_csum[idx+1]]#.astype(float)
        shift_test = shift
        mask = self.mask[self.n_csum[idx]:self.n_csum[idx+1]].astype(bool)

        if self.target == '1H':
            # A carbon carries N attached protons with N distinct shifts, so an
            # atom label is the mean of its shifts while evaluation needs every
            # individual shift and how many there are.
            shift = np.hstack([np.mean(s) for s in self.shift[idx]])
            
            numHshifts = np.hstack([len(s) for s in self.shift[idx][mask]])
            shift_test = np.hstack([np.hstack(s) for s in self.shift[idx][mask]])
            
            
        return g, n_node, numHshifts, shift_test, shift, mask
        
        
    def __len__(self):
        """Number of molecules in the dataset."""
        return self.n_node.shape[0]










