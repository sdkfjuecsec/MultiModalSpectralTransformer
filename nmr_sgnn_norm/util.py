"""Collation and evaluation helpers shared by the SGNN code.

``collate_reaction_graphs`` turns a list of per-molecule samples (each holding
one DGL graph) into a single batched graph plus flat tensors, which is what the
``forward`` implementations expect. ``MC_dropout`` re-enables dropout after
``net.eval()`` so that inference can sample several sub-networks.
"""

import torch
import dgl
import numpy as np

                        
def collate_reaction_graphs(batch):
    """Collate per-molecule samples into one DGL batch and flat label tensors.

    ``dgl.batch`` concatenates the node/edge blocks of every molecule, keeping
    the molecules contiguous; the label arrays are flattened the same way so
    index i in ``shift`` still refers to node i of the batched graph.
    """
    gs, n_nodes, numHshifts, shift_test, shift, masks = map(list, zip(*batch))
    
    gs = dgl.batch(gs)

    # Concatenate every per-molecule array in the same molecule order used by
    # dgl.batch, so tensor positions line up with node positions in `gs`.
    n_nodes = torch.LongTensor(np.hstack(n_nodes))
    numHshifts = torch.LongTensor(np.hstack(numHshifts))
    shift_test = torch.FloatTensor(np.hstack(shift_test))
    shift = torch.FloatTensor(np.hstack(shift))
    masks = torch.BoolTensor(np.hstack(masks))
    
    return gs, n_nodes, numHshifts, shift_test, shift, masks


def MC_dropout(model):
    """Put every Dropout module of ``model`` back into training mode."""
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    
    pass