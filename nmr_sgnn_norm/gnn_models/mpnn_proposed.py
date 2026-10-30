"""Proposed (PAGTN) message-passing network for per-atom NMR shift prediction.

Instead of an NNConv matrix per edge, this variant uses Path-Augmented Graph
Transformer Network layers: the attention score of every edge combines the
source atom, the destination atom and the bond descriptor, softmax is taken
per destination atom, and the edge message is a learned mix of destination atom
and bond features. Several layers are stacked and averaged over attention heads.

Together with the ``readout_mode`` switch this covers the remaining ablations:
``'baseline'`` = per-atom readout, ``'proposed'`` = per-atom readout that also
sees a Set2Set graph embedding.
"""

# This code is based on the implementation of Path-Augmented Graph Transformer Network in DGL-LifeSci
# https://github.com/awslabs/dgl-lifesci/blob/master/python/dgllife/model/gnn/pagtn.py

import torch
import torch.nn as nn

from dgl.nn.pytorch import Set2Set

from dgl.nn.functional import edge_softmax
import dgl.function as fn



class nmr_mpnn_PROPOSED(nn.Module):
    """PAGTN encoder plus the same two readout heads as the baseline model.

    Parameters
    ----------
    node_in_feats : int
        Width of the input node (atom) descriptors.
    edge_feats : int
        Width of the input bond descriptors.
    readout_mode : str
        ``'baseline'`` (per-atom readout) or ``'proposed'`` (per-atom + graph
        embedding readout).
    node_feats : int
        Hidden width of the PAGTN layers (per attention head).
    pred_hid_feats : int
        Width of the readout MLP.
    depth : int
        Number of stacked PAGTN layers (default 5).
    n_heads : int
        Number of attention heads per layer (default 10).
    activation : callable
        Activation used inside the PAGTN layers.
    prob_dropout : float
        Dropout probability inside the readout MLPs.
    """

    def __init__(self, node_in_feats, edge_feats, readout_mode,
                 node_feats,
                 pred_hid_feats, 
                 depth=5,
                 n_heads=10,
                 activation=nn.ReLU(),
                 prob_dropout = 0.1,
                 ):
        
        super(nmr_mpnn_PROPOSED, self).__init__()

        self.readout_mode = readout_mode
        
        
        self.gnn = PAGTNGNN(node_in_feats, node_feats, edge_feats,
                              depth, n_heads, activation)
        

        
        
        self.readout_g = Set2Set(input_dim = node_feats + node_in_feats,
                            n_iters = 3,
                            n_layers = 1)
        
        # Input width = per-atom context (node_feats + node_in_feats)
        # + Set2Set graph embedding (2 * (node_feats + node_in_feats)).
        self.readout_n = nn.Sequential(
            nn.Linear(node_feats * 3 + node_in_feats * 3, pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(pred_hid_feats, pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(pred_hid_feats, pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(pred_hid_feats, 1)
        )
        

        self.readout_n_naive = nn.Sequential(
            nn.Linear(node_feats + node_in_feats, pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(pred_hid_feats, pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(pred_hid_feats, pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(pred_hid_feats, 1)
        )       


    def forward(self, g, n_nodes, masks):
        """Encode the batched graph with PAGTN and predict shifts.

        Parameters
        ----------
        g : DGLGraph
            Batched molecule graph with ``node_attr`` / ``edge_attr`` features.
        n_nodes : torch.LongTensor
            Number of nodes of each molecule, used to broadcast the graph
            embedding back to its atoms.
        masks : torch.BoolTensor
            Per-node flag; only True nodes are predicted and returned.

        Returns
        -------
        torch.Tensor
            One predicted shift per masked node, shape ``(n_masked,)``.
        """
        
        def embed(g):
            """Concatenate the PAGTN output with the raw atom descriptors."""
            
            node_feats = g.ndata['node_attr']
            edge_feats = g.edata['edge_attr']
            
            
            node_feats_embedding = self.gnn(g, node_feats, edge_feats)
            node_feats = torch.cat([node_feats_embedding, node_feats], dim=1)
            
            return node_feats
            
        node_embed_feats = embed(g)
        
        
        # NOTE (known issue, left unchanged): `out` is only assigned for the two
        # recognised modes, so any other readout_mode reaches `out[:,0]` with the
        # variable unbound and raises UnboundLocalError. main.py restricts the
        # CLI choice, direct callers are not protected.
        if self.readout_mode == 'baseline':
            out = self.readout_n_naive(node_embed_feats[masks])
        
        elif self.readout_mode == 'proposed':
            # One graph vector per molecule, expanded back to every node so it
            # can be concatenated with the per-node context.
            graph_embed_feats = self.readout_g(g, node_embed_feats)      
            graph_embed_feats = torch.repeat_interleave(graph_embed_feats, n_nodes, dim = 0)
            out = self.readout_n(torch.hstack([node_embed_feats, graph_embed_feats])[masks])
        
        return out[:,0]



class PAGTNLayer(nn.Module):
    """
    Single PAGTN layer from `Path-Augmented Graph Transformer Network
    <https://arxiv.org/abs/1905.12712>`__
    This will be used for incorporating the information of edge features
    into node features for message passing.
    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    node_out_feats : int
        Size for the output node features.
    edge_feats : int
        Size for the input edge features.
    dropout : float
        The probability for performing dropout. Default to 0.1
    activation : callable
        Activation function to apply. Default to LeakyReLU.
    """
    def __init__(self,
                 node_in_feats,
                 node_out_feats,
                 edge_feats,
                 activation=nn.ReLU()):
        """Build the five attention/message projections of one PAGTN layer."""
        super(PAGTNLayer, self).__init__()
        # Separate projections for attention vs. message: src/dst atom and edge.
        self.attn_src = nn.Linear(node_in_feats, node_in_feats)
        self.attn_dst = nn.Linear(node_in_feats, node_in_feats)
        self.attn_edg = nn.Linear(edge_feats, node_in_feats)
        
        # Scalar attention score per edge (one weight per head).
        self.attn_dot = nn.Linear(node_in_feats, 1)
        
        self.msg_src = nn.Linear(node_in_feats, node_out_feats)
        self.msg_dst = nn.Linear(node_in_feats, node_out_feats)
        self.msg_edg = nn.Linear(edge_feats, node_out_feats)
        self.wgt_n = nn.Linear(node_in_feats, node_out_feats)
        self.act = activation
        

    

    def forward(self, g, node_feats, edge_feats):
        """Update node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        node_feats : float32 tensor of shape (V, node_in_feats) or (V, n_head, node_in_feats)
            Input node features. V for the number of nodes in the batch of graphs.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges in the batch of graphs.
        Returns
        -------
        float32 tensor of shape (V, node_out_feats) or (V, n_head, node_out_feats)
            Updated node features.
        """

        g = g.local_var()
        # In the paper node_src, node_dst, edge feats are concatenated
        # and multiplied with the matrix. We have optimized this step
        # by having three separate matrix multiplication.
        
        g.ndata['src'] = self.attn_src(node_feats)
        g.ndata['dst'] = self.attn_dst(node_feats)
        edg_atn = self.attn_edg(edge_feats).unsqueeze(-2)

        # Additive attention: src + dst (per edge) plus the bond contribution.
        g.apply_edges(fn.u_add_v('src', 'dst', 'e'))
        atn_scores = self.act(g.edata.pop('e') + edg_atn)

        atn_scores = self.attn_dot(atn_scores)
        # edge_softmax normalises per destination atom, i.e. every atom attends
        # over all incoming bonds.
        atn_scores = edge_softmax(g, atn_scores)



        g.ndata['src'] = self.msg_src(node_feats)
        g.ndata['dst'] = self.msg_dst(node_feats)
        # Message on edge (u -> v) = dst atom transform (+ bond transform).
        g.apply_edges(fn.copy_src('dst', 'e'))
        atn_inp = g.edata.pop('e') + self.msg_edg(edge_feats).unsqueeze(-2)
        
        g.edata['msg'] = atn_scores * atn_inp
        # Weighted (attention) sum over incoming messages, plus a skip connection
        # of the node's own features.
        g.update_all(fn.copy_e('msg', 'm'), fn.sum('m', 'feat'))
        out = g.ndata.pop('feat') + self.wgt_n(node_feats)
        
        return out


class PAGTNGNN(nn.Module):
    """Multilayer PAGTN model for updating node representations.
    PAGTN is introduced in `Path-Augmented Graph Transformer Network
    <https://arxiv.org/abs/1905.12712>`__.
    Parameters
    ----------
    node_in_feats : int
        Size for the input node features.
    node_out_feats : int
        Size for the output node features.
    node_hid_feats : int
        Size for the hidden node features.
    edge_feats : int
        Size for the input edge features.
    depth : int
        Number of PAGTN layers to be applied.
    nheads : int
        Number of attention heads.
    dropout : float
        The probability for performing dropout. Default to 0.1
    activation : callable
        Activation function to apply. Default to LeakyReLU.
    """

    def __init__(self,
                 node_in_feats,
                 node_hid_feats,
                 edge_feats,
                 depth=5,
                 nheads=10,
                 activation=nn.ReLU()):
        """Stack ``depth`` PAGTN layers behind a linear multi-head projection."""
        super(PAGTNGNN, self).__init__()
        self.depth = depth
        self.nheads = nheads
        self.node_hid_feats = node_hid_feats

        self.atom_inp = nn.Linear(node_in_feats, node_hid_feats * nheads)
                        
        self.model = nn.ModuleList([PAGTNLayer(node_hid_feats, node_hid_feats,
                                               edge_feats,
                                               activation)
                                    for _ in range(depth)])
        
        self.act = activation

        

    def forward(self, g, node_feats, edge_feats):
        """Update node representations.
        Parameters
        ----------
        g : DGLGraph
            DGLGraph for a batch of graphs
        node_feats : float32 tensor of shape (V, node_in_feats)
            Input node features. V for the number of nodes in the batch of graphs.
        edge_feats : float32 tensor of shape (E, edge_in_feats)
            Input edge features. E for the number of edges in the batch of graphs.
        Returns
        -------
        float32 tensor of shape (V, node_out_feats)
            Updated node features.
        """
        g = g.local_var()
                
        # Split the projected atom features into (n_nodes, n_heads, node_hid_feats)
        # so each head attends independently.
        atom_input = self.atom_inp(node_feats).view(-1, self.nheads, self.node_hid_feats)
        atom_input = self.act(atom_input)

        atom_h = atom_input
        for i in range(self.depth):
            # Residual connection back to the multi-head input after every layer.
            attn_h = self.model[i](g, atom_h, edge_feats)            
            atom_h = torch.nn.functional.relu(attn_h + atom_input)
        
        # Collapse the head dimension by averaging, giving one vector per atom.
        atom_h = atom_h.mean(1)
        
        return atom_h