"""Baseline (NNConv) message-passing network for per-atom NMR shift prediction.

Message passing is a plain MPNN: edge features are mapped to a node_feats x
node_feats matrix and used by ``NNConv`` to produce one message per edge; a GRU
cell then mixes that message with the previous node state, so the network keeps
a residual-like memory of every hop. Node states from all hops are concatenated
and fed to the chosen readout, which is why a shift is predicted per node.

Paired with the ``readout_mode`` switch this covers the two "SG-" ablations:
``'baseline'`` = per-atom readout only, ``'proposed'`` = readout that also sees
a Set2Set graph embedding.
"""

import torch
import torch.nn as nn

from dgl.nn.pytorch import NNConv, Set2Set


        
class nmr_mpnn_BASELINE(nn.Module):
    """MPNN that predicts one chemical shift for every (masked) atom.

    Parameters
    ----------
    node_in_feats : int
        Width of the input node (atom) descriptors.
    edge_in_feats : int
        Width of the input bond descriptors.
    readout_mode : str
        ``'baseline'`` (per-atom readout) or ``'proposed'`` (per-atom + graph
        embedding readout).
    node_feats : int
        Width of the hidden node state used inside message passing.
    node_hid_feats : int
        Width of the MLPs that project input nodes / edges.
    pred_hid_feats : int
        Width of the readout MLP.
    num_step_message_passing : int
        Number of NNConv + GRU hops (default 5).
    num_step_set2set, num_layer_set2set : int
        Set2Set readout iterations / layers.
    prob_dropout : float
        Dropout probability inside the readout MLPs.
    """

    def __init__(self, node_in_feats, edge_in_feats, readout_mode, 
                 node_feats, node_hid_feats, pred_hid_feats,
                 num_step_message_passing = 5,
                 num_step_set2set = 3, num_layer_set2set = 1,
                 prob_dropout = 0.1):
        
        super(nmr_mpnn_BASELINE, self).__init__()

        self.readout_mode = readout_mode
        
        # Map the raw atom descriptors to the hidden node-state width used by the
        # message passing loop.
        self.project_node_feats = nn.Sequential(
            nn.Linear(node_in_feats, node_hid_feats), nn.ReLU(),
            nn.Linear(node_hid_feats, node_hid_feats), nn.ReLU(),
            nn.Linear(node_hid_feats, node_hid_feats), nn.ReLU(),
            nn.Linear(node_hid_feats, node_feats), nn.Tanh(),
        )

        
        self.num_step_message_passing = num_step_message_passing
        

        edge_network = nn.Sequential(
            nn.Linear(edge_in_feats, node_hid_feats), nn.ReLU(),
            nn.Linear(node_hid_feats, node_hid_feats), nn.ReLU(),
            nn.Linear(node_hid_feats, node_hid_feats), nn.ReLU(),
            # NNConv needs one full node_feats x node_feats matrix per edge.
            nn.Linear(node_hid_feats, node_feats * node_feats), nn.ReLU()
        )

        
        self.gnn_layer = NNConv(
            in_feats = node_feats,
            out_feats = node_feats,
            edge_func = edge_network,
            aggregator_type = 'sum'
        )
        
        self.gru = nn.GRU(node_feats, node_feats)
        
        
        
        # Set2Set compresses the node states into one fixed-size graph vector;
        # it returns 2 * input_dim values (lstm hidden + cell state).
        self.readout_g = Set2Set(input_dim = node_feats + node_in_feats,
                            n_iters = num_step_set2set,
                            n_layers = num_layer_set2set)
        
        # Input width = concatenated node states of every hop
        # (node_feats * (1 + n_hops)) + the Set2Set graph embedding
        # ((node_feats + node_in_feats) * 2).
        self.readout_n = nn.Sequential(
            nn.Linear(node_feats * (1 + num_step_message_passing) + (node_feats + node_in_feats) * 2, pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(pred_hid_feats, pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(pred_hid_feats, pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(pred_hid_feats, 1)
        )    

        


        # Per-atom readout without the graph embedding: only the concatenated
        # node states of every hop are used.
        self.readout_n_naive = nn.Sequential(
            nn.Linear(node_feats * (1 + num_step_message_passing), pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(pred_hid_feats, pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(pred_hid_feats, pred_hid_feats), nn.PReLU(), nn.Dropout(prob_dropout),
            nn.Linear(pred_hid_feats, 1)
        )                   

    def forward(self, g, n_nodes, masks):
        """Run message passing on the batched graph and predict shifts.

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
            """Return per-hop node states (for the readout) and their raw inputs."""
            
            node_feats_in = g.ndata['node_attr']
            node_feats = self.project_node_feats(node_feats_in)

            edge_feats = g.edata['edge_attr']
            
            # Accumulate the initial projection and every hop's state; the raw
            # descriptors are kept alongside as extra context (node_aggr2).
            node_aggr1 = [node_feats]
            node_aggr2 = [node_feats_in]
            for _ in range(self.num_step_message_passing):
                msg = self.gnn_layer(g, node_feats, edge_feats).unsqueeze(0)
                _, node_feats = self.gru(msg, node_feats.unsqueeze(0))
                node_feats = node_feats.squeeze(0)
                
                node_aggr1.append(node_feats)
            
            node_aggr2.append(node_feats)    
            
            node_aggr1 = torch.cat(node_aggr1, 1)
            node_aggr2 = torch.cat(node_aggr2, 1)
            
            return node_aggr1, node_aggr2

        node_embed_feats, node_embed_feats2 = embed(g)

        # NOTE (known issue, left unchanged): `out` is only assigned for the two
        # recognised modes, so any other readout_mode reaches `out[:,0]` with the
        # variable unbound and raises UnboundLocalError. main.py restricts the
        # CLI choice, direct callers are not protected.
        if self.readout_mode == 'baseline':
            out = self.readout_n_naive(node_embed_feats[masks])


        elif self.readout_mode == 'proposed':
            # One graph vector per molecule, expanded back to every node so it
            # can be concatenated with the per-node states.
            graph_embed_feats = self.readout_g(g, node_embed_feats2)        
            graph_embed_feats = torch.repeat_interleave(graph_embed_feats, n_nodes, dim = 0)

            out = self.readout_n(torch.hstack([node_embed_feats, graph_embed_feats])[masks])
        

        return out[:,0]
