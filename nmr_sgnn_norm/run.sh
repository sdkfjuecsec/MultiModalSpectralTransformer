#!/bin/bash
#
# Experiment sweep for the SGNN NMR shift predictor.
#
# Purpose
#   Reproduce the paper's four ablations over all 10 folds and both nuclei:
#     Proposed  : PAGTN message passing + graph-aware (Set2Set) readout
#     SG-IR     : NNConv message passing  + graph-aware readout
#     SG-IMP    : PAGTN message passing   + per-atom readout
#     SG-Only   : NNConv message passing  + per-atom readout
#
# Usage
#   bash run.sh          (run from inside nmr_sgnn_norm/; it writes to ./result/)
#
# The block of commented-out commands at the top is the older variant that
# trained with main.py directly; the live block below evaluates an existing
# checkpoint instead.
#
# NOTE (known issue, left unchanged): the live calls below pass
#   --readout_mode proposed_set2set
# but main.py only accepts the choices 'proposed' and 'baseline', so argparse
# exits with "invalid choice: 'proposed_set2set'". Additionally eval_by_range.py
# and make_all_preds.py are not part of this repository, so every command fails
# with "can't open file" even after the choice error is worked around.


# for fold_seed in {0..2}
# do
#   for target in 13C 1H
#   do
    
#     # Proposed
#     #python main.py --fold_seed $fold_seed --target $target --node_embedding_dim 256 --readout_n_hidden_dim 512 --message_passing_mode proposed --readout_mode proposed_set2set --graph_representation sparsified

#     # SG-IR
#     #python main.py --fold_seed $fold_seed --target $target --node_embedding_dim 128 --node_hidden_dim 256 --readout_n_hidden_dim 512 --message_passing_mode baseline --readout_mode proposed_set2set --graph_representation sparsified

#     #SG-IMP
#     #python main.py --fold_seed $fold_seed --target $target --node_embedding_dim 256 --readout_n_hidden_dim 512 --message_passing_mode proposed --readout_mode baseline --graph_representation sparsified  

#     # #SG-Only
#     # python main.py --fold_seed $fold_seed --target $target --node_embedding_dim 128 --node_hidden_dim 256 --readout_n_hidden_dim 512 --message_passing_mode baseline --readout_mode baseline --graph_representation sparsified

#     # # #FCG
#     #python main.py --fold_seed $fold_seed --target $target --node_embedding_dim 50 --node_hidden_dim 250 --readout_n_hidden_dim 500 --message_passing_mode baseline --readout_mode baseline --graph_representation fully_connected --feat_mode onehot

    
#   done
# done


# --- Per-fold evaluation of the four ablations -----------------------------
# fold_seed selects the held-out tenth of the data (see k_fold_split in train.py).
for fold_seed in {0..9}
do
  for target in 13C 1H
  do
    
    # Proposed
    python eval_by_range.py --fold_seed $fold_seed --target $target --node_embedding_dim 256 --readout_n_hidden_dim 512 --message_passing_mode proposed --readout_mode proposed_set2set --graph_representation sparsified

    # SG-IR
    python eval_by_range.py --fold_seed $fold_seed --target $target --node_embedding_dim 128 --node_hidden_dim 256 --readout_n_hidden_dim 512 --message_passing_mode baseline --readout_mode proposed_set2set --graph_representation sparsified

    #SG-IMP
    python eval_by_range.py --fold_seed $fold_seed --target $target --node_embedding_dim 256 --readout_n_hidden_dim 512 --message_passing_mode proposed --readout_mode baseline --graph_representation sparsified  

    # #SG-Only
    python eval_by_range.py --fold_seed $fold_seed --target $target --node_embedding_dim 128 --node_hidden_dim 256 --readout_n_hidden_dim 512 --message_passing_mode baseline --readout_mode baseline --graph_representation sparsified

    # # #FCG
    #python main.py --fold_seed $fold_seed --target $target --node_embedding_dim 50 --node_hidden_dim 250 --readout_n_hidden_dim 500 --message_passing_mode baseline --readout_mode baseline --graph_representation fully_connected --feat_mode onehot

    
  done
done


# --- Dump per-atom predictions for the proposed model ---------------------
# Only the fully proposed variant is exported; the predictions are consumed
# downstream by the analysis notebooks.
for fold_seed in {0..9}
do
  for target in 13C 1H
  do
    
    # Proposed
    python make_all_preds.py --fold_seed $fold_seed --target $target --node_embedding_dim 256 --readout_n_hidden_dim 512 --message_passing_mode proposed --readout_mode proposed_set2set --graph_representation sparsified

    
  done
done