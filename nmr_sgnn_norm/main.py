
"""Command line entry point for the SGNN NMR shift predictor.

Parses the experiment switches (target nucleus, message-passing / readout
variant, graph representation, fold seed, widths), pre-processes the
nmrshiftdb2 graphs on first use, runs training + test evaluation and appends
one line per run to ``result/result.csv``.

Usage (from inside ``nmr_sgnn_norm``)::

    python main.py --target 13C --message_passing_mode proposed \\
                   --readout_mode proposed --graph_representation sparsified \\
                   --fold_seed 0
"""

import os, csv
import argparse

# NOTE (known issue, left unchanged): `data` is neither a sibling module nor part
# of this repo (only dataset.py exists), so importing this file raises
# ModuleNotFoundError: No module named 'data' before anything runs, even though
# the imported helper is only needed on a cache miss below.
from data import nmrshiftdb2_get_data
from train import train

# Path to the new location in the models/sgnn directory
rel_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
data_path = os.path.join(rel_path, 'models', 'sgnn', 'data')

# Create the data directory if it doesn't exist
os.makedirs(data_path, exist_ok=True)
arg_parser = argparse.ArgumentParser()

# One switch per paper ablation: the two variants of message passing and readout
# can be mixed freely, graph_representation picks sparsified vs fully connected.
arg_parser.add_argument('--target', help ='13C or 1H', choices=['13C', '1H'], default='13C', type = str)
arg_parser.add_argument('--message_passing_mode', help ='proposed or baseline', choices=['proposed', 'baseline'], default='proposed', type = str)
arg_parser.add_argument('--readout_mode', help ='proposed or baseline', choices = ['proposed', 'baseline'], default='proposed', type = str)
arg_parser.add_argument('--graph_representation', help ='sparsified or fully_connected', choices=['sparsified', 'fully_connected'], default='sparsified', type = str)
arg_parser.add_argument('--fold_seed', default=0, type = int)
arg_parser.add_argument('--node_embedding_dim', default=256, type=int)
arg_parser.add_argument('--node_hidden_dim', default=256, type=int)
arg_parser.add_argument('--readout_n_hidden_dim', default=256, type=int)
arg_parser.add_argument('--memo', default=None, type=int)



args = arg_parser.parse_args()

print('-- CONFIGURATIONS')
print(f'--- current mode: target: {args.target} message_passing_mode: {args.message_passing_mode}, readout_mode: {args.readout_mode}, graph_representation: {args.graph_representation}, fold_seed: {args.fold_seed}')

data_filename = os.path.join(data_path, f'nmrshiftdb2_graph_{args.graph_representation}_{args.target}.npz')

# Build the graph cache lazily: the .npz is expensive to create and is reused by
# every fold / ablation of the same (graph_representation, target) pair.
if not os.path.isfile(data_filename):
    print('no data found, preprocessing...')
    nmrshiftdb2_get_data.preprocess(args)
    
model, test_mae, test_rmse, time_per_mol = train(args)

# save result to csv
# NOTE: './result' is resolved against the *current working directory*, not the
# repo, so results only land in nmr_sgnn_norm/result/ when main.py is started
# from that folder.
if not os.path.exists('./result'):
        os.mkdir('./result')

# Write the header only for the very first run so later runs append cleanly.
if not os.path.isfile(f'result/result.csv'):
    with open(f'result/result.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['message_passing_mode', 'readout_mode', 'graph_representation', 'target', 'fold_seed', 'test_mae', 'test_rmse', 'time_per_mol'])

with open(f'result/result.csv', 'a', newline='') as f:
    w = csv.writer(f)
    w.writerow([args.message_passing_mode, args.readout_mode, args.graph_representation, args.target, args.fold_seed, test_mae, test_rmse, time_per_mol])

