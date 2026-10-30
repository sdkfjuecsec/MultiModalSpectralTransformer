"""Batch IR spectrum simulation via the ``chemprop-IR`` predictor.

This module is a thin wrapper around the vendored ``chemprop-IR`` package
(``<repo>/chemprop-IR``). For every molecule in a CSV it predicts an IR
spectrum with a pretrained message-passing model and writes one small CSV per
sample so that the downstream MMST pipeline can read spectra by ``sample-id``.

Pipeline position
-----------------
``3.0_Chemprop_IR_Data_Generation__.ipynb`` builds the ``args``/``config``
consumed here, runs :func:`run_IR_simulation`, and then ``1.0`` reads the
resulting ``<sample-id>.csv`` files.

Runtime requirements
--------------------
Needs the ``chemprop-IR`` sub-module, its pretrained checkpoint and a
(optionally GPU) torch install. Annotated statically only; never executed here.

NOTE (known issue, left unchanged)
    导入区重复：``os``（第 2、22 行）、``pandas``（第 21、23 行）、
    ``Namespace``（第 14、32 行）各出现两次；``roundrobin_sid`` /
    ``get_available_features_generators`` / ``csv`` / ``np`` / ``torch`` /
    ``tqdm`` / ``Chem`` 在本文件中均未使用。只是冗余，不影响行为。
"""

# ---------------------------------------------------------------------------
# Make the vendored ``chemprop-IR`` importable.
# ---------------------------------------------------------------------------
# ``__file__`` is ``<repo>/utils_MMT/ir_simulation_v15_4.py``; going up one level
# and into ``chemprop-IR`` resolves to ``<repo>/chemprop-IR``. This sys.path
# hack is required because ``chemprop-IR`` ships no setuptools package and is
# not pip-installed in the cluster environment.
import sys
import os

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Build the path to the chemprop-IR directory relative to the script's location
chemprop_ir_path = os.path.abspath(os.path.join(script_dir, '../chemprop-IR'))

# Add the chemprop-IR directory to sys.path
if chemprop_ir_path not in sys.path:
    sys.path.append(chemprop_ir_path)
    
from argparse import Namespace
import csv
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
import os
import pandas as pd
from rdkit import Chem

# ``predict`` is the low-level single-model entry point; ``make_predictions``
# (imported further down) is the higher-level helper actually used here.
from chemprop.train.predict import predict
from chemprop.data import MoleculeDataset
from chemprop.data.utils import get_data, get_data_from_smiles
from chemprop.utils import load_args, load_checkpoint, load_scalers
from chemprop.train.spectral_loss import roundrobin_sid
from chemprop.features import get_available_features_generators
from argparse import Namespace

from chemprop.train import make_predictions
from chemprop.parsing import modify_predict_args

#get_available_features_generators()


def parse_arguments(hyperparameters):
    """Turn a ``{key: [value, ...]}`` mapping into an ``argparse.Namespace``.

    The notebook config stores every hyper-parameter as a **single-element
    list** (an artefact inherited from the original ``argparse``/sweep format),
    so this helper unwraps them with ``val[0]``.

    Args:
        hyperparameters (dict): mapping ``str -> list``.

    Returns:
        argparse.Namespace: one attribute per key.

    NOTE (known issue, left unchanged)
        ``val[0]`` blindly takes the first element. For a *string* value such as
        ``{"project": ["mmst"]}`` you get ``"mmst"`` correctly, but a bare
        string (not wrapped in a list) would yield its first character
        (``"a"`` for ``"abc"``). Also, an empty list raises ``IndexError``.
    """
    parsed_args = {key: val[0] for key, val in hyperparameters.items()}
    return Namespace(**parsed_args)


# Function to generate spectral data from smiles (dummy function for demonstration)
def generate_spectral_data_batch(args, smiles_list):
    """Predict IR spectra for a batch of SMILES.

    Args:
        args (argparse.Namespace): chemprop predict args (must contain
            ``checkpoint_dir``/``checkpoint_paths``, ``spectra_type``,
            ``batch_size`` ... see the notebook's cell 4 comment).
        smiles_list (List[str]): SMILES of the batch.

    Returns:
        Tuple[Any, pandas.DataFrame]:
            ``avg_preds`` is the array of predicted spectra (one row per
            molecule; ``make_predictions`` averages over an ensemble when
            several checkpoints are given). ``predictions_df`` is the
            per-molecule dataframe; **currently unused** by the caller
            (only the averaged predictions are written to disk).

    NOTE (known issue, left unchanged)
        The neighbouring ``TODO`` is stale: the neural prediction *is*
        implemented (``make_predictions``); the comment was left over from an
        older placeholder version.
    """
    avg_preds, predictions_df = make_predictions(args, smiles=smiles_list)
    # TODO: Implement the neural network-based spectral data generation
    return avg_preds, predictions_df


def run_IR_simulation(config, args, mode):
    """Simulate IR spectra for a whole CSV and write one CSV per sample.

    Args:
        config (argparse.Namespace): experiment config. Attributes used:
            ``csv_SMI_targets`` or ``csv_1H_path_SGNN`` (input CSV),
            ``SGNN_gen_folder_path`` (output root), ``ran_num`` (run tag).
        args (argparse.Namespace): chemprop predict args (see
            :func:`parse_arguments`).
        mode (str): ``"target"`` -> read ``config.csv_SMI_targets``;
            ``"1H"`` -> read ``config.csv_1H_path_SGNN``.

    Returns:
        str: path of the directory holding the per-sample spectra CSVs,
            i.e. ``<SGNN_gen_folder_path>/IR_data_<ran_num>``.

    Side effects:
        Writes ``<output_directory>/<sample-id>.csv`` with a single column
        ``spectra`` (the 1-D predicted IR intensity vector).

    NOTE (known issue, left unchanged)
        1. ``mode`` has no ``else`` branch: any other value leaves ``df``
           unbound and raises ``NameError`` on ``len(df)``.
        2. ``parent_folder`` is computed but never used (dead variable).
        3. ``sample_ids[j]`` is assumed to line up 1:1 with
           ``spectral_data_batch`` (true for ``make_predictions``, which keeps
           input order). If a SMILES fails to featurise, chemprop raises rather
           than silently dropping it, so the alignment holds in practice.
    """
    if mode == "target":
        df = pd.read_csv(config.csv_SMI_targets)
    if mode == "1H":
        df = pd.read_csv(config.csv_1H_path_SGNN)

    # Create a directory to save the spectral data files
    folder_path = config.SGNN_gen_folder_path
    # Get the parent folder path
    parent_folder = os.path.dirname(folder_path)
    # One directory per run tag so repeated runs do not overwrite each other.
    output_directory = os.path.join(config.SGNN_gen_folder_path, "IR_data_" + str(config.ran_num))
   
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Cap the batch at 64 molecules (the chemprop default) but never ask for
    # more than what is actually in the CSV.
    if len(df) < 64:
        batch_size = len(df)
    else:
        batch_size = 64

    # Step 2 and 3: Process in batches of 64
    for i in range(0, len(df), batch_size):
        batch = df.iloc[i:i+batch_size]
        smiles_list = batch['SMILES'].tolist()
        sample_ids = batch['sample-id'].tolist()

        # Generate spectral data for the batch
        spectral_data_batch, predictions_df = generate_spectral_data_batch(args, smiles_list)

        # Save spectral data for each sample in the batch
        for j, spectral_data in enumerate(spectral_data_batch):
            #import IPython; IPython.embed();
            # ``sample_ids[j]`` is the MMST identifier; downstream code looks
            # spectra up by this name, so it must match the dataset sample-id.
            output_file_path = os.path.join(output_directory, f"{sample_ids[j]}.csv")
            pd.DataFrame({'spectra': spectral_data}).to_csv(output_file_path, index=False)

    return output_directory
