"""Training entry points for the Multimodal Spectral Transformer (MMST).

This module is the thin "glue" layer used by the experiment notebooks to turn a
parsed ``argparse.Namespace`` into (a) train/test dataloaders and (b) a PyTorch
Lightning training run of :class:`~utils_MMT.models_MMT_v15_4.MultimodalTransformer`.

Layout
------
``load_dataloaders(config, stoi, stoi_MF)``
    Build the train/test :class:`~utils_MMT.dataloaders_pl_v15_4.MultimodalData`
    datasets and wrap them in ``DataLoader`` objects.
``run_training_MMT(train_dataloader, test_dataloader, config)``
    Persist the config, build callbacks/logger/model and call ``Trainer.fit``.
``run_MMT(config, stoi, stoi_MF)``
    Convenience wrapper that chains the two steps above.

Notes on runtime requirements
-----------------------------
Everything here needs a GPU (``accelerator='ddp'`` + ``gpus=config.gpu_num``),
W&B credentials and the real checkpoint/token dictionaries. Therefore this file
is *annotated statically only* in this repository; it is never executed as part
of the annotation work.

NOTE (known issue, left unchanged)
    导入区存在重复/迟到的导入：``random`` 出现两次（第 5、35 行），
    ``WandbLogger`` 出现两次（第 30、37 行），``os`` 与
    ``pytorch_lightning as pl`` 也放在自定义 import 之后（第 34、36 行）。
    这不影响运行，只是 PEP 8 风格问题；保持原样以免改变 import 顺序语义。
"""

# Standard library imports
import ast
import json
import pickle
import random

import re
import threading
from ast import literal_eval

# Third-party imports
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolDescriptors
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# Custom utility/module imports
# NOTE: ``MultimodalData`` is the dataset used below; ``collate_fn`` performs the
#        per-modality padding/bucketing needed to batch variable-length spectra.
from utils_MMT.dataloaders_pl_v15_4 import MultimodalData, collate_fn
# NOTE: wildcard import. ``load_std_mean`` (only used by the commented-out
#        ``make_pl`` helper) comes from here; a wildcard keeps the historical
#        behaviour, e.g. names such as ``np``/``Chem`` may also be re-bound.
from utils_MMT.nmr_calculation_from_dft_v15_4 import *
from utils_MMT.smi_augmenter_v15_4 import SMILESAugmenter
# ``TransformerMultiGPU`` wraps the model so that it can be trained with DDP;
# ``MultimodalTransformer`` is the raw nn.Module (imported for type users).
from utils_MMT.models_MMT_v15_4 import MultimodalTransformer, TransformerMultiGPU


from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from torch.utils.data.distributed import DistributedSampler
import os
import random
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger
from torch.nn import functional as F


# Makes ``Series.progress_apply`` available (``df.progress_apply``) used in the
# data-generation helpers of other modules; harmless here.
tqdm.pandas()

# ---------------------------------------------------------------------------
# Legacy / disabled helper: ``make_pl``
# ---------------------------------------------------------------------------
# NOTE: the whole function is commented out in the source history. It used to
#       assemble the model + loss functions + token dictionaries + SGNN
#       mean/std normalisation constants in one place. It is kept verbatim as
#       documentation of the original wiring. Deleted code, kept for reference:
#
# def make_pl(config):
#
#     ### Load dictionaries:
#     # Load from file
#     with open(config.itos_path, 'r') as f:
#         itos = json.load(f)
#
#     with open(config.stoi_path, 'r') as f:
#         stoi = json.load(f)
#
#     with open(config.itos_MF_path, 'r') as f:
#         itos_MF = json.load(f)
#
#     with open(config.stoi_MF_path, 'r') as f:
#         stoi_MF = json.load(f)
#
#     # Load Model
#     model = MultimodalTransformer(config, src_pad_idx=0)
#
#     # Define a loss function
#     SMI_loss_fn = torch.nn.CrossEntropyLoss().to(config.device)
#     #smiles_loss_fn = nn.NLLLoss(reduction="none", ignore_index=0)
#     MF_loss_fn = torch.nn.CrossEntropyLoss().to(config.device)       #not used yet
#     MW_loss_fn = torch.nn.MSELoss().to(config.device)
#     FP_loss_fn = torch.nn.BCEWithLogitsLoss().to(config.device)
#
#     #Load SGNN mean values
#     graph_representation = "sparsified"
#     target = "13C"
#     #train_y_mean_C, train_y_std_C, train_y_mean_H, train_y_std_H = None, None, None, None
#     train_y_mean_C, train_y_std_C = load_std_mean(target,graph_representation)
#     target = "1H"
#     train_y_mean_H, train_y_std_H = load_std_mean(target,graph_representation)
#     sgnn_means_stds = (train_y_mean_C, train_y_std_C, train_y_mean_H, train_y_std_H)
#
#     return model, SMI_loss_fn, MF_loss_fn, MW_loss_fn, FP_loss_fn, itos, stoi, itos_MF, stoi_MF, sgnn_means_stds


def load_dataloaders(config, stoi, stoi_MF):
    """Build the train and test dataloaders used by :func:`run_training_MMT`.

    Two ``MultimodalData`` instances are created with the same config/token
    dictionaries but different ``mode``; the split is therefore *inside*
    ``MultimodalData`` (it reads ``config.train_test_split_ratio`` /
    ``config.test_set_ratio`` and the smiles file paths), not here.

    Args:
        config: parsed experiment config (``argparse.Namespace``). Required
            attributes used here: ``batch_size``, ``num_workers``. The dataset
            itself reads many more (file paths, modalities, max lengths, ...).
        stoi: character-to-index dict for SMILES strings.
        stoi_MF: character-to-index dict for molecular formulas.

    Returns:
        Tuple[List[DataLoader], DataLoader]:
            ``(train_dataloader, test_dataloader)``. The train loader shuffles
            each epoch; the test loader keeps a deterministic order.
            ``drop_last=False`` because the last incomplete batch is still
            wanted (it contains distinct molecules, not padding artefacts).

    Note:
        ``num_workers`` is taken verbatim from the config. On Windows / when
        the dataset holds non-picklable objects this must stay ``0``; the
        canonical config value in ``1.0_Experiment_Notebook___.ipynb`` is
        ``[16]`` for the Linux cluster.
    """

    train_dataset = MultimodalData(config, stoi, stoi_MF, mode="train")    
    test_dataset = MultimodalData(config, stoi, stoi_MF, mode="test")

    # ``collate_fn`` does the modality-specific padding (see
    # ``utils_MMT/dataloaders_pl_v15_4.py``); without it PyTorch's default
    # collate would fail on ragged spectral arrays.
    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=config.num_workers, drop_last=False)
    test_dataloader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=config.num_workers, drop_last=False)
    return train_dataloader, test_dataloader



def run_training_MMT(train_dataloader, test_dataloader, config):
    """Set up callbacks/logger/model and run ``pl.Trainer.fit``.

    Side effects:
        * creates ``config.model_save_dir`` if missing (single level only);
        * writes ``<model_save_dir>/config_<random>.json`` with the full config
          (a random suffix de-duplicates repeated runs in the same directory);
        * creates one checkpoint per epoch and logs them all to W&B;
        * when ``config.load_model`` is true, restores weights from
          ``config.checkpoint_path``;
        * when ``config.use_real_data`` is true, freezes the whole network and
          unfreezes only the ``real_data_linear`` head (a probe/linear-probe run).

    Args:
        train_dataloader: output of :func:`load_dataloaders` (train part).
        test_dataloader: output of :func:`load_dataloaders` (test/val part).
            It is passed as ``val_dataloaders`` to ``Trainer.fit``.
        config: parsed ``argparse.Namespace``.

    Returns:
        None. The trained weights are only on disk / in W&B.

    NOTE (known issue, left unchanged)
        1. ``os.mkdir`` (not ``os.makedirs``) raises ``FileNotFoundError`` if
           the parent of ``model_save_dir`` does not exist. The config in the
           notebook uses a nested path (``.../all_models/...``), so the parent
           directory must already exist.
        2. The ``except`` block calls ``trainer.save_checkpoint``. If the
           failure happened while *constructing* ``pl.Trainer(...)`` (e.g.
           missing GPUs or a PL version that dropped ``gpus=``), ``trainer``
           is unbound and the except block raises ``UnboundLocalError``
           instead of saving anything.
        3. ``ModelCheckpoint(filepath=..., period=...)`` is the PyTorch
           Lightning 0.9 / 1.0 API. On PL >= 1.5 it must be
           ``dirpath=..., filename=...``; likewise ``gpus=`` and
           ``progress_bar_refresh_rate=`` were removed. This pins the repo to
           an old PL version.
    """
    # NOTE: single-level mkdir, see note 1 above.
    if not os.path.exists(config.model_save_dir):
        os.mkdir(config.model_save_dir)

    # convert the Namespace to dictionary
    config_dict = vars(config)
    # assuming the model_save_dir is a list with one element
    # (it is, in the notebook config, but it has already been unwrapped to a
    # plain string by ``parse_arguments`` before reaching this function)
    config_save_path = config.model_save_dir
    # save the dictionary as a JSON file
    # NOTE: json.dump on raw ``vars(Namespace)`` will fail on non-serialisable
    #       values (Path objects, torch devices, ...). The MMST config only
    #       holds JSON-friendly Python builtins, so it works in practice.
    random_num = random.randint(0,10000)
    with open(f"{config_save_path}/config_{str(random_num)}.json", 'w') as f:
        json.dump(config_dict, f, indent=4) 

    filepath= config.model_save_dir
    # ``save_top_k=-1`` => keep *every* epoch checkpoint (needed because the
    # downstream greedy/IBM search picks whichever epoch scored best).
    # ``monitor='loss'`` + ``mode='min'`` => the training loss, not val loss.
    checkpoint_callback = ModelCheckpoint(
        monitor='loss',
        mode='min',
        filepath=os.path.join(filepath,'model-{epoch:02d}-{loss:.4f}'),
        save_top_k=-1,  # Keeps all checkpoints.
    )

    # If i use that the saving doesn't work well
    # (i.e. ``AdvancedProfiler`` breaks checkpoint saving in this PL version)
    profiler = SimpleProfiler()

    wandb_logger = WandbLogger(project=config.project, 
                               log_model='all') # log all new checkpoints during training
    # ``TransformerMultiGPU`` is a ``pl.LightningModule``; it owns the loss
    # computation and the optimiser/scheduler for DDP.
    transformer_multi_gpu_model = TransformerMultiGPU(config)

    if config.load_model:
        checkpoint_path = config.checkpoint_path
        # NOTE: keyword ``checkpoint_path=`` (instead of the usual
        #       ``ckpt_path=``/positional argument) is accepted by the (older)
        #       PL version pinned here; modern PL would treat it as an
        #       unknown kwarg passed to ``__init__``.
        transformer_multi_gpu_model = transformer_multi_gpu_model.load_from_checkpoint(config=config, checkpoint_path=checkpoint_path)
    if config.use_real_data:
        # Linear-probe mode: everything frozen, only the "real data" head learns.
        # Freeze all parameters in the model
        for param in transformer_multi_gpu_model.parameters():
            param.requires_grad = False
        # Unfreeze parameters in self.real_data_linear
        # Check if the real_data_linear layer exists in the model's 'model' attribute
        if hasattr(transformer_multi_gpu_model.model, 'real_data_linear'):
            print("has layer")
            for param in transformer_multi_gpu_model.model.real_data_linear.parameters():
                param.requires_grad = True
        # NOTE: if the attribute is missing this silently trains *nothing*
        #       (all grads are None) and the run fails later or no-ops; there
        #       is no error message in that branch.

    config_dict = vars(config)
    wandb_logger.log_hyperparams(config_dict)
    
    # NOTE: ``trainer`` is only bound inside the ``try``; see note 2 above.
    try:
        trainer = pl.Trainer(profiler=profiler,
                             gpus=config.gpu_num, 
                             progress_bar_refresh_rate=10, 
                             accelerator='ddp', 
                             logger=wandb_logger,
                             checkpoint_callback=checkpoint_callback,
                             #callbacks=[checkpoint_callback, batch_checkpoint_callback],  # Add the batch_checkpoint_callback here
                             max_epochs=config.num_epochs,
                             #fast_dev_run=True,
                             #early_stop_callback=early_stopping,
                             #limit_train_batches=1,
                             #limit_val_batches=1
                             )

        # The second positional argument is the *validation* dataloader, i.e.
        # ``test_dataloader`` is used to compute the monitored metrics.
        trainer.fit(transformer_multi_gpu_model, train_dataloader, test_dataloader)
    except Exception as e:
        print(f"Error occurred: {e}")
        # Fallback: keep whatever state exists so a crashed run is not lost.
        backup_ckpt_path = config.model_save_dir+"/last_backup_checkpoint.ckpt"
        trainer.save_checkpoint(backup_ckpt_path)
        print("Model saved.")


def run_MMT(config, stoi, stoi_MF):
    """One-call wrapper: build dataloaders, then train.

    Args:
        config: parsed ``argparse.Namespace``.
        stoi: SMILES char -> index.
        stoi_MF: molecular-formula char -> index.

    Returns:
        None (training is a side effect).
    """
    train_dataloader, test_dataloader = load_dataloaders(config, stoi, stoi_MF)
    run_training_MMT(train_dataloader, test_dataloader, config)
