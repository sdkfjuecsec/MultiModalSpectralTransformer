"""Helper functions for the legacy BLIP training branch.

Contents
--------
create_batches / filter_molecules_by_weight / tanimoto_similarity
    Similarity- and molecular-weight-aware batch construction.
WeightSortedBatchSampler / TanimotoBatchSampler
    ``torch.utils.data.Sampler`` subclasses built on those helpers.
create_BLIP_dataloaders
    Builds the train / test / val dataloaders for the BLIP model.
AvgMeter / get_lr
    Small training-loop utilities.
run_training_BLIP / run_BLIP
    Lightning training entry points.

Legacy note: this module is kept under ``_old/``; the current pipeline trains
``TransformerMultiGPU`` from ``models_MMT_v15_4*.py`` instead.
"""

# Standard library imports
import random

# Third-party imports
import rdkit
import torch
import json
from tqdm import tqdm
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem
from torch.utils.data import DataLoader, Sampler
from transformers import RobertaModel, RobertaTokenizer
import pytorch_lightning as pl
from pytorch_lightning.loggers.wandb import WandbLogger

from sklearn.model_selection import train_test_split
from torch.utils.data.distributed import DistributedSampler
import os
import random
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.profiler import SimpleProfiler, AdvancedProfiler
from pytorch_lightning.loggers import WandbLogger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from IPython.display import display, HTML, SVG


# Local module imports
from utils_MMT.dataloaders_pl_v15_4 import collate_fn, MultimodalData
#from utils.models_MMT_v14_1 import MultimodalTransformer
#from utils.models_CLIP_v15_3 import CLIPModel, ChembertaFingerprint, CLIPMultiGPU
from utils_MMT.models_BLIP_v15_4 import BLIPMultiGPU, ChembertaFingerprint, BLIPMultiGPU

import utils_MMT.data_generation_v15_4 as dg

def create_batches(dataset, similarity_threshold, batch_size, max_search_size, weight_delta):
    """
    Create batches of molecules from a dataset based on the Tanimoto similarity and molecular weight.
    
    Args:
        dataset (pd.DataFrame): Dataset containing the molecular information.
        similarity_threshold (float): Minimum Tanimoto similarity to be considered similar.
        batch_size (int): The desired batch size.
        max_search_size (int): The maximum number of molecules to search for each anchor molecule.
        weight_delta (float): The molecular weight range allowed for molecules in the same batch.
        
    Returns:
        batches (list): A list of lists containing indices of molecules in each batch.
    """
    # NOTE (known issue, left unchanged): ``Chem.MolFromSmiles`` returns ``None``
    # for an unparseable SMILES string, and ``CalcExactMolWt`` / the Morgan
    # fingerprint below then raise on ``None``, aborting the whole batching
    # pass. No validity check is performed here.
    molecules = [Chem.MolFromSmiles(smi) for smi in tqdm(dataset.ref_data["SMILES"])]
    available_indices = set(range(len(molecules)))
    batches = []

    while available_indices:
        current_batch = []
        similar_indices = []

        # Choose a new anchor molecule
        anchor_index = available_indices.pop()
        current_batch.append(anchor_index)
        anchor_mol = molecules[anchor_index]

        # Get the molecular weight range based on the anchor molecule
        anchor_weight = rdkit.Chem.rdMolDescriptors.CalcExactMolWt(anchor_mol)
        min_weight, max_weight = anchor_weight - weight_delta, anchor_weight + weight_delta
        weight_filtered_indices = filter_molecules_by_weight(molecules, min_weight, max_weight)
        # Filter out the unavailable indices
        search_indices = list(set(weight_filtered_indices).intersection(available_indices))
        random.shuffle(search_indices)

        searched_molecules = 0

        for i in tqdm(search_indices):
            if len(current_batch) >= batch_size:
                break
            if tanimoto_similarity(anchor_mol, molecules[i]) >= similarity_threshold:
                current_batch.append(i)
                available_indices.discard(i)

            searched_molecules += 1

            # Stop searching if max_search_size is reached
            if searched_molecules >= max_search_size:
                break

        # Add the remaining spots in the batch with molecules within the weight range
        if len(current_batch) < batch_size:
            for i in search_indices:
                if len(current_batch) >= batch_size:
                    break
                current_batch.append(i)
                available_indices.discard(i)

        # Add the current batch to the list of batches
        batches.append(current_batch)

    return batches

def filter_molecules_by_weight(molecules, min_weight, max_weight):
    """
    Filter molecules based on their molecular weight.
    
    Args:
        molecules (list): A list of RDKit Mol objects.
        min_weight (float): The minimum molecular weight to be included in the filtered list.
        max_weight (float): The maximum molecular weight to be included in the filtered list.
        
    Returns:
        filtered_indices (list): A list of indices corresponding to the molecules with molecular weights within the specified range.
    """
    filtered_indices = []

    for i, mol in enumerate(molecules):
        mol_weight = rdkit.Chem.rdMolDescriptors.CalcExactMolWt(mol)
        if min_weight <= mol_weight <= max_weight:
            filtered_indices.append(i)

    return filtered_indices

# class WeightSortedBatchSampler(Sampler):
#     def __init__(self, dataset, batch_size, drop_last, window_size=None):
#         self.dataset = dataset
#         self.batch_size = batch_size
#         self.drop_last = drop_last
#         self.window_size = window_size if window_size is not None else max(batch_size, 4)

#         # Sort the dataset by molecular weight
#         self.sorted_indices = self.dataset.ref_data.sort_values("MW").index.tolist()
        
#     def __iter__(self):
#         batch = []
#         window = []
#         for idx in self.sorted_indices:
#             window.append(idx)
#             if len(window) == self.window_size:
#                 random.shuffle(window)
#                 batch.extend(window)
#                 window = []

#                 if len(batch) == self.batch_size:
#                     yield batch
#                     batch = []

#         if len(window) > 0:
#             random.shuffle(window)
#             batch.extend(window)

#         if not self.drop_last and len(batch) > 0:
#             yield batch

#     def __len__(self):
#         if self.drop_last:
#             return len(self.sorted_indices) // self.batch_size
#         else:
#             return (len(self.sorted_indices) + self.batch_size - 1) // self.batch_size

#### Works as well ####
class WeightSortedBatchSampler(Sampler):
    """Yield batches from a molecular-weight-sorted, window-shuffled index list.

    Sorting by weight keeps the molecules of a batch similar in size (less
    padding), while shuffling inside windows restores enough randomness.

    Args:
        dataset: ``MultimodalData`` exposing ``ref_data`` with an ``MW`` column.
        batch_size (int): Number of samples per batch.
        drop_last (bool): Drop the trailing partial batch when True.
        window_size (int): Number of consecutive indices shuffled together;
            defaults to ``max(batch_size, 4)``.
    """

    def __init__(self, dataset, batch_size, drop_last, window_size=None):
        """Store the configuration and precompute the weight-sorted indices."""
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.window_size = window_size if window_size is not None else max(batch_size, 4)
        self.sorted_indices = self.dataset.ref_data.sort_values("MW").index.tolist()
    
    def __iter__(self):
        """Yield lists of indices, one list per batch."""
        # Create windows of indices, where each window is `self.window_size` long
        windows = [self.sorted_indices[i:i + self.window_size] for i in range(0, len(self.sorted_indices), self.window_size)]
        
        # Shuffle each window
        for window in windows:
            random.shuffle(window)

        # Now iterate through the shuffled windows to yield batches
        batch = []
        for window in windows:
            for idx in window:
                batch.append(idx)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []

        # Handle the last batch
        if not self.drop_last and len(batch) > 0:
            yield batch

    def __len__(self):
        """Number of batches produced by ``__iter__``."""
        if self.drop_last:
            return len(self.sorted_indices) // self.batch_size
        else:
            return (len(self.sorted_indices) + self.batch_size - 1) // self.batch_size


class TanimotoBatchSampler(Sampler):
    """
    A custom batch sampler for RDKit Mol datasets using Tanimoto similarity and molecular weight for creating batches.
    
    Args:
        dataset (Dataset): A dataset containing RDKit Mol objects.
        batch_size (int): The number of samples per batch.
        drop_last (bool): Whether to drop the last incomplete batch.
        similarity_threshold (float): The minimum Tanimoto similarity threshold for a molecule to be included in the same batch as the anchor molecule.
        max_search_size (int): The maximum number of molecules to search for similarity within a batch.
        weight_delta (float): The range of molecular weights to search for similar molecules around the anchor molecule.
        
    Attributes:
        dataset (Dataset): The dataset containing RDKit Mol objects.
        batch_size (int): The number of samples per batch.
        drop_last (bool): Whether to drop the last incomplete batch.
        batches (list): A list of lists, where each inner list contains indices corresponding to a batch of molecules.

    Note: no call site in this repository instantiates this sampler; the BLIP
    dataloaders use ``WeightSortedBatchSampler`` instead.
    """

    def __init__(self, dataset, batch_size, drop_last, similarity_threshold, max_search_size, weight_delta):
        """Precompute the Tanimoto-based batches via ``create_batches``."""
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        self.batches = create_batches(self.dataset, similarity_threshold, batch_size, max_search_size, weight_delta)

    def __iter__(self):
        """Yield the precomputed batches, honouring ``drop_last``."""
        for batch in self.batches:
            if not self.drop_last or len(batch) == self.batch_size:
                yield batch

    def __len__(self):
        """Number of batches that will be yielded."""
        if self.drop_last:
            return sum(1 for b in self.batches if len(b) == self.batch_size)
        else:
            return len(self.batches)


def tanimoto_similarity(mol1, mol2, radius=2, num_class=1024):
    """Return the Tanimoto similarity of two Morgan fingerprints.

    Args:
        mol1, mol2: RDKit ``Mol`` objects.
        radius (int): Morgan fingerprint radius.
        num_class (int): Number of fingerprint bits.

    Returns:
        float: Tanimoto similarity in ``[0, 1]``.
    """
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, radius, nBits=num_class)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(mol2, radius, nBits=num_class)
    return DataStructs.TanimotoSimilarity(fp1, fp2)



def create_BLIP_dataloaders(config, stoi, stoi_MF):
    """Create the train / test / val dataloaders for the BLIP model.

    The train and test loaders use ``WeightSortedBatchSampler`` with
    ``config.BLIP_batch_size``; the validation loader deliberately uses
    ``batch_size=1`` (single-sample inference) and no sampler.

    Args:
        config: Experiment configuration (reads ``BLIP_batch_size``).
        stoi (dict): SMILES token -> id mapping.
        stoi_MF (dict): Formula token -> id mapping.

    Returns:
        dict: ``{"train": DataLoader, "test": DataLoader, "val": DataLoader}``.

    Side effects:
        Instantiates three ``MultimodalData`` datasets, which may cache /
        preprocess data on disk. All loaders use ``num_workers=0``.
    """
    #NUM_WORKERS = os.cpu_count()
    batch_size = config.BLIP_batch_size

    train_dataset = MultimodalData(config, 
                                   stoi, 
                                   stoi_MF, 
                                   mode="train")
    
    train_sampler = WeightSortedBatchSampler(train_dataset, 
                                             batch_size, 
                                             drop_last=False)

    test_dataset = MultimodalData(config, 
                                  stoi, 
                                  stoi_MF, 
                                  mode="test")

    test_sampler = WeightSortedBatchSampler(test_dataset, 
                                            batch_size, 
                                            drop_last=False)
    
    val_dataset = MultimodalData(config, 
                                stoi, 
                                stoi_MF, 
                                mode="val")

    #val_sampler = WeightSortedBatchSampler(val_dataset, 
    #                                        1, 
    #                                        drop_last=False)
        
    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, 
                                #batch_size=config.BLIP_batch_size,  # because will run multimodal in parallel on many samples
                                  batch_sampler=train_sampler,
                                  #shuffle=True, 
                                  collate_fn=collate_fn, 
                                  #num_workers=config.num_workers, 
                                  num_workers=0,#config.num_workers, 
                                  drop_last=False)
    
    test_dataloader = DataLoader(test_dataset, 
                               # batch_size=config.BLIP_batch_size,  # because will run multimodal in parallel on many samples
                                  batch_sampler=test_sampler,
                                  #shuffle=False, 
                                  collate_fn=collate_fn, 
                                  num_workers=0,#config.num_workers, 
                                  #num_workers=config.num_workers, 
                                  drop_last=False)

    val_dataloader = DataLoader(val_dataset, 
                                batch_size=1,  # because will run multimodal in parallel on many samples
                                  shuffle=False, 
                                  collate_fn=collate_fn, 
                                  num_workers=0,#config.num_workers, 
                                  drop_last=False)

    dataloaders = {"train":train_dataloader, "test":test_dataloader, "val":val_dataloader} #, "all":all_dataloader}
    return dataloaders

class AvgMeter:
    """Running-average accumulator used to report a scalar metric.

    Args:
        name (str): Label printed by ``__repr__``.
    """

    def __init__(self, name="Metric"):
        """Set the label and reset the counters."""
        self.name = name
        self.reset()

    def reset(self):
        """Reset ``avg``, ``sum`` and ``count`` to zero."""
        self.avg, self.sum, self.count = [0] * 3

    def update(self, val, count=1):
        """Add ``val * count`` to the running sum and refresh the average."""
        self.count += count
        self.sum += val * count
        self.avg = self.sum / self.count

    def __repr__(self):
        """Return ``"<name>: <avg>"`` with four decimals."""
        text = f"{self.name}: {self.avg:.4f}"
        return text

def get_lr(optimizer):
    """Return the learning rate of the first parameter group (or ``None`` if empty)."""
    for param_group in optimizer.param_groups:
        return param_group["lr"]




def run_training_BLIP(train_dataloader, test_dataloader, config):
    """Train ``BLIPMultiGPU`` with PyTorch Lightning and save checkpoints.

    Args:
        train_dataloader: Training loader (``batch_sampler`` based).
        test_dataloader: Loader passed to Lightning as the validation loader.
        config: Experiment configuration; ``BLIP_model_save_dir``,
            ``project``, ``gpu_num``, ``BLIP_NUM_EPOCHS``,
            ``BLIP_continue_training`` and ``BLIP_model_path`` are read.

    Returns:
        None. The trained checkpoint paths are written under
        ``config.BLIP_model_save_dir``.

    Side effects:
        Creates the output directory, dumps ``config`` (as JSON), starts a
        Weights & Biases run, trains, and on error writes
        ``last_backup_checkpoint.ckpt``.

    NOTE (known issue, left unchanged): ``ModelCheckpoint`` is constructed with
    the removed ``filepath=`` keyword (newer Lightning versions only accept
    ``dirpath`` / ``filename``), so this line raises ``TypeError`` before
    training starts.

    NOTE (known issue, left unchanged): if ``pl.Trainer(...)`` itself raises,
    the ``except`` block calls ``trainer.save_checkpoint`` although ``trainer``
    was never assigned, so a ``NameError``/``UnboundLocalError`` replaces the
    original exception instead of the "Model saved." recovery path.
    """
    if not os.path.exists(config.BLIP_model_save_dir):
        os.mkdir(config.BLIP_model_save_dir)

    # convert the Namespace to dictionary
    config_dict = vars(config)
    # assuming the CLIP_model_save_dir is a list with one element
    config_save_path = config.BLIP_model_save_dir
    # save the dictionary as a JSON file
    random_num = random.randint(0,10000)
    with open(f"{config_save_path}/config_{str(random_num)}.json", 'w') as f:
        json.dump(config_dict, f, indent=4) 

    filepath= config.BLIP_model_save_dir
    checkpoint_callback = ModelCheckpoint(
        monitor='loss',
        mode='min',
        filepath=os.path.join(filepath,'model_BLIP-{epoch:02d}-{loss:.2f}'),
        save_top_k=-1,  # Keeps all checkpoints.
    )

    # If i use that the saving doesn't work well
    profiler = SimpleProfiler()

    wandb_logger = WandbLogger(project=config.project, 
                               log_model='all') # log all new checkpoints during training
    BLIP_multi_gpu_model = BLIPMultiGPU(config)

    if config.BLIP_continue_training:
        checkpoint_path = config.BLIP_model_path
        BLIP_multi_gpu_model = BLIP_multi_gpu_model.load_from_checkpoint(config=config, checkpoint_path=checkpoint_path)

    config_dict = vars(config)
    wandb_logger.log_hyperparams(config_dict)

    try:
        trainer = pl.Trainer(profiler=profiler,
                             gpus=config.gpu_num, 
                             progress_bar_refresh_rate=10, 
                             accelerator='ddp', 
                             logger=wandb_logger,
                             checkpoint_callback=checkpoint_callback,
                             max_epochs=config.BLIP_NUM_EPOCHS,
                             #fast_dev_run=True,
                             #early_stop_callback=early_stopping,
                             #limit_train_batches=1,
                             #limit_val_batches=1
                             )
        trainer.fit(BLIP_multi_gpu_model, train_dataloader, test_dataloader)
    except Exception as e:
        print(f"Error occurred: {e}")
        backup_ckpt_path = config.BLIP_model_save_dir+"/last_backup_checkpoint.ckpt"
        trainer.save_checkpoint(backup_ckpt_path)
        print("Model saved.")

def run_BLIP(config, stoi, stoi_MF):
    """Build the BLIP dataloaders and launch training.

    Args:
        config: Experiment configuration.
        stoi (dict): SMILES token -> id mapping.
        stoi_MF (dict): Formula token -> id mapping.

    Returns:
        None. Delegates to ``run_training_BLIP`` with the train / test loaders.
    """
    dataloaders = create_BLIP_dataloaders(config, stoi, stoi_MF)

    #train_dataloader, test_dataloader = load_dataloaders(config, stoi, stoi_MF)
    run_training_BLIP(dataloaders["train"], dataloaders["test"], config)
   
    
########### Decide if I want to use  dg.create_CLIP_dataloaders or from here ###########