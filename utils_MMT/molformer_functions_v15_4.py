"""Molformer-based molecule generation helpers.

This module wraps the external ``deep-molecular-optimization`` (Molformer)
code base so that the NMR pipeline can generate structurally related
molecules ("analogues") for a given input SMILES string.

Responsibilities:
    * resolve the path to the vendored Molformer package at import time,
    * define the ``GenerateRunner`` class that loads a trained Molformer
      checkpoint and samples valid, drug-like analogues,
    * provide small RDKit utility helpers (standardisation, Murcko scaffolds,
      Lipinski filtering),
    * expose ``main_molformer(config)`` as the single entry point used by the
      rest of the project.

The module performs import-time side effects (``sys.path`` mutation,
``argparse`` parser construction, global matplotlib/warnings settings).
Importing it therefore requires the vendored package and RDKit dependencies
to be installed.
"""

import argparse
import copy
import glob
import json
import pandas as pd
import pickle as pkl
import random
#sys.path.append('/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer/deep-molecular-optimization')
import os
import sys

# Get the directory of the current script
script_dir = os.path.dirname(__file__)

# Build the path to the deep-molecular-optimization directory relative to the script's location
deep_molecular_optimization_path = os.path.abspath(os.path.join(script_dir, '../deep-molecular-optimization'))

# Add the deep-molecular-optimization directory to sys.path
if deep_molecular_optimization_path not in sys.path:
    sys.path.append(deep_molecular_optimization_path)

import torch
import wandb
import warnings
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import SVG, display
from matplotlib.image import NonUniformImage
from matplotlib.pyplot import figure
from molvs import Standardizer
from rdkit import Chem, DataStructs
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem import AllChem, Descriptors, Draw, Lipinski, MolFromSmiles, MolToSmiles, PandasTools, rdMolDescriptors
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D
from rdkit.Chem.MolStandardize.rdMolStandardize import ChargeParent
from rdkit.Chem.Scaffolds import MurckoScaffold
from torch.utils.data import DataLoader
from tqdm import tqdm

# Custom imports
import configuration.config_default as cfgd
import configuration.opts as opts
import models.dataset as md
import models.seq2seq.model as seq2seq_model
import models.transformer.encode_decode.model as encode_decode_model
import models.transformer.module.decode as decode_module
from models.transformer.encode_decode.model import EncoderDecoder
from models.transformer.module.decode import decode
from models.seq2seq.model import Model


import preprocess.vocabulary as mv
#import utils.train_test_functions_pl_v15_3 as ttf
import utils_MMT.helper_functions_pl_v15_4 as hf
from utils_MMT.dataloaders_pl_v15_4 import MultimodalData, collate_fn

import utils_MF.chem as uc
import utils_MF.log as ul
import utils_MF.plot as up
import utils_MF.torch_util as ut


# Miscellaneous settings
tqdm.pandas()
warnings.filterwarnings('ignore')
IPythonConsole.ipython_useSVG = True

def convert_to_standardized_smiles(smiles):
    """Standardize a SMILES string into the form expected by Molformer.

    The molecule is parsed without sanitisation, reduced to its charge parent,
    then standardised with ``molvs`` and re-serialised without stereo
    information. If explicit hydrogens ``[H]`` remain in the output the
    function recurses until they disappear.

    Args:
        smiles: A (possibly non-canonical) SMILES string.

    Returns:
        A standardised, non-isomeric SMILES string (no explicit ``[H]``).

    Side effects:
        None. Recursion depth is bounded by the number of explicit hydrogens.
    """
    mol = MolFromSmiles(smiles, sanitize=False)
    mol = ChargeParent(mol)
    smol = Standardizer().standardize(mol)
    smi = MolToSmiles(smol, isomericSmiles=False)
    if '[H]' in smi:
        return convert_to_standardized_smiles(smi)
    else:
        return smi
    

def calculate_scaffold(smile):
    """Return the canonical Murcko scaffold SMILES of a molecule.

    Args:
        smile: SMILES string of the full molecule.

    Returns:
        The canonical SMILES of the Murcko scaffold, or an empty string when
        the molecule cannot be parsed or its scaffold is invalid/empty.
    """
    mol = Chem.MolFromSmiles(smile)
    if mol:
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smiles = Chem.MolToSmiles(scaffold)
        except ValueError:
            scaffold_smiles = ""
    else:
        scaffold_smiles = ""
    return scaffold_smiles

    
def follows_lipinski(smiles):
    """Check whether a molecule passes a relaxed Lipinski "rule of five".

    Rules evaluated: H-bond donors <= 5, H-bond acceptors <= 10,
    exact molecular weight < 550 (relaxed from 500), and logP <= 5.
    The molecule is accepted when it violates at most one rule.

    Args:
        smiles: SMILES string to evaluate.

    Returns:
        True when the SMILES parses and has <= 1 violation, otherwise False.
    """
    mol = Chem.MolFromSmiles(smiles)
    if not mol:
        return False

    # Getting Lipinski descriptors
    h_donors = Lipinski.NumHDonors(mol)
    h_acceptors = Lipinski.NumHAcceptors(mol)
    mol_weight = Descriptors.ExactMolWt(mol)
    log_p = Descriptors.MolLogP(mol)

    # Check against rules
    violations = 0
    if h_donors > 5: 
        violations += 1
    if h_acceptors > 10: 
        violations += 1
    if mol_weight >= 550: ### to give it a bit more flexibility to generate molecules close to 500
        violations += 1
    if log_p > 5: 
        violations += 1

    # Check if it adheres to Lipinski's Rule of Five
    return violations <= 1



class GenerateRunner():
    """Load a trained Molformer checkpoint and sample molecule analogues.

    A runner owns the output directory, the loaded vocabulary/tokenizer and
    the logger. ``generate(opt)`` iterates the test CSV, calls ``sample`` for
    every batch and writes the generated SMILES back into a CSV.
    """

    def __init__(self, opt):
        """Initialise output paths, logging, vocabulary and tokenizer.

        Args:
            opt: argparse namespace produced by ``generate_opts``. Must expose
                ``save_directory``, ``test_file_name``, ``epoch`` and
                ``vocab_path``.

        Side effects:
            Creates ``experiments/<save_directory>/<test_file_name>/
            evaluation_<epoch>`` indirectly via the logger file handler and
            sets the module-global ``LOG``.
        """

        self.save_path = os.path.join('experiments', opt.save_directory, opt.test_file_name,
                                      f'evaluation_{opt.epoch}')
        global LOG
        LOG = ul.get_logger(name="generate",
                            log_path=os.path.join(self.save_path, 'generate.log'))
        LOG.info(opt)
        LOG.info("Save directory: {}".format(self.save_path))

        # Load vocabulary
        #with open(os.path.join(opt.data_path, 'vocab.pkl'), "rb") as input_file:
        with open(opt.vocab_path, "rb") as input_file:
            vocab = pkl.load(input_file)

        # Wrap the vocab object with the VocabularyWrapper
        
        self.vocab = vocab
        self.tokenizer = mv.SMILESTokenizer()
        print("__init__")


    def initialize_dataloader(self, opt, vocab, test_file):
        """
        Initialize dataloader that feeds the source molecules into Molformer.

        Args:
            opt: argparse namespace; uses ``data_path``, ``prediction_mode``,
                ``without_property`` and ``batch_size``.
            vocab: SMILES vocabulary used to encode/decode tokens.
            test_file: CSV file stem (without extension) inside ``data_path``.

        Returns:
            A torch ``DataLoader`` over ``models.dataset.Dataset`` yielding
            collated batches with the source token matrix and a metadata
            DataFrame.

        Side effects:
            Reads ``<data_path>/<test_file>.csv`` from disk; prints the
            namespace and a progress marker.
        """
        print(opt)
        # Read test
        data = pd.read_csv(os.path.join(opt.data_path, test_file + '.csv'), sep=",")
        #import IPython; IPython.embed();

        ########################### add prediction mode as opt ###########################
        dataset = md.Dataset(data=data, vocabulary=vocab, tokenizer=self.tokenizer, prediction_mode=opt.prediction_mode, without_property=opt.without_property)
        dataloader = torch.utils.data.DataLoader(dataset, opt.batch_size, shuffle=False, collate_fn=md.Dataset.collate_fn)
        print("initialize_dataloader")
        #import IPython; IPython.embed();

        return dataloader

    def generate(self, opt):
        """Generate analogues for every row of the configured test CSV.

        Loads the model selected by ``opt.model_choice``, decodes
        ``opt.num_generations`` analogues per source molecule and stores them
        as ``Predicted_smi_<n>`` columns next to the original metadata.

        Args:
            opt: argparse namespace produced by ``generate_opts``.

        Returns:
            None.

        Side effects:
            Writes ``generated_molecules.csv`` into ``self.save_path`` and
            logs progress through the module-global ``LOG``.
        """
        print("generate start")
        # import IPython; IPython.embed(); exit(1)

        # set device
        device = ut.allocate_gpu()

        # Data loader
        dataloader_test = self.initialize_dataloader(opt, self.vocab, opt.test_file_name)

        # Load model
        #file_name = os.path.join(opt.model_path, f'model_{opt.epoch}.pt')
        file_name = opt.model_full_path
        #import IPython; IPython.embed();

        if opt.model_choice == 'transformer':
            model = EncoderDecoder.load_from_file(file_name)
            model.to(device)
            model.eval()
        elif opt.model_choice == 'seq2seq':
            model = Model.load_from_file(file_name, evaluation_mode=True)
            # move to GPU
            model.network.encoder.to(device)
            model.network.decoder.to(device)
        max_len = cfgd.DATA_DEFAULT['max_sequence_length']
        df_list = []
        sampled_smiles_list = []
        for j, batch in tqdm(enumerate(ul.progress_bar(dataloader_test, total=len(dataloader_test)))):
            src, source_length, _, src_mask, _, _, df = batch
            # Move to GPU
            src = src.to(device)
            src_mask = src_mask.to(device)
            smiles, scaffold_dict = self.sample(opt.model_choice, model, src, src_mask,
                                                            source_length,
                                                            opt.decode_type,
                                                            opt.without_property,   #########
                                                            num_generations=opt.num_generations,
                                                            delta_weight=opt.delta_weight,
                                                            tanimoto_filter=opt.tanimoto_filter,
                                                            filter_higher=opt.filter_higher,
                                                            max_trials=opt.max_trials,
                                                            max_len=max_len,
                                                            isomericSmiles=opt.isomericSmiles,
                                                            device=device,
                                                            max_scaffold_generations=opt.max_scaffold_generations)

            #print(smiles)
            df_list.append(df)
            sampled_smiles_list.extend(smiles)

        # prepare dataframe
        data_sorted = pd.concat(df_list)
        sampled_smiles_list = np.array(sampled_smiles_list)
        sampled_smiles_list = [list(lst) if isinstance(lst, np.ndarray) else lst for lst in sampled_smiles_list]

        # Rows can hold a different number of analogues, so pad each list to the
        # longest one before transposing. The transposed matrix has one row per
        # generation index and one column per source molecule.
        max_length = max(len(lst) for lst in sampled_smiles_list)
        padded_smiles_list = [lst + [None] * (max_length - len(lst)) for lst in sampled_smiles_list]
        transposed_smiles = list(map(list, zip(*padded_smiles_list)))

        #sampled_smiles_list = sampled_smiles_list.T
        print("sampled_smiles_list")
        for i in range(opt.num_generations):
            try:
                data_sorted['Predicted_smi_{}'.format(i + 1)] = transposed_smiles[i]
            except:
                # NOTE (known issue, left unchanged): if a batch produced fewer
                # rows than num_generations (transposed_smiles is shorter than
                # opt.num_generations), this raises IndexError; the bare except
                # swallows it and the remaining Predicted_smi_* columns are
                # silently omitted from the output CSV.
                print("fail")
                continue

        
        result_path = os.path.join(self.save_path, "generated_molecules.csv")
        LOG.info("Save to {}".format(result_path))
        data_sorted.to_csv(result_path, index=False)
        print("generate")


    def sample(self, model_choice, model, src, src_mask, source_length, decode_type, without_property, num_generations=10, delta_weight=30, tanimoto_filter=0.2, filter_higher=1, max_trials=100,
               max_len=cfgd.DATA_DEFAULT['max_sequence_length'], isomericSmiles=False, device=None, max_scaffold_generations=10):
        """Sample up to ``num_generations`` analogues per source molecule.

        The routine repeatedly decodes 10 sequences from the current source
        molecule and keeps those that are valid, Lipinski-compliant, differ in
        molecular weight by less than ``delta_weight``, are new, and satisfy the
        Tanimoto filter (above ``tanimoto_filter`` when ``filter_higher == 1``,
        below it when ``filter_higher == 0``). Generated molecules are grouped
        by Murcko scaffold; when a scaffold reaches
        ``max_scaffold_generations`` the source is switched to a molecule from
        a less populated scaffold to encourage diversity.

        Args:
            model_choice: ``'transformer'`` (only supported branch).
            model: already loaded (and, for transformer, eval-mode) model.
            src: LongTensor of source token ids, shape ``(batch, seq_len)``.
            src_mask: attention mask matching ``src``.
            source_length: per-row source lengths (unused here).
            decode_type: decoder sampling strategy string.
            without_property: whether property tokens are absent.
            num_generations: target number of analogues per source molecule.
            delta_weight: maximum allowed absolute molecular-weight difference.
            tanimoto_filter: initial similarity threshold.
            filter_higher: 1 to require similarity above the threshold, 0 below.
            max_trials: maximum decode attempts per source molecule.
            max_len: maximum decoded sequence length.
            isomericSmiles: passed to canonicalisation.
            device: torch device for the model inputs.
            max_scaffold_generations: cap of molecules sharing one scaffold.

        Returns:
            Tuple ``(smiles_lists, scaffold_dict)`` where ``smiles_lists`` is
            one list of generated SMILES per source molecule and
            ``scaffold_dict`` is the scaffold grouping of the last processed
            molecule.

        Side effects:
            Progress bars/prints, and mutates ``tanimoto_filter`` locally when
            no alternative scaffold is available. Note that ``src_current``,
            ``mask_current`` and the returned ``scaffold_dict`` are (re)assigned
            inside the loop, so the returned grouping only reflects the last
            source molecule.
        """
        # NOTE (known issue, left unchanged): ``src`` is dereferenced here before
        # the later ``if src is not None`` guard. Calling with src=None raises
        # AttributeError immediately; the guard below is therefore unreachable
        # for that case.
        batch_size = src.shape[0]
        batch_index = torch.LongTensor(range(batch_size))
        batch_index_current = torch.LongTensor(range(batch_size)).to(device)
        start_mols = []
        #max_scaffold_generations = config.get('max_scaffold_generations', 5)  # Defaulting to 5 if not specified

        # Set of unique starting molecules
        if src is not None:
            for ibatch in range(batch_size):
                source_smi = self.tokenizer.untokenize(self.vocab.decode(src[ibatch].tolist()[:]))
                source_smi = uc.get_canonical_smile(source_smi, isomericSmiles)
                start_mols.append(source_smi)

        with torch.no_grad(): 
            # Valid, ibatch index is different from original, need map back
            batch_size = len(start_mols)     

            smiles_lists = []
            for ibatch in tqdm(range(batch_size)):
                current_trials = 1
                smiles_list = []
                found = 0   
                print(f"Source Molecule: {start_mols[ibatch]}")
                # Take 10 times the same sample as input for the network and iterate over it later
                if src is not None:
                    src_current = src.index_select(0, batch_index_current)
                    src_current = torch.stack([src_current[ibatch] for i in range(10)])
                if src_mask is not None:
                    mask_current = src_mask.index_select(0, batch_index_current)
                    mask_current = torch.stack([mask_current[ibatch] for i in range(10)])
              

                scaffold_dict = defaultdict(list)
                scaffold_trials = 0
                #import IPython; IPython.embed();
                while found < num_generations and current_trials < max_trials:

                    if current_trials%100==0:
                        print("current_trials")
                        print(current_trials)
                    current_trials += 1
                    scaffold_trials += 1
                    # NOTE (known issue, left unchanged): only the transformer
                    # branch assigns ``sequences``. With model_choice ==
                    # 'seq2seq' the name is undefined, raising NameError that
                    # the bare except below swallows; the loop then spins until
                    # max_trials without generating anything.
                    if model_choice == 'transformer':
                        try:
                            #import IPython; IPython.embed();
                            sequences = decode(model, src_current, mask_current, max_len, decode_type)
                            padding = (0, max_len-sequences.shape[1], 0, 0)
                            sequences = torch.nn.functional.pad(sequences, padding)
                        except:
                            print("Molformer decode failed")
                            continue

                    # Check valid and unique
                    for seq in sequences:
                    # seq = sequences[0]
                        smi = self.tokenizer.untokenize(self.vocab.decode(seq.cpu().numpy()))
                        smi = uc.get_canonical_smile(smi, isomericSmiles)

                        if found>num_generations:
                            break
                        # valid and not same as starting molecules
                        ####################################
                        if uc.is_valid(smi) and follows_lipinski(smi):

                            #import IPython; IPython.embed();
                            #### My modification to filter on delta molecular weight  ####
                            smi_weight = Descriptors.ExactMolWt(Chem.MolFromSmiles(smi))
                            source_smi_weight = Descriptors.ExactMolWt(Chem.MolFromSmiles(start_mols[ibatch]))
                            delta = abs(source_smi_weight-smi_weight)
                            if (delta < delta_weight) :       ################ weight delta ######
                            #if delta==0:       ################ weight delta StereoIsomers ######
                                #Check if already in list
                                if smi not in smiles_list and smi != start_mols[ibatch]:
                                    #Check similarity
                                    tanimoto_val = hf.calculate_tanimoto_from_two_smiles(start_mols[ibatch], smi)
                                    if filter_higher == 1:
                                        if tanimoto_val > tanimoto_filter:#             ################ filter on Tanomoto score
                                            smiles_list.append(smi)
                                            #print(smi)
                                            found +=1
                                            #print(f"Molecule: {ibatch}, found: {found} of {num_generations}")
                                            scaffold_smiles = calculate_scaffold(smi)
                                            scaffold_dict[scaffold_smiles].append(smi)
                                            
                                    elif filter_higher == 0:
                                        if tanimoto_val < tanimoto_filter:#             ################ filter on Tanomoto score
                                            smiles_list.append(smi)
                                            #print(smi)
                                            found +=1
                                            #print(f"Molecule: {ibatch}, found: {found} of {num_generations}")
                                            scaffold_smiles = calculate_scaffold(smi)
                                            scaffold_dict[scaffold_smiles].append(smi)     

                                    # Modification: Check scaffold generation limit
                                    base_scaffold_smiles = calculate_scaffold(start_mols[ibatch])
                                    molecules = scaffold_dict[base_scaffold_smiles]

                                    if len(molecules) >= max_scaffold_generations or current_trials%1000==0:
                                        # If the scaffold limit is reached, we find the next most populous scaffold
                                        #next_scaffolds = sorted(scaffold_dict.items(), key=lambda item: len(item[1]), reverse=True)
                                        if len(scaffold_dict)>1:
                                            eligible_keys = []
                                            for scaf, mols in scaffold_dict.items():
                                                if len(mols) < max_scaffold_generations:
                                                    eligible_keys.append(scaf)
                                            # Select a random key from the filtered list
                                            if eligible_keys:
                                                random_key = random.choice(eligible_keys)
                                                new_source_smi_list = scaffold_dict[random_key]
                                                if not new_source_smi_list:
                                                    continue
                                                else:
                                                    new_source_smi = random.choice(new_source_smi_list)
                                            else:
                                                # stay with the starting molecule
                                                new_source_smi = start_mols[ibatch]
                                            print("scaffold_trials len(molecules) new_source_smi")                                              
                                            print(new_source_smi)  
                                            try: # To cover cases that are not encodable for Molformer such as [SH] - maybe find a better way of coping with that

                                                new_source_encoded = self.vocab.encode(self.tokenizer.tokenize(new_source_smi))
                                                src_current = torch.tensor(new_source_encoded, dtype=torch.long).unsqueeze(0).repeat(10, 1).to(device)
                                                mask_current = torch.ones_like(src_current).to(device)
                                                mask_current = mask_current.unsqueeze(1) # to get the necessary dimensions for Molformer
                                                max_len = src_current.shape[1]     
                                                scaffold_trials = 0      
                                                scaffold_dict = defaultdict(list)
                                            except:
                                                print("FAILED")
                                                # No alternative scaffold available, adjust the tanimoto_filter
                                                if filter_higher == 1:
                                                    tanimoto_filter = max(tanimoto_filter - 0.1, 0)  # Ensuring filter doesn't go below zero
                                                else:
                                                    tanimoto_filter = min(tanimoto_filter + 0.1, 1)  # Ensuring filter doesn't exceed one
                                
                                        else:
                                            # No alternative scaffold available, adjust the tanimoto_filter
                                            if filter_higher == 1:
                                                tanimoto_filter = max(tanimoto_filter - 0.1, 0)  # Ensuring filter doesn't go below zero
                                            else:
                                                tanimoto_filter = min(tanimoto_filter + 0.1, 1)  # Ensuring filter doesn't exceed one
                    
                    scaffold_trials+=1
                    if scaffold_trials>=30:   
                        # If the scaffold limit is reached, we find the next most populous scaffold
                        if len(scaffold_dict)>1:
                            eligible_keys = []
                            for scaf, mols in scaffold_dict.items():
                                if len(mols) < max_scaffold_generations:
                                    eligible_keys.append(scaf)
                            # Select a random key from the filtered list
                            try:
                                if eligible_keys:
                                    random_key = random.choice(eligible_keys)
                                    new_source_smi_list = scaffold_dict[random_key]
                                    new_source_smi = random.choice(new_source_smi_list)
                                else:
                                    # stay with the starting molecule
                                    new_source_smi = start_mols[ibatch]
                            except:
                                # stay with the starting molecule
                                new_source_smi = start_mols[ibatch]   
                            try: # To cover cases that are not encodable for Molformer such as [SH] - maybe find a better way of coping with that
                                new_source_encoded = self.vocab.encode(self.tokenizer.tokenize(new_source_smi))
                                src_current = torch.tensor(new_source_encoded, dtype=torch.long).unsqueeze(0).repeat(10, 1).to(device)
                                mask_current = torch.ones_like(src_current).to(device)
                                mask_current = mask_current.unsqueeze(1) # to get the necessary dimensions for Molformer
                                max_len = src_current.shape[1]     
                                scaffold_trials = 0      
                                scaffold_dict = defaultdict(list)
                            except:
                                print("FAILED")
                                # No alternative scaffold available, adjust the tanimoto_filter
                                if filter_higher == 1:
                                    tanimoto_filter = max(tanimoto_filter - 0.1, 0)  # Ensuring filter doesn't go below zero
                                else:
                                    tanimoto_filter = min(tanimoto_filter + 0.1, 1)  # Ensuring filter doesn't exceed one

                print(f"Num generations: {num_generations}")
                # NOTE (known issue, left unchanged): the comparison is inverted.
                # When fewer than num_generations were found no padding happens
                # (the list stays short), and when enough were found the range
                # is empty so the line is a no-op. Consequently downstream
                # transposition sees ragged rows and may drop columns.
                if len(smiles_list) >= num_generations:
                    smiles_list = smiles_list + ["NAN" for i in range(num_generations-len(smiles_list))]
                smiles_lists.append(smiles_list)
        #print("sample")

        return smiles_lists, scaffold_dict

    def sample_seq2seq(self, model, mask, batch_index_current, decoder_hidden, encoder_outputs, max_len, device):
        """Greedy/ancestral decoding loop for a seq2seq model (legacy helper).

        Repeatedly feeds the previous token through the decoder and samples the
        next token from the softmax distribution.

        Args:
            model: seq2seq ``Model`` exposing ``network.decoder``.
            mask: encoder padding mask.
            batch_index_current: indices of the rows still being generated.
            decoder_hidden: initial decoder hidden state (tensor or tuple).
            encoder_outputs: encoder states for the selected rows.
            max_len: number of decoding steps.
            device: torch device.

        Returns:
            LongTensor of token ids with shape ``(batch, max_len)``.

        Side effects:
            None beyond allocating tensors on ``device``. Not called by the
            current ``sample`` implementation.
        """
        # batch size will change when some of the generated molecules are valid
        encoder_outputs_current = encoder_outputs.index_select(0, batch_index_current)
        batch_size = encoder_outputs_current.shape[0]

        # start token
        start_token = torch.zeros(batch_size, dtype=torch.long)
        start_token[:] = self.vocab["^"]
        decoder_input = start_token.to(device)
        sequences = []
        mask = torch.squeeze(mask, 1).to(device)

        # initial decoder hidden states
        if isinstance(decoder_hidden, tuple):
            decoder_hidden_current = (decoder_hidden[0].index_select(1, batch_index_current),
                                      decoder_hidden[1].index_select(1, batch_index_current))
        else:
            decoder_hidden_current = decoder_hidden.index_select(1, batch_index_current)
        for i in range(max_len):
            logits, decoder_hidden_current = model.network.decoder(decoder_input.unsqueeze(1),
                                                                  decoder_hidden_current,
                                                                  encoder_outputs_current, mask)
            logits = logits.squeeze(1)
            probabilities = logits.softmax(dim=1)  # torch.Size([batch_size, vocab_size])
            topi = torch.multinomial(probabilities, 1)  # torch.Size([batch_size, 1])
            decoder_input = topi.view(-1).detach()
            sequences.append(decoder_input.view(-1, 1))

        sequences = torch.cat(sequences, 1)
        print("sample_seq2seq")
        return sequences

# def calculate_tanimoto_from_two_smiles(smi1, smi2, nbits=2048):
#     """This function takes two smile_stings and 
#     calculates the Tanimoto similarity and returns it and prints it out"""
    
#     pattern1 = Chem.MolFromSmiles(smi1)
#     pattern2 = Chem.MolFromSmiles(smi2)
#     fp1 = AllChem.GetMorganFingerprintAsBitVect(pattern1, 2, nBits=nbits)
#     fp2 = AllChem.GetMorganFingerprintAsBitVect(pattern2, 2, nBits=nbits)

#     tan_sim = DataStructs.TanimotoSimilarity(fp1, fp2)
#     tan_sim = round(tan_sim,4)
      
#     return tan_sim


# def calculate_weight(smiles_str):
#     """ used in compare_molformer input smiles string"""
#     m = Chem.MolFromSmiles(smiles_str)
#     return ExactMolWt(m)

def check_delta_weights(data_sorted):
    """Print the molecular-weight difference for every generated molecule.

    Debug helper used in the Molformer notebook: for every ``SMILES`` row it
    iterates the ``Predicted_smi_*`` columns and prints
    ``source_weight - generated_weight``.

    Args:
        data_sorted: DataFrame with a ``SMILES`` column and one or more
            ``Predicted_smi_*`` columns.

    Returns:
        None.

    Side effects:
        Prints one difference per comparable column; unparsable/invalid
        entries are silently skipped by a bare ``except``.
    """
    for i in range(len(data_sorted["SMILES"])):
        source = data_sorted.iloc[i]["SMILES"]

        source_weight = ExactMolWt(Chem.MolFromSmiles(source))
        for column in data_sorted.columns:
            if "Predicted_smi_" in column:
                try:
                    sim = data_sorted.iloc[i][column]
                    sim_weight = ExactMolWt(Chem.MolFromSmiles(sim))
                    print(source_weight-sim_weight)
                except:
                    pass
                

def generate_opts(parser):
    """Register all CLI options used by ``GenerateRunner`` on ``parser``.

    Args:
        parser: an ``argparse.ArgumentParser`` instance (normally the module
            level ``parser``).

    Returns:
        None.

    Side effects:
        Mutates ``parser`` in place by adding arguments/groups. This is called
        at import time immediately after ``parser`` is created.
    """
    # Transformer or Seq2Seq
    parser.add_argument('--model-choice', required=True, help="transformer or seq2seq")
    """Input output settings"""
    group = parser.add_argument_group('Input-Output')
    group.add_argument('--data-path', required=True,
                       help="""Input data path""")
    group.add_argument('--vocab-path', required=True, help="""Vocabulary path""")
    group.add_argument('--test-file-name', default='test', help="""test file name without .csv""")
    group.add_argument('--save-directory', default='evaluation',
                       help="""Result save directory""")
    # Model to be used for generating molecules
    group = parser.add_argument_group('Model')
    group.add_argument('--model-path', help="""Model path""", required=True)
    group.add_argument('--model-full-path', help="""Full Model path to ckp file""", required=True)
    group.add_argument('--epoch', type=int, help="""Which epoch to use""", required=True)
    # General
    group = parser.add_argument_group('General')
    group.add_argument('--batch-size', type=int, default=128,
                       help='Batch size for training')
    group.add_argument('--num-generations', type=int, default=10,
                       help='Number of molecules to be generated')
    group.add_argument('--decode-type',type=str, default='multinomial',help='decode strategy')
    group.add_argument('--without-property', help="""Without property tokens as input""",
                       action="store_true")
    group.add_argument('--delta-weight', type=int, default=30,
                       help='weight to deviate from template')
    group.add_argument('--tanimoto-filter', type=float, default=0.2,
                       help='defines the minimum similarity that needs to be met in the generation') 
    group.add_argument('--filter-higher', type=int, default=1, 
                       help="""0=False, 1=True; If it should generate lower or higher tanimoto values""")
    group.add_argument('--max-scaffold-generations', type=int, default=10, 
                       help="""defines the maximum number of generated molecules with the same scaffold""")                       
    # NOTE (known issue, left unchanged): ``action="store_false"`` combined with
    # ``default=False`` means the flag can only ever be False (passing it also
    # sets False), so isomeric SMILES can never be enabled from the CLI.
    group.add_argument('--isomericSmiles', default=False,
                       help='isomericSmiles', action="store_false")     
    group.add_argument('--max-trials', type=int, default=100,
                       help='maximum number of trails until while loop breaks')
    group.add_argument('--prediction-mode', help="""Without property tokens as input""",
                       action="store_true")  ###########################
    
parser = argparse.ArgumentParser(
        description='generate.py',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
generate_opts(parser)


def prep_dataframe_with_generated_molecules(csv_file):    
    """Reshape the generated-molecules CSV into one column per source molecule.

    The CSV produced by ``GenerateRunner.generate`` stores one source molecule
    per row with ``Predicted_smi_*`` columns. This helper pivots that layout so
    each source molecule becomes a column named by its ``sample-id``, holding
    ``SMILES`` followed by all generated analogues.

    Args:
        csv_file: path to ``generated_molecules.csv``.

    Returns:
        DataFrame whose columns are sample ids and whose rows are the original
        molecule plus each generation index.

    Side effects:
        Reads ``csv_file`` from disk. Columns named ``sample-id`` are required.
        Note that rows with differing lengths cannot be represented column-wise;
        the loop assumes every row yields the same number of entries.
    """
    df = pd.read_csv(csv_file)
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    # columns_ = [f"Predicted_smi_{i}" for i in range(1,len(df.columns)-1)]
    columns_ = list(df.columns[df.columns.str.contains('Predicted_smi_')])
    # import IPython; IPython.embed();
    columns = ["SMILES"] + columns_
    
    df_samples = pd.DataFrame()
    for i in range(len(df)):
        smi_list = []
        sample_id_list = []
        ID = str(df.iloc[i]["sample-id"])

        for idx, column in enumerate(columns):

            smi = df.iloc[i][column]
            smi_list.append(smi)
            sample_id_list.append(ID+f"_{idx}")
        df_samples[ID] = smi_list
    return df_samples   


def worker_init_fn(worker_id):
    """Seed numpy and ``random`` per DataLoader worker for reproducibility.

    Args:
        worker_id: worker index supplied by torch's DataLoader.

    Returns:
        None.

    Side effects:
        Sets the global numpy and Python ``random`` seeds from the worker's
        torch seed. ``worker_id`` itself is unused.
    """
    np.random.seed(torch.initial_seed() % (2**32))
    random.seed(torch.initial_seed() % (2**32))
    
    
def load_data(config, mode, stoi, stoi_MF, itos, itos_MF):
    """Load the multimodal dataset and write its SMILES to the Molformer CSV.

    Iterates a batch-size-1 DataLoader, decodes each target SMILES, canonicalises
    it and assigns a random ``MF_<id>`` sample id. The result is written to
    ``<config.MF_csv_source_folder_location>/<config.MF_csv_source_file_name>.csv``
    and also returned, so the same file can be consumed by Molformer.

    Args:
        config: experiment config exposing the ``MF_csv_source_*`` paths.
        mode: dataset split mode (e.g. ``'train'``/``'val'``/``'test'``).
        stoi/stoi_MF: token-to-index maps for SMILES and molecular formula.
        itos/itos_MF: index-to-token maps (only ``itos`` is used here).

    Returns:
        DataFrame with columns ``SMILES`` and ``sample-id``.

    Side effects:
        Reads the dataset, writes the CSV to disk, prints unparsable SMILES.
    Note ``drop_last=True`` with ``batch_size=1`` silently discards the final
    sample of the dataset.
    """
    data = MultimodalData(config, stoi, stoi_MF, mode=mode)
    dataloader = DataLoader(data, 
                                batch_size=1, 
                                shuffle=False, 
                                collate_fn=collate_fn, 
                                drop_last=True,
                                worker_init_fn=worker_init_fn)
    list_smi = []
    list_id = []
    for idx, i in enumerate(dataloader):
        smi = hf.tensor_to_smiles(i["trg_enc_SMI"].transpose(1,0)[1:], itos)
            # Canonicalize the SMILES string
        mol = Chem.MolFromSmiles(smi[0])
        if mol is not None:
            canonical_smi = Chem.MolToSmiles(mol)
        else:
            # Handle cases where the SMILES string could not be parsed
            canonical_smi = None
            print(f"Could not parse SMILES: {smi}")
            continue
        random_number = "MF_" + str(random.randint(1, 10000000))
        list_id.append(random_number)
        list_smi.append(canonical_smi)
        #if idx == config.n_samples:
        #    break

    df = pd.DataFrame({'SMILES': list_smi, 'sample-id': list_id})
    csv_file_path = os.path.join(config.MF_csv_source_folder_location, config.MF_csv_source_file_name) +".csv"
    df.to_csv(csv_file_path, index=False)
    return df

# def main_molformer(config):
#     df_samples_MMP = pd.DataFrame()
#     df_samples_scaffold = pd.DataFrame()

#     if "MMP" in config.MF_methods:
#         config_MMP = {
#               "model-choice": "transformer",
#               "data-path": "/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer/deep-molecular-optimization/data/MMP",
#               "test-file-name": "test_selection_1",
#               "epoch": 60,
#               "num-generations": 20,
#               "save-directory": "/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer/deep-molecular-optimization/experiments/evaluation_transformer",
#               "model-path": "/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer/deep-molecular-optimization/experiments/trained/Transformer-U/MMP/checkpoint",
#               "vocab-path": "/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer/deep-molecular-optimization/data/MMP/vocab.pkl",
#               "without-property": True,
#               "delta-weight": 250,
#               "tanimoto-filter": 0.15,
#               "max-trials": 1000,
#               "prediction-mode": True
#             }
#         config_MMP["max-trials"] = config.MF_max_trails
#         config_MMP["tanimoto-filter"] = config.MF_tanimoto_filter
#         config_MMP["filter-higher"] = config.MF_filter_higher
#         config_MMP["max-scaffold-generations"] = config.max_scaffold_generations
#         config_MMP["delta-weight"] = config.MF_delta_weight
#         config_MMP["num-generations"] = config.MF_generations
#         config_MMP["model-full-path"] = config.MF_model_path 
#         config_MMP["data-path"] = config.MF_csv_source_folder_location
#         config_MMP["test-file-name"] = config.MF_csv_source_file_name
#         config_MMP["vocab-path"] = config.MF_vocab
#         arg_list = []
#         for key, value in config_MMP.items():
#             arg_list.append(f"--{key}")
#             if value is not True:  # To handle flags like --without-property
#                 arg_list.append(str(value))

#         args = parser.parse_args(args=arg_list)
#         runner = GenerateRunner(args)
#         runner.generate(args)

#         csv_file = os.path.join(config_MMP["save-directory"], config_MMP["test-file-name"], f"evaluation_{config_MMP['epoch']}/generated_molecules.csv")
#         #csv_file = "/projects/cc/knlr326/2_git_repos/deep-molecular-optimization/experiments/evaluation_transformer/test_selection_1/evaluation_60/generated_molecules.csv"
#         df_samples_MMP = prep_dataframe_with_generated_molecules(csv_file)    


#     if "scaffold" in config.MF_methods:
    
#         config_scaf = {
#               "model-choice": "transformer",
#               "data-path": "/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer/deep-molecular-optimization/data/MMP",
#               "test-file-name": "test_selection_1",
#               "epoch": 60,
#               "num-generations": 3,
#               "save-directory": "/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer/deep-molecular-optimization/experiments/evaluation_transformer",
#               "model-path": "/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer/deep-molecular-optimization/experiments/trained/Transformer-U/scaffold/checkpoint",
#               "vocab-path": "/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer/deep-molecular-optimization/data/scaffold/vocab.pkl",
#               "without-property": True,
#               "delta-weight": 150,
#               "tanimoto-filter": 0.1,
#               "max-trials": 100,
#               "prediction-mode": True
#             }

#         config_scaf["max-trials"] = config.MF_max_trails
#         config_scaf["tanimoto-filter"] = config.MF_tanimoto_filter
#         config_scaf["filter-higher"] = config.MF_filter_higher
#         config_scaf["delta-weight"] = config.MF_delta_weight
#         config_scaf["num-generations"] = config.MF_generations
#         config_scaf["data-path"] = config.MF_csv_source_folder_location
#         config_scaf["test-file-name"] = config.MF_csv_source_file_name

#         arg_list = []
#         for key, value in config_scaf.items():
#             arg_list.append(f"--{key}")
#             if value is not True:  # To handle flags like --without-property
#                 arg_list.append(str(value))

#         args = parser.parse_args(args=arg_list)
#         runner = GenerateRunner(args)
#         runner.generate(args)

#         csv_file = os.path.join(config_MMP["save-directory"], config_MMP["test-file-name"], f"evaluation_{config_MMP['epoch']}/generated_molecules.csv")
#         #csv_file = "/projects/cc/knlr326/2_git_repos/deep-molecular-optimization/experiments/evaluation_transformer/test_selection_1/evaluation_60/generated_molecules.csv"
#         df_samples_scaffold = prep_dataframe_with_generated_molecules(csv_file)   
#     combined_df = pd.concat([df_samples_MMP, df_samples_scaffold], axis=0, ignore_index=True)
        
#     return combined_df


def main_molformer(config):
    """Generate analogues with Molformer and return them as a DataFrame.

    Runs one generation pass per method listed in ``config.MF_methods``
    (``'MMP'`` for matched-molecular-pair generation and ``'scaffold'`` for
    scaffold hopping), rebuilds the CLI argument list from the config, invokes
    ``GenerateRunner`` for each method and merges the resulting per-molecule
    DataFrames.

    Args:
        config: experiment config providing ``MF_methods`` plus the
            ``MF_*`` / ``max_scaffold_generations`` hyper-parameters listed
            below.

    Returns:
        DataFrame combining the generated molecules from all selected methods
        (one column per sample id).

    Side effects:
        Runs model inference, writes ``generated_molecules.csv`` files under
        ``experiments/...`` and reads them back. Note that both branches reuse
        ``test_file_name`` and ``data_path`` overrides from ``config``.
    """
    df_samples_MMP = pd.DataFrame()
    df_samples_scaffold = pd.DataFrame()

    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)

    # Define base path relative to the script's location
    base_path = os.path.abspath(os.path.join(script_dir, '../deep-molecular-optimization'))

    if "MMP" in config.MF_methods:
        config_MMP = {
            "model-choice": "transformer",
            "data-path": os.path.join(base_path, 'data/MMP'),
            "test-file-name": "test_selection_1",
            "epoch": 60,
            "num-generations": 20,
            "save-directory": os.path.join(base_path, 'experiments/evaluation_transformer'),
            "model-path": os.path.join(base_path, 'experiments/trained/Transformer-U/MMP/checkpoint'),
            "vocab-path": os.path.join(base_path, 'data/MMP/vocab.pkl'),
            "without-property": True,
            "delta-weight": 250,
            "tanimoto-filter": 0.15,
            "max-trials": 1000,
            "prediction-mode": True
        }
        # Update config with values from the config object
        config_MMP["max-trials"] = config.MF_max_trails
        config_MMP["tanimoto-filter"] = config.MF_tanimoto_filter
        config_MMP["filter-higher"] = config.MF_filter_higher
        config_MMP["max-scaffold-generations"] = config.max_scaffold_generations
        config_MMP["delta-weight"] = config.MF_delta_weight
        config_MMP["num-generations"] = config.MF_generations
        config_MMP["model-full-path"] = config.MF_model_path
        config_MMP["data-path"] = config.MF_csv_source_folder_location
        config_MMP["test-file-name"] = config.MF_csv_source_file_name
        config_MMP["vocab-path"] = config.MF_vocab
        
        # Prepare arguments
        arg_list = []
        for key, value in config_MMP.items():
            arg_list.append(f"--{key}")
            if value is not True:
                arg_list.append(str(value))

        args = parser.parse_args(args=arg_list)
        runner = GenerateRunner(args)
        runner.generate(args)

        csv_file = os.path.join(config_MMP["save-directory"], config_MMP["test-file-name"], f"evaluation_{config_MMP['epoch']}/generated_molecules.csv")
        df_samples_MMP = prep_dataframe_with_generated_molecules(csv_file)

    if "scaffold" in config.MF_methods:
        config_scaf = {
            "model-choice": "transformer",
            "data-path": os.path.join(base_path, 'data/MMP'),
            "test-file-name": "test_selection_1",
            "epoch": 60,
            "num-generations": 3,
            "save-directory": os.path.join(base_path, 'experiments/evaluation_transformer'),
            "model-path": os.path.join(base_path, 'experiments/trained/Transformer-U/scaffold/checkpoint'),
            "vocab-path": os.path.join(base_path, 'data/scaffold/vocab.pkl'),
            "without-property": True,
            "delta-weight": 150,
            "tanimoto-filter": 0.1,
            "max-trials": 100,
            "prediction-mode": True
        }

        # Update config with values from the config object
        config_scaf["max-trials"] = config.MF_max_trails
        config_scaf["tanimoto-filter"] = config.MF_tanimoto_filter
        config_scaf["filter-higher"] = config.MF_filter_higher
        config_scaf["delta-weight"] = config.MF_delta_weight
        config_scaf["num-generations"] = config.MF_generations
        config_scaf["data-path"] = config.MF_csv_source_folder_location
        config_scaf["test-file-name"] = config.MF_csv_source_file_name

        # Prepare arguments
        arg_list = []
        for key, value in config_scaf.items():
            arg_list.append(f"--{key}")
            if value is not True:
                arg_list.append(str(value))

        args = parser.parse_args(args=arg_list)
        runner = GenerateRunner(args)
        runner.generate(args)

        csv_file = os.path.join(config_scaf["save-directory"], config_scaf["test-file-name"], f"evaluation_{config_scaf['epoch']}/generated_molecules.csv")
        df_samples_scaffold = prep_dataframe_with_generated_molecules(csv_file)

    combined_df = pd.concat([df_samples_MMP, df_samples_scaffold], axis=0, ignore_index=True)
    
    return combined_df