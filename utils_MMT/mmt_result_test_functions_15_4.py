"""Result-analysis utilities for the v15.4 Multimodal Transformer (MMT).

This module wraps a trained MMT (and optionally a CLIP) checkpoint and provides
the downstream analyses used to benchmark molecule generation from spectra:

1.  Data / model loading helpers (``load_data``, ``load_MMT_model``,
    ``load_CLIP_model``).
2.  Correct-token / max-token probability statistics over a validation set
    (``calculate_corr_max_prob``, ``predict_prop_correct_max_sequence_3``,
    ``calculate_corr_max_prob_2``, ``predict_corr_max_performance_metric``,
    ``run_model_analysis``, ``run_test_performance_CLIP_3``).
3.  Greedy decoding and evaluation (``greedy_sequence_2``,
    ``evaluate_greedy_2``, ``run_greedy_sampling``, ``run_greedy_sampling_v2``,
    ``run_test_performance_CLIP_greedy_3``).
4.  Multinomial sampling and evaluation (``multinomial_sequence_multi_2``,
    ``run_multinomial_sampling``, ``run_multinomial_sampling_v2``,
    ``run_test_mns_performance_CLIP_3``, ``generate_mns_list``,
    ``run_mns_hit_counter_experiment``).
5.  Post-processing / filtering: SMILES de-duplication, canonicalisation,
    molecular-weight and molecular-formula filters, invalid-pair bookkeeping.
6.  Similarity scoring (Tanimoto against the ground-truth molecule, CLIP dot
    similarity, simulated HSQC/COSY error) and hit-rate statistics
    (``calc_percentage_top_x_correct``, ``run_precentage_calculation*``).
7.  Plotting helpers: Tanimoto histograms, rank histograms, molecule grids.

The ``prepare_*_data_from_src_2`` helpers re-apply the same per-dimension
scaling used by :mod:`utils_MMT.run_batch_gen_val_MMT_v15_4` before handing
tensors to the SGNN spectrum simulator.

Most functions take a ``config`` namespace carrying checkpoint paths,
``config.device``, ``config.temperature``, ``config.max_len`` and
``config.multinom_runs``. ``config.temperature`` is frequently mutated in place
to escape sampling collapse and restored afterwards, so callers must not rely
on it staying constant across a call.

This file only performs analysis: it never trains models and never writes
checkpoints. Large commented-out sections below preserve earlier versions of
``calculate_max_prob``, ``predict_prop_max_sequence_3`` and
``greedy_sequence_2``; they are not executed.
"""

# Standard library imports
import glob
import json
import os
import random
from collections import defaultdict
from functools import reduce
from argparse import Namespace

# Data processing and scientific computing
import numpy as np
import pandas as pd
from tqdm import tqdm
import statistics
import operator

# Visualization libraries
import matplotlib.pyplot as plt

# PyTorch for deep learning
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

# RDKit for cheminformatics
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, Descriptors, MolFromSmiles, MolToSmiles

# Weights & Biases for experiment tracking
import wandb

# Local utilities/modules
import utils_MMT.sgnn_code_pl_v15_4 as sc
import utils_MMT.train_test_functions_pl_v15_4 as ttf
import utils_MMT.helper_functions_pl_v15_4 as hf
import utils_MMT.validate_generate_MMT_v15_4 as vgmmt
import utils_MMT.run_batch_gen_val_MMT_v15_4 as rbgvm
from utils_MMT.dataloaders_pl_v15_4 import MultimodalData, collate_fn
from utils_MMT.models_MMT_v15_4 import MultimodalTransformer, TransformerMultiGPU
from utils_MMT.models_CLIP_v15_4 import CLIPMultiGPU  # CHANGE WHEN I HAVE A CLIP V8 model trained
# from utils_MMT.models_BLIP_v15_4 import BLIPMultiGPU  # CHANGE WHEN I HAVE A CLIP V8 model trained




def worker_init_fn(worker_id):
    """Seed a DataLoader worker deterministically from the loader's base seed.

    Args:
        worker_id (int): Worker index supplied by PyTorch; unused because both
            seeds derive from ``torch.initial_seed()``.

    Returns:
        None

    Side effects:
        Sets the global NumPy and ``random`` module seeds of the worker process.
    """
    np.random.seed(torch.initial_seed() % (2**32))
    random.seed(torch.initial_seed() % (2**32))

def prepare_HSQC_data_from_src_2(src_HSQC_list):
    """Keep the non-empty HSQC rows and rescale them for the SGNN simulator.

    Args:
        src_HSQC_list (list[torch.Tensor]): Per-sample HSQC tensors of shape
            (n_peaks, 2) holding (1H shift, 13C shift) pairs.

    Returns:
        list[torch.Tensor]: One tensor per input sample with the same shape,
            where column 0 (1H) is multiplied by 10 and column 1 (13C) by 200.
            Samples without any surviving row are returned as empty tensors so
            the output stays aligned with the input list.

    Side effects:
        None; the input tensors are not modified in place.

    Notes:
        The row filter keeps rows where *both* coordinates are non-zero; the
        inline comment in the body is misleading.
    """
    processed_HSQC = []

    for src in src_HSQC_list:
        #for src in src_HSQC:
            # Filter out rows where both elements are not zero
            non_zero_mask = (src != 0).all(dim=1)
            filtered_src = src[non_zero_mask]

            if filtered_src.nelement() != 0:  # Check if tensor is not empty
                scaled_tensors = [filtered_src[:, 0] * 10, filtered_src[:, 1] * 200]
                combined_tensor = torch.stack(scaled_tensors, dim=1)
                processed_HSQC.append(combined_tensor)
            else:
                processed_HSQC.append(torch.tensor([]))  # Append an empty tensor for consistency

    return processed_HSQC

def prepare_COSY_data_from_src_2(src_COSY_list):
    """Keep the non-empty COSY rows and rescale them for the SGNN simulator.

    Args:
        src_COSY_list (list[torch.Tensor]): Per-sample COSY tensors of shape
            (n_peaks, 2) holding (1H shift, 1H shift) pairs.

    Returns:
        list[torch.Tensor]: One tensor per input sample with the same shape,
            where both columns are multiplied by 10. Samples without any
            surviving row are returned as empty tensors so the output stays
            aligned with the input list.

    Side effects:
        None; the input tensors are not modified in place.
    """
    processed_COSY = []

    for src in src_COSY_list:
        #for src in src_HSQC:

            # Filter out rows where both elements are not zero
            non_zero_mask = (src != 0).all(dim=1)
            filtered_src = src[non_zero_mask]

            if filtered_src.nelement() != 0:  # Check if tensor is not empty
                scaled_tensors = [filtered_src[:, 0] * 10, filtered_src[:, 1] * 10]
                combined_tensor = torch.stack(scaled_tensors, dim=1)
                processed_COSY.append(combined_tensor)
            else:
                processed_COSY.append(torch.tensor([]))  # Append an empty tensor for consistency

    return processed_COSY

def generate_df_for_HSQC_calculations(gen_conv_SMI_list, trg_conv_SMI_list, src_HSQC_list):
    """Split generated molecules into parseable and failed buckets (HSQC variant).

    Args:
        gen_conv_SMI_list (list[str]): Generated SMILES strings.
        trg_conv_SMI_list (list[str]): Matching ground-truth SMILES strings.
        src_HSQC_list (list[torch.Tensor]): HSQC tensors aligned with both lists.

    Returns:
        tuple:
            pandas.DataFrame: Columns ``sample-id`` (``"<index>_<random>"``),
                ``SMILES`` and ``trg_SMILES`` for the parseable molecules.
            list[torch.Tensor]: ``src_HSQC_list`` restricted to those molecules.
            list[list[str]]: ``[gen_smi, trg_smi]`` pairs that failed to parse.

    Side effects:
        Consumes the global ``random`` state when minting sample ids, so the ids
        are not reproducible without seeding.
    """
    succ_gen_list =[]
    src_HSQC_list_new = []
    failed_list = []
    for i, (gen_smi, trg_smi, src_HSQC) in enumerate(zip(gen_conv_SMI_list, trg_conv_SMI_list, src_HSQC_list)):
            mol = Chem.MolFromSmiles(gen_smi)
            if mol is not None:
                ran_num = random.randint(0, 100000)
                sample_id = f"{i}_{ran_num}"
                succ_gen_list.append([sample_id, gen_smi, trg_smi])
                src_HSQC_list_new.append(src_HSQC)
            else:
                failed_list.append([gen_smi,trg_smi])
                continue

    # Create a DataFrame of successful SMILES generations
    df_succ_smis = pd.DataFrame(succ_gen_list, columns=['sample-id', 'SMILES', "trg_SMILES"])
    return df_succ_smis, src_HSQC_list_new, failed_list

def generate_df_for_COSY_calculations(gen_conv_SMI_list, trg_conv_SMI_list, src_COSY_list):
    """Split generated molecules into parseable and failed buckets (COSY variant).

    Args:
        gen_conv_SMI_list (list[str]): Generated SMILES strings.
        trg_conv_SMI_list (list[str]): Matching ground-truth SMILES strings.
        src_COSY_list (list[torch.Tensor]): COSY tensors aligned with both lists.

    Returns:
        tuple:
            pandas.DataFrame: Columns ``sample-id`` (``"<index>_<random>"``),
                ``SMILES`` and ``trg_SMILES`` for the parseable molecules.
            list[torch.Tensor]: ``src_COSY_list`` restricted to those molecules.
            list[list[str]]: ``[gen_smi, trg_smi]`` pairs that failed to parse.

    Side effects:
        Consumes the global ``random`` state when minting sample ids.
    """
    succ_gen_list =[]
    src_COSY_list_new = []
    failed_list = []
    for i, (gen_smi, trg_smi, src_HSQC) in enumerate(zip(gen_conv_SMI_list, trg_conv_SMI_list, src_COSY_list)):
            mol = Chem.MolFromSmiles(gen_smi)
            if mol is not None:
                ran_num = random.randint(0, 100000)
                sample_id = f"{i}_{ran_num}"
                succ_gen_list.append([sample_id, gen_smi, trg_smi])
                src_COSY_list_new.append(src_HSQC)
            else:
                failed_list.append([gen_smi,trg_smi])
                continue

    # Create a DataFrame of successful SMILES generations
    df_succ_smis = pd.DataFrame(succ_gen_list, columns=['sample-id', 'SMILES', "trg_SMILES"])
    return df_succ_smis, src_COSY_list_new, failed_list


def calculate_corr_max_prob(config, model_MMT, val_dataloader, stoi, itos):
    """
    Token-probability statistics for the teacher-forced (ground-truth) token stream.

    Args:
        config: Experiment namespace (``config.device``, ``config.temperature``).
        model_MMT (torch.nn.Module): Trained MMT decoder.
        val_dataloader (DataLoader): Validation batches collated by ``collate_fn``.
        stoi (dict[str, int]): Token-to-index map; ``"<EOS>"`` terminates a sequence.
        itos (dict[int, str]): Index-to-token map (accepted for API symmetry, unused).

    Returns:
        dict[str, list[float]]: Four parallel lists, one entry per sequence:
        ``aggregated_corr_prob_multi`` (product of the probabilities assigned to
        the ground-truth token), ``aggregated_corr_prob_avg`` (their mean),
        ``aggregated_max_prob_multi`` (product of the arg-max token probabilities)
        and ``aggregated_max_prob_avg`` (their mean).

    Side effects:
        Runs the model in eval/no-grad mode; prints a tqdm progress bar.
    """
    aggregated_corr_prob_multi, aggregated_corr_prob_avg, aggregated_max_prob_multi, aggregated_max_prob_avg =[],[],[],[]
    prob_dict_results = {}
    for idx, data_dict in enumerate(tqdm(val_dataloader)):

        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC, src_COSY = vgmmt.run_model(model_MMT, data_dict, config) 
        trg_tensor, corr_token_prob, trg_tensor_max, max_token_prob = predict_prop_correct_max_sequence_3(model_MMT, stoi, memory, src_padding_mask, trg_enc_SMI, config)

        # Ensure we're working with 2D tensors even for single molecules
        if corr_token_prob.dim() == 1:
            corr_token_prob = corr_token_prob.unsqueeze(1)
            max_token_prob = max_token_prob.unsqueeze(1)
        if isinstance(trg_tensor, int):
            trg_tensor = [trg_tensor]
        if isinstance(trg_tensor_max, int):
            trg_tensor_max = [trg_tensor_max]

        for corr_prob_list, max_prob_list, token_list, token_list_max in zip(corr_token_prob.T, max_token_prob.T, trg_tensor, trg_tensor_max):
            seq_corr_probs, seq_max_probs  = [], []
            for corr_prob, max_prob, token in zip(corr_prob_list, max_prob_list, token_list):                                                  
                if token == stoi["<EOS>"]:  # End of sequence
                    break
                seq_corr_probs.append(corr_prob.detach().item())
                seq_max_probs.append(max_prob.detach().item())
                    # Populate dictionaries with calculated probabilities

            prob_corr_multi = reduce(operator.mul, seq_corr_probs, 1)
            prob_corr_avg = statistics.mean(seq_corr_probs) if seq_corr_probs else 0
            prob_max_multi = reduce(operator.mul, seq_max_probs, 1)
            prob_max_avg = statistics.mean(seq_max_probs) if seq_max_probs else 0

            aggregated_corr_prob_multi.append(prob_corr_multi)
            aggregated_corr_prob_avg.append(prob_corr_avg)  
            aggregated_max_prob_multi.append(prob_max_multi)
            aggregated_max_prob_avg.append(prob_max_avg) 
        #import IPython; IPython.embed();


    prob_dict_results["aggregated_corr_prob_multi"] = aggregated_corr_prob_multi
    prob_dict_results["aggregated_corr_prob_avg"] = aggregated_corr_prob_avg
    prob_dict_results["aggregated_max_prob_multi"] = aggregated_max_prob_multi
    prob_dict_results["aggregated_max_prob_avg"] = aggregated_max_prob_avg
    return prob_dict_results



# def calculate_max_prob(config, model_MMT, val_dataloader, stoi, itos):
#     aggregated_corr_prob_multi, aggregated_corr_prob_avg, aggregated_max_prob_multi, aggregated_max_prob_avg =[],[],[],[]
#     prob_dict_results = {}
#     for idx, data_dict in enumerate(tqdm(val_dataloader)):
#         memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC = vgmmt.run_model(model_MMT, data_dict, config)
#         trg_tensor_max, max_token_prob = predict_prop_max_sequence_3(model_MMT, stoi, memory, src_padding_mask, trg_enc_SMI, config)

#         for max_prob_list, token_list_max in zip(max_token_prob.T, trg_tensor_max):
#             seq_max_probs  = []
#             for max_prob, token in zip(max_prob_list, token_list_max):                                                  
#                 if token == stoi["<EOS>"]:  # End of sequence
#                     break
#                 seq_max_probs.append(max_prob.detach().item())
#                     # Populate dictionaries with calculated probabilities

#             prob_max_multi = reduce(operator.mul, seq_max_probs, 1)
#             prob_max_avg = statistics.mean(seq_max_probs) if seq_max_probs else 0

#             aggregated_max_prob_multi.append(prob_max_multi)
#             aggregated_max_prob_avg.append(prob_max_avg) 
#         #import IPython; IPython.embed();

#     prob_dict_results["aggregated_max_prob_multi"] = aggregated_max_prob_multi
#     prob_dict_results["aggregated_max_prob_avg"] = aggregated_max_prob_avg
#     return prob_dict_results


# def predict_prop_max_sequence_3(model, stoi, memory, src_padding_mask, trg_enc_SMI, config):
#     """
#     Predicts the properties of each token in a sequence generated by a transformer model.

#     Parameters:
#     model (torch.nn.Module): The transformer model used for prediction.
#     stoi (dict): Dictionary mapping tokens to indices.
#     memory (torch.Tensor): The memory tensor output from the transformer's encoder.
#     src_padding_mask (torch.Tensor): The source padding mask.
#     trg_enc_SMI (torch.Tensor): The target encoded Simplified Molecular Input Line Entry System (SMILES).
#     gen_num (int): Number of generations for multinomial sampling.
#     config (obj): Configuration object containing model and operation settings.

#     Returns:
#     tuple: Contains tensors for target sequence, correct token probabilities, 
#            maximum probability sequence, maximum token probabilities, 
#            and multinomial token probabilities.
#     """

#     # Ensure the model is in evaluation mode
#     model.eval()

#     # Define the initial target tensor with <SOS> tokens
#     N = memory.size(1)
#     trg_tensor = torch.full((1, N), stoi["<SOS>"], dtype=torch.long, device=config.device)
#     trg_tensor_max = torch.full((1, N), stoi["<SOS>"], dtype=torch.long, device=config.device)

#     # Token probabilities containers
#     corr_token_prob, max_token_prob = [], []

#     # Transpose target encoded SMILES and remove <EOS> token
#     trg_enc_SMI_T = trg_enc_SMI.transpose(0, 1)
#     real_trg = trg_enc_SMI_T[1:, :]
    
#     # Iterate over each token in the target sequence
#     with torch.no_grad():
#         for idx in range(real_trg.shape[0]):
#             # Prepare input for the decoder
#             gen_seq_length, N = trg_tensor_max.shape
#             gen_positions = torch.arange(gen_seq_length).unsqueeze(1).expand(gen_seq_length, N).to(config.device)
#             embedding_gen = model.dropout2(model.embed_trg(trg_tensor) + model.pe_trg(gen_positions))
#             gen_mask = model.generate_square_subsequent_mask(gen_seq_length).to(config.device)

#             # Generate output from the decoder
#             output = model.decoder(embedding_gen, memory, tgt_mask=gen_mask, memory_key_padding_mask=src_padding_mask)
#             output = model.fc_out(output)
#             probabilities = F.softmax(output / config.temperature, dim=2)

#             # Process token probabilities
#             next_word = torch.argmax(probabilities[-1], dim=1)
#             max_prob = probabilities[-1].gather(1, next_word.unsqueeze(-1)).squeeze()
#             max_token_prob.append(max_prob)

#             # Update target tensor with max probability token
#             trg_tensor_max = torch.cat((trg_tensor_max, next_word.unsqueeze(0)), dim=0)

#     #import IPython; IPython.embed();
#     # Organize and return probabilities
#     max_token_prob = torch.stack(max_token_prob)#.transpose(0, 1)
#     #import IPython; IPython.embed();

#     # Remove <SOS> token from target sequences
#     trg_tensor_max = trg_tensor_max.transpose(0, 1)[:, 1:]
#     return trg_tensor_max, max_token_prob


def evaluate_greedy_2(model, stoi, itos, val_dataloader, config, randomize=False):
    """
    Evaluate greedy generation over a validation loader and score it.

    Args:
        model (torch.nn.Module): Trained MMT decoder.
        stoi (dict): Token-to-index map.
        itos (dict): Index-to-token map used to decode tensors back to SMILES.
        val_dataloader (DataLoader): Validation batches; each batch must be a
            ``dict`` with ``"trg_enc_SMI"`` and the spectral keys read by
            ``vgmmt.run_model``.
        config: Experiment namespace (``config.device``, ``config.max_len``,
            ``config.temperature``).
        randomize (bool, optional): If True, shuffle the generated SMILES before
            pairing them with the targets; this destroys the 1:1 pairing and is
            only meant as an ablation baseline. Defaults to False.

    Returns:
        tuple:
            dict: Keys ``gen_conv_SMI_list``, ``trg_conv_SMI_list``,
                ``gen_conv_SMI_list_`` / ``trg_conv_SMI_list_`` (pairs that
                produced a Tanimoto score), ``idx_list``, ``token_probs_list``,
                ``validity_term``, ``tanimoto_scores_``, ``tanimoto_scores_all``,
                ``data_dict_list`` and ``failed``.
            list: Source HSQC tensors.
            list: Source COSY tensors.

    Side effects:
        Prints a tqdm bar; leaves ``config`` untouched (no temperature change).
    """    
    gen_conv_SMI_list = []
    trg_conv_SMI_list = []
    src_HSQC_list = []
    src_COSY_list = []
    token_probs_list = []
    data_dict_list = []
    # generate all the smiles of trg and greedy gen
    for i, data_dict in tqdm(enumerate(val_dataloader)):
        #import IPython; IPython.embed();
        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC, src_COSY = vgmmt.run_model(model,
                                                                       data_dict, 
                                                                       config)
        #print("eval_greedy")
        #print(src_HSQC)
        greedy_tensor, greedy_token_prob = greedy_sequence_2(model, stoi, itos, memory, src_padding_mask, config)
        gen_conv_SMI = hf.tensor_to_smiles(greedy_tensor, itos)
        #gen_conv_SMI, token_probs = ttf.tensor_to_smiles_and_prob(greedy_tensor.squeeze(1), greedy_token_prob, itos)
        token_probs_list.append(greedy_token_prob)
        gen_conv_SMI_list.extend(gen_conv_SMI)
        #gen_conv_SMI_list = gen_conv_SMI_list + gen_conv_SMI
        
        trg_enc_SMI = data_dict["trg_enc_SMI"]
        trg_enc_SMI = trg_enc_SMI.transpose(0, 1)
        trg_SMI_input = trg_enc_SMI[1:, :] # Remove <EOS> token from target sequence
        trg_conv_SMI = hf.tensor_to_smiles(trg_SMI_input, itos)
        trg_conv_SMI_list = trg_conv_SMI_list + trg_conv_SMI
        src_HSQC_list.extend(src_HSQC)
        src_COSY_list.extend(src_COSY)
        data_dict_list.append(data_dict)
    # Calculate validity of gen smiles
    validity_term = hf.get_validity_term(gen_conv_SMI_list) 
    # Calculate tanimoto similarity
    if randomize == True:
        random.shuffle(gen_conv_SMI_list)

    tanimoto_mean, tanimoto_std_dev, failed_pairs, tanimoto_scores_, tanimoto_scores_all, gen_conv_SMI_list_, trg_conv_SMI_list_, idx_list = hf.calculate_tanimoto_similarity_2(gen_conv_SMI_list, trg_conv_SMI_list)

    results_dict = {
            'gen_conv_SMI_list': gen_conv_SMI_list,
            'trg_conv_SMI_list': trg_conv_SMI_list,
            'gen_conv_SMI_list_': gen_conv_SMI_list_,
            'trg_conv_SMI_list_': trg_conv_SMI_list_,            
            'idx_list': idx_list,
            'token_probs_list': token_probs_list,
            'validity_term': validity_term,
            'tanimoto_scores_': tanimoto_scores_,
            'tanimoto_scores_all': tanimoto_scores_all,
            'data_dict_list': data_dict_list,
            'failed':failed_pairs}
    return results_dict, src_HSQC_list, src_COSY_list    



def predict_prop_correct_max_sequence_3(model, stoi, memory, src_padding_mask, trg_enc_SMI, config):
    """
    Predict the per-token probabilities of the ground-truth and arg-max tokens.

    Teacher-forces the *ground-truth* SMILES into the decoder while separately
    tracking the greedy (arg-max) path, so both probability streams are produced
    in a single pass over the sequence.

    Args:
        model (torch.nn.Module): Trained MMT decoder.
        stoi (dict[str, int]): Token-to-index map (``"<SOS>"`` start token).
        memory (torch.Tensor): Encoder memory, shape (seq_len, batch, d_model).
        src_padding_mask (torch.BoolTensor): Encoder padding mask.
        trg_enc_SMI (torch.Tensor): Ground-truth target tokens (``<SOS>`` included).
        config: Experiment namespace (``config.device``, ``config.temperature``).

    Returns:
        tuple: ``trg_tensor`` (ground-truth tokens, ``<SOS>`` removed),
        ``corr_token_prob`` (probability of the ground-truth token per step),
        ``trg_tensor_max`` (arg-max tokens, ``<SOS>`` removed) and
        ``max_token_prob`` (their probabilities).

    Side effects:
        Switches ``model`` to eval mode; all computation runs under
        ``torch.no_grad``.
    """

    # Ensure the model is in evaluation mode
    model.eval()

    # Define the initial target tensor with <SOS> tokens
    N = memory.size(1)
    trg_tensor = torch.full((1, N), stoi["<SOS>"], dtype=torch.long, device=config.device)
    trg_tensor_max = torch.full((1, N), stoi["<SOS>"], dtype=torch.long, device=config.device)

    # Token probabilities containers
    corr_token_prob, max_token_prob = [], []

    # Transpose target encoded SMILES and remove <EOS> token
    trg_enc_SMI_T = trg_enc_SMI.transpose(0, 1)
    real_trg = trg_enc_SMI_T[1:, :]
    
    # Iterate over each token in the target sequence
    with torch.no_grad():
        for idx in range(real_trg.shape[0]):
            # Prepare input for the decoder
            gen_seq_length, N = trg_tensor.shape
            gen_positions = torch.arange(gen_seq_length).unsqueeze(1).expand(gen_seq_length, N).to(config.device)
            embedding_gen = model.dropout2(model.embed_trg(trg_tensor) + model.pe_trg(gen_positions))
            gen_mask = model.generate_square_subsequent_mask(gen_seq_length).to(config.device)

            # Generate output from the decoder
            output = model.decoder(embedding_gen, memory, tgt_mask=gen_mask, memory_key_padding_mask=src_padding_mask)
            output = model.fc_out(output)
            probabilities = F.softmax(output / config.temperature, dim=2)

            # Process token probabilities
            next_word = torch.argmax(probabilities[-1], dim=1)
            max_prob = probabilities[-1].gather(1, next_word.unsqueeze(-1)).squeeze()
            max_token_prob.append(max_prob)

            # Update target tensor with max probability token
            trg_tensor_max = torch.cat((trg_tensor_max, next_word.unsqueeze(0)), dim=0)

            # Correct token probability
            if idx <= real_trg.shape[0]:
                corr_probability = probabilities[-1].gather(1, real_trg[idx].unsqueeze(-1)).squeeze()
                corr_token_prob.append(corr_probability)

            # Update target tensor with actual next token
            next_word = real_trg[idx].unsqueeze(0)
            trg_tensor = torch.cat((trg_tensor, next_word), dim=0)

    #import IPython; IPython.embed();
    # Organize and return probabilities
    max_token_prob = torch.stack(max_token_prob)#.transpose(0, 1)
    corr_token_prob = torch.stack(corr_token_prob)#.transpose(0, 1)
    #import IPython; IPython.embed();

    # Remove <SOS> token from target sequences
    trg_tensor = trg_tensor.transpose(0, 1)[:, 1:]
    trg_tensor_max = trg_tensor_max.transpose(0, 1)[:, 1:]
    return trg_tensor, corr_token_prob, trg_tensor_max, max_token_prob


def run_model_analysis(config, model_MMT, val_dataloader, stoi, itos):
    """Run the full probability + greedy analysis pipeline for one loader.

    Args:
        config: Experiment namespace.
        model_MMT (torch.nn.Module): Trained MMT decoder.
        val_dataloader (DataLoader): Validation batches.
        stoi (dict): Token-to-index map.
        itos (dict): Index-to-token map.

    Returns:
        tuple[dict, dict]: ``prob_dict_results`` from
        :func:`calculate_corr_max_prob`, and the enriched ``results_dict`` from
        :func:`evaluate_greedy_2` (with SGNN HSQC/COSY errors, the successful
        SMILES frame and the prepared spectra tensors added).

    Side effects:
        Prints stage names; runs the SGNN simulator through
        ``ttf.run_sgnn_sim_calculations_if_possible_return_spectra``.

    NOTE (known issue, left unchanged):
        The ``try`` block binds ``final_prob_*`` locals that are never used or
        returned, and it catches everything with a bare ``except``. If the
        aggregation fails, the statistics are silently dropped and the caller
        receives no confidence summary at all.
    """
    print("calculate_corr_max_prob")

    prob_dict_results = calculate_corr_max_prob(config, model_MMT, val_dataloader, stoi, itos)
    try:
        final_prob_max_multi_sum = sum(prob_dict_results["aggregated_max_prob_multi"])
        final_prob_max_multi_avg = statistics.mean(prob_dict_results["aggregated_max_prob_multi"])
        final_prob_max_avg_avg = statistics.mean(prob_dict_results["aggregated_max_prob_avg"])
        final_prob_corr_multi_sum = sum(prob_dict_results["aggregated_corr_prob_multi"])
        final_prob_corr_multi_avg = statistics.mean(prob_dict_results["aggregated_corr_prob_multi"])
        final_prob_corr_avg_avg = statistics.mean(prob_dict_results["aggregated_corr_prob_avg"])
    except:
        print("failed statistics")

    print("evaluate_greedy_2")
    results_dict, src_HSQC_list, src_COSY_list = evaluate_greedy_2(model_MMT, stoi, itos, val_dataloader, config, randomize=False)

    trg_conv_SMI_list = results_dict["trg_conv_SMI_list"]
    gen_conv_SMI_list = results_dict["gen_conv_SMI_list"]

    print("generate_df_for_HSQC_calculations")
    df_succ_smis, src_HSQC_list, failed_list = generate_df_for_HSQC_calculations(gen_conv_SMI_list, trg_conv_SMI_list, src_HSQC_list)
    df_succ_smis, src_COSY_list, failed_list = generate_df_for_COSY_calculations(gen_conv_SMI_list, trg_conv_SMI_list, src_COSY_list)
    tensor_HSQC = prepare_HSQC_data_from_src_2(src_HSQC_list)
    tensor_COSY = prepare_COSY_data_from_src_2(src_COSY_list)

    print("run_sgnn_sim_calculations_if_possible_return_spectra")
    #sgnn_avg_sim_error, HSQC_sim_error_list = ttf.run_sgnn_sim_calculations_if_possible(df_succ_smis, tensor_HSQC, vgmmt.sgnn_means_stds, config)
    avg_sim_error_HSQC, avg_sim_error_COSY, HSQC_sim_error_list, COSY_sim_error_list, batch_data = ttf.run_sgnn_sim_calculations_if_possible_return_spectra(df_succ_smis, tensor_HSQC, tensor_COSY, vgmmt.sgnn_means_stds, config)

    results_dict["HSQC_sim_error_list"] = HSQC_sim_error_list
    results_dict["COSY_sim_error_list"] = COSY_sim_error_list
    results_dict["batch_data"] = batch_data
    results_dict["df_succ_smis"] = df_succ_smis
    results_dict["tensor_HSQC"] = tensor_HSQC
    results_dict["tensor_COSY"] = tensor_COSY
    return prob_dict_results, results_dict



def load_data(config, stoi, stoi_MF, single=True, mode="val"):
    """Build the validation ``DataLoader`` for the requested mode.

    Args:
        config: Experiment namespace (batch size, dataset settings).
        stoi (dict): Token-to-index map for SMILES tokens.
        stoi_MF (dict): Token-to-index map for the molecular-formula aux task.
        single (bool, optional): If True, force ``batch_size=1`` with
            ``drop_last=True`` (one molecule per step, required by the
            per-molecule generation loops; the final sample is silently dropped).
            Defaults to True.
        mode (str, optional): Split selector handed to ``MultimodalData``,
            normally ``"val"``. Defaults to ``"val"``.

    Returns:
        torch.utils.data.DataLoader: Non-shuffled loader using ``collate_fn``
        and :func:`worker_init_fn`.

    Side effects:
        Instantiates ``MultimodalData``, which reads the CSV / spectra files.
    """

    # Load and prepare the validation dataset
    data = MultimodalData(config, stoi, stoi_MF, mode=mode)
    if single:
        dataloader = DataLoader(data, 
                                    batch_size=1, 
                                    shuffle=False, 
                                    collate_fn=collate_fn,
                                    drop_last=True, 
                                    worker_init_fn=worker_init_fn)
    else:
        dataloader = DataLoader(data, 
                            batch_size=config.batch_size, 
                            shuffle=False, 
                            collate_fn=collate_fn,
                            drop_last=False, 
                            worker_init_fn=worker_init_fn)
    return dataloader



def load_MMT_model(config):
    """Load the MMT checkpoint through its Lightning wrapper.

    Args:
        config: Experiment namespace; needs ``config.checkpoint_path`` and the
            architecture fields consumed by ``TransformerMultiGPU``.

    Returns:
        torch.nn.Module: The inner ``.model`` of the loaded ``TransformerMultiGPU``,
        already moved to CUDA.

    Side effects:
        Reads the checkpoint from disk, allocates GPU memory, and leaves the
        returned module in whatever mode the checkpoint restored (usually
        training mode, so callers relying on dropout-free output must call
        ``.eval()`` themselves).

    NOTE (known issue, left unchanged):
        ``.to("cuda")`` is hard-coded instead of using ``config.device``, so this
        fails on CPU-only machines even though the rest of the module is
        device-agnostic.
    """
    # Initialize and load the multi-GPU model

    multi_gpu_model = TransformerMultiGPU(config)
    multi_gpu_model = multi_gpu_model.load_from_checkpoint(config.checkpoint_path, config=config)
    multi_gpu_model.model.to("cuda")

    return multi_gpu_model.model



def load_CLIP_model(config):
    """Load the CLIP checkpoint and return its inner scoring module.

    Args:
        config: Experiment namespace; needs ``config.CLIP_model_path`` and
            ``config.device``.

    Returns:
        torch.nn.Module: The ``CLIP_model`` attribute of the loaded
        ``CLIPMultiGPU`` wrapper, moved to ``config.device``.

    Side effects:
        Reads the checkpoint from disk and allocates device memory.
    """
    CLIP_multi_gpu_model = CLIPMultiGPU(config)
    checkpoint_path = config.CLIP_model_path
    CLIP_model = CLIP_multi_gpu_model.load_from_checkpoint(config=config, checkpoint_path=checkpoint_path)

    #CLIP_model, optimizer = CLIP_make(config, stoi, stoi_MF, itos)
    CLIP_model.to(config.device)

    return CLIP_model.CLIP_model



# def load_BLIP_model(config):

#     BLIP_multi_gpu_model = BLIPMultiGPU(config)
#     checkpoint_path = config.BLIP_model_path
#     BLIP_model = BLIP_multi_gpu_model.load_from_checkpoint(config=config, checkpoint_path=checkpoint_path)

#     #CLIP_model, optimizer = CLIP_make(config, stoi, stoi_MF, itos)
#     BLIP_model.to(config.device)

#     return BLIP_model.BLIP_model


def run_test_mns_performance_CLIP_3(config, 
                                model_MMT,
                                model_CLIP,
                                val_dataloader,                                
                                 stoi, 
                                 itos, 
                                 MW_filter):
    """Multinomial sampling + CLIP + HSQC/COSY scoring for every validation sample.

    For each target it keeps sampling until ``config.multinom_runs`` unique
    canonical molecules are found (or 20 attempts), temporarily raising
    ``config.temperature`` by 0.1 per attempt to escape repetition collapse.

    Args:
        config: Experiment namespace (``multinom_runs``, ``temperature``,
            ``max_len``, ``device``).
        model_MMT (torch.nn.Module): Trained MMT decoder.
        model_CLIP (torch.nn.Module): CLIP wrapper exposing
            ``inference(data_dict, smiles_list)`` -> ``(mean_loss, losses, logits,
            targets, dot_similarity)``.
        val_dataloader (DataLoader): Validation batches.
        stoi (dict): Token-to-index map.
        itos (dict): Index-to-token map.
        MW_filter (bool): If True, keep only generated molecules whose rounded
            molecular weight equals the target's.

    Returns:
        dict[str, list]: ``{target SMILES: [sorted_combined_list, batch_data]}``,
        where each row is ``[SMILES, CLIP loss, dot similarity, token probability,
        Tanimoto]`` sorted by Tanimoto descending; ``[None, None]`` is stored when
        nothing valid was generated.

    Side effects:
        Mutates ``config.temperature`` while sampling and restores it per target;
        runs both models in inference mode; prints progress counters.
    """
    ### Same code as function: run_multinomial_sampling
    n_times = config.multinom_runs
    results_dict = {} #defaultdict(list)
    temperature_orig = config.temperature
    for idx, data_dict in enumerate(val_dataloader):
        if idx % 10 == 0:
            print(idx)
        gen_conv_SMI_list, trg_conv_SMI_list, token_probs_list, src_HSQC_list, prob_list = [], [], [], [], []
        data_dict_dup = rbgvm.duplicate_dict(data_dict, 128)
        trg_enc_SMI = data_dict["trg_enc_SMI"][0]
        trg_conv_SMI = hf.tensor_to_smiles(trg_enc_SMI[1:], itos)
        # to confirm that this smies is valid
        if Chem.MolFromSmiles(trg_conv_SMI) == None:
            continue
        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC, src_COSY = vgmmt.run_model(model_MMT,
                                                                                    data_dict_dup, 
                                                                                    config)
        counter = 1
        while len(gen_conv_SMI_list)<n_times:
            # increase the temperature if not enough different molecules get generated               
            print(counter, len(gen_conv_SMI_list), config.temperature)
            if counter%20==0:
                print(trg_conv_SMI)
                break
            multinom_tensor, multinom_token_prob = multinomial_sequence_multi_2(model_MMT, memory, src_padding_mask, stoi, config)
            gen_conv_SMI, token_probs = hf.tensor_to_smiles_and_prob_2(multinom_tensor, multinom_token_prob, itos)

            # import IPython; IPython.embed();
            gen_conv_SMI, token_probs = filter_probs_and_valid_smiles_and_canonicolize(gen_conv_SMI, token_probs)   ### for 10.000 commented out
            if MW_filter == True:
                gen_conv_SMI, token_probs = filter_for_MW_2(trg_conv_SMI, gen_conv_SMI, token_probs)
            gen_conv_SMI_list.extend(gen_conv_SMI)
            prob_list.extend(token_probs)
            gen_conv_SMI_list, prob_list = deduplicate_smiles(gen_conv_SMI_list, prob_list)
            #gen_conv_SMI_list = list(set(gen_conv_SMI_list)) ### for 10.000 commented out
            counter += 1
            config.temperature = config.temperature + 0.1  
        config.temperature = temperature_orig
        gen_conv_SMI_list = gen_conv_SMI_list[:n_times]
        prob_list = prob_list[:n_times]
        trg_conv_SMI_list = [trg_conv_SMI for i in range(len(gen_conv_SMI_list))]
        data_dict_dup_CLIP = rbgvm.duplicate_dict(data_dict, len(gen_conv_SMI_list))
        if len(gen_conv_SMI_list) != 0:
            mean_loss, losses, logits, targets, dot_similarity= model_CLIP.inference(data_dict_dup_CLIP, 
                                                                                gen_conv_SMI_list)

            combined_list = [[smile, num.item(), dot_sim.item(), prob] for smile, num, dot_sim, prob in zip(gen_conv_SMI_list, losses, dot_similarity, prob_list)]
            ### Sort by the lowest similarity
            #sorted_list = sorted(combined_list, key=lambda x: x[1])

            combined_list, failed_combined_list = add_tanimoto_similarity(trg_conv_SMI, combined_list)
            combined_list, batch_data = rbgvm.add_HSQC_COSY_error(config, combined_list, data_dict_dup, gen_conv_SMI_list, trg_conv_SMI, config.multinom_runs) # config.MMT_batch

            sorted_list = sorted(combined_list, key=lambda x: -x[4]) # SMILES = 0, losses =1, dot_sim= 2, propb = 3, tanimoto = 4

            results_dict[trg_conv_SMI] = [sorted_list, batch_data]
        else:
            results_dict[trg_conv_SMI] = [None, None]

    return results_dict

def run_test_performance_CLIP_greedy_3(config, 
                                 model_MMT,
                                 model_CLIP,
                                 val_dataloader,                                        
                                 stoi, 
                                 stoi_MF, 
                                 itos, 
                                 itos_MF):
    """Greedy-decoding evaluation with CLIP dot similarity and SGNN HSQC error.

    Args:
        config: Experiment namespace.
        model_MMT (torch.nn.Module): Trained MMT decoder.
        model_CLIP (torch.nn.Module): CLIP wrapper exposing ``inference``.
        val_dataloader (DataLoader): Validation batches.
        stoi, stoi_MF, itos, itos_MF (dict): Token maps; the ``*_MF`` pair is
            accepted for call-site symmetry and is not used here.

    Returns:
        tuple[dict, list]: ``{target SMILES: [combined_list, batch_data]}`` where
        rows are ``[gen SMILES, CLIP loss, dot similarity, Tanimoto, HSQC error]``,
        plus the list of targets whose Tanimoto computation failed.

    Side effects:
        Runs MMT and CLIP inference; prints the batch index every 10 batches.

    NOTE (known issue, left unchanged):
        ``vgmmt.run_model`` returns **six** values, but this line unpacks only
        five, so every call raises ``ValueError: too many values to unpack`` and
        the function cannot run.
    """


    results_dict = defaultdict(list)
    gen_conv_SMI_list = []
    token_probs_list = []
    src_HSQC_list = []
    failed = []

    # generate all the smiles of trg and greedy gen
    for idx, data_dict in enumerate(val_dataloader):
        if idx % 10 == 0:
            print(idx)
        data_dict_dup = rbgvm.duplicate_dict(data_dict, 1)  # Maybe should hardcode it here as 64 - it will always cut it down to the number needed with ntimes

        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC = vgmmt.run_model(model_MMT,
                                                                        data_dict_dup, 
                                                                        config)
        trg_enc_SMI = data_dict["trg_enc_SMI"][0]
        trg_conv_SMI = hf.tensor_to_smiles(trg_enc_SMI[1:], itos)

        greedy_tensor, greedy_token_prob = greedy_sequence_2(model_MMT, stoi, itos, memory, src_padding_mask, config)
        gen_conv_SMI, token_probs = hf.tensor_to_smiles_and_prob(greedy_tensor.squeeze(1), greedy_token_prob, itos)
        tan_sim = try_calculate_tanimoto_from_two_smiles(trg_conv_SMI, gen_conv_SMI, 512, extra_info = False)
        mean_loss, losses, logits, targets, dot_similarity = model_CLIP.inference(data_dict_dup, gen_conv_SMI)

        sgnn_avg_sim_error, sim_error_list, batch_data = rbgvm.calculate_HSQC_error(config, data_dict_dup, gen_conv_SMI)
        combined_list =[gen_conv_SMI, losses.item(), dot_similarity.item(), tan_sim, sgnn_avg_sim_error]
        if tan_sim == None:
            failed.append([trg_conv_SMI, gen_conv_SMI, combined_list, batch_data])
            continue
        else:
            results_dict[trg_conv_SMI] = [combined_list, batch_data]
    #if i == config.n_samples:
    #    break
    #import IPython; IPython.embed();

    return results_dict, failed

def predict_corr_max_performance_metric(trg_tensor, corr_token_prob, max_token_prob, stoi):
    """
    Aggregate token probabilities of one sequence into four summary scores.

    Args:
        trg_tensor (torch.Tensor): Ground-truth token tensor; only
            ``trg_tensor[0]`` (first sequence of the batch) is inspected.
        corr_token_prob (torch.Tensor): Probability of the ground-truth token per
            step, aligned with ``trg_tensor[0]``.
        max_token_prob (torch.Tensor): Probability of the arg-max token per step.
        stoi (dict[str, int]): Token-to-index map; ``"<EOS>"`` stops the scan.

    Returns:
        tuple[dict, dict]: ``prop_dict`` with ``prob_corr_multi``,
        ``prob_corr_avg``, ``prob_max_multi``, ``prob_max_avg``, and
        ``sample_dict`` with ``sample_prob_list_corr`` / ``sample_prob_list_max``.

    Side effects:
        None.

    NOTE (known issue, left unchanged):
        Only the first sequence of the batch is scored, and the ``zip`` over
        three differently-shaped inputs truncates to the shortest one, so any
        remaining molecules are silently ignored.
    """

    # Initialize dictionaries for storing results
    prop_dict, sample_dict = {}, {}
    sample_prob_list_corr, sample_prob_list_max = [], []
    prob_corr_multi, prob_corr_avg, prob_max_multi, prob_max_avg = [], [], [], []

    # Iterate over each sequence to calculate probabilities
    seq_corr_probs, seq_max_probs, seq_multinom_probs = [], [], []

    for corr_prob, max_prob, token in zip(corr_token_prob, max_token_prob, trg_tensor[0]):
        # Initialize lists for individual sequence probabilities
        #import IPython; IPython.embed();
        # Iterate over each token in the sequence
        #for idx, (corr_prob, max_prob, multinom_prob, token) in enumerate(zip(corr_probs, max_probs, multinom_probs, tokens)):
        if token == stoi["<EOS>"]:  # End of sequence
            break
        seq_corr_probs.append(corr_prob.item())
        seq_max_probs.append(max_prob.item())

    # Calculate and append aggregated probabilities
    prob_corr_multi = reduce(operator.mul, seq_corr_probs, 1)
    prob_corr_avg = statistics.mean(seq_corr_probs) if seq_corr_probs else 0
    prob_max_multi = reduce(operator.mul, seq_max_probs, 1)
    prob_max_avg = statistics.mean(seq_max_probs) if seq_max_probs else 0

    # Populate dictionaries with calculated probabilities
    sample_dict = {
        "sample_prob_list_corr": seq_corr_probs,
        "sample_prob_list_max": seq_max_probs    
        }

    prop_dict = {
        "prob_corr_multi": prob_corr_multi,
        "prob_corr_avg": prob_corr_avg,
        "prob_max_multi": prob_max_multi,
        "prob_max_avg": prob_max_avg
        }

    return prop_dict, sample_dict

## There is something wrong with the predict_corr_max_performance_metric function
def calculate_corr_max_prob_2(config, model, stoi, val_dataloader, gen_num):
    """Batch-wise wrapper around :func:`predict_corr_max_performance_metric`.

    Args:
        config: Experiment namespace.
        model (torch.nn.Module): Trained MMT decoder.
        stoi (dict): Token-to-index map.
        val_dataloader (DataLoader): Validation batches.
        gen_num (int): Kept for API compatibility; unused (callers always pass 1).

    Returns:
        tuple[dict, dict]: Aggregated per-batch probability lists, and the
        ``sample_dict`` of the last processed batch.

    Side effects:
        Runs the model in eval/no-grad mode; prints a tqdm bar.

    NOTE (known issue, left unchanged):
        ``sample_dict`` is only bound inside the loop, so an empty loader raises
        ``UnboundLocalError`` on ``return``; even for a non-empty loader only the
        last batch's per-sample probabilities are returned.
    """
    
    prob_dict_results = {}
    aggregated_corr_prob_multi = []
    aggregated_corr_prob_avg = []
    aggregated_max_prob_multi = []
    aggregated_max_prob_avg = []
    #sample_dict = {}
    #for _ in range(2):  # Num_Runs is the number of times you want to run the entire process for randomized smiles

    for idx, data_dict in enumerate(tqdm(val_dataloader)):
        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC, src_COSY = vgmmt.run_model(model, data_dict, config) 
        trg_tensor, corr_token_prob, trg_tensor_max, max_token_prob = predict_prop_correct_max_sequence_3(model, stoi, memory, src_padding_mask, trg_enc_SMI, config)
        #import IPython; IPython.embed();

        prop_dict, sample_dict = predict_corr_max_performance_metric(trg_tensor, corr_token_prob, max_token_prob, stoi)
        aggregated_corr_prob_multi.append(prop_dict["prob_corr_multi"])
        aggregated_corr_prob_avg.append(prop_dict["prob_corr_avg"])  
        aggregated_max_prob_multi.append(prop_dict["prob_max_multi"])
        aggregated_max_prob_avg.append(prop_dict["prob_max_avg"]) 
    #import IPython; IPython.embed();

    prob_dict_results["aggregated_corr_prob_multi"] = aggregated_corr_prob_multi
    prob_dict_results["aggregated_corr_prob_avg"] = aggregated_corr_prob_avg
    prob_dict_results["aggregated_max_prob_multi"] = aggregated_max_prob_multi
    prob_dict_results["aggregated_max_prob_avg"] = aggregated_max_prob_avg
    #print("calculate_corr_max_prob_2")
    #import IPython; IPython.embed();

    return prob_dict_results, sample_dict


def run_test_performance_CLIP_3(config, 
                                model_MMT,
                                val_dataloader,                                   
                                stoi):
    """Collect model-confidence statistics for a validation split.

    Args:
        config: Experiment namespace.
        model_MMT (torch.nn.Module): Trained MMT decoder.
        val_dataloader (DataLoader): Validation batches.
        stoi (dict): Token-to-index map.

    Returns:
        dict[str, list[float]]: ``statistics_multiplication_avg``,
        ``statistics_multiplication_sum`` and ``statistics_avg_avg``, each a
        ``[correct-token, max-token]`` pair.

    Side effects:
        Runs the model; prints ``"failed statistics"`` on failure.

    NOTE (known issue, left unchanged):
        The ``try`` block is the only place that binds ``final_prob_*``; when the
        aggregation raises (e.g. ``statistics.mean`` of an empty list) the
        following ``total_results[...] = ...`` lines raise ``NameError`` and mask
        the original error.
    """


    total_results = {}
    # Number of times to duplicate each tensor 
    gen_num = 1 # Multinomial sampling number
    prob_dict_results, _ = calculate_corr_max_prob_2(config, model_MMT, stoi, val_dataloader, gen_num)
    try:
        final_prob_max_multi_sum = sum(prob_dict_results["aggregated_max_prob_multi"])
        final_prob_max_multi_avg = statistics.mean(prob_dict_results["aggregated_max_prob_multi"])
        final_prob_max_avg_avg = statistics.mean(prob_dict_results["aggregated_max_prob_avg"])
        final_prob_corr_multi_sum = sum(prob_dict_results["aggregated_corr_prob_multi"])
        final_prob_corr_multi_avg = statistics.mean(prob_dict_results["aggregated_corr_prob_multi"])
        final_prob_corr_avg_avg = statistics.mean(prob_dict_results["aggregated_corr_prob_avg"])
    except:
        print("failed statistics")
    total_results["statistics_multiplication_avg"] = [final_prob_corr_multi_avg,
                                                    final_prob_max_multi_avg]

    total_results["statistics_multiplication_sum"] = [final_prob_corr_multi_sum,
                                                    final_prob_max_multi_sum]

    total_results["statistics_avg_avg"] = [final_prob_corr_avg_avg,
                                            final_prob_max_avg_avg]
    return total_results


#__________________________________________________
### Sample 1000 unique molecules and plot their tanimoto similartyy 
### compared to the target
def run_multinomial_sampling(config, model_MMT, val_dataloader, itos, stoi, MW_filter=False):
    """Multinomial-sampling baseline with per-target Tanimoto statistics.

    Args:
        config: Experiment namespace (``multinom_runs``, ``temperature``,
            ``max_len``, ``device``).
        model_MMT (torch.nn.Module): Trained MMT decoder.
        val_dataloader (DataLoader): Validation batches.
        itos (dict): Index-to-token map.
        stoi (dict): Token-to-index map.
        MW_filter (bool, optional): Keep only same-molecular-weight generations.
            Defaults to False.

    Returns:
        tuple: ``(config, results_dict)`` where ``results_dict[idx]`` is a
        single-element list holding ``gen_conv_SMI_list``, ``trg_conv_SMI_list``,
        ``tanimoto_sim``, ``tanimoto_mean``, ``tanimoto_std_dev``, ``failed`` and
        ``prob_list``.

    Side effects:
        Mutates ``config.temperature`` (+0.1 per attempt) and restores it per
        sample; prints a batch index every 10 batches.
    """
    n_times = config.multinom_runs
    results_dict = defaultdict(list)
    temperature_orig = config.temperature
    for idx, data_dict in tqdm(enumerate(val_dataloader)):
        if idx % 10 ==0:
            print(idx)

        gen_conv_SMI_list, trg_conv_SMI_list, token_probs_list, src_HSQC_list, prob_list = [], [], [], [], []
        data_dict_dup = rbgvm.duplicate_dict(data_dict, 128)
        trg_enc_SMI = data_dict["trg_enc_SMI"][0]
        trg_conv_SMI = hf.tensor_to_smiles(trg_enc_SMI[1:], itos)
        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC = vgmmt.run_model(model_MMT,
                                                                   data_dict_dup, 
                                                                   config)
        counter = 1
        while len(gen_conv_SMI_list)<n_times:
            # increase the temperature if not enough different molecules get generated               
            if counter%20==0:
                print(trg_conv_SMI)
                break
            multinom_tensor, multinom_token_prob = multinomial_sequence_multi_2(model_MMT, memory, src_padding_mask, stoi, config)
            gen_conv_SMI, token_probs = hf.tensor_to_smiles_and_prob_2(multinom_tensor, multinom_token_prob, itos)

            # import IPython; IPython.embed();
            gen_conv_SMI, token_probs = filter_probs_and_valid_smiles_and_canonicolize(gen_conv_SMI, token_probs)   ### for 10.000 commented out
            if MW_filter == True:
                gen_conv_SMI, token_probs = filter_for_MW_2(trg_conv_SMI, gen_conv_SMI, token_probs)
            gen_conv_SMI_list.extend(gen_conv_SMI)
            prob_list.extend(token_probs)
            gen_conv_SMI_list, prob_list = deduplicate_smiles(gen_conv_SMI_list, prob_list)
            #gen_conv_SMI_list = list(set(gen_conv_SMI_list)) ### for 10.000 commented out
            counter += 1
            config.temperature = config.temperature + 0.1    
        gen_conv_SMI_list = gen_conv_SMI_list[:n_times]
        prob_list = prob_list[:n_times]
        trg_conv_SMI_list = [trg_conv_SMI for i in range(len(gen_conv_SMI_list))]

        tanimoto_mean, tanimoto_std_dev, failed, tanimoto_list_all = vgmmt.calculate_tanimoto_similarity(gen_conv_SMI_list, trg_conv_SMI_list)
        results_dict[idx].append({
            'gen_conv_SMI_list': gen_conv_SMI_list,
            'trg_conv_SMI_list': trg_conv_SMI_list,
            'tanimoto_sim': tanimoto_list_all,
            'tanimoto_mean': tanimoto_mean,
            'tanimoto_std_dev': tanimoto_std_dev,
            'failed': failed,
            'prob_list': prob_list,
        })
        config.temperature = temperature_orig 
    return config, results_dict



# Multinomial
def multinomial_sequence_multi_2(model, memory, src_padding_mask, stoi, config):
    """Sample one token per step from ``softmax(logits / temperature)``.

    Args:
        model (torch.nn.Module): Trained MMT decoder.
        memory (torch.Tensor): Encoder memory, shape (seq_len, batch, d_model).
        src_padding_mask (torch.BoolTensor): Encoder padding mask.
        stoi (dict[str, int]): Token-to-index map (``"<SOS>"`` start token).
        config: Experiment namespace (``max_len``, ``temperature``, ``device``).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``multinom_tensor`` of shape
        (max_len, batch) with the sampled token ids and
        ``multinom_token_prob`` with the probability that was sampled at each
        step. The ``<SOS>`` row is removed from both.

    Side effects:
        Switches ``model`` to eval mode; draws from the global torch RNG
        (``torch.multinomial``), so runs are reproducible only after seeding.

    NOTE (known issue, left unchanged):
        The loop never stops at ``<EOS>``; it always decodes ``config.max_len``
        tokens and relies on downstream SMILES parsing to reject the trailing
        tokens.
    """
    # Initialization

    model.eval()
    N = memory.size(1)
    multinom_tensor = torch.full((1, N), stoi["<SOS>"], dtype=torch.long, device=config.device)
    multinom_token_prob = []  

    # Sequence Prediction
    with torch.no_grad():
        for idx in range(0, config.max_len):
            # [The same logic you already had]
            gen_seq_length, N = multinom_tensor.shape
            gen_positions = (
                torch.arange(0, gen_seq_length)
                .unsqueeze(1)
                .expand(gen_seq_length, N)
                .to(config.device))

            embedding_gen = model.dropout2((model.embed_trg(multinom_tensor) + model.pe_trg(gen_positions)))
            gen_mask = model.generate_square_subsequent_mask(gen_seq_length).to(config.device)
            output = model.decoder(embedding_gen, memory, tgt_mask=gen_mask, memory_key_padding_mask=src_padding_mask)
            output = model.fc_out(output)

            probabilities = F.softmax(output / config.temperature, dim=2)
            ## Capturing the probability of the next predicted token with multinomial sampling
            next_word = torch.multinomial(probabilities[-1, :, :], 1)  
            sel_prob = probabilities[-1, :, :].gather(1, next_word).squeeze()  # Get the probability of the predicted token
            multinom_token_prob.append(sel_prob)
            next_word = next_word.squeeze(1).unsqueeze(0)
            multinom_tensor = torch.cat((multinom_tensor, next_word), dim=0)
    # import IPython; IPython.embed();
    multinom_token_prob = torch.stack(multinom_token_prob) 

    # remove "SOS" token
    multinom_tensor = multinom_tensor[1:,:]
    multinom_token_prob = multinom_token_prob[1:,:]
    
    return multinom_tensor, multinom_token_prob


def run_greedy_sampling(config, model_MMT, val_dataloader, itos, stoi):
    """Greedy-decode a whole loader and report Tanimoto statistics.

    Args:
        config: Experiment namespace.
        model_MMT (torch.nn.Module): Trained MMT decoder.
        val_dataloader (DataLoader): Validation batches.
        itos (dict): Index-to-token map.
        stoi (dict): Token-to-index map.

    Returns:
        tuple: ``(config, results_dict)`` with ``gen_conv_SMI_list``,
        ``trg_conv_SMI_list``, ``tanimoto_sim``, ``tanimoto_mean``,
        ``tanimoto_std_dev``, ``failed``, ``prob_list``, ``src_HSQC_list`` and
        ``src_COSY_list``.

    Side effects:
        Runs MMT inference only; does not mutate ``config``.
    """

    results_dict = defaultdict(list)
    #model_MMT, val_dataloader = vgmmt.load_data_and_MMT_model(config, stoi, stoi_MF, single=True, mode="val")

    gen_conv_SMI_list, trg_conv_SMI_list, prob_list, src_HSQC_list, src_COSY_list = [], [], [], [], []
    for idx, data_dict in enumerate(val_dataloader):
        trg_enc_SMI = data_dict["trg_enc_SMI"]
        trg_conv_SMI = hf.tensor_to_smiles(trg_enc_SMI.T[1:], itos)
        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC, src_COSY = vgmmt.run_model(model_MMT,
                                                                   data_dict, 
                                                                   config)

        greedy_tensor, greedy_token_prob = greedy_sequence_2(model_MMT, stoi, itos, memory, src_padding_mask, config)

        gen_conv_SMI, token_probs = hf.tensor_to_smiles_and_prob(greedy_tensor, greedy_token_prob.T, itos)

        gen_conv_SMI_list.extend(gen_conv_SMI)
        trg_conv_SMI_list.extend(trg_conv_SMI)
        prob_list.extend(token_probs)
        src_HSQC_list.extend(src_HSQC)
        src_COSY_list.extend(src_COSY)
        
    tanimoto_mean, tanimoto_std_dev, failed, tanimoto_list_all = vgmmt.calculate_tanimoto_similarity(gen_conv_SMI_list, trg_conv_SMI_list)
    results_dict = {
        'gen_conv_SMI_list': gen_conv_SMI_list,
        'trg_conv_SMI_list': trg_conv_SMI_list,
        'tanimoto_sim': tanimoto_list_all,
        'tanimoto_mean': tanimoto_mean,
        'tanimoto_std_dev': tanimoto_std_dev,
        'failed': failed,
        'prob_list': prob_list,
        "src_HSQC_list": src_HSQC_list,
        "src_COSY_list": src_COSY_list,
        }
    
    return config, results_dict

    
# def greedy_sequence_2(model, stoi, itos, memory, src_padding_mask, config):
#     """
#     Generates a sequence of tokens using a greedy approach.

#     Parameters:
#     model (torch.nn.Module): The trained model for sequence generation.
#     stoi (dict): Mapping of tokens to indices.
#     itos (dict): Mapping of indices to tokens.
#     memory (Tensor): Memory tensor from the model.
#     src_padding_mask (Tensor): Source padding mask.
#     config (object): Configuration containing model parameters.

#     Returns:
#     tuple: Tensor of generated tokens and their probabilities.
#     """
#     model.eval()
#     N = memory.size(1)
#     greedy_tensor = torch.full((1, N), stoi["<SOS>"], dtype=torch.long, device=config.device)
#     greedy_token_prob = []

#     with torch.no_grad():
#         for _ in range(config.max_len):
#             gen_seq_length = greedy_tensor.size(0)
#             gen_positions = torch.arange(gen_seq_length, device=config.device).unsqueeze(1).expand(gen_seq_length, N)
#             embedding_gen = model.embed_trg(greedy_tensor) + model.pe_trg(gen_positions)
#             if model.training:
#                 embedding_gen = model.dropout2((embedding_gen))
#             gen_mask = model.generate_square_subsequent_mask(gen_seq_length).to(config.device)

#            #import IPython; IPython.embed();
#             output = model.decoder(embedding_gen, memory, tgt_mask=gen_mask, memory_key_padding_mask=src_padding_mask)
           
#             output = model.fc_out(output)
#             probabilities = F.softmax(output / config.temperature, dim=2)

#             ## Capturing the probability of the best next predicted token
#             next_word = torch.argmax(probabilities[-1, :, :], dim=1)
#             max_prob = probabilities[-1, :, :].gather(1, next_word.unsqueeze(-1)).squeeze()
#             greedy_token_prob.append(max_prob)
#             next_word = next_word.unsqueeze(0)

#             greedy_tensor = torch.cat((greedy_tensor, next_word), dim=0)
#             if (next_word == 0).all():
#                 break
                


#     greedy_token_prob = torch.stack(greedy_token_prob) 
#     #greedy_token_prob = greedy_token_prob.transpose(0, 1)
#     # import IPython; IPython.embed();

#     # remove "SOS" token
#     greedy_tensor = greedy_tensor[1:,:]
#     greedy_token_prob = greedy_token_prob[1:,:]
    
#     return greedy_tensor, greedy_token_prob


# def greedy_sequence_2(model, stoi, itos, memory, src_padding_mask, config):
#     """
#     Generates a sequence of tokens using a greedy approach.

#     Parameters:
#     model (torch.nn.Module): The trained model for sequence generation.
#     stoi (dict): Mapping of tokens to indices.
#     itos (dict): Mapping of indices to tokens.
#     memory (Tensor): Memory tensor from the model.
#     src_padding_mask (Tensor): Source padding mask.
#     config (object): Configuration containing model parameters.

#     Returns:
#     tuple: Tensor of generated tokens and their probabilities.
#     """
#     model.eval()
#     N = memory.size(1)
#     greedy_tensor = torch.full((1, N), stoi["<SOS>"], dtype=torch.long, device=config.device)
#     greedy_token_prob = []

#     with torch.no_grad():
#         for _ in range(config.max_len):
#             gen_seq_length = greedy_tensor.size(0)
#             gen_positions = torch.arange(gen_seq_length, device=config.device).unsqueeze(1).expand(gen_seq_length, N)
#             embedding_gen = model.dropout2((model.embed_trg(greedy_tensor) + model.pe_trg(gen_positions)))
#             gen_mask = model.generate_square_subsequent_mask(gen_seq_length).to(config.device)

#            #import IPython; IPython.embed();
#             output = model.decoder(embedding_gen, memory, tgt_mask=gen_mask, memory_key_padding_mask=src_padding_mask)
           
#             output = model.fc_out(output)
#             probabilities = F.softmax(output / config.temperature, dim=2)

#             ## Capturing the probability of the best next predicted token
#             next_word = torch.argmax(probabilities[-1, :, :], dim=1)
#             max_prob = probabilities[-1, :, :].gather(1, next_word.unsqueeze(-1)).squeeze()
#             greedy_token_prob.append(max_prob)
#             next_word = next_word.unsqueeze(0)

#             greedy_tensor = torch.cat((greedy_tensor, next_word), dim=0)
#             if (next_word == 0).all():
#                 break
                


#     greedy_token_prob = torch.stack(greedy_token_prob) 
#     #greedy_token_prob = greedy_token_prob.transpose(0, 1)

#     # remove "SOS" token
#     greedy_tensor = greedy_tensor[1:,:]
#     greedy_token_prob = greedy_token_prob[:,:]
    
#     return greedy_tensor, greedy_token_prob


def greedy_sequence_2(model, stoi, itos, memory, src_padding_mask, config):
    """
    Generate a sequence of tokens with a greedy (arg-max) decoder.

    Args:
        model (torch.nn.Module): Trained MMT decoder.
        stoi (dict[str, int]): Token-to-index map (``"<SOS>"`` start token).
        itos (dict[int, str]): Index-to-token map (accepted but unused; decoding
            happens in the callers).
        memory (torch.Tensor): Encoder memory, shape (seq_len, batch, d_model).
        src_padding_mask (torch.BoolTensor): Encoder padding mask.
        config: Experiment namespace (``max_len``, ``temperature``, ``device``).

    Returns:
        tuple[torch.Tensor, torch.Tensor]: ``greedy_tensor`` of shape
        (steps+1, batch) with the ``<SOS>`` row removed, and
        ``greedy_token_prob`` holding the arg-max probability per step. For a
        single molecule the probability tensor is unsqueezed to (steps, 1) so
        callers can always transpose it the same way.

    Side effects:
        Switches ``model`` to eval mode; no RNG use (fully deterministic).

    NOTE (known issue, left unchanged):
        The stopping test ``(next_word == 0).all()`` compares against token id 0
        instead of ``stoi["<EOS>"]``. Unless id 0 happens to be the EOS index the
        loop always runs the full ``config.max_len`` steps, and it stops only
        when *all* batch rows emit 0 at the same step.
    """
    model.eval()
    N = memory.size(1)  # Batch size
    greedy_tensor = torch.full((1, N), stoi["<SOS>"], dtype=torch.long, device=config.device)
    greedy_token_prob = []

    with torch.no_grad():
        for _ in range(config.max_len):
            gen_seq_length = greedy_tensor.size(0)
            gen_positions = torch.arange(gen_seq_length, device=config.device).unsqueeze(1).expand(gen_seq_length, N)
            embedding_gen = model.embed_trg(greedy_tensor) + model.pe_trg(gen_positions)
            if model.training:
                embedding_gen = model.dropout2((embedding_gen))
            gen_mask = model.generate_square_subsequent_mask(gen_seq_length).to(config.device)

            output = model.decoder(embedding_gen, memory, tgt_mask=gen_mask, memory_key_padding_mask=src_padding_mask)
           
            output = model.fc_out(output)
            probabilities = F.softmax(output / config.temperature, dim=2)

            next_word = torch.argmax(probabilities[-1, :, :], dim=1)
            max_prob = probabilities[-1, :, :].gather(1, next_word.unsqueeze(-1)).squeeze()
            greedy_token_prob.append(max_prob)
            next_word = next_word.unsqueeze(0)

            greedy_tensor = torch.cat((greedy_tensor, next_word), dim=0)
            if (next_word == 0).all():
                break

    greedy_token_prob = torch.stack(greedy_token_prob)
    #import IPython; IPython.embed();

    # Remove "SOS" token
    greedy_tensor = greedy_tensor[1:]
    greedy_token_prob = greedy_token_prob


    # Handle single molecule case
    if N == 1:
        greedy_tensor = greedy_tensor#.squeeze(1)
        greedy_token_prob = greedy_token_prob.unsqueeze(-1)
    # else:
    #     greedy_tensor = greedy_tensor.transpose(0, 1)
    #     greedy_token_prob = greedy_token_prob.transpose(0, 1)
    
    return greedy_tensor, greedy_token_prob


def deduplicate_smiles(smiles_list, prob_list):
    """Remove duplicate SMILES, keeping the probability of the first occurrence.

    Args:
        smiles_list (list[str]): Possibly duplicated SMILES strings.
        prob_list (list[float]): Probabilities aligned 1:1 with ``smiles_list``.

    Returns:
        tuple[list[str], list[float]]: Deduplicated SMILES and their
        probabilities in first-occurrence (generation) order.

    Side effects:
        None; the insertion-ordered dict preserves the sampling order.
    """
    # Create a dictionary to hold unique smiles and their corresponding probabilities
    unique_smiles = {}

    # Loop over the SMILES and their corresponding probabilities
    for smi, prob in zip(smiles_list, prob_list):
        if smi not in unique_smiles:
            unique_smiles[smi] = prob

    # Extracting the deduplicated lists
    deduped_smiles = list(unique_smiles.keys())
    deduped_probs = list(unique_smiles.values())

    return deduped_smiles, deduped_probs

# Function to filter valid SMILES
def filter_probs_and_valid_smiles_and_canonicolize(smiles_list, token_probs, canonical=True, isomericSmiles=False):
    """Drop unparseable SMILES, canonicalise the rest and keep their probabilities.

    Args:
        smiles_list (list[str]): Raw decoded SMILES.
        token_probs (list[float]): Probabilities aligned with ``smiles_list``.
        canonical (bool, optional): Forwarded to ``Chem.MolToSmiles``. Defaults
            to True.
        isomericSmiles (bool, optional): Keep stereochemistry when True. Defaults
            to False.

    Returns:
        tuple[list[str], list[float]]: Valid canonical SMILES and their matching
        probabilities; entries rejected by ``Chem.MolFromSmiles`` are dropped.

    Side effects:
        None (``MolToSmiles`` is called with ``doRandom=False``).
    """
    valid_smiles = []
    valid_token_probs = []
    for smi, prob in zip(smiles_list, token_probs):
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            gen_smi = Chem.MolToSmiles(mol, canonical=canonical, doRandom=False, isomericSmiles=isomericSmiles)
            valid_smiles.append(gen_smi)
            valid_token_probs.append(prob)
    return valid_smiles, valid_token_probs


# Function to calculate the rounded molecular weight
def calc_rounded_mw(smi):
    """Molecular weight of ``smi`` rounded to the nearest integer.

    Args:
        smi (str): SMILES string.

    Returns:
        int | None: ``round(Descriptors.MolWt(mol))``, or ``None`` when RDKit
        cannot build the molecule (the bare ``except`` also hides any other
        error, e.g. ``MolWt(None)``).

    Side effects:
        None.
    """
    try:
        mol = Chem.MolFromSmiles(smi)
        return round(Descriptors.MolWt(mol))
    except:
        return None



def filter_for_MW(trg_conv_SMI, gen_conv_SMI):
    """Keep generated SMILES whose rounded molecular weight matches the target.

    Args:
        trg_conv_SMI (str): Ground-truth SMILES.
        gen_conv_SMI (list[str]): Generated SMILES to filter.

    Returns:
        list[str]: Sub-list for which
        ``calc_rounded_mw(smi) == calc_rounded_mw(trg_conv_SMI)``.

    Side effects:
        None; the target weight is recomputed for every candidate.
    """

    # Calculate the rounded molecular weight of the target molecule
    trg_mw = calc_rounded_mw(trg_conv_SMI)

    # Filter the list based on molecular weight
    filtered_gen_smis = [smi for smi in gen_conv_SMI if calc_rounded_mw(smi) == trg_mw]  
    return filtered_gen_smis


def filter_for_MW_2(trg_conv_SMI, gen_conv_SMI, prob_list):
    """Same as :func:`filter_for_MW`, but also returns the matching probabilities.

    Args:
        trg_conv_SMI (str): Ground-truth SMILES.
        gen_conv_SMI (list[str]): Generated SMILES.
        prob_list (list[float]): Probabilities aligned with ``gen_conv_SMI``.

    Returns:
        tuple[list[str], list[float]]: Filtered SMILES and their probabilities.

    Side effects:
        None; ``calc_rounded_mw`` runs twice per candidate (once per list
        comprehension).
    """

    # Calculate the rounded molecular weight of the target molecule
    trg_mw = calc_rounded_mw(trg_conv_SMI)

    # Filter the list based on molecular weight
    filtered_gen_smis = [smi for (smi, token_prob) in zip(gen_conv_SMI, prob_list) if calc_rounded_mw(smi) == trg_mw]  
    filtered_prob_list = [token_prob for (smi, token_prob) in zip(gen_conv_SMI, prob_list) if calc_rounded_mw(smi) == trg_mw]  
    return filtered_gen_smis, filtered_prob_list

    
def filter_dict_results_for_MW(results_dict):
    """Rebuild Tanimoto statistics after applying the molecular-weight filter.

    Args:
        results_dict (dict[int, list[dict]]): Output of
            :func:`run_multinomial_sampling` (``idx -> [{'tanimoto_sim': ...}]``).

    Returns:
        tuple[list[list[float]], list[list]]: Per-target Tanimoto lists and the
        ``[filtered_gen_smis, trg_conv_SMI_list]`` pairs used to compute them.

    Side effects:
        None; assumes every ``result`` dict has ``trg_conv_SMI_list`` and
        ``gen_conv_SMI_list`` keys (``KeyError`` otherwise).
    """
    tanimoto_values = []
    gen_trg_smi_lists = []
    for i, idx in enumerate(results_dict.keys()):
        # Collect all tanimoto_list_all for the current idx

        for result in results_dict[idx]:
            trg_smi = result["trg_conv_SMI_list"][0]
            smi_list = result["gen_conv_SMI_list"]

            # Calculate the rounded molecular weight of the target molecule
            trg_mw = calc_rounded_mw(trg_smi)

            # Filter the list based on molecular weight
            filtered_gen_smis = [smi for smi in smi_list if calc_rounded_mw(smi) == trg_mw]  
            trg_conv_SMI_list = [trg_smi for i in range(len(filtered_gen_smis))]
            gen_trg_smi_lists.append([filtered_gen_smis, trg_conv_SMI_list])
            tanimoto_mean, tanimoto_std_dev, failed, tanimoto_list_all = vgmmt.calculate_tanimoto_similarity(filtered_gen_smis, trg_conv_SMI_list)
            tanimoto_values.append(tanimoto_list_all)
    return tanimoto_values, gen_trg_smi_lists


def plot_hist_MN_sampling(config, results_dict):
    """Overlay one Tanimoto histogram per target on a single figure.

    Args:
        config: Experiment namespace (unused, kept for call-site symmetry).
        results_dict (dict[int, list[dict]]): Output of
            :func:`run_multinomial_sampling`; ``'tanimoto_sim'`` lists are pooled
            per target.

    Returns:
        None

    Side effects:
        Creates and shows a matplotlib figure (blocking ``plt.show()``).
    """
    # Number of unique idx values
    num_idx = len(results_dict)

    # Create a figure with subplots (if there are many idx, you might need to adjust the size and layout)
    plt.figure(figsize=(10, 6))

    for i, idx in enumerate(results_dict.keys()):
        # Collect all tanimoto_list_all for the current idx
        tanimoto_values = []
        for result in results_dict[idx]:
            tanimoto_values.extend(result['tanimoto_sim'])
        #tanimoto_values = [val for result in results_dict[idx] for val in result['tanimoto_sim'] if val != 0]

        # Create a subplot for each idx
       
        plt.hist(tanimoto_values, bins=100+idx*3, alpha=0.7, label=f'Idx {idx}, Nr: {len(tanimoto_values)}')
    plt.xlabel('Tanimoto Similarity')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Tanimoto Similarity for {idx+1} Molecule generations')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()

    
def plot_hist_MN_sampling_all(config, results_dict):
    """Pool every target's Tanimoto scores into one histogram.

    Args:
        config: Experiment namespace (unused, kept for call-site symmetry).
        results_dict (dict[int, list[dict]]): As in :func:`plot_hist_MN_sampling`.

    Returns:
        None

    Side effects:
        Creates and shows one matplotlib figure.

    NOTE (known issue, left unchanged):
        ``idx`` is read outside the loop that defines it and ``i`` only exists if
        the loop body ran, so an empty ``results_dict`` raises ``NameError``;
        otherwise stale loop values drive the bin count and the title.
    """
    # Number of unique idx values
    num_idx = len(results_dict)

    # Create a figure with subplots (if there are many idx, you might need to adjust the size and layout)
    plt.figure(figsize=(10, 6))
    tanimoto_values = []
    for i, key in enumerate(results_dict.keys()):
        # Collect all tanimoto_list_all for the current idx
        for result in results_dict[key]:
            tanimoto_values.extend(result['tanimoto_sim'])
        #tanimoto_values = [val for result in results_dict[idx] for val in result['tanimoto_sim'] if val != 0]

        # Create a subplot for each idx
       
    plt.hist(tanimoto_values, bins=100+idx*3, alpha=0.7, label=f'Idx {idx}, Nr: {len(tanimoto_values)}')
    plt.xlabel('Tanimoto Similarity')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Tanimoto Similarity for {i+1} Molecule generations')
    plt.legend()

    # Show the plot
    plt.tight_layout()
    plt.show()
    
def plot_hist_MN_sampling_filtered(config, results_dict):
    """Histogram of Tanimoto scores restricted to same-molecular-weight molecules.

    Args:
        config: Experiment namespace (unused, kept for call-site symmetry).
        results_dict (dict[int, list[dict]]): Output of
            :func:`run_multinomial_sampling`.

    Returns:
        tuple[list[list[float]], list[list]]: Forwarded from
        :func:`filter_dict_results_for_MW`.

    Side effects:
        Pops up a matplotlib window; Tanimoto scores are recomputed from scratch
        instead of reusing the cached ``'tanimoto_sim'`` values.
    """
    ##filter out all molecules with the same MW as the target

    # Number of unique idx values
    num_idx = len(results_dict)
    tanimoto_values, gen_trg_smi_lists = filter_dict_results_for_MW(results_dict)

    # Plotting the histogram
    plt.figure(figsize=(10, 6))

    # Plot each list in the histogram
    for idx, lst in enumerate(tanimoto_values):
        
        plt.hist(lst, bins=20, alpha=0.5, label=f'List {idx+1}, Nr: {len(lst)}')

    # Adding labels and title
    plt.xlabel('Tanimoto Similarity')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of Tanimoto Similarity with same MW')
    plt.legend()
    plt.tight_layout()

    # Show the plot
    plt.show()
    return tanimoto_values, gen_trg_smi_lists


def calc_percentage_top_x_correct(results_dict, top_x):
    """Hit rate: fraction of targets found within their top-``top_x`` samples.

    Args:
        results_dict (dict[int, list[dict]]): Output of
            :func:`run_multinomial_sampling` (or its ``v2`` variant).
        top_x (int): How many leading ``tanimoto_sim`` entries are inspected.

    Returns:
        float: Cumulative ``count_yes / (count_yes + count_no)`` over every
        target processed, where a "yes" is ``1 in tanimoto_sim[:top_x]``.

    Side effects:
        None.

    NOTE (known issue, left unchanged):
        With an empty ``results_dict`` the function falls through to ``return``
        without ever binding ``percentage`` and raises ``UnboundLocalError``;
        the hit test also requires an *exact* Tanimoto value of 1.
    """
    count_yes = 0
    count_no = 0
    for i, idx in enumerate(results_dict.keys()):
        # Collect all tanimoto_list_all for the current idx

        for result in results_dict[idx]:
            tanimoto_sim = result["tanimoto_sim"]
            tanimoto_sim = tanimoto_sim[:top_x]
            if 1 in tanimoto_sim:
                count_yes += 1
            else:
                count_no += 1
        percentage = count_yes/(count_yes +count_no)
    return percentage
    
def calc_percentage_top_x_correct_greedy(results_dict):
    """Share of exact Tanimoto hits for a greedy run.

    Args:
        results_dict (dict): Output of :func:`run_greedy_sampling`; only the
            ``'tanimoto_sim'`` list is read.

    Returns:
        float: ``count_yes / (count_yes + count_no)``, where a "yes" means
        ``tanimoto == 1``.

    Side effects:
        None.

    NOTE (known issue, left unchanged):
        An empty Tanimoto list raises ``ZeroDivisionError`` (0/0).
    """
    count_yes = 0
    count_no = 0

    for i in results_dict["tanimoto_sim"]:
        if i==1:
            count_yes += 1
        else:
            count_no += 1

    percentage = count_yes/(count_yes + count_no)
    return percentage
    

def run_precentage_calculation(config, itos, stoi, stoi_MF, MW_filter):
    """Full hit-rate sweep (top 1/3/5/10) for multinomial sampling plus greedy.

    Args:
        config: Experiment namespace.
        itos (dict): Index-to-token map.
        stoi (dict): Token-to-index map.
        stoi_MF (dict): Molecular-formula token map.
        MW_filter (bool): Forwarded to :func:`run_multinomial_sampling`.

    Returns:
        tuple: ``percentage_collection`` = ``[greedy_top1, mns_top1, mns_top3,
        mns_top5, mns_top10]``, plus the two raw results dictionaries.

    Side effects:
        Loads the MMT checkpoint, forces ``config.temperature = 1`` and runs two
        full validation passes (single-molecule batches for multinomial,
        ``config.batch_size`` batches for greedy).
    """
    
    model_MMT = load_MMT_model(config)
    val_dataloader = load_data(config, stoi, stoi_MF, single=True, mode="val")  
    config.temperature = 1
    config, results_dict_mns_10 = run_multinomial_sampling(config, model_MMT, val_dataloader, itos, stoi, MW_filter=MW_filter)
    top_x = 10
    percentage_top_10 = calc_percentage_top_x_correct(results_dict_mns_10, top_x)
    top_x = 5
    percentage_top_5 = calc_percentage_top_x_correct(results_dict_mns_10, top_x)
    top_x = 3
    percentage_top_3 = calc_percentage_top_x_correct(results_dict_mns_10, top_x)
    top_x = 1
    percentage_top_1 = calc_percentage_top_x_correct(results_dict_mns_10, top_x)
    
    val_dataloader_multi = load_data(config, stoi, stoi_MF, single=False, mode="val")  
    config, results_dict_greedy = run_greedy_sampling(config, model_MMT, val_dataloader_multi, itos, stoi)

    percentage_1_greedy = calc_percentage_top_x_correct_greedy(results_dict_greedy)

    percentage_collection = [percentage_1_greedy, percentage_top_1, percentage_top_3, percentage_top_5, percentage_top_10]

    return percentage_collection, results_dict_mns_10, results_dict_greedy

    
def add_tanimoto_similarity(trg_conv_SMI, combined_list):
    """Append the 512-bit Morgan Tanimoto score against the target to each row.

    Args:
        trg_conv_SMI (str): Ground-truth SMILES.
        combined_list (list[list]): Rows ``[SMILES, loss, dot_sim, prob]``.

    Returns:
        tuple[list[list], list[list]]: ``(new_combined_list, failed_combined_list)``;
        valid rows get the score appended at index 4, unparseable rows get
        ``None`` and are moved to the failed list.

    Side effects:
        Mutates the input row lists in place via ``item.append`` before returning
        them.
    """
    # Generate fingerprint for the ground truth molecule
    ground_truth_mol = Chem.MolFromSmiles(trg_conv_SMI)
    ground_truth_fp = AllChem.GetMorganFingerprintAsBitVect(ground_truth_mol, 2, nBits=512)
    # Calculate Tanimoto similarity and add to the list
    new_combined_list = []
    failed_combined_list = []
    for item in combined_list:

        smiles = item[0]
        mol = Chem.MolFromSmiles(smiles)
        if mol:  # Check if the molecule is valid
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
            tanimoto_similarity = DataStructs.TanimotoSimilarity(ground_truth_fp, fp)
            item.append(tanimoto_similarity)
            new_combined_list.append(item)
        else:
            item.append(None)
            failed_combined_list.append(item)
    return new_combined_list, failed_combined_list
#print(combined_list)


def try_calculate_tanimoto_from_two_smiles(smi1, smi2, nbits, extra_info = False):
    """Tanimoto similarity of two SMILES via Morgan fingerprints.

    Args:
        smi1 (str): First SMILES (the target in most call sites).
        smi2 (str): Second SMILES (the generated molecule).
        nbits (int): Fingerprint length, e.g. 512.
        extra_info (bool, optional): Print both SMILES and the score when True.
            Defaults to False.

    Returns:
        float | None: Score rounded to 4 decimals, or ``None`` if RDKit fails.

    Side effects:
        Optional ``print``; the bare ``except`` also hides programming errors.
    """
    
    try:
        pattern1 = Chem.MolFromSmiles(smi1)
        pattern2 = Chem.MolFromSmiles(smi2)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(pattern1, 2, nBits=nbits)
        fp2 = AllChem.GetMorganFingerprintAsBitVect(pattern2, 2, nBits=nbits)

        tan_sim = DataStructs.TanimotoSimilarity(fp1, fp2)
        tan_sim = round(tan_sim,4)
        if extra_info:
            print(f"Smiles 1: {smi1} \n Target Smiles: {smi2} \nTanimoto score:{tan_sim}")

        return tan_sim
    except:
        return None

from collections import defaultdict

def calculate_tanimoto_of_all_compared_to_trg(trg_conv_SMI, gen_conv_SMI_list):
    """Tanimoto score of every generated SMILES against one target (512 bits).

    Args:
        trg_conv_SMI (str): Ground-truth SMILES.
        gen_conv_SMI_list (list[str]): Generated SMILES.

    Returns:
        list[float]: One score per *parseable* generated SMILES; unparseable
        strings are skipped without placeholders, so the result may be shorter
        than the input list.

    Side effects:
        Calls ``Chem.MolFromSmiles`` once for the target and reuses its
        fingerprint for every candidate.
    """
    tani_list = []
    ground_truth_mol = Chem.MolFromSmiles(trg_conv_SMI)
    ground_truth_fp = AllChem.GetMorganFingerprintAsBitVect(ground_truth_mol, 2, nBits=512)
    for smi in gen_conv_SMI_list:
        #smiles = item[0]
        mol = Chem.MolFromSmiles(smi)
        if mol:  # Check if the molecule is valid
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
            tanimoto_similarity = DataStructs.TanimotoSimilarity(ground_truth_fp, fp)
            tani_list.append(tanimoto_similarity)
    return tani_list


def run_mns_hit_counter_experiment(config, model_MMT, val_dataloader, itos, stoi, MW_filter, max_runs):
    """Measure *when* (at which sample index) a perfect hit first appears.

    Args:
        config: Experiment namespace (``multinom_runs``, ``temperature``).
        model_MMT (torch.nn.Module): Trained MMT decoder.
        val_dataloader (DataLoader): Validation batches.
        itos (dict): Index-to-token map.
        stoi (dict): Token-to-index map.
        MW_filter (bool): Apply the molecular-weight filter while sampling.
        max_runs (int): Upper bound on sampling iterations per target.

    Returns:
        tuple[list[int], list[list]]: For each target either the 1-based index of
        the first perfect Tanimoto hit or ``-5`` when none was found; and the
        ``[gen_smis, trg_smis, tani_list]`` triples used to decide.

    Side effects:
        Mutates ``config.temperature`` (+0.1 per iteration, saved/restored per
        target); prints counters every 10 iterations.

    NOTE (known issue, left unchanged):
        ``1 in tani_list`` / ``tani_list.index(1.0)`` need an *exact* float
        match, but :func:`calculate_tanimoto_of_all_compared_to_trg` returns
        un-rounded values, so a hit almost never registers and every target
        typically burns all ``max_runs`` iterations. If ``max_runs <= 0`` the
        while-loop body never runs and ``tani_list`` is still undefined at the
        ``try`` statement.
    """
    percentage_collection = []
    molecule_tani_comparison_lists = []
    one_finder_list = []
    n_times = config.multinom_runs

    results_dict = defaultdict(list)
    # generate all the smiles of trg and greedy gen
    for i, data_dict in enumerate(val_dataloader):
        gen_conv_SMI_list = []
        gen_conv_SMI_list, trg_conv_SMI_list,  prob_list = [], [], []

        trg_enc_SMI = data_dict["trg_enc_SMI"][0]
        trg_conv_SMI = hf.tensor_to_smiles(trg_enc_SMI[1:], itos)

        ### multiply the input to paralellize the generation
        data_dict_dup = rbgvm.duplicate_dict(data_dict, 128)

        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC = vgmmt.run_model(model_MMT,
                                                                       data_dict_dup, 
                                                                       config)
        counter = 0
        ### Here I increase the temperature in case if it does not generate enough diverse molecules but sets it back to the original
        ### value after enough molecules were found.
        temp_orig = config.temperature
        while len(gen_conv_SMI_list)<n_times and counter<max_runs:
            # increase the temperature if not enough different molecules get generated
            if counter %10 == 0:
                print(counter, len(gen_conv_SMI_list), config.temperature )
                
            multinom_tensor, multinom_token_prob = multinomial_sequence_multi_2(model_MMT, memory, src_padding_mask, stoi, config)
            gen_conv_SMI, token_probs = hf.tensor_to_smiles_and_prob_2(multinom_tensor, multinom_token_prob, itos)

            # import IPython; IPython.embed();
            gen_conv_SMI, token_probs = filter_probs_and_valid_smiles_and_canonicolize(gen_conv_SMI, token_probs)   ### for 10.000 commented out
            if MW_filter == True:
                gen_conv_SMI, token_probs = filter_for_MW_2(trg_conv_SMI, gen_conv_SMI, token_probs)
            gen_conv_SMI_list.extend(gen_conv_SMI)
            prob_list.extend(token_probs)
            gen_conv_SMI_list, prob_list = deduplicate_smiles(gen_conv_SMI_list, prob_list)
            
            tani_list = calculate_tanimoto_of_all_compared_to_trg(trg_conv_SMI, gen_conv_SMI_list)
            if 1 in tani_list:
                break
            #print(counter, len(gen_conv_SMI_list))
            counter += 1
            config.temperature = config.temperature + 0.1    
            
        try:
            one_finder_list.append(tani_list.index(1.0)+1)  # because indices start with 0
        except:
            one_finder_list.append(-5)
        trg_SMI_list = [trg_conv_SMI for i in range(len(gen_conv_SMI_list))]
        molecule_tani_comparison_lists.append([gen_conv_SMI_list,trg_SMI_list, tani_list])
        config.temperature = temp_orig 
        #break
    return one_finder_list, molecule_tani_comparison_lists

def sel_data_slice_and_save_as_csv(config):
    """Truncate the validation SMILES CSV to the first ``config.data_size`` rows.

    Args:
        config: Experiment namespace; ``config.csv_path_val`` is read and
            ``config.data_size`` rows are kept.

    Returns:
        config: The same namespace, with ``config.csv_SMI_targets`` pointing at
        the newly written ``*_sel_<n>.csv`` file.

    Side effects:
        Writes a new CSV next to the original and mutates the ``config`` object.
    """
    # File path
    file_path = config.csv_path_val  # Replace with your file path

    # Read the CSV file
    df = pd.read_csv(file_path)

    # Select the first X rows
    df_selected = df.head(config.data_size)

    # Save the selected rows to a new CSV file
    new_file_path = file_path.replace('.csv', f'_sel_{config.data_size}.csv')
    df_selected.to_csv(new_file_path, index=False)
    config.csv_SMI_targets = new_file_path
    return config



def filter_smiles(df, smi_list):
    """
    Filter out SMILES strings from smi_list that are already present in df.

    Comparison is exact string matching on the raw ``'SMILES'`` column (no
    canonicalisation), so differently written forms of the same molecule are
    not considered duplicates.

    Parameters:
    df (pandas.DataFrame): DataFrame with a 'SMILES' column.
    smi_list (list): List of SMILES strings.

    Returns:
    list: Filtered list of SMILES strings.
    """
    df_smiles_set = set(df['SMILES'])
    filtered_list = [smiles for smiles in smi_list if smiles not in df_smiles_set]
    return filtered_list


import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw

import matplotlib.pyplot as plt

def analyze_and_plot(results_dict, mode):
    """Rank the ground-truth molecule among generations and histogram those ranks.

    Args:
        results_dict (dict[str, list]): ``{target SMILES: [[SMILES, loss,
            dot_sim, tanimoto, HSQC_error], ...]}`` from
            :func:`run_test_mns_performance_CLIP_3` combined with
            :func:`combine_CLIP_HSQC_data`.
        mode (str): ``"dot_sim"`` (higher is better, descending sort) or
            ``"HSQC_sim"`` (lower error is better, ascending sort).

    Returns:
        list: ``[key, value]`` pairs whose first row has ``tanimoto != 1``, i.e.
        the targets the run failed to reproduce.

    Side effects:
        Draws and shows a rank histogram.

    NOTE (known issue, left unchanged):
        Any ``mode`` other than those two literals leaves ``similaritys``
        unbound (``UnboundLocalError``); the histogram bins and ticks also use
        only the last ``value`` list.
    """
    ranks = []
    failed = []
    
    for key, value in results_dict.items():
        # Extracting second and third numbers from each sublist
        if mode == "HSQC_sim":
            similaritys = [item[4] for item in value]
        elif mode == "dot_sim":
            similaritys = [item[2] for item in value]
        tani_sims = [item[3] for item in value]
        
        if tani_sims[0] == 1:
            # First number
            first_number = similaritys[0]

            # Sorting the numbers in descending order
            if mode == "HSQC_sim":
                sorted_numbers = sorted(similaritys, reverse=False)
            elif mode == "dot_sim":
                sorted_numbers = sorted(similaritys, reverse=True)

            # Finding the rank of the first number
            rank_of_first_number = sorted_numbers.index(first_number) + 1  # Adding 1 because index starts from 0

            ranks.append(rank_of_first_number)
        else:
            failed.append([key, value])
            #print(third_numbers)
            
    # Plotting the histogram of ranks
    plt.hist(ranks, bins=range(1, len(value) + 2), align='left')
    plt.xlabel('Rank')
    plt.ylabel('Frequency')
    plt.title(f'Rank Histogram of {mode} corresponding correct Molecule')
    plt.xticks(range(1, len(value) + 1))
    plt.show()
    return failed
    
    
# Function to plot a molecule with additional data
def plot_molecule_with_data(smiles, cosine_sim, tanimoto, HSQC_error):
    """Render one molecule with its CLIP similarity, Tanimoto and HSQC error.

    Args:
        smiles (str): SMILES to draw.
        cosine_sim (float | str): CLIP dot similarity; ``''`` is displayed as blank.
        tanimoto (float | str): Tanimoto score; ``''`` is displayed as blank.
        HSQC_error (float | str): Simulated-spectrum error; ``''`` is blank.

    Returns:
        None

    Side effects:
        Opens a blocking matplotlib window per molecule (``plt.show()``).
    """
    mol = Chem.MolFromSmiles(smiles)
    cosine_sim_rounded = round(float(cosine_sim), 3) if cosine_sim else ''
    tanimoto_rounded = round(float(tanimoto), 3) if tanimoto else ''
    HSQC_error_rounded = round(float(HSQC_error), 4) if HSQC_error else ''
    fig, ax = plt.subplots()
    img = Draw.MolToImage(mol, size=(300, 300))
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(f"CLIP: {cosine_sim_rounded}, Tanimoto: {tanimoto_rounded},  HSQC_err: {HSQC_error_rounded}")
    plt.show()

    ### PLOT Molecules

def plot_CLIP_molecules(results_dict, investigate_number, stop_nr):
    """Visual walk-through of the generations for a single target.

    Args:
        results_dict (dict): ``{target SMILES: [combined_list]}`` as returned by
            :func:`run_clip_similarity_check`.
        investigate_number (int): Position (in dict order) of the target to show.
        stop_nr (int): Stop after this many generated molecules.

    Returns:
        None

    Side effects:
        Draws one figure per molecule (target first) and blocks on ``plt.show()``.
    """
    #investigate_number = 5
    # Plot the key SMILES


    # Plot the first item of each list with the third and fourth elements
    for idx, data_val in enumerate(results_dict.values()):
        
        if idx == investigate_number:
            key_smiles = list(results_dict.keys())[investigate_number]   
            plot_molecule_with_data(key_smiles, '', '', '')
            
            for idy, lists in enumerate(data_val[0]):
                first_smiles, _, cosine_sim, tanimoto, HSQC_error = lists
                plot_molecule_with_data(first_smiles, cosine_sim, tanimoto, HSQC_error)
                if idy == stop_nr:
                    break
                    
def generate_mns_list(config, 
                  model_MMT, 
                  val_dataloader,
                  stoi, 
                  itos, 
                  MW_filter):
    """Sample ``config.multinom_runs`` molecules per target and keep the batch.

    Args:
        config: Experiment namespace (``multinom_runs``, ``temperature``).
        model_MMT (torch.nn.Module): Trained MMT decoder.
        val_dataloader (DataLoader): Validation batches.
        stoi (dict): Token-to-index map.
        itos (dict): Index-to-token map.
        MW_filter (bool): Apply the molecular-weight filter while sampling.

    Returns:
        dict[str, list]: ``{target SMILES: [gen_conv_SMI_list,
        trg_conv_SMI_list, data_dict]}``; the raw collated batch is kept so CLIP
        and HSQC scoring can be re-run later.

    Side effects:
        Mutates ``config.temperature`` (+0.1 per attempt, restored per target);
        prints a counter on every attempt.

    NOTE (known issue, left unchanged):
        ``ttf.tensor_to_smiles`` does not exist in
        ``utils_MMT.train_test_functions_pl_v15_4`` (the helper lives in
        ``hf.tensor_to_smiles``), so the first batch raises ``AttributeError``.
    """
    ### Same code as function: run_multinomial_sampling
    n_times = config.multinom_runs
    gen_dict = {} #defaultdict(list)
    temperature_orig = config.temperature
    for idx, data_dict in enumerate(val_dataloader):
        if idx % 10 == 0:
            print(idx)
        gen_conv_SMI_list, trg_conv_SMI_list, token_probs_list, src_HSQC_list, prob_list = [], [], [], [], []
        data_dict_dup = rbgvm.duplicate_dict(data_dict, 128)
        trg_enc_SMI = data_dict["trg_enc_SMI"][0]
        trg_conv_SMI = ttf.tensor_to_smiles(trg_enc_SMI[1:], itos)
        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC = vgmmt.run_model(model_MMT,
                                                                   data_dict_dup, 
                                                                   config)
        counter = 1
        while len(gen_conv_SMI_list)<n_times:
            # increase the temperature if not enough different molecules get generated               
            print(counter, len(gen_conv_SMI_list), config.temperature)
            if counter%30==0:
                print(trg_conv_SMI)
                break

            multinom_tensor, multinom_token_prob = multinomial_sequence_multi_2(model_MMT, memory, src_padding_mask, stoi, config)
            gen_conv_SMI, token_probs = hf.tensor_to_smiles_and_prob_2(multinom_tensor, multinom_token_prob, itos)
            # import IPython; IPython.embed();
            gen_conv_SMI, token_probs = filter_probs_and_valid_smiles_and_canonicolize(gen_conv_SMI, token_probs)   ### for 10.000 commented out
            if MW_filter == True:
                gen_conv_SMI, token_probs = filter_for_MW_2(trg_conv_SMI, gen_conv_SMI, token_probs)
            gen_conv_SMI_list.extend(gen_conv_SMI)
            prob_list.extend(token_probs)
            gen_conv_SMI_list, prob_list = deduplicate_smiles(gen_conv_SMI_list, prob_list)
            #gen_conv_SMI_list = list(set(gen_conv_SMI_list)) ### for 10.000 commented out
            counter += 1
            config.temperature = config.temperature + 0.1  
            #print(time.time() -start)
            #start = time.time()
        config.temperature = temperature_orig

        gen_conv_SMI_list = gen_conv_SMI_list[:n_times]
        prob_list = prob_list[:n_times]
        trg_conv_SMI_list = [trg_conv_SMI for i in range(len(gen_conv_SMI_list))]
        
        gen_dict[trg_conv_SMI] = [gen_conv_SMI_list, trg_conv_SMI_list, data_dict]
    return gen_dict

def run_clip_similarity_check(config, model_CLIP, gen_dict):
    """Score every sampled molecule of every target with the CLIP model.

    Args:
        config: Experiment namespace.
        model_CLIP (torch.nn.Module): CLIP wrapper exposing ``inference``.
        gen_dict (dict): Output of :func:`generate_mns_list`.

    Returns:
        dict[str, list]: ``{target SMILES: [combined_list]}`` where rows are
        ``[SMILES, CLIP loss, dot similarity, Tanimoto]``.

    Side effects:
        Runs CLIP inference; no sorting is applied (that line is commented out).
    """
    results_dict ={}
    for trg_conv_SMI, values in gen_dict.items():
        gen_conv_SMI_list, trg_conv_SMI_list, data_dict = values
        data_dict_dup = rbgvm.duplicate_dict(data_dict, len(gen_conv_SMI_list))

        mean_loss, losses, logits, targets, dot_similarity= model_CLIP.inference(data_dict_dup, 
                                                                    gen_conv_SMI_list)

        combined_list = [[smile, num.item(), dot_sim.item()] for smile, num, dot_sim in zip(gen_conv_SMI_list, losses, dot_similarity)]
        ### Sort by the lowest similarity
        #sorted_list = sorted(combined_list, key=lambda x: x[1])

        combined_list, failed_combined_list = add_tanimoto_similarity(trg_conv_SMI, combined_list)
        #combined_list, batch_data = rbgvm.add_HSQC_error(config, combined_list, data_dict_dup, gen_conv_SMI_list, trg_conv_SMI, config.multinom_runs) # config.MMT_batch
        #sorted_list = sorted(combined_list, key=lambda x: -x[3]) # SMILES = 0, losses =1, dot_sim= 2, tanimoto = 3 


        results_dict[trg_conv_SMI] = [combined_list]
    return results_dict

def run_HSQC_similarity_check(config, gen_dict, results_dict):
    """Simulate the HSQC spectrum of every sampled molecule and store the errors.

    Args:
        config: Experiment namespace.
        gen_dict (dict): Output of :func:`generate_mns_list`.
        results_dict (dict): Accepted for call-site symmetry; not read.

    Returns:
        dict[str, list]: ``{target SMILES: [gen_conv_SMI_list, sim_error_list]}``.

    Side effects:
        Runs the SGNN simulator through ``rbgvm.calculate_HSQC_error``.
    """
    results_dict_HSQC ={}
    for trg_conv_SMI, values in gen_dict.items():
        gen_conv_SMI_list, trg_conv_SMI_list, data_dict = values
        data_dict_dup = rbgvm.duplicate_dict(data_dict, len(gen_conv_SMI_list))

        src_HSQC_list = [data_dict_dup["src_HSQC"]]
        tensor_HSQC = rbgvm.prepare_HSQC_data_from_src(src_HSQC_list)

        sgnn_avg_sim_error, sim_error_list, batch_data = rbgvm.calculate_HSQC_error(config, data_dict_dup, gen_conv_SMI_list)
        results_dict_HSQC[trg_conv_SMI] = [gen_conv_SMI_list, sim_error_list]# batch_data]
    return results_dict_HSQC

def combine_CLIP_HSQC_data(results_dict_CLIP, results_dict_HSQC):
    """Merge CLIP scores and HSQC errors, then sort by Tanimoto descending.

    Args:
        results_dict_CLIP (dict): Output of :func:`run_clip_similarity_check`.
        results_dict_HSQC (dict): Output of :func:`run_HSQC_similarity_check`.

    Returns:
        dict[str, list]: ``{target SMILES: sorted_rows}`` where each row is
        ``[SMILES, loss, dot_sim, tanimoto, HSQC_error]``.

    Side effects:
        Mutates the row lists of ``results_dict_CLIP`` in place; a missing HSQC
        error is replaced by the sentinel ``-9``.
    """

    final_results = {}
    for trg, data in results_dict_CLIP.items():
        for idx, data_list in enumerate(data[0]):
            try:
                hsqc_error = results_dict_HSQC[trg][1][idx]
                data[0][idx].append(hsqc_error)
            except:
                data[0][idx].append(-9)            
        data = sorted(data[0], key=lambda x: -x[3]) # SMILES = 0, losses =1, dot_sim= 2, tanimoto = 3 
        final_results[trg] = data
    return final_results


def filter_invalid_inputs(results_dict):
    """Drop the ``[None, None]`` placeholders written by the sampling loop.

    Args:
        results_dict (dict): Output of :func:`run_test_mns_performance_CLIP_3`.

    Returns:
        tuple[dict, int]: The filtered dictionary (only keys whose first value is
        not ``None``) and the number of dropped keys.

    Side effects:
        None.
    """
    filtered_dict = {}
    counter = 0
    for key, value in results_dict.items():
        # Assuming 'combined_list' is the first item in the list which is the value of the dictionary.
        # And we're checking if the first item in 'combined_list' is not '[None, None]'.
        if value[0] is not None:# and value[1] is not None:
            filtered_dict[key] = value
        else:
            counter+=1
    return filtered_dict, counter

def calc_molecular_formula(smi):
    """Molecular formula of a SMILES string (used by the MF filter).

    Args:
        smi (str): SMILES string.

    Returns:
        str | None: Formula such as ``"C6H6"``, or ``None`` on any error.

    Side effects:
        None.

    NOTE (known issue, left unchanged):
        ``CalcMolFormula`` is never imported in this module, so the ``NameError``
        raised here is swallowed by the bare ``except`` and this function
        **always** returns ``None``. As a consequence
        :func:`filter_for_MF_2` compares ``None == None`` and silently keeps
        every molecule instead of filtering by formula.
    """
    try:
        mol = Chem.MolFromSmiles(smi)
        return CalcMolFormula(mol)
    except:
        return None

def filter_for_MF_2(trg_conv_SMI, gen_conv_SMI, prob_list):
    """Keep generated SMILES whose molecular formula equals the target's.

    Args:
        trg_conv_SMI (str): Ground-truth SMILES.
        gen_conv_SMI (list[str]): Generated SMILES.
        prob_list (list[float]): Probabilities aligned with ``gen_conv_SMI``.

    Returns:
        tuple[list[str], list[float]]: Filtered SMILES and their probabilities.

    Side effects:
        None; the formula is recomputed for every candidate.

    NOTE (known issue, left unchanged):
        Because :func:`calc_molecular_formula` currently always returns ``None``
        (missing import), this filter is a no-op and keeps all candidates.
    """
    # Calculate the molecular formula of the target molecule
    trg_mf = calc_molecular_formula(trg_conv_SMI)

    # Filter the list based on molecular formula
    filtered_gen_smis = [smi for (smi, token_prob) in zip(gen_conv_SMI, prob_list) if calc_molecular_formula(smi) == trg_mf]
    filtered_prob_list = [token_prob for (smi, token_prob) in zip(gen_conv_SMI, prob_list) if calc_molecular_formula(smi) == trg_mf]
    return filtered_gen_smis, filtered_prob_list


def run_multinomial_sampling_v2(config, model_MMT, val_dataloader, itos, stoi, MW_filter=False, MF_filter=False):
    """Multinomial sampling with optional molecular-weight / formula filtering.

    Args:
        config: Experiment namespace (``multinom_runs`` etc.).
        model_MMT (torch.nn.Module): Trained MMT decoder.
        val_dataloader (DataLoader): Validation batches.
        itos (dict): Index-to-token map.
        stoi (dict): Token-to-index map.
        MW_filter (bool, optional): Apply :func:`filter_for_MW_2`. Defaults to False.
        MF_filter (bool, optional): Apply :func:`filter_for_MF_2`. Defaults to False.

    Returns:
        tuple: ``(config, results_dict)`` with the same schema as
        :func:`run_multinomial_sampling` (per-target Tanimoto statistics).

    Side effects:
        Mutates ``config.temperature`` (+0.2 per attempt, restored per target);
        duplicates each batch to 16 generations; prints progress.

    NOTE (known issue, left unchanged):
        The loop gives up after 5 attempts, so ``config.multinom_runs`` unique
        molecules are frequently not reached. The ``removed`` bookkeeping is
        computed but never reported to the caller.
    """
    n_times = config.multinom_runs
    results_dict = defaultdict(list)
    temperature_orig = config.temperature

    for idx, data_dict in tqdm(enumerate(val_dataloader)):
        if idx % 10 == 0:
            print(idx)

        gen_conv_SMI_list, trg_conv_SMI_list, token_probs_list, src_HSQC_list, prob_list = [], [], [], [], []
        data_dict_dup = rbgvm.duplicate_dict(data_dict, 16)

        trg_enc_SMI = data_dict["trg_enc_SMI"][0]
        trg_conv_SMI = hf.tensor_to_smiles(trg_enc_SMI[1:], itos)
        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC, src_COSY = vgmmt.run_model(model_MMT,
                                                                   data_dict_dup, 
                                                                   config)
        counter = 1
        removed = 0
        while len(gen_conv_SMI_list) < n_times:
            # Increase the temperature if not enough different molecules get generated               
            if counter % 5 == 0:
                print(trg_conv_SMI)
                break
            multinom_tensor, multinom_token_prob = multinomial_sequence_multi_2(model_MMT, memory, src_padding_mask, stoi, config)
            gen_conv_SMI, token_probs = hf.tensor_to_smiles_and_prob_2(multinom_tensor, multinom_token_prob, itos)

            # Filter valid SMILES and canonicalize
            gen_conv_SMI, token_probs = filter_probs_and_valid_smiles_and_canonicolize(gen_conv_SMI, token_probs)
            #print(f"valid smi: {len(set(gen_conv_SMI))}")
            #import IPython; IPython.embed();
            before_gen = len(gen_conv_SMI)
            if MW_filter:
                gen_conv_SMI, token_probs = filter_for_MW_2(trg_conv_SMI, gen_conv_SMI, token_probs)
                #print(f"MW_filter: {before_gen - len(gen_conv_SMI)}")
                removed_1 = before_gen - len(gen_conv_SMI)
                removed += removed_1
            if MF_filter:
                gen_conv_SMI, token_probs = filter_for_MF_2(trg_conv_SMI, gen_conv_SMI, token_probs)
                #print(f"MF_filter: {before_gen - len(gen_conv_SMI)}")
                removed_2 = before_gen - len(gen_conv_SMI)
                removed += removed_2

            gen_conv_SMI_list.extend(gen_conv_SMI)
            prob_list.extend(token_probs)
            gen_conv_SMI_list, prob_list = deduplicate_smiles(gen_conv_SMI_list, prob_list)
            counter += 1
            config.temperature = config.temperature + 0.2
        #print(removed, len(gen_conv_SMI_list))
        gen_conv_SMI_list = gen_conv_SMI_list[:n_times]
        prob_list = prob_list[:n_times]
        trg_conv_SMI_list = [trg_conv_SMI for _ in range(len(gen_conv_SMI_list))]

        tanimoto_mean, tanimoto_std_dev, failed, tanimoto_list_all = vgmmt.calculate_tanimoto_similarity(gen_conv_SMI_list, trg_conv_SMI_list)
        results_dict[idx].append({
            'gen_conv_SMI_list': gen_conv_SMI_list,
            'trg_conv_SMI_list': trg_conv_SMI_list,
            'tanimoto_sim': tanimoto_list_all,
            'tanimoto_mean': tanimoto_mean,
            'tanimoto_std_dev': tanimoto_std_dev,
            'failed': failed,
            'prob_list': prob_list,
        })
        config.temperature = temperature_orig
    return config, results_dict

def run_precentage_calculation_v2(config, itos, stoi, stoi_MF, MW_filter=False, MF_filter=False):
    """Same hit-rate sweep as :func:`run_precentage_calculation`, on the v2 samplers.

    Args:
        config: Experiment namespace.
        itos (dict): Index-to-token map.
        stoi (dict): Token-to-index map.
        stoi_MF (dict): Molecular-formula token map.
        MW_filter (bool, optional): Forwarded to the multinomial sampler.
            Defaults to False.
        MF_filter (bool, optional): Forwarded to the multinomial sampler.
            Defaults to False.

    Returns:
        tuple: ``[greedy_top1, mns_top1, mns_top3, mns_top5, mns_top10]`` plus the
        two raw results dictionaries.

    Side effects:
        Loads the MMT checkpoint, forces ``config.temperature = 1`` and runs two
        validation passes.

    NOTE (known issue, left unchanged):
        The greedy pass is called with ``MW_filter=False, MF_filter=False`` even
        when filtering was requested, so the greedy and multinomial halves of the
        comparison do not use the same candidate set.
    """
    model_MMT = load_MMT_model(config)
    val_dataloader = load_data(config, stoi, stoi_MF, single=True, mode="val")  
    config.temperature = 1
    config, results_dict_mns_10 = run_multinomial_sampling_v2(config, model_MMT, val_dataloader, itos, stoi, MW_filter=MW_filter, MF_filter=MF_filter)
    top_x = 10
    percentage_top_10 = calc_percentage_top_x_correct(results_dict_mns_10, top_x)
    top_x = 5
    percentage_top_5 = calc_percentage_top_x_correct(results_dict_mns_10, top_x)
    top_x = 3
    percentage_top_3 = calc_percentage_top_x_correct(results_dict_mns_10, top_x)
    top_x = 1
    percentage_top_1 = calc_percentage_top_x_correct(results_dict_mns_10, top_x)
    
    val_dataloader_multi = load_data(config, stoi, stoi_MF, single=False, mode="val")  
    config, results_dict_greedy = run_greedy_sampling_v2(config, model_MMT, val_dataloader_multi, itos, stoi, MW_filter=False, MF_filter=False)
    #config, results_dict_greedy = run_greedy_sampling(config, model_MMT, val_dataloader_multi, itos, stoi, MW_filter=MW_filter)

    percentage_1_greedy = calc_percentage_top_x_correct_greedy(results_dict_greedy)

    percentage_collection = [percentage_1_greedy, percentage_top_1, percentage_top_3, percentage_top_5, percentage_top_10]

    return percentage_collection, results_dict_mns_10, results_dict_greedy

def run_greedy_sampling_v2(config, model_MMT, val_dataloader, itos, stoi, MW_filter=False, MF_filter=False):
    """Greedy-decoding variant that accepts (but currently ignores) the filters.

    Args:
        config: Experiment namespace.
        model_MMT (torch.nn.Module): Trained MMT decoder.
        val_dataloader (DataLoader): Validation batches.
        itos (dict): Index-to-token map.
        stoi (dict): Token-to-index map.
        MW_filter (bool, optional): Accepted for API parity; the filtering block
            is commented out, so nothing is filtered. Defaults to False.
        MF_filter (bool, optional): Same. Defaults to False.

    Returns:
        tuple: ``(config, results_dict)`` with the generated/target SMILES,
        Tanimoto statistics and the source HSQC/COSY lists.

    Side effects:
        Runs MMT inference; does not mutate ``config``.
    """
    results_dict = defaultdict(list)
    gen_conv_SMI_list, trg_conv_SMI_list, prob_list, src_HSQC_list, src_COSY_list = [], [], [], [], []
    for idx, data_dict in enumerate(val_dataloader):
        trg_enc_SMI = data_dict["trg_enc_SMI"]
        trg_conv_SMI = hf.tensor_to_smiles(trg_enc_SMI.T[1:], itos)
        memory, src_padding_mask, trg_enc_SMI, fingerprint, src_HSQC, src_COSY = vgmmt.run_model(model_MMT,
                                                                   data_dict, 
                                                                   config)

        greedy_tensor, greedy_token_prob = greedy_sequence_2(model_MMT, stoi, itos, memory, src_padding_mask, config)

        gen_conv_SMI, token_probs = hf.tensor_to_smiles_and_prob(greedy_tensor, greedy_token_prob.T, itos)

        # Filter valid SMILES and canonicalize
        """gen_conv_SMI, token_probs = filter_probs_and_valid_smiles_and_canonicolize(gen_conv_SMI, token_probs)
        if MW_filter:
            gen_conv_SMI, token_probs = filter_for_MW_2(trg_conv_SMI[0], gen_conv_SMI, token_probs)
        if MF_filter:
            gen_conv_SMI, token_probs = filter_for_MF_2(trg_conv_SMI[0], gen_conv_SMI, token_probs)
        """
        gen_conv_SMI_list.extend(gen_conv_SMI)
        trg_conv_SMI_list.extend(trg_conv_SMI)
        prob_list.extend(token_probs)
        src_HSQC_list.extend(src_HSQC)
        src_COSY_list.extend(src_COSY)
        
    tanimoto_mean, tanimoto_std_dev, failed, tanimoto_list_all = vgmmt.calculate_tanimoto_similarity(gen_conv_SMI_list, trg_conv_SMI_list)
    results_dict = {
        'gen_conv_SMI_list': gen_conv_SMI_list,
        'trg_conv_SMI_list': trg_conv_SMI_list,
        'tanimoto_sim': tanimoto_list_all,
        'tanimoto_mean': tanimoto_mean,
        'tanimoto_std_dev': tanimoto_std_dev,
        'failed': failed,
        'prob_list': prob_list,
        "src_HSQC_list": src_HSQC_list,
        "src_COSY_list": src_COSY_list,
        }
    
    return config, results_dict
