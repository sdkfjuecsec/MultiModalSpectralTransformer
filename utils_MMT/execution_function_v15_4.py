"""High-level orchestration steps of the MMST "execution" cycles.

This module wires the pieces together for the iterative experiment loops used
in ``1.0_Experiment_Notebook___.ipynb``:

``simulate_syn_data``      generate simulated 1H/13C/HSQC/COSY/IR for the
                           target SMILES and point every config path at it.
``SMI_generation_MMT``     load the CLIP+MMST models and run candidate
                           generation, returning ``{target: [candidates]}``.
``SMI_generation_MF``      molecular-formula based generation (MolFormer).
``gen_sim_aug_data``       same as ``simulate_syn_data`` but with a unique run
                           tag and config backup/restore (used for the
                           "augment with simulated data" cycle).
``blend_aug_with_train_data``  mix generated molecules into the training CSV.
``update_model_path``      pick the newest checkpoint as ``config.checkpoint_path``.
``filter_invalid_criteria`` / ``clean_dataset``
                           CSV clean-up helpers (drop exotic elements/charges,
                           remove stereochemistry, canonicalise).

Each function **mutates ``config`` in place** and also returns it, so callers
can write either ``config = f(config)`` or just ``f(config)``. This dual
contract is intentional but easy to misread - see the per-function notes.

NOTE (known issue, left unchanged)
    1. ``tqdm`` / ``os`` / ``pd`` 依赖文件顶部的 ``from
       utils_MMT.nmr_calculation_from_dft_v15_4 import *`` 间接导入，
       没有显式 ``import``（本文件只在第 220 行才 ``import pandas as pd``，
       而第 147 行已经在用 ``pd``）。
    2. ``simulate_syn_data`` 用 ``os.mkdir`` 且目录名不含随机数：
       同一 ``ran_num`` 重复执行会 ``FileExistsError``（``gen_sim_aug_data``
       已用 ``makedirs(exist_ok=True)`` + 随机数规避）。
    3. 多个过滤函数用裸 ``except:``（第 236 行），会吞掉 ``KeyboardInterrupt``。
"""

import utils_MMT.clip_functions_v15_4 as cf
import utils_MMT.dataloaders_pl_v15_4 as dl
# NOTE: wildcard import is what makes ``tqdm`` / ``os`` / ``pd`` resolvable in
#       this module (see module-level note 1).
from utils_MMT.nmr_calculation_from_dft_v15_4 import *
import utils_MMT.MT_functions_v15_4 as mtf
import utils_MMT.data_generation_v15_4 as dg
import utils_MMT.molformer_functions_v15_4 as mff
import utils_MMT.validate_generate_MMT_v15_4 as vgmmt
import utils_MMT.run_batch_gen_val_MMT_v15_4 as rbgvm
import utils_MMT.clustering_visualization_v15_4 as cv
import utils_MMT.ir_simulation_v15_4 as irs
import utils_MMT.plotting_v15_4 as pt
import utils_MMT.train_test_functions_pl_v15_4 as ttf

from sklearn.utils import shuffle
from argparse import Namespace
import random
import glob


def parse_arguments(hyperparameters):
    """Unwrap the notebook's ``{key: [value]}`` dict into a Namespace.

    Args:
        hyperparameters (dict): values are single-element lists (argparse
            legacy format).

    Returns:
        argparse.Namespace: one attribute per key.

    NOTE: identical to ``utils_MMT.MT_functions_v15_4``'s counterpart; the
          ``val[0]`` unwrapping gives the first *character* if a value is a
          bare string instead of a list.
    """
    # Using dictionary comprehension to simplify your code
    parsed_args = {key: val[0] for key, val in hyperparameters.items()}
    return Namespace(**parsed_args)

def simulate_syn_data(config, IR_config):
    """Generate simulated spectra for the target SMILES and re-point all config paths.

    Steps:
        1. create ``<SGNN_gen_folder_path>_syn_<ran_num>`` and make it the new
           ``SGNN_gen_folder_path`` (so later steps write inside it);
        2. run ``dg.main_run_data_generation`` to produce 1H/13C/HSQC/COSY CSV;
        3. run the chemprop-IR simulation (``irs.run_IR_simulation``);
        4. patch ``config`` so the subsequent MMST evaluation reads these
           simulated files as both "SGNN input" and "ground truth".

    Args:
        config (argparse.Namespace): mutated in place; needs
            ``SGNN_gen_folder_path``, ``ran_num``, ``csv_SMI_targets``.
        IR_config (argparse.Namespace): chemprop predict args.

    Returns:
        argparse.Namespace: the same (mutated) ``config``.

    NOTE (known issue, left unchanged)
        ``os.mkdir`` is single-level and the folder name is deterministic
        (``..._syn_<ran_num>``), so a second run with the same ``ran_num``
        raises ``FileExistsError``. Also, the whole body is inside
        ``if os.path.exists(config.SGNN_gen_folder_path):`` - if that base
        folder is missing, the function silently returns the untouched config
        and no simulation happens at all.
    """
    if os.path.exists(config.SGNN_gen_folder_path):
        ran_num = config.ran_num
        print(ran_num)
        config.SGNN_ran_num = ran_num
        path_gen_folder = config.SGNN_gen_folder_path+ "_syn_" + ran_num
        os.mkdir(path_gen_folder)
        config.SGNN_gen_folder_path = path_gen_folder
    # the input molecules for the simulation are the target SMILES
    config.SGNN_csv_gen_smi = config.csv_SMI_targets
    # returns the per-modality CSV paths written by the data generation step
    combined_df, data_1H, data_13C, data_COSY, data_HSQC, csv_1H_path, csv_13C_path, csv_COSY_path, csv_HSQC_path = dg.main_run_data_generation(config)

    #args = parse_arguments(IR_config_dict)
    # Only molecules that produced a 1H spectrum are kept (successful ones).
    config.csv_SMI_targets = csv_1H_path # it should just generate the smiles that has been successfully generated for 1H 
    IR_output_path = irs.run_IR_simulation(config, IR_config, "target")

    # ---- "input" side: what the model reads -------------------------------
    config.csv_1H_path_SGNN = csv_1H_path
    config.csv_13C_path_SGNN = csv_13C_path 
    config.csv_HSQC_path_SGNN = csv_HSQC_path
    config.csv_COSY_path_SGNN = csv_COSY_path 
    config.IR_data_folder = IR_output_path
    #config.IR_data_folder = ""     ############
    # empty => the multimodal model falls back to simulated spectra only
    config.pickle_file_path = ""

    # ---- "ground truth" side: what the metrics compare against ------------
    # Here the simulated data doubles as the reference, i.e. this is a
    # self-consistency run (model is scored against its own simulations).
    config.csv_1H_path_GT_sim = csv_1H_path
    config.csv_13C_path_GT_sim = csv_13C_path 
    config.csv_HSQC_path_GT_sim = csv_HSQC_path
    config.csv_COSY_path_GT_sim = csv_COSY_path
    config.IR_data_folder_GT_sim = IR_output_path
    #config.IR_data_folder_GT_sim = ""  ############
    
    # in case if I get IR as  well
    config.data_size = len(data_1H)
    
    # to make sure that failed molecules get ignored and just the successful
    # generations are processed
    config.csv_path_val = csv_1H_path
    
    #sort automatic model saving in new folder
    # the training_mode string is used to build the model_save_dir name
    config.training_mode = "1H_13C_HSQC_COSY_IR_MF_MW"
    return config


    
def SMI_generation_MMT(config, stoi, stoi_MF, itos, itos_MF):
    """Run MMST candidate generation for all targets.

    Args:
        config (argparse.Namespace): generation settings (``data_size`` etc.).
        stoi, stoi_MF: token dicts for SMILES / formula.
        itos, itos_MF: inverse token dicts (decoding).

    Returns:
        Tuple[argparse.Namespace, Dict[str, List[str]]]:
            the config and ``{target_smiles: [candidate_smiles, ...]}``.

    Note on the odd loading sequence:
        The CLIP model is loaded first and its dataloader (``mode="val"``) is
        what generation iterates over; ``MMT_m`` is then replaced by the model
        loaded inside ``vgmmt.load_data_and_MMT_model``. The first assignment
        (``MMT_m = CLIP_model.CLIP_model.MT_model``) is therefore discarded -
        the inline comment says this is a workaround "as long as the CLIP model
        is not working correctly".
    """
    #config.data_size = 10 # number of molecules that should be taken for generation of new molecules
    mode = "val"
    sort_num = 3 # 3=tanimoto score
    CLIP_model, dataloader = rbgvm.load_data_and_CLIP_model(config, mode, stoi, stoi_MF, itos, itos_MF)
    MMT_m = CLIP_model.CLIP_model.MT_model
    CLIP_m = CLIP_model.CLIP_model
    # NOTE: overwrites the MMT model taken from CLIP (see docstring note)
    MMT_m, val_dataloader = vgmmt.load_data_and_MMT_model(config, stoi, stoi_MF)    ######################## as long as the CLIP model is not working correctly

    results_dict = rbgvm.run_MMT_generations(config, MMT_m, CLIP_m, dataloader, stoi, itos, sort_num)

    return config, results_dict




def SMI_generation_MF(config, stoi, stoi_MF, itos, itos_MF): 
    """Run molecular-formula (MolFormer) candidate generation.

    Args:
        config (argparse.Namespace): MolFormer settings.
        stoi, stoi_MF, itos, itos_MF: token dicts; accepted for signature
            symmetry, **unused** here.

    Returns:
        Tuple[argparse.Namespace, Dict[str, List[str]]]:
            ``{target_smiles: [formula_candidates...]}`` built from the MolFormer
            output frame (row 0 = source SMILES, remaining rows = candidates).

    Note:
        The commented-out block shows the richer visualisation path
        (``cv.plot_cluster_MF`` + ``pt.plot_molecules_from_list``) that was
        disabled.
    """

    #mode = "val"
    #df = mff.load_data(config, mode, stoi, stoi_MF, itos, itos_MF)
    #import IPython; IPython.embed();
    combined_df = mff.main_molformer(config)
    #print("combined_df")
    #print(combined_df)
    # Generating the dictionary
    results_dict_MF = {}
    for column in tqdm(combined_df.columns):
        # row 0 is the target/query SMILES; the rest are generated candidates
        results_dict_MF[combined_df[column][0]] = combined_df[column][1:].tolist()

    # combined_list_MF, html_TSNE, html_UMAP, html_PCA = cv.plot_cluster_MF(results_dict_MF, config, mode, stoi, stoi_MF, itos, itos_MF)
    # #import IPython; IPython.embed();
    # max_num = 20
    # mol_html_plot = pt.plot_molecules_from_list(combined_list_MF, max_num)
    
    return config, results_dict_MF

import copy
def gen_sim_aug_data(config, IR_config):
    """Simulated-data variant of :func:`simulate_syn_data` used for augmentation.

    Differences:
        * a fresh 6-digit ``ran_num`` is generated and written back to
          ``config.ran_num`` (unique output folders across runs);
        * ``makedirs(exist_ok=True)`` instead of ``mkdir``;
        * the original ``SGNN_gen_folder_path`` is backed up and restored
          before returning, so the *caller's* config keeps pointing at the
          training folder while all generated file paths are absolute;
        * string config fields are ``copy.deepcopy``-ed (defensive; strings are
          immutable so this is a no-op in practice).

    Args:
        config (argparse.Namespace): mutated: ``ran_num``,
            ``SGNN_gen_folder_path_backup``, the SGNN/IR CSV paths,
            ``data_size``, ``pickle_file_path``, ``csv_path_val``.
        IR_config (argparse.Namespace): chemprop predict args.

    Returns:
        argparse.Namespace: the same (mutated) ``config``.
    """
    ran_num = random.randint(100000, 999999)
    config.ran_num = ran_num
    print("outside ran_num")
    print(ran_num)
    config.SGNN_gen_folder_path_backup = config.SGNN_gen_folder_path

    if os.path.exists(config.SGNN_gen_folder_path):
        print(ran_num)
        path_gen_folder = config.SGNN_gen_folder_path + "_" + str(ran_num)
        os.makedirs(path_gen_folder, exist_ok=True)
        config.SGNN_gen_folder_path = path_gen_folder

    combined_df, data_1H, data_13C, data_COSY, data_HSQC, csv_1H_path, csv_13C_path, csv_COSY_path, csv_HSQC_path = dg.main_run_data_generation(config)

    config.csv_SMI_targets = copy.deepcopy(csv_1H_path) # it should just generate the smiles that has been successfully generated for 1H 
    data_IR = irs.run_IR_simulation(config, IR_config, "target")
    # ANSI escape: bold + yellow "IR Generation: DONE"
    print("\033[1m\033[33m IR Generation: DONE\033[0m")    
    config.csv_1H_path_SGNN = copy.deepcopy(csv_1H_path)
    config.csv_13C_path_SGNN = copy.deepcopy(csv_13C_path) 
    config.csv_HSQC_path_SGNN = copy.deepcopy(csv_HSQC_path)
    config.csv_COSY_path_SGNN = copy.deepcopy(csv_COSY_path) 
    config.IR_data_folder = copy.deepcopy(data_IR)
    #config.IR_data_folder = ""  ### Because for now we don't use it but 15.1 will

    # in case if I get IR as well
    config.data_size = len(data_1H)
    config.pickle_file_path = ""
    # to make sure that failed molecules get ignored and just the successful
    # generations are processed
    config.csv_path_val = csv_1H_path
    # restore the training folder: only the generated file paths stay absolute
    config.SGNN_gen_folder_path = config.SGNN_gen_folder_path_backup
    #sort automatic model saving in new folder
    #config.training_mode = "1H_13C_HSQC_COSY_IR_MF_MW"
    #config.execution_type = "transformer_improvement"
    return config


def blend_aug_with_train_data(config, aug_mol_df):
    """Mix generated molecules with a random slice of the training set.

    Args:
        config (argparse.Namespace): needs ``csv_train_path``,
            ``train_data_blend`` (number of training rows to add),
            ``gen_mol_csv_folder_path`` and ``ran_num`` (used in the filename).
        aug_mol_df (pandas.DataFrame): generated molecules, must have the same
            columns as the training CSV (``SMILES`` and ``sample-id``).

    Returns:
        Tuple[argparse.Namespace, pandas.DataFrame]: the config (with
        ``SGNN_csv_gen_smi`` and ``data_size`` updated) and the blended frame.

    Side effects:
        Writes ``<gen_mol_csv_folder_path>/<ran_num>.csv``.

    Note:
        ``sample(n=...)`` has no ``random_state`` -> a different training
        blend every run (intentional for augmentation cycles, but makes the
        experiment non-reproducible).
    """
    # Assuming df is already created with 'SMILES' and 'sample-id'
    # df = pd.DataFrame({'SMILES': all_gen_smis, 'sample-id': random_number_strings})
    train_df = pd.read_csv(config.csv_train_path)
    random_subset = train_df.sample(n=config.train_data_blend)
    # Combine the dataframes
    combined_df = pd.concat([random_subset, aug_mol_df], ignore_index=True)
    # Shuffle the combined dataframe
    # ``sklearn.utils.shuffle`` returns a *new* shuffled frame
    final_df = shuffle(combined_df)
    # Reset index if needed
    final_df.reset_index(drop=True, inplace=True)
    
    csv_file_path = os.path.join(config.gen_mol_csv_folder_path, f"{config.ran_num}.csv")
    print(csv_file_path)                                                             
    # NOTE: index=True (the default) -> the written CSV gains an unnamed index
    #       column that the readers in ``dataloaders_pl_v15_4`` ignore.
    final_df.to_csv(csv_file_path)

    config.SGNN_csv_gen_smi = csv_file_path
    #config.execution_type = "data_generation"
    config.data_size = len(final_df)
    return config, final_df


def update_model_path(config):
    """Point ``config.checkpoint_path`` at the most recently modified ``*.ckpt``.

    Args:
        config (argparse.Namespace): needs ``model_save_dir``.

    Returns:
        argparse.Namespace: the same (mutated) ``config``.

    NOTE (known issue, left unchanged)
        ``flist[0]`` raises ``IndexError`` when the directory holds no
        checkpoint (e.g. the previous training run crashed before saving).
        Selection is by file mtime, not by parsed epoch/loss in the name.
    """
    flist = glob.glob(config.model_save_dir+"/*.ckpt")
    #filtered_flist = [file for file in flist if f"epoch={config.num_epochs}" in file]
    flist.sort(key=os.path.getmtime, reverse=True)
    new_model_path = flist[0]
    config.checkpoint_path = new_model_path
    print(config.checkpoint_path)
    return config


# ---------------------------------------------------------------------------
# Legacy version of ``filter_invalid_criteria`` (commented out, kept verbatim)
# ---------------------------------------------------------------------------
# The active version below additionally canonicalises SMILES and drops the
# failed ones. The old version kept ``-``/``+`` anywhere in the string (which
# also removed legitimate triple bonds ``C#C``? no - only charges) and did not
# canonicalise. Kept as documentation of the evolution.
#
# def filter_invalid_criteria(file_path):
#     # Path to the original CSV file
#     #file_path = config.csv_1H_path_SGNN
#
#     # Read the CSV file into a DataFrame
#     df = pd.read_csv(file_path)
#
#     # Filter out rows where the SMILES string contains 'P' or '[NH+]'
#     df_filtered = df[~df['SMILES'].str.contains('P')]
#     df_filtered = df_filtered[~df_filtered['SMILES'].str.contains('Si')]
#     df_filtered = df_filtered[~df_filtered['SMILES'].str.contains(r'\-')]
#     df_filtered = df_filtered[~df_filtered['SMILES'].str.contains(r'\+')]
#
#     df_filtered = df_filtered[~df_filtered['SMILES'].str.contains(r'\[B\]')]
#     df_filtered = df_filtered[~df_filtered['SMILES'].str.contains('B')]
#     #df_filtered = df_filtered[~df_filtered['SMILES'].str.contains(r'\[NH2\+\]')]
#     #df_filtered = df_filtered[~df_filtered['SMILES'].str.contains(r'\[c\-\]')]
#     #df_filtered = df_filtered[~df_filtered['SMILES'].str.contains(r'\[C\+\]')]
#     df_filtered = df_filtered[~df_filtered['SMILES'].str.contains(r'\[NH\+\]', regex=True)]
#     df_filtered = df_filtered[~df_filtered['SMILES'].str.contains(r'\[SH\]', regex=True)]
#
#     def remove_stereo(smiles):
#         smiles = smiles.replace('[C@@H]', 'C').replace('[C@H]', 'C')
#         smiles = smiles.replace('[C@@]', 'C').replace('[C@]', 'C')
#         return smiles
#
#     # Apply the function to each SMILES string in the DataFrame
#     df_filtered['SMILES'] = df_filtered['SMILES'].apply(remove_stereo)
#
#     # Generate a random number for the new file name
#     random_number = random.randint(10000, 99999)
#
#     # Construct new file name with the random number
#     new_file_name = file_path.rsplit('.', 1)[0] + f"_{random_number}.csv"
#
#     # Save the filtered DataFrame to the new file
#     df_filtered.to_csv(new_file_name, index=False)
#
#     # Update the config with the new file path
#     #config.csv_path_val = new_file_name
#     #config.csv_1H_path_SGNN = new_file_name
#     return new_file_name


import pandas as pd
import random
from rdkit import Chem

def filter_invalid_criteria(file_path):
    """Clean a SMILES CSV: canonicalise, drop exotic elements/charges, de-stereo.

    Filtering rules (applied to the canonical SMILES):
        * no phosphorus (``P``), no silicon (``Si``);
        * no boron ``[B]`` and no isolated ``B`` atom (but ``Br`` is kept via
          the negative lookahead);
        * no protonated amine ``[NH+]``, no ``[SH]``;
        * no charge markers inside brackets (``[X-]`` / ``[X+]``);
        * stereochemistry markers are stripped (``[C@@H]`` -> ``C`` ...).

    Args:
        file_path (str): input CSV with at least a ``SMILES`` column.

    Returns:
        str: path of the new CSV, named
        ``<original_stem>_<random 5 digits>.csv`` (written next to the input;
        the input file is untouched).

    NOTE (known issue, left unchanged)
        1. ``except:`` is bare -> swallows ``KeyboardInterrupt``/``SystemExit``.
        2. ``df['SMILES'].str.contains`` returns NaN for the ``None`` values
           produced by a failed canonicalisation; ``~`` on NaN yields NaN and
           the row is **kept**, which is why the explicit ``dropna`` later is
           needed. Order matters: filtering runs before ``dropna``.
        3. ``remove_stereo`` is a plain string replace, so it also mangles
           atom maps / isotopes written with the same bracket syntax.
    """
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Function to canonicalize SMILES strings
    def canonicalize_smiles(smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                return Chem.MolToSmiles(mol, canonical=True)
            else:
                return None
        except:
            # invalid/unparsable SMILES -> marker for the later dropna
            return None

    # Apply the canonicalization function to each SMILES string in the DataFrame
    df['SMILES'] = df['SMILES'].apply(canonicalize_smiles)

    # Filter out rows where the SMILES string contains specific unwanted elements or characters
    df_filtered = df[~df['SMILES'].str.contains('P')]
    # NOTE: the mask is always built from the *full* ``df`` while the selection
    #       target is ``df_filtered``. pandas aligns boolean masks by index, so
    #       this is correct (just redundant), and it keeps the row order of
    #       ``df_filtered``.
    df_filtered = df_filtered[~df['SMILES'].str.contains('Si')]
    df_filtered = df_filtered[~df['SMILES'].str.contains(r'\[B\]')]
    # isolated "B" but not "Br" (negative lookahead)
    df_filtered = df_filtered[~df['SMILES'].str.contains(r'\bB\b(?!r)', regex=True)]
    # df_filtered = df_filtered[~df['SMILES'].str.contains('B')]
    df_filtered = df_filtered[~df['SMILES'].str.contains(r'\[NH\+\]', regex=True)]
    df_filtered = df_filtered[~df['SMILES'].str.contains(r'\[SH\]', regex=True)]

    # Specifically match '-' and '+' as charges, not as part of the structure
    df_filtered = df_filtered[~df_filtered['SMILES'].str.contains(r'\[.*?\-.*?\]', regex=True)]
    df_filtered = df_filtered[~df_filtered['SMILES'].str.contains(r'\[.*?\+.*?\]', regex=True)]

    def remove_stereo(smiles):
        smiles = smiles.replace('[C@@H]', 'C').replace('[C@H]', 'C')
        smiles = smiles.replace('[C@@]', 'C').replace('[C@]', 'C')
        return smiles

    # Apply the function to remove stereochemistry from each SMILES string in the DataFrame
    df_filtered['SMILES'] = df_filtered['SMILES'].apply(remove_stereo)

    # Drop rows where the canonicalization failed (i.e., resulted in None)
    df_filtered = df_filtered.dropna(subset=['SMILES'])

    # Generate a random number for the new file name
    random_number = random.randint(10000, 99999)
    
    # Construct new file name with the random number
    new_file_name = file_path.rsplit('.', 1)[0] + f"_{random_number}.csv"

    # Save the filtered and canonicalized DataFrame to the new file
    df_filtered.to_csv(new_file_name, index=False)
    
    return new_file_name


def clean_dataset(config):
    """Simpler, in-place variant of :func:`filter_invalid_criteria`.

    Removes stereochemistry, drops ``P``/``Si`` containing SMILES and writes
    ``<original_stem>_<random>.csv``, then stores that path back into
    ``config.csv_SMI_targets``.

    Args:
        config (argparse.Namespace): needs ``csv_SMI_targets``; mutated.

    Returns:
        argparse.Namespace: the same (mutated) ``config``.

    Note:
        Unlike :func:`filter_invalid_criteria` this one does **not**
        canonicalise or drop unparsable strings, and it keeps charges.
    """
    # Path to the original CSV file
    # ANSI escape: bold + red
    print("\033[1m\033[31mThis is: clean_dataset\033[0m")
    file_path = config.csv_SMI_targets

    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path)

    # Function to remove stereochemistry information from a SMILES string
    def remove_stereo(smiles):
        smiles = smiles.replace('[C@@H]', 'C').replace('[C@H]', 'C')
        smiles = smiles.replace('[C@@]', 'C').replace('[C@]', 'C')
        return smiles

    # Apply the function to each SMILES string in the DataFrame
    df['SMILES'] = df['SMILES'].apply(remove_stereo)

    # Remove rows containing 'P' or '+'
    # NOTE: the comment is stale - the code only filters 'P' and 'Si',
    #       no '+' rule is applied here.
    df = df[~df['SMILES'].str.contains('P', regex=True)]
    df = df[~df['SMILES'].str.contains('Si')]

    # Generate a random number for the new file name
    random_number = random.randint(10000, 99999)

    # Construct new file name with the random number
    new_file_name = file_path.rsplit('.', 1)[0] + f"_{random_number}.csv"

    # Save the modified DataFrame to the new file
    df.to_csv(new_file_name, index=False)

    # Update the config with the new file path
    config.csv_SMI_targets = new_file_name
    return config
