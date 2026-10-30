"""Central import hub for the MMST web app.

Importing this module does three things, in order:

1. put the repository root and ``models/chemprop-ir`` on ``sys.path`` so the
   ``utils_MMT`` package and the vendored chemprop fork become importable;
2. import the ``utils_MMT`` modules under short aliases used everywhere else
   (``mrtf`` = result tests, ``ex`` = execution, ``mtf`` = transformer training,
   ``dl`` = data generation, ``irs`` = IR simulation, ``cv``/``pt`` = plots,
   ``hf`` = helpers);
3. load the four token dictionaries (``itos``, ``stoi``, ``stoi_MF``,
   ``itos_MF``) and a random run id.

It also provides the small config helpers used by ``app.py``: JSON <-> config
object conversion (attributes are read as ``config.foo``) and loading of the
paired ``config_V8.json`` / ``ir_config_V8.json``. The JSON files are resolved
relative to this file, so the app must keep its Runfolder layout.
"""

# Standard library imports
import json
import os
import random
import glob
import pickle
import sys
import os
from argparse import Namespace
from collections import defaultdict
from types import SimpleNamespace

# Third-party imports
## Data processing and scientific computing
import numpy as np
import pandas as pd
from tqdm import tqdm
from tqdm.autonotebook import tqdm

## Machine learning and data visualization
import matplotlib.pyplot as plt
#import seaborn as sns
from umap import UMAP
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances
from sklearn.model_selection import train_test_split

## PyTorch and related libraries
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.optim.lr_scheduler import ReduceLROnPlateau

## RDKit for cheminformatics
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, MolFromSmiles, MolToSmiles, Descriptors
from rdkit.Chem.Descriptors import MolWt

## Miscellaneous
from IPython.display import HTML, SVG

# Local imports
#sys.path.append('/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer')
# Dynamically add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if project_root not in sys.path:
    sys.path.append(project_root)
# Chemprop imports
#sys.path.append("/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer/chemprop-IR")
chemprop_ir_path = os.path.join(project_root, 'models', 'chemprop-ir')

if chemprop_ir_path not in sys.path:
    sys.path.append(chemprop_ir_path)

import utils_MMT.MT_functions_v15_4 as mtf
import utils_MMT.run_batch_gen_val_MMT_v15_4 as rbgvm
import utils_MMT.clustering_visualization_v15_4 as cv
import utils_MMT.plotting_v15_4 as pt
import utils_MMT.execution_function_v15_4 as ex
import utils_MMT.ir_simulation_v15_4 as irs
import utils_MMT.helper_functions_pl_v15_4 as hf
import utils_MMT.mmt_result_test_functions_15_4 as mrtf
import utils_MMT.data_generation_v15_4 as dl


from chemprop.train import make_predictions
from chemprop.parsing import modify_predict_args


# Setting up environment
torch.cuda.device_count()



def parse_arguments(hyperparameters):
    """Turn a config dict into an attribute namespace (``config.target`` ...).

    JSON stores everything as lists/dicts, so any single-element list or tuple is
    unwrapped to its scalar before the namespace is built.
    """
    parsed_args = {key: val[0] if isinstance(val, (list, tuple)) else val for key, val in hyperparameters.items()}
    return Namespace(**parsed_args)


# def load_json_dics():
#     with open('/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer/itos.json', 'r') as f:
#         itos = json.load(f)
#     with open('/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer/stoi.json', 'r') as f:
#         stoi = json.load(f)
 
#     with open('/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer/stoi_MF.json', 'r') as f:
#         stoi_MF = json.load(f)
#     with open('/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer/itos_MF.json', 'r') as f:
#         itos_MF = json.load(f)    
#     return itos, stoi, stoi_MF, itos_MF
# itos, stoi, stoi_MF, itos_MF = load_json_dics()
# rand_num = str(random.randint(1, 10000000))
 


def load_json_dics():
    """Load the four token dictionaries from the repository root.

    Returns ``(itos, stoi, stoi_MF, itos_MF)`` — plain token <-> index maps used
    by the transformer and the matched-molecular-pair generator.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)
    
    # Build paths relative to the script's location
    base_path = os.path.abspath(os.path.join(script_dir, '../..'))
    
    itos_path = os.path.join(base_path, 'itos.json')
    stoi_path = os.path.join(base_path, 'stoi.json')
    stoi_MF_path = os.path.join(base_path, 'stoi_MF.json')
    itos_MF_path = os.path.join(base_path, 'itos_MF.json')

    # Load JSON files
    with open(itos_path, 'r') as f:
        itos = json.load(f)
    with open(stoi_path, 'r') as f:
        stoi = json.load(f)
    with open(stoi_MF_path, 'r') as f:
        stoi_MF = json.load(f)
    with open(itos_MF_path, 'r') as f:
        itos_MF = json.load(f)
    
    return itos, stoi, stoi_MF, itos_MF

# Example usage
itos, stoi, stoi_MF, itos_MF = load_json_dics()
rand_num = str(random.randint(1, 10000000))

 
def save_config(config, path):
    """Dump a plain ``dict`` to ``path`` as JSON.

    NOTE (known issue, left unchanged): every config in this app is an
    ``argparse.Namespace`` (see ``parse_arguments``), and ``json.dump`` cannot
    serialise that, so calling this with such an object raises
    "TypeError: Object of type Namespace is not JSON serializable". Use
    ``save_updated_config`` (which calls ``vars``) for namespace configs.
    """
    with open(path, 'w') as f:
        json.dump(config, f)
 
def load_config(path):
    """Read one JSON config file; returns ``None`` when the file is missing."""
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None    

 
def save_updated_config(config, path):
    """Persist a config namespace as JSON by dumping its ``__dict__``."""
    config_dict = vars(config)
    with open(path, 'w') as f:
        json.dump(config_dict, f, indent=4)
 
 
# def load_configs():
#     # Load IR config
#     ir_config_path = '/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer/_ISAK/Runfolder/ir_config_V8.json'
#     IR_config_dict = load_config(ir_config_path)
#     if IR_config_dict is None:
#         raise FileNotFoundError(f"IR config file not found at {ir_config_path}")
#     IR_config = parse_arguments(IR_config_dict)
#     modify_predict_args(IR_config)
    
#     # Load main config
#     config_path = '/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/2_paper_code/Experiments_SLURM/20.0_SLURM_MasterTransformer/_ISAK/Runfolder/config_V8.json'
#     config_dict = load_config(config_path)
#     if config_dict is None:
#         raise FileNotFoundError(f"Main config file not found at {config_path}")
#     config = parse_arguments(config_dict)
 
#     return IR_config, config

def load_configs():
    """Load both configs from ``Runfolder`` and return ``(IR_config, config)``.

    ``IR_config`` is post-processed by chemprop's ``modify_predict_args`` so it
    carries the defaults chemprop expects; both are attribute namespaces.
    """
    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)
    
    # Build paths relative to the script's location
    base_path = os.path.abspath(os.path.join(script_dir, ''))
    
    ir_config_path = os.path.join(base_path, 'ir_config_V8.json')
    config_path = os.path.join(base_path, 'config_V8.json')
    
    # Load IR config
    IR_config_dict = load_config(ir_config_path)
    if IR_config_dict is None:
        raise FileNotFoundError(f"IR config file not found at {ir_config_path}")
    IR_config = parse_arguments(IR_config_dict)
    modify_predict_args(IR_config)
    
    # Load main config
    config_dict = load_config(config_path)
    if config_dict is None:
        raise FileNotFoundError(f"Main config file not found at {config_path}")
    config = parse_arguments(config_dict)
 
    return IR_config, config

# IR_config, config = load_config()
# config = parse_arguments(config)
# rand_num = random.randint(0, 1000000)
# itos, stoi, stoi_MF, itos_MF = load_json_dics()

def plot_first_smiles(csv_path):
    """Quick debug helper: draw and pop up the first SMILES of a CSV."""
    # Load the CSV file
    df = pd.read_csv(csv_path)
    # Extract the first SMILES string
    first_smiles = df['SMILES'].iloc[0]
    # Generate the molecule from SMILES
    mol = Chem.MolFromSmiles(first_smiles)
    # Draw the molecule
    img = Draw.MolToImage(mol)
    # Display the image
    img.show()

def sim_and_display():
    """Simulate all spectra for the SMILES CSV named in the config.

    Cleans the SGNN input CSV, runs the full data-generation pipeline with the
    IR model, then copies the generated paths into the ``*_display`` config keys
    (which the plotting routes read), saves the config and returns it.
    """
    print("="*50)
    print("sim_and_display function called")
    try:
        print("Loading dictionaries...")
        itos, stoi, stoi_MF, itos_MF = load_json_dics()
        print("Loading configs...")
        IR_config, config = load_configs()
        
        print(f"SGNN_csv_gen_smi path: {config.SGNN_csv_gen_smi}")
        
        # Ensure the path doesn't have a leading slash when checking file existence
        # NOTE (known issue, left unchanged): only this local copy is stripped.
        # config.csv_SGNN_gen_smi / csv_SMI_targets keep the leading slash, so on
        # Windows (where "/data/x.csv" is not a valid absolute path) the guard
        # passes but the generation step below fails to open the file.
        csv_path = config.SGNN_csv_gen_smi
        if csv_path.startswith('/'):
            csv_path = csv_path[1:]
            
        config.csv_SMI_targets = config.SGNN_csv_gen_smi
        
        # Check if the file exists
        print(f"Checking if file exists: {csv_path}")
        if not os.path.exists(csv_path):
            print(f"ERROR: File not found at path: {csv_path}")
            raise FileNotFoundError(f"File not found at path: {csv_path}")
        else:
            print(f"File exists: {csv_path}")
            
        # Check if the file is readable
        try:
            with open(csv_path, 'r') as f:
                first_line = f.readline()
                print(f"First line of CSV: {first_line}")
        except Exception as e:
            print(f"ERROR reading file: {str(e)}")
            raise
            
        print("Cleaning dataset...")
        config = ex.clean_dataset(config)
        print("Dataset cleaned successfully")
        
        print("\033[1m\033[31mThis is: simulate_syn_data\033[0m")
        
        print("Generating simulated data...")
        print(f"IR_config checkpoint_dir: {IR_config.checkpoint_dir}")
        print(f"IR_config test_path: {IR_config.test_path}")
        print(f"IR_config preds_path: {IR_config.preds_path}")
        
        # Check if IR model files exist
        if not os.path.exists(IR_config.checkpoint_dir):
            print(f"WARNING: IR checkpoint directory not found: {IR_config.checkpoint_dir}")
        else:
            print(f"IR checkpoint directory exists: {IR_config.checkpoint_dir}")
            # List files in the directory
            print("Files in checkpoint directory:")
            for file in os.listdir(IR_config.checkpoint_dir):
                print(f"  - {file}")
        
        config = ex.gen_sim_aug_data(config, IR_config) 
        print("Simulated data generated successfully")

        print("Setting display paths...")
        config.csv_1H_path_display = config.csv_1H_path_SGNN
        config.csv_13C_path_display = config.csv_13C_path_SGNN
        config.csv_HSQC_path_display = config.csv_HSQC_path_SGNN
        config.csv_COSY_path_display = config.csv_COSY_path_SGNN
        config.IR_data_folder_display = config.IR_data_folder
        
        print(f"1H path: {config.csv_1H_path_display}")
        print(f"13C path: {config.csv_13C_path_display}")
        print(f"HSQC path: {config.csv_HSQC_path_display}")
        print(f"COSY path: {config.csv_COSY_path_display}")
        print(f"IR folder: {config.IR_data_folder_display}")
        
        print("Saving updated config...")
        save_updated_config(config, config.config_path)
        print("sim_and_display completed successfully")
        print("="*50)
        return config
    except Exception as e:
        print(f"ERROR in sim_and_display: {str(e)}")
        import traceback
        print(traceback.format_exc())
        print("="*50)
        raise