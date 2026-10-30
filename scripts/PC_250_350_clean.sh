#!/bin/bash
# =============================================================================
# SLURM launcher for scripts/script_PC_250_350_clean.py
#
# Purpose : same experiment as PC_250_350.sh (baseline evaluation of the
#           pre-trained V8i Multimodal Transformer on the PubChem 250-350 Da
#           test split), but it runs the path-portable *_clean.py driver, which
#           resolves its data/model folders relative to the repository root.
# Usage   : from the repository root,  sbatch scripts/PC_250_350_clean.sh
#           Progress and errors are written to the --output path below.
#
# Lines marked "USER CHANGE REQUIRED" must be adapted to your own cluster
# before the job can start (output location, conda install, project path).
# =============================================================================
#SBATCH --job-name=PC_250_350
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=128g
#SBATCH --time=4-1:00:00
#SBATCH --constraints=volta
# USER CHANGE REQUIRED: Update output path to your actual project location
#SBATCH --output=/projects/cc/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/scripts/PC_250_350.txt

echo "SBATCH";
# Drop every inherited module before loading the CUDA stack below.
module purge
echo $(which nvcc)

# Load Anaconda module
# Activate the desired conda environment
echo "ACTIVATE";
# USER CHANGE REQUIRED: Update these paths to your actual conda installation and environment
source /projects/cc/se_users/knlr326/miniconda_SE/bin/activate  /projects/cc/se_users/knlr326/miniconda_SE/envs/NMR_Structure_Elucidator 
# CUDA toolkit that matches the PyTorch build inside the conda environment.
module load CUDA/11.3.1
echo $(which python)

echo "nvidia-smi";    
nvidia-smi
# USER CHANGE REQUIRED: Update this path to your actual miniconda installation
# Make the shared libraries shipped with conda visible to the GPU stack.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/cc/se_users/knlr326/miniconda_all/lib/

echo "python";
# USER CHANGE REQUIRED: Update this path to your actual script location
# This should point to your cleaned script in the MultiModalSpectralTransformer_cleaned project
# Start the cleaned evaluation driver.
python /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/scripts/script_PC_250_350_clean.py

