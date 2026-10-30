#!/bin/bash
# =============================================================================
# SLURM launcher for scripts/script_PC_0_250.py
#
# Purpose : baseline (non-improvement) evaluation of the pre-trained V8i
#           Multimodal Transformer on the PubChem 0-250 Da test split.
#           The pipeline itself is described in the module docstring of the
#           Python file.
# Usage   : from the repository root,  sbatch scripts/PC_0_250_new.sh
#           Progress and errors are written to the --output path below.
#
# What the job does: load the CUDA module and the NMR_Structure_Elucidator
# conda environment, print the GPU inventory, then start the evaluation.
# The SBATCH lines request 1 node / 1 GPU / 128 GB / 4 days.
# =============================================================================
#SBATCH --job-name=PC_0_250
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=128g
#SBATCH --time=4-1:00:00
#SBATCH --constraints=volta
#SBATCH --output=/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/scripts/PC_0_250_v4.txt

echo "SBATCH";
# Drop every inherited module before loading the CUDA stack below.
module purge
echo $(which nvcc)

# Load Anaconda module
# Activate the desired conda environment
echo "ACTIVATE";
source /projects/cc/se_users/knlr326/miniconda_SE/bin/activate  /projects/cc/se_users/knlr326/miniconda_SE/envs/NMR_Structure_Elucidator 
# CUDA toolkit that matches the PyTorch build inside the conda environment.
module load CUDA/11.3.1
echo $(which python)
# NOTE (known issue, left unchanged): this stray `d` is executed as a command,
# so bash reports "d: command not found" and the job continues.
d
echo "nvidia-smi";    
nvidia-smi
# Make the shared libraries shipped with conda visible to the GPU stack.
# NOTE (known issue, left unchanged): the environment is activated from
# miniconda_SE while this path points at miniconda_all, so a mismatched
# MKL/CUDA runtime can be picked up at run time.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/cc/se_users/knlr326/miniconda_all/lib/

echo "python";
# Start the baseline evaluation (paths inside the script are cluster absolute).
python /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/scripts/script_PC_0_250.py
