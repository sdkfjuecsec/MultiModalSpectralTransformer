#!/bin/bash
# =============================================================================
# SLURM launcher for scripts/script_PC_250_350.py
#
# Purpose : baseline evaluation of the pre-trained V8i Multimodal Transformer
#           on the PubChem 250-350 Da test split (see the module docstring of
#           the Python file for the pipeline).
# Usage   : from the repository root,  sbatch scripts/PC_250_350.sh
#           Progress and errors are written to the --output path below.
#
# What the job does: load the CUDA module and the NMR_Structure_Elucidator
# conda environment, print the GPU inventory, then start the evaluation.
# The SBATCH lines request 1 node / 1 GPU / 128 GB / 4 days.
# =============================================================================
#SBATCH --job-name=PC_250_350
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=128g
#SBATCH --time=4-1:00:00
#SBATCH --output=/projects/cc/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/scripts/PC_250_350.txt

echo "SBATCH";
# Drop every inherited module before loading the CUDA stack below.
module purge
echo $(which nvcc)

# Load Anaconda module
# Activate the desired conda environment
echo "ACTIVATE";
source /projects/cc/knlr326/miniconda_all/bin/activate  /projects/cc/knlr326/miniconda_all/envs/NMR_Structure_Elucidator 
# CUDA toolkit that matches the PyTorch build inside the conda environment.
module load CUDA/11.1.0
echo $(which python)

echo "nvidia-smi";    
nvidia-smi
# Make the shared libraries shipped with conda visible to the GPU stack.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/cc/knlr326/miniconda_all/lib/

echo "python";
# Start the baseline evaluation.
# NOTE (known issue, left unchanged): this path points at the repository root,
# but script_PC_250_350.py was moved into scripts/, so the job fails with
# "No such file or directory" unless the path is corrected.
python /projects/cc/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/script_PC_250_350.py
