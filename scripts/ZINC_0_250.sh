#!/bin/bash
# =============================================================================
# SLURM launcher for the ZINC baseline evaluation.
#
# Purpose : runs scripts/script_ZINC_250_350.py, i.e. the baseline evaluation
#           of the pre-trained V8i Multimodal Transformer on the ZINC test set
#           (see the module docstring of the Python file for the pipeline).
# Usage   : from the repository root,  sbatch scripts/ZINC_0_250.sh
#           Progress and errors are written to the --output path below.
#
# What the job does: load the CUDA module and the NMR_Structure_Elucidator
# conda environment, print the GPU inventory, then start the evaluation.
# The SBATCH lines request 1 node / 1 GPU / 128 GB / 4 days.
# =============================================================================
#SBATCH --job-name=ZINC_0_250
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=128g
#SBATCH --time=4-1:00:00
#SBATCH --output=/projects/cc/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/scripts/ZINC_0_250.txt

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
# Start the baseline evaluation of the 250-350 ZINC split.
# NOTE (known issue, left unchanged): the file name says 0_250 while the script
# it launches is the 250-350 variant, and the path points at the repository
# root although the script lives in scripts/.
python /projects/cc/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/script_ZINC_250_350.py
