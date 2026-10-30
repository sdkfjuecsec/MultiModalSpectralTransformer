#!/bin/bash
# =============================================================================
# SLURM launcher for scripts/script_ZINC_250_350_4000.py
#
# Purpose : baseline evaluation of the pre-trained V8i Multimodal Transformer
#           on the 4000-molecule ZINC test set (see the module docstring of the
#           Python file for the pipeline).
# Usage   : from the repository root,  sbatch scripts/ZINC_250_350_4000.sh
#           Progress and errors are written to the --output path below.
#
# What the job does: load the CUDA module and the NMR_Structure_Elucidator
# conda environment, print the GPU inventory, then start the evaluation.
# The SBATCH lines request 1 node / 1 GPU / 128 GB / 7 days because the set
# is four times larger than the 1000-molecule baseline.
# =============================================================================
#SBATCH --job-name=ZINC_250_350_4000_v1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=128g
#SBATCH --time=7-1:00:00
#SBATCH --output=/projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/scripts/ZINC_250_350_4000_v1.txt

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

echo "nvidia-smi";    
nvidia-smi
# Make the shared libraries shipped with conda visible to the GPU stack.
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/projects/cc/se_users/knlr326/miniconda_all/lib/

echo "python";
# Start the baseline evaluation on the 4000-molecule set.
python /projects/cc/se_users/knlr326/1_NMR_project/2_Notebooks/MultiModalTransformer/scripts/script_ZINC_250_350_4000.py