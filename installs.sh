#!/bin/bash

# System libraries the Python packages dlopen at runtime. Without them the
# notebooks fail with ImportError before any Python code can help:
#   rdkit.Chem.Draw  -> libXrender.so.1 / libXext.so.6 / libSM.so.6
#   cairosvg         -> no library called "cairo-2" was found (libcairo.so.2)
# Needs root (apt-get). Skip this block on a non-Debian image.
apt-get update -qq
apt-get install -y -qq libxrender1 libxext6 libsm6 libcairo2 \
                       libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0

# conda env remove -n NMR_Structure_Elucidator 
# conda create -y -c conda-forge -n NMR_Structure_Elucidator python=3.7.6

# conda  activate NMR_Structure_Elucidator
pip install jupyter-client==7.0.6
pip install jupyter_core==4.11.1
pip install jupyter-server==1.23.4
pip install jupyterlab==3.5.0
pip install jupyterlab-pygments==0.2.2
pip install jupyterlab_server==2.19.0
pip install jupyterlab-widgets==3.0.7
pip install jupyter_core==4.11.1
conda install -y -c conda-forge nbclassic=0.5.1
conda install -y -c conda-forge nbconvert=7.2.9
conda install -y -c conda-forge nbformat=5.7.3
conda install -y -c conda-forge nbclient=0.6.8
conda install -y nb_conda

pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 torchaudio==0.9.1 -f https://download.pytorch.org/whl/torch_stable.html
pip install pytorch-lightning==0.7.3
pip install six==1.16.0
pip install setuptools==59.5.0
pip install dgl-cu111==0.6.1

pip install svgwrite
pip install wandb==0.15.4
pip install tqdm

pip install Flask==2.2.5
pip install Flask-SocketIO==5.3.6

pip install pandas==1.3.5
pip install matplotlib==3.5.3
pip install plotly==5.18.0

pip install umap-learn==0.5.4
# umap-learn pulls pynndescent, whose 0.7.x line imports scipy.sparse.csr_array
# (scipy>=1.8). Python 3.7 caps scipy at 1.7.3, so pin a compatible release.
pip install pynndescent==0.5.7
pip install rdkit==2023.3.2
pip install ipython==7.15.0
pip install CairoSVG==2.7.0

pip install MolVS==0.1.1
pip install dgllife==0.3.0
pip install transformers==4.26.1
pip install ipykernel==6.15.0 ipython==7.33.0
pip install tensorboardX==2.6.2.2

# The notebooks call plt.savefig('./_FIGURES/...'), and the release does not
# ship that folder, so create it once to avoid FileNotFoundError.
mkdir -p ./_FIGURES

echo "Installation process completed."

