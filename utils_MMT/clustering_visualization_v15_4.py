"""Chemical-space clustering visualisation for generated molecules.

Given a dict of ``{target_SMILES: [generated_SMILES, ...]}`` (the output of the
MMST generation step) this module embeds all SMILES into 2-D with t-SNE, UMAP
and PCA and renders three PNG charts (returned as base64 ``<img>`` HTML so they
can be dropped into a notebook/report cell).

Three embedding families are provided because they answer different questions:
    * **t-SNE** – local neighbourhoods (is the target surrounded by its
      generated candidates?)
    * **UMAP** – similar local structure but better global layout
    * **PCA**  – linear baseline; large scale structure only

Helper legend
-------------
``smiles_to_fps``          Morgan fingerprints (radius 2, 512 bits)
``plot_2D``                generic 2-D scatter for the ``number_point_list`` API
``plot_cluster_MMT``       embedding for the MMST (transformer) candidates
``plot_cluster_MMT_2``     same, for the ``{key: [smi, ...]}`` dict shape
``plot_cluster_MF``        embedding for molecular-formula candidates
``plot_cluster_target``    embedding of targets vs a random training subset
``run_cluster_comparision`` end-to-end wrapper reading two CSVs

NOTE (known issue, left unchanged)
    1. ``save_molecule_to_svg`` 调用 ``prob.item()``，要求 ``prob`` 是 torch
       tensor；传 numpy / python float 会 ``AttributeError``。
    2. ``plot_2D`` **会原地修改** ``config.n_samples``（>30 时裁剪成 30），
       而颜色表在裁剪前就按原值创建，二者不一致；且 ``config`` 被多个绘图
       函数共享，副作用会串到后续调用。
    3. ``plot_2D`` / ``plot_2D_similarity`` 的散点从未设置 ``label``，
       所以 ``plt.legend()`` 只会给出 “No artists with labels found” 警告。
    4. ``smiles_to_fps`` 会**静默丢弃**无法解析的 SMILES，
       但调用方仍按原始列表长度切片索引 → 一旦有条目被丢弃，
       指纹与 SMILES 就会错位。
"""

from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import Image
import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Standard library imports
from IPython.display import display, HTML, Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Third-party imports
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, AllChem
from rdkit.Chem.Draw import rdMolDraw2D
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from tqdm import tqdm
from io import BytesIO
import base64

def display_gen_and_trg_molecules(selected_best_SMI_list, trg_conv_SMI_list, show_number):
    """Provide two lists of smiles - first column are generated molecules - second column the target
    molecules then the tanimoto similarity

    For the first ``show_number`` pairs a 1x3 figure is drawn: generated
    structure, target structure, and their Morgan/Tanimoto similarity value.

    Args:
        selected_best_SMI_list (List[str]): generated SMILES (e.g. the best
            candidate per target).
        trg_conv_SMI_list (List[str]): the corresponding target SMILES,
            same order and same length.
        show_number (int): how many pairs to render.

    Returns:
        None (figures are shown inline).

    Note:
        Each pair is wrapped in ``try/except`` so one bad SMILES does not stop
        the whole grid; failures are printed with their index.
    """
    selected_best_mol_list = [Chem.MolFromSmiles(smiles) for smiles in tqdm(selected_best_SMI_list)]
    trg_conv_mol_list = [Chem.MolFromSmiles(smiles) for smiles in tqdm(trg_conv_SMI_list)]

    # Assume the following lists
    mol_list1  = selected_best_mol_list 
    mol_list2  = trg_conv_mol_list
    list1 = selected_best_SMI_list

    # Loop through the molecules and create a separate plot for each pair
    for i in tqdm(range(0, show_number)):
        try:
            fig, axs = plt.subplots(1, 3, figsize=(15, 5))  # 1 row, 3 columns per plot
            # Draw molecules from list 1
            img1 = Draw.MolToImage(mol_list1[i])
            axs[0].imshow(img1)
            axs[0].axis('off')

            # Draw molecules from list 2
            img2 = Draw.MolToImage(mol_list2[i])
            axs[1].imshow(img2)
            axs[1].axis('off')

            # Compute Tanimoto similarity
            # Morgan fingerprint, radius 2 (a.k.a. ECFP4), 512 bits.
            fp1 = AllChem.GetMorganFingerprintAsBitVect(mol_list1[i],  2, nBits=512)
            fp2 = AllChem.GetMorganFingerprintAsBitVect(mol_list2[i],  2, nBits=512)
            similarity = DataStructs.TanimotoSimilarity(fp1, fp2)

            # Display Tanimoto similarity
            axs[2].text(0.5, 0.5, f'Tanimoto Similarity: {similarity:.2f}', fontsize=12, ha='center', va='center')
            axs[2].axis('off')

            plt.tight_layout()
            plt.show()
        except Exception as e:
            # e.g. ``MolFromSmiles`` returned None -> MolToImage/Draw fails
            print(f"An error occurred for index {i}: {e}")

    
################################################################    


def generate_colored_html(smiles, probabilities):
    """Render a SMILES string with a red->green background per character.

    Colour encoding: ``prob`` 0 (model very unsure) -> red, 1 -> green.
    This is the "confidence strip" used under molecule depictions in the
    explainability notebook.

    Args:
        smiles (str): SMILES to colour.
        probabilities (Sequence[float]): per-character probabilities; zipped
            with ``smiles`` so a length mismatch simply truncates.

    Returns:
        str: an HTML ``<span>`` fragment.
    """
    # Start the HTML string
    html_string = '<span style="font-family: monospace;">'

    # Iterate over each character and its associated probability
    for char, prob in zip(smiles, probabilities):
        # Calculate the RGB values for the color gradient
        # NOTE: applied per *character*, so multi-char tokens like "Cl" / "[nH]"
        #       are coloured character-by-character.
        red_value = int((1.0 - prob) * 255)
        green_value = int(prob * 255)
        
        # Convert RGB values to a CSS-readable format
        color = f'rgb({red_value}, {green_value}, 0)'
        
        # Append the character with its associated background color to the HTML string
        html_string += f'<span style="background-color: {color}">{char}</span>'

    # Close the HTML string
    html_string += '</span>'
    
    return html_string



# def save_molecule_to_svg(smiles, probabilities, filename="molecule.svg"):
#     mol = Chem.MolFromSmiles(smiles)
#     d = rdMolDraw2D.MolDraw2DSVG(400, 200)
#     d.drawOptions().useBWAtomPalette()
    
#     atom_colors = {i: get_color(prob.item()) for i, prob in enumerate(probabilities)}
    
#     d.DrawMolecule(mol, highlightAtoms=list(range(mol.GetNumAtoms())), highlightAtomColors=atom_colors)
#     d.FinishDrawing()
    
#     with open(filename, "w") as f:
#         f.write(d.GetDrawingText())

def save_molecule_to_svg(smiles, probabilities):
    """Draw the molecule with atoms coloured by per-atom confidence, as inline HTML.

    Args:
        smiles (str): molecule SMILES.
        probabilities (Sequence[torch.Tensor]): one 0-d tensor per atom
            (the model's per-position confidence).

    Returns:
        str: ``<img src='data:image/svg+xml;base64,...'/>``.

    NOTE (known issue, left unchanged)
        ``prob.item()`` requires a **torch tensor**. Passing plain floats
        (e.g. from a numpy array) raises ``AttributeError``.
        Also, ``probabilities`` is indexed by atom order of *this* SMILES,
        so a canonicalised SMILES would mismatch (see notebook 4.0 notes).
    """
    mol = Chem.MolFromSmiles(smiles)
    d = rdMolDraw2D.MolDraw2DSVG(400, 200)
    # Black & white atom palette so the highlight colours are the only colour.
    d.drawOptions().useBWAtomPalette()
    
    atom_colors = {i: get_color(prob.item()) for i, prob in enumerate(probabilities)}
    
    d.DrawMolecule(mol, highlightAtoms=list(range(mol.GetNumAtoms())), highlightAtomColors=atom_colors)
    d.FinishDrawing()

    # Instead of saving to a file, we'll return the SVG as an HTML image
    svg = d.GetDrawingText().replace('\n', '')
    html = f"<img src='data:image/svg+xml;base64,{base64.b64encode(svg.encode('utf-8')).decode()}'/>"
    
    return html

def get_color(prob):
    """Map a probability to an (R, G, 0) colour: 0 -> red, 1 -> green."""
    red_value = (1.0 - prob)
    green_value = prob
    return (red_value, green_value, 0)

    

def plot_2D(X, number_point_list, config, title):
    """Scatter the 2-D embedding, colouring each group of generated points.

    The first ``config.n_samples`` points are the *targets* (drawn black, large);
    the remaining points come in runs whose lengths are given by
    ``number_point_list``, one colour per run (the candidates generated for
    that target).

    Args:
        X (np.ndarray): ``(n, 2)`` embedding coordinates.
        number_point_list (List[int]): run lengths of the candidate blocks.
        config: object with ``n_samples`` (number of targets). **Mutated**:
            clamped to 30.
        title (str): plot title.

    Returns:
        str: base64 encoded PNG wrapped in an ``<img>`` tag.

    NOTE (known issue, left unchanged)
        1. The colormap is created with the *original* ``config.n_samples``
           **before** the ``>30`` clamp, so the colour count can be larger than
           the number of groups actually drawn (harmless, just wasteful).
        2. ``config.n_samples`` is written back -> shared-config side effect.
        3. Scatter calls have no ``label`` -> ``legend`` warns and is empty.
    """
    # Initialize variables
    colors = cm.get_cmap('tab20', config.n_samples+1) # Adjust the colormap size based on the number of points left after black points
    counter = 0

    if config.n_samples > 30:
        config.n_samples = 30
    # Create a new figure
    fig, ax = plt.subplots()
    # Plot the first few points in black
    for i in range(config.n_samples+1):
        ax.scatter(X[i, 0], X[i, 1], c='black', s=100, alpha=0.7)
    
    # Iterate over the remaining points and plot them in different colors
    # NOTE: the step of this ``range`` is re-evaluated every iteration using the
    # current ``counter``; the ``break`` below prevents indexing past
    # ``number_point_list``.
    for i in range(config.n_samples, len(X), number_point_list[counter]):
        for j in range(number_point_list[counter]-1):
            if i + j < len(X)-1:  # To ensure we don't go out of bounds
                plt.scatter(X[i+j, 0], X[i+j, 1], c=[colors(counter)], s=40, alpha=0.3)
        counter += 1  # Update the counter
        if counter>len(number_point_list)-1:
            break

    plt.title(title)
    plt.legend(loc='best')  # Adjusting legend location for best fit
    plt.show()

    # Save the plot to a BytesIO object and encode it as base64
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
    plt.close(fig)  # Close the figure to free memory
    return html
# Example usage:
# Assuming X is your data, config.MMT_samples is 10, and the title is "2D Plot"
# plot_2D(X, 5, 20, config, "2D Plot")


# Convert SMILES to fingerprints
def smiles_to_fps(smiles_list):
    """Morgan fingerprints for a list of SMILES (invalid entries are dropped).

    Args:
        smiles_list (Iterable[str]): SMILES strings.

    Returns:
        np.ndarray: object array of ``ExplicitBitVect`` (radius 2, 512 bits).
        **Shorter than the input** if some SMILES could not be parsed.

    NOTE (known issue, left unchanged)
        Silent drop -> the caller's ``combined_list_new`` and ``all_fps`` can
        have different lengths, which desynchronises "which point is which
        molecule". Downstream index arithmetic still runs, so the plot would
        be misleading rather than crash.
        Also, sklearn's TSNE/UMAP/PCA expect a numeric matrix; feeding RDKit
        bit-vector objects requires a conversion step (e.g.
        ``np.array([list(fp) for fp in fps])``) that the original code relies
        on implicitly.
    """
    fps = []
    for smiles in tqdm(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512)
            fps.append(fp)
    return np.array(fps)



def plot_cluster_MMT(results_dict, config, mode, stoi, stoi_MF, itos, itos_MF):
    """Embed MMST candidates (``{target: [[candidates...], ...]}`` dict) and plot.

    Args:
        results_dict (Dict[str, Any]): ``{src_smiles: [candidate_list, ...]}``.
            The ``"0"``/``0`` key is **removed in place** (it holds the
            target's own entry, which is not a candidate).
        config: plotting config (``n_samples`` used/clamped by :func:`plot_2D`).
        mode, stoi, stoi_MF, itos, itos_MF: accepted for API symmetry with the
            generation code; **unused** in this function.

    Returns:
        Tuple[List[str], str, str, str]:
            ``(combined_candidates, html_tsne, html_umap, html_pca)``.

    Note:
        The highest-Tanimoto candidate (``[0]`` of each group) is skipped
        (``[1:]``) on the assumption that it is the target itself.

    NOTE (known issue, left unchanged): ``del results_dict[...]`` mutates the
        caller's dict.
    """
    if "0" in results_dict:
        del results_dict["0"]
    if 0 in results_dict:
        del results_dict[0]

    transformed_list_MMT = [[key, [item[0] for item in value[0]]] for key, value in results_dict.items()]
    src_smi_MMT, key_len = list(results_dict.keys()), len(list(results_dict.keys()))
    # this leaves out the molecules with the highest tanimoto similarity to the target 
    # (assuming that it might be the correct molecules)
    combined_list_MMT = [item for sublist in transformed_list_MMT for item in sublist[1][1:]]  
    number_point_list_MMT = [len(sublist[1]) for sublist in transformed_list_MMT ]
    # targets first, then candidates: the first block of points is black
    combined_list_new_MMT = src_smi_MMT+combined_list_MMT
    all_fps_MMT = smiles_to_fps(combined_list_new_MMT)
    
    # Dimensionality Reduction: t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne_MMT = tsne.fit_transform(all_fps_MMT)

    # Dimensionality Reduction: UMAP
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    X_umap_MMT = umap_model.fit_transform(all_fps_MMT)

    # Dimensionality Reduction: PCA
    pca = PCA(n_components=2)
    X_pca_MMT = pca.fit_transform(all_fps_MMT)


    # For testing purposes, let's use some dummy data

    html_TSNE = plot_2D(X_tsne_MMT, number_point_list_MMT, config, 't-SNE Plot')

    # Plot UMAP
    html_UMAP = plot_2D(X_umap_MMT, number_point_list_MMT, config, 'UMAP Plot')

    # Plot PCA
    html_PCA = plot_2D(X_pca_MMT, number_point_list_MMT, config, 'PCA Plot')
    
    return combined_list_MMT, html_TSNE, html_UMAP, html_PCA
    


def plot_cluster_MMT_2(config, gen_dict):
    """Same as :func:`plot_cluster_MMT` but for a ``{key: [smi, ...]}`` dict.

    Here ``config.n_samples`` is set from the number of keys (targets) so that
    the black block in :func:`plot_2D` matches exactly.

    Args:
        config: plotting config; ``n_samples`` is **overwritten**.
        gen_dict (Dict[str, List[str]]): ``{target_smiles: [candidate_smiles]}``.

    Returns:
        Tuple[List[str], str, str, str]: same contract as :func:`plot_cluster_MMT`.
    """
    transformed_list_MMT = [[key, value[0]] for key, value in gen_dict.items()]                           
    src_smi_MMT, key_len = list(gen_dict.keys()), len(list(gen_dict.keys()))
    config.n_samples = key_len                     
    # this leaves out the molecules with the highest tanimoto similarity to the target 
    # (assuming that it might be the correct molecules)
    combined_list_MMT = [item for sublist in transformed_list_MMT for item in sublist[1]] #[1:]]  
    number_point_list_MMT = [len(sublist[1]) for sublist in transformed_list_MMT ]
    combined_list_new_MMT = src_smi_MMT+combined_list_MMT
    all_fps_MMT = smiles_to_fps(combined_list_new_MMT)
    
    # Dimensionality Reduction: t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne_MMT = tsne.fit_transform(all_fps_MMT)

    # Dimensionality Reduction: UMAP
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    X_umap_MMT = umap_model.fit_transform(all_fps_MMT)

    # Dimensionality Reduction: PCA
    pca = PCA(n_components=2)
    X_pca_MMT = pca.fit_transform(all_fps_MMT)

    # For testing purposes, let's use some dummy data
    html_TSNE = plot_2D(X_tsne_MMT, number_point_list_MMT, config, 't-SNE Plot')

    # Plot UMAP
    html_UMAP = plot_2D(X_umap_MMT, number_point_list_MMT, config, 'UMAP Plot')

    # Plot PCA
    html_PCA = plot_2D(X_pca_MMT, number_point_list_MMT, config, 'PCA Plot')
    
    return combined_list_MMT, html_TSNE, html_UMAP, html_PCA        


def plot_cluster_MF(results_dict, config):
    """Embed molecular-formula candidates (``{formula_key: [candidates]}`` dict).

    Args:
        results_dict (Dict[str, List[str]]): ``{source_smiles: [candidate_smiles]}``.
        config: plotting config (``n_samples``).

    Returns:
        Tuple[List[str], str, str, str]: same contract as :func:`plot_cluster_MMT`.
    """
    transformed_list_MF = [[key, value] for key, value in results_dict.items()]
    src_smi_MF, key_len_MF = list(results_dict.keys()), len(list(results_dict.keys()))

    # ``[:]`` is a full slice (a copy of the list); kept for symmetry with the
    # ``[1:]`` variant in the MMT version.
    combined_list_MF = [item for sublist in transformed_list_MF for item in sublist[1][:]]
    number_point_list_MF = [len(sublist[1]) for sublist in transformed_list_MF ]
    combined_list_new_MF = src_smi_MF + combined_list_MF
    all_fps_MF = smiles_to_fps(combined_list_new_MF)
    
    # Dimensionality Reduction: t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne_MF = tsne.fit_transform(all_fps_MF)

    # Dimensionality Reduction: UMAP
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    X_umap_MF = umap_model.fit_transform(all_fps_MF)

    # Dimensionality Reduction: PCA
    pca = PCA(n_components=2)
    X_pca_MF = pca.fit_transform(all_fps_MF)


    # For testing purposes, let's use some dummy data

    html_TSNE = plot_2D(X_tsne_MF, number_point_list_MF, config, 't-SNE Plot')

    # Plot UMAP
    html_UMAP = plot_2D(X_umap_MF, number_point_list_MF, config, 'UMAP Plot')

    # Plot PCA
    html_PCA = plot_2D(X_pca_MF, number_point_list_MF, config, 'PCA Plot')
    
    return combined_list_MF, html_TSNE, html_UMAP, html_PCA
    
    

def plot_cluster_target(smi_list_target, smi_list_trainset, config):
    """Embed the target molecules together with a sample of the training set.

    Args:
        smi_list_target (List[str]): target SMILES (drawn black).
        smi_list_trainset (List[str]): background/training SMILES (drawn blue).
        config: must expose ``data_size`` and ``comparision_number``
            (see :func:`plot_2D_similarity`).

    Returns:
        Tuple[List[str], str, str, str]: ``(combined_smiles, html_tsne,
        html_umap, html_pca)``.
    """
    # targets first so the black block is contiguous at the start of X
    combined_list_new = smi_list_target+smi_list_trainset
    all_fps = smiles_to_fps(combined_list_new)
    
    # Dimensionality Reduction: t-SNE
    tsne = TSNE(n_components=2, random_state=0)
    X_tsne = tsne.fit_transform(all_fps)

    # Dimensionality Reduction: UMAP
    umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, n_components=2)
    X_umap = umap_model.fit_transform(all_fps)

    # Dimensionality Reduction: PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(all_fps)


    # For testing purposes, let's use some dummy data
    number_point_list = len(smi_list_target)
    html_TSNE = plot_2D_similarity(X_tsne, number_point_list, config, 't-SNE Plot')

    # Plot UMAP
    html_UMAP = plot_2D_similarity(X_umap, number_point_list, config, 'UMAP Plot')

    # Plot PCA
    html_PCA = plot_2D_similarity(X_pca, number_point_list, config, 'PCA Plot')
    
    return combined_list_new, html_TSNE, html_UMAP, html_PCA


def plot_2D_similarity(X, number_point_list, config, title):
    """Scatter targets (black) and the first ``config.comparision_number`` others (blue).

    Args:
        X (np.ndarray): ``(n, 2)`` embedding.
        number_point_list (int): number of target points at the start of ``X``.
        config: ``data_size`` (colormap size) and ``comparision_number``
            (how many background points to draw).
        title (str): plot title.

    Returns:
        str: base64 PNG in an ``<img>`` tag.

    NOTE (known issue, left unchanged)
        1. ``i`` is leaked from the first loop; if ``number_point_list == 0``
           the second loop hits ``NameError`` on the unbound ``i``.
        2. ``colors`` is computed but never used here.
        3. No ``label`` on the scatter -> empty legend warning again.
    """
    # Initialize variables
    colors = cm.get_cmap('tab20', config.data_size+1) # Adjust the colormap size based on the number of points left after black points
    counter = 0

    # Create a new figured
    fig, ax = plt.subplots()

    # Plot the first few points in black
    for i in range(number_point_list):
        ax.scatter(X[i, 0], X[i, 1], c='black', s=100, alpha=0.7)

    # Iterate over the remaining points and plot them in different colors
    for j in range(config.comparision_number):
        if i + j < len(X)-1:  # To ensure we don't go out of bounds
            plt.scatter(X[i+j, 0], X[i+j, 1], c="blue", s=40, alpha=0.3)

    plt.title(title)
    plt.legend(loc='best')  # Adjusting legend location for best fit
    plt.show()

    # Save the plot to a BytesIO object and encode it as base64
    tmpfile = BytesIO()
    fig.savefig(tmpfile, format='png')
    encoded = base64.b64encode(tmpfile.getvalue()).decode('utf-8')
    html = '<img src=\'data:image/png;base64,{}\'>'.format(encoded)
    plt.close(fig)  # Close the figure to free memory

    # Return the HTML for the plot image
    return html



def run_cluster_comparision(config):
    """Read the training CSV, sample background molecules, and plot targets vs train.

    Args:
        config: needs ``csv_train_path``, ``csv_SMI_targets`` and
            ``comparision_number``.

    Returns:
        Tuple[pandas.DataFrame, pandas.DataFrame, str, str, str]:
            ``(df_train_sample, df_target, html_tsne, html_umap, html_pca)``.
            The dataframes are the *filtered* single-column (``SMILES``) views.

    Note:
        ``random_state=1`` makes the background sample reproducible.
    """
    # File path to the CSV file
    print("run_cluster_comparision")
    # train_file_path = '/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/15_ZINC270M/ML_NMR_5M_XL_1H_train.csv'

    # Attempting to read the CSV file and select a random subset
    df_train = pd.read_csv(config.csv_train_path)

    # Filtering just the 'SMILES' column
    df_train = df_train[['SMILES']]

    # Selecting a random subset of 32 rows
    # (the actual count is ``config.comparision_number``; the comment is stale)
    df_train = df_train.sample(n=config.comparision_number, random_state=1)
    smi_list_trainset = list(df_train["SMILES"])
    
    # File path to the CSV file
    target_file_path = config.csv_SMI_targets
    # Attempting to read the CSV file and select a random subset
    df_target = pd.read_csv(target_file_path)

    # Filtering just the 'SMILES' column
    df_target = df_target[['SMILES']]
    smi_list_targets = list(df_target["SMILES"])

    combined_list_new, html_TSNE, html_UMAP, html_PCA = plot_cluster_target(smi_list_targets, smi_list_trainset, config)
    
    return df_train, df_target, html_TSNE, html_UMAP, html_PCA
