"""Similarity and plotting utilities for 2D HSQC spectra.

The module compares an experimental HSQC spectrum with simulated spectra
(ACD, ML, DFT, etc.). Peak positions are taken from DataFrames with columns
``F2 (ppm)`` (1H) and ``F1 (ppm)`` (13C), rescaled into a unit-ish range,
aligned with several point-matching strategies, and finally scored with a
cosine, Euclidean, or Pearson metric.

Typical workflow:
    1. ``similarity_calculations`` selects an alignment ``mode`` and a
       ``similarity_type`` and returns the score plus rescaled point pairs.
    2. ``plot_assignment_points`` / ``plot_all_similarities`` visualise the
       matching for debugging.
    3. Small RDKit/MolVS helpers convert SMILES and fingerprints.

All plotting functions call ``plt.show()`` and therefore have visible side
effects. No module-level state is modified.
"""

# Standard library imports
import os
import random
import shutil
import sys

# Third-party imports for scientific computing
import numpy as np
import scipy.stats
from scipy.optimize import linear_sum_assignment
from scipy.spatial import distance
from scipy.spatial.distance import cdist

# Visualization libraries
import matplotlib.pyplot as plt

# RDKit imports for cheminformatics
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, MolFromSmiles, MolToSmiles, rdMolDescriptors
from rdkit.Chem.Descriptors import ExactMolWt
from rdkit.Chem.MolStandardize.rdMolStandardize import ChargeParent
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

# MolVS for molecule standardization
from molvs import Standardizer

# Local path adjustments



def calculate_weight(smiles_str):
    """Return the exact molecular weight of a SMILES string.

    Args:
        smiles_str: A SMILES string.

    Returns:
        The exact molecular weight in Dalton (float). If the SMILES cannot be
        parsed, RDKit raises/propagates an exception because no None check is
        performed.

    Side effects:
        None.
    """
    m = Chem.MolFromSmiles(smiles_str)
    return ExactMolWt(m)

def generate_fingerprint(input_df_or_smiles, output_type = ["list","tensor", "array"], nbits = 512):
    """Compute a Morgan (ECFP4-like, radius 2) bit fingerprint.

    Accepts either a DataFrame with a ``SMILES`` column or a raw SMILES
    string. The fingerprint is returned as a Python list by default; if
    ``output_type`` is exactly ``"array"`` it is returned as a float16 numpy
    array.

    Args:
        input_df_or_smiles: DataFrame containing ``SMILES``, or a SMILES str.
        output_type: ``"array"`` selects numpy output; anything else yields a
            list. (The default is a list, not a scalar string.)
        nbits: Fingerprint length in bits.

    Returns:
        list or numpy.ndarray of fingerprint bits.

    Side effects:
        None.

    NOTE (known issue, left unchanged): the ``output_type == "tensor"`` branch
    calls ``torch.Tensor`` but ``torch`` is never imported in this module, so
    that path raises NameError. Also, ``output_type`` defaults to the list
    ``["list", "tensor", "array"]``, which never equals the string ``"array"``,
    so a plain list is always returned unless the caller passes a string.
    """
    # just calculate Morgan fingerprint of single example
    try:
        sample = input_df_or_smiles['SMILES']
        pattern1 = Chem.MolFromSmiles(sample)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(pattern1, 2, nBits=nbits)
        fingerprint = list(fp1)
    except:
        pattern1 = Chem.MolFromSmiles(input_df_or_smiles)
        fp1 = AllChem.GetMorganFingerprintAsBitVect(pattern1, 2, nBits=nbits)
        fingerprint = list(fp1)
        
    if output_type == "tensor":
        fingerprint = torch.Tensor(fingerprint)
    elif output_type == "array":
        fingerprint = np.array(fingerprint, dtype=np.float16)
    return fingerprint

  
def mol2mol_file(mol, folder=None, name=None):
    """Write a single RDKit molecule to ``<folder>/<name>.mol``.

    Adds explicit hydrogens and 3D coordinates before embedding, which the
    downstream ICOLOS workflow requires. The file is written as a one-record
    MDL mol/SDF file (SDWriter with a single molecule).

    Args:
        mol: RDKit ``Mol`` object (2D/3D coordinates not required).
        folder: destination directory; defaults to a hard-coded absolute Linux
            path if omitted.
        name: file stem; a random 4-digit number if omitted.

    Returns:
        The full path of the written ``.mol`` file.

    Side effects:
        Creates the output file on disk and runs ETKDG embedding (random
        seed fixed to ``0xf00d``).
    """
    if name==None:
        rand_num = random.randint(0,1000)
        name = "%04d" % (rand_num)
    if folder==None:
        folder= "/projects/cc/knlr326/4_Everything/trash"
        
    # add 3D information to mol
    mol = Chem.AddHs(mol, addCoords=True)
    AllChem.EmbedMolecule(mol,randomSeed=0xf00d)
    save_path = os.path.join(folder,name+".mol")
    writer = Chem.SDWriter(save_path)
    writer.write(mol)
    writer.close()
    return save_path
    

def filter_out_zeros(input_list_1, input_list_2):
    """
    Filters out pairs of elements from input_list_1 and input_list_2 where both
    elements in the pair are close to zero (Close because I modified the numbers slightly because of duplicate entries for some simulation techniques which would otherwise get lost). Returns two new lists containing the
    filtered elements from input_list_1 and input_list_2, respectively.

    Args:
    - input_list_1: A list of pairs of numbers.
    - input_list_2: A list of pairs of numbers.

    Returns:
    - new_list_1: A list containing the elements from input_list_1 that are not
                  close to zero.
    - new_list_2: A list containing the elements from input_list_2 that correspond
                  to the elements in new_list_1 and are not close to zero.
    """
    new_list_1, new_list_2 = [],[]
    for i,j in zip(input_list_1, input_list_2 ):
        if (abs(i[0])+abs(i[1]))<0.0001 or (abs(j[0]) + abs(j[1]))<0.0001:
            continue
        else:
            new_list_1.append(list(i))
            new_list_2.append(list(j))
    new_list_1 = np.array(new_list_1)
    new_list_2 = np.array(new_list_2)
    return new_list_1, new_list_2


#from utils.functions_1 import euclidean_distance_zero_padded, #euclidean_distance_advanced,min_sum_adv,hungarian_zero_padded,hungarian_advanced_euc,plot_assignment_points
def similarity_calculations(df_1, df_2, mode=["min_sum_zero", "min_sum_nn", "min_sum_trunc", "euc_dist_zero","euc_dist_nn", "euc_dist_trunc","hung_dist_zero","hung_dist_trunc", "hung_dist_nn" ], \
                            similarity_type=["euclidean","cosine_similarity","pearson_similarity"],  \
                            error=["sum","avg", "avg_scaled"], \
                            assignment_plot = True, sim_technique=None):
    """Align two HSQC peak lists and score their similarity.

    Both DataFrames are rescaled (1H by ``ppm/11 - 0.5``, 13C by
    ``ppm/180 - 0.5``), transposed into ``(n, 2)`` point arrays, aligned
    according to ``mode`` and then scored according to ``similarity_type``.

    Alignment modes:
        * ``min_sum_*``: sort by decreasing ``x + y`` (optionally pad to the
          longer list / truncate to the shorter / nearest-neighbour).
        * ``euc_dist_*``: greedy nearest Euclidean matching (zero padding,
          truncation, or advanced re-matching).
        * ``hung_dist_*``: Hungarian (linear sum assignment) matching.

    Args:
        df_1: experimental peak DataFrame with ``F2 (ppm)``/``F1 (ppm)``.
        df_2: simulated peak DataFrame with the same columns.
        mode: alignment mode name (a single string, despite the list default).
        similarity_type: ``"cosine_similarity"``, ``"euclidean"`` or
            ``"pearson_similarity"`` (single string despite list default).
        error: Euclidean error aggregation: ``"sum"``, ``"avg"`` or
            ``"avg_scaled"``.
        assignment_plot: when True, calls ``plot_assignment_points``.
        sim_technique: optional label used to jitter duplicate ACD/ML/DFT peaks.

    Returns:
        Tuple ``(score, rescaled_points_1, rescaled_points_2)`` for the
        selected metric. ``score`` is a numpy scalar/float. Returns ``None``
        if an unsupported ``similarity_type`` is supplied.

    Side effects:
        May open a matplotlib window (``assignment_plot=True``).

    NOTE (known issue, left unchanged): the defaults are lists, but the body
    compares them with strings (``mode == "min_sum_zero"`` etc.). Calling the
    function without explicit ``mode``/``similarity_type``/``error`` therefore
    performs no alignment and falls through every branch, returning ``None``.
    """
    # Rescale and transpose both peak lists into (n, 2) coordinate arrays.

    h_dim_1 = list(np.array(df_1['F2 (ppm)'].astype(float))/11-0.5)
    c_dim_1 = list(np.array(df_1['F1 (ppm)'].astype(float))/180-0.5)
    
    h_dim_2 = list(np.array(df_2['F2 (ppm)'].astype(float))/11-0.5)
    c_dim_2 = list(np.array(df_2['F1 (ppm)'].astype(float))/180-0.5)

    input_list_1 = [h_dim_1, c_dim_1]
    input_list_2 = [h_dim_2, c_dim_2]
    
    # convert it into the right shape
    input_list_1 = np.array(input_list_1).transpose()
    input_list_2 = np.array(input_list_2).transpose()
    
    #sorting by sum of x,y high to low
    ######### Do alignment algorithm instead

    if mode == "min_sum_zero":
        input_list_1,input_list_2, pad_num = padding_to_max_length(input_list_1,input_list_2)
        input_list_1 = np.array(sorted(input_list_1, key = lambda x: -(x[0]+x[1])))
        input_list_2 = np.array(sorted(input_list_2, key = lambda x: -(x[0]+x[1])))

    elif mode == "min_sum_trunc":
        min_len = min(len(input_list_1), len(input_list_2))
        input_list_1 = np.array(sorted(input_list_1, key = lambda x: -(x[0]+x[1])))[:min_len]
        input_list_2 = np.array(sorted(input_list_2, key = lambda x: -(x[0]+x[1])))[:min_len]
    
    elif mode == "min_sum_nn":
        input_list_1, input_list_2 = min_sum_adv(input_list_1,input_list_2, sim_technique=sim_technique)
        
    elif mode == "euc_dist_zero": 
        input_list_1, input_list_2, pad_num = padding_to_max_length(input_list_1,input_list_2)
        input_list_1, input_list_2 = euclidean_distance_zero_padded(input_list_1,input_list_2, pad_num, sim_technique=sim_technique)
        
    elif mode == "euc_dist_trunc": 
        input_list_1, input_list_2, pad_num = padding_to_max_length(input_list_1,input_list_2)
        input_list_1, input_list_2 = euclidean_distance_zero_padded(input_list_1,input_list_2, pad_num, sim_technique=sim_technique)
        input_list_1, input_list_2 = filter_out_zeros(input_list_1, input_list_2)     
        
    elif mode == "euc_dist_nn": 
        input_list_1, input_list_2 = euclidean_distance_advanced(input_list_1,input_list_2)

    elif mode == "hung_dist_zero":
        input_list_1, input_list_2, pad_num = padding_to_max_length(input_list_1,input_list_2)
        input_list_1, input_list_2 = hungarian_zero_padded(input_list_1,input_list_2)
        
    elif mode == "hung_dist_trunc":
        input_list_1, input_list_2, pad_num = padding_to_max_length(input_list_1,input_list_2)
        input_list_1, input_list_2 = euclidean_distance_zero_padded(input_list_1,input_list_2, pad_num, sim_technique=sim_technique)
        input_list_1, input_list_2 = filter_out_zeros(input_list_1, input_list_2)     
        input_list_1, input_list_2 = hungarian_zero_padded(input_list_1,input_list_2)
   
    elif mode == "hung_dist_nn": #NO PADDING
        input_list_1, input_list_2 = hungarian_advanced_euc(input_list_1,input_list_2)
    
    
    if assignment_plot == True:
        plot_assignment_points(input_list_1, input_list_2, mode)
        pass
    #import IPython; IPython.embed();    
    
    from scipy.spatial import distance
    if similarity_type == "cosine_similarity":
        from scipy.spatial import distance
        list_points_1 = np.array(input_list_1, dtype=object)
        list_points_2 = np.array(input_list_2, dtype=object)
        Aflat = np.hstack(list_points_1)
        Bflat = np.hstack(list_points_2)
        # Aflat = Aflat - Aflat.mean()
        # Bflat = Bflat - Bflat.mean()
        cos_sim = 1 - distance.cosine(Aflat, Bflat)
        return cos_sim, np.array([(input_list_1[:,0]+0.5)*11,(input_list_1[:,1]+0.5)*180]).transpose(), np.array([(input_list_2[:,0]+0.5)*11,(input_list_2[:,1]+0.5)*180]).transpose()
    
    elif similarity_type == "euclidean":
        sum_dist = 0
        max_dist = 0
        for sample_1, sample_2 in zip(input_list_1, input_list_2):
            dst = distance.euclidean(sample_1, sample_2)
            sum_dist+=dst
            max_dist = max(dst, max_dist)
            
        if error=="avg":
            ############# new addition #############
            if not "trunc" in mode:
                avg_dist = sum_dist/max(len(input_list_1),len(input_list_2))
            elif  "trunc" in mode:
                avg_dist = sum_dist/min(len(input_list_1),len(input_list_2))    
            return np.array(avg_dist), np.array([(input_list_1[:,0]+0.5)*11,(input_list_1[:,1]+0.5)*180]).transpose(), np.array([(input_list_2[:,0]+0.5)*11,(input_list_2[:,1]+0.5)*180]).transpose()
        
        elif error=="sum":
            return np.array(sum_dist), np.array([(input_list_1[:,0]+0.5)*11,(input_list_1[:,1]+0.5)*180]).transpose(), np.array([(input_list_2[:,0]+0.5)*11,(input_list_2[:,1]+0.5)*180]).transpose()  
        
        elif error=="avg_scaled":
            avg_dist=sum_dist/len(input_list_1)
            avg_dist_scaled = avg_dist/max_dist
            return avg_dist_scaled, np.array([(input_list_1[:,0]+0.5)*11,(input_list_1[:,1]+0.5)*180]).transpose(), np.array([(input_list_2[:,0]+0.5)*11,(input_list_2[:,1]+0.5)*180]).transpose()  
        
    elif similarity_type == "pearson_similarity":
        pearson_sim = calculate_pearson_similarity(input_list_1, input_list_2)
        return pearson_sim, np.array([(input_list_1[:,0]+0.5)*11,(input_list_1[:,1]+0.5)*180]).transpose(), np.array([(input_list_2[:,0]+0.5)*11,(input_list_2[:,1]+0.5)*180]).transpose()

    

def calculate_pearson_similarity(spectrum1, spectrum2):
    """Computes the Pearson correlation similarity between two 2D HSQC spectra.

    The coefficient is computed independently for the 1H and 13C coordinates
    and then averaged.

    Parameters:
    spectrum1 (numpy.ndarray): The first spectrum, with shape (n, 2).
    spectrum2 (numpy.ndarray): The second spectrum, with shape (m, 2).

    Returns:
    float: The mean Pearson correlation of the two coordinate axes.

    Side effects:
        None.

    NOTE (known issue, left unchanged): ``scipy.stats.pearsonr`` requires the
    two inputs to have the same length. When the spectra have different peak
    counts (n != m) this raises a ValueError; callers must pad/truncate first.
    """
    # Extract the x and y coordinates from the spectra
    x1 = spectrum1[:, 0]
    y1 = spectrum1[:, 1]
    x2 = spectrum2[:, 0]
    y2 = spectrum2[:, 1]

    # Compute the Pearson correlation coefficient
    correlation, _ = scipy.stats.pearsonr(x1, x2)
    correlation += scipy.stats.pearsonr(y1, y2)[0]
    correlation /= 2
    return correlation


def padding_to_max_length(input_list_1,input_list_2,  dim = 2):
    """Zero-pad the shorter of two point arrays to match the longer one.

    Args:
        input_list_1: numpy array of shape ``(n, dim)``.
        input_list_2: numpy array of shape ``(m, dim)``.
        dim: number of coordinates per point (2 for HSQC).

    Returns:
        Tuple ``(input_list_1, input_list_2, pad_num)`` where both arrays have
        equal length and ``pad_num`` is the number of appended zero rows.

    Side effects:
        None (new arrays are allocated).
    """
    
    if len(input_list_2) > len(input_list_1):
        pad_num = len(input_list_2)-len(input_list_1)
        padding_matrix = np.zeros((pad_num,dim))
        input_list_1 = np.concatenate((input_list_1, padding_matrix), axis=0)
    elif len(input_list_2) < len(input_list_1):
        pad_num = len(input_list_1)-len(input_list_2)
        padding_matrix = np.zeros((pad_num,dim))
        input_list_2 = np.concatenate((input_list_2, padding_matrix), axis=0)
    else:
        pad_num=0
        pass
    return input_list_1,input_list_2, pad_num
    


def plot_assignment_points(input_list_orig, input_list_sim, mode):
    """Plot the two matched point sets and draw lines between paired points.

    Args:
        input_list_orig: array of experimental points, shape ``(n, 2)``.
        input_list_sim: array of simulated points, same length as the original
            (padding should already have been applied).
        mode: alignment mode label used as the plot title.

    Returns:
        None.

    Side effects:
        Opens a matplotlib figure via ``plt.show()``.

    NOTE (known issue, left unchanged): pairing is by list position
    (``assignment = range(len(input_list_orig))``), not by the actual
    alignment computed by the caller, so the drawn lines need not correspond
    to the real point assignment.
    """
    assignment = list(range(len(input_list_orig)))

    points1 = input_list_orig
    points2 = input_list_sim

    # fig, ax = plt.subplots(figsize=(5,5), dpi=80)

    plt.plot(points1[:,0], points1[:,1],'bo', markersize = 10)
    plt.plot(points2[:,0], points2[:,1],'rs',  markersize = 7)
    for p in range(len(points1)):
        plt.plot([points1[p,0], points2[assignment[p],0]], [points1[p,1], points2[assignment[p],1]], 'k')

    plt.title(mode)
    # plt.axes().set_aspect('equal')
    plt.show()

#######################################################################################################


def min_sum_adv(input_list_1,input_list_2, sim_technique=None):
    """Match by min-sum order, then re-match unmatched extra points.

    The first ``min(len(a), len(b))`` points of each set are sorted by
    decreasing ``x + y`` and paired by index. Points left over in the longer
    set are matched again against the full shorter set with
    ``euclidean_distance_uneven`` and appended to both arrays so the outputs
    have equal length.

    Args:
        input_list_1: array of points, shape ``(n, 2)``.
        input_list_2: array of points, shape ``(m, 2)``.
        sim_technique: passed through for duplicate-peak jittering.

    Returns:
        Tuple of two equal-length numpy arrays with the aligned points.

    Side effects:
        None.
    """
    input_list_1 = np.array(sorted(input_list_1, key = lambda x: -(x[0]+x[1])))
    input_list_2 = np.array(sorted(input_list_2, key = lambda x: -(x[0]+x[1])))
    min_length = min(len(input_list_1),len(input_list_2))
    input_list_1_part_1 = input_list_1[:min_length]
    input_list_2_part_1 = input_list_2[:min_length]
    input_list_1_align = list(input_list_1_part_1)
    input_list_2_align = list(input_list_2_part_1)
    
    if len(input_list_2) > len(input_list_1):
        input_list_2_part_2 = input_list_2[min_length:]
        
        input_list_1_part_2, input_list_2_part_2  = euclidean_distance_uneven(input_list_1, input_list_2_part_2, sim_technique=sim_technique)
        input_list_1_align.extend(list(input_list_1_part_2))
        input_list_2_align.extend(list(input_list_2_part_2))        

        
    elif len(input_list_2) < len(input_list_1):
        input_list_1_part_2 = input_list_1[min_length:]
        
        input_list_1_part_2, input_list_2_part_2 = euclidean_distance_uneven(input_list_1_part_2, input_list_2, sim_technique=sim_technique)
        input_list_1_align.extend(list(input_list_1_part_2))
        input_list_2_align.extend(list(input_list_2_part_2))     


    return np.array(input_list_1_align), np.array(input_list_2_align)


def hungarian_zero_padded(input_list_1,input_list_2):
    """Optimal one-to-one assignment of two point sets (Hungarian algorithm).

    A cost matrix of pairwise Euclidean distances is built with ``cdist`` and
    solved with ``scipy.optimize.linear_sum_assignment``; each point is then
    reordered so both output arrays list the matched pairs in the same order.

    Adapted from https://stackoverflow.com/questions/39016821/
    minimize-total-distance-between-two-sets-of-points-in-python

    Args:
        input_list_1: array of points, shape ``(n, 2)``.
        input_list_2: array of points, shape ``(m, 2)``.

    Returns:
        Tuple of two numpy arrays, each of length ``min(n, m)``, containing
        the assigned points.

    Side effects:
        None. The local name ``b`` shadows the row index returned by
        ``linear_sum_assignment`` but is re-bound by ``enumerate``, which is
        intentional here.
    """
    C = cdist(input_list_1, input_list_2)
    b, assigment = linear_sum_assignment(C)

    # make array to list
    input_list_1_list = [list(i) for i in list(input_list_1)]
    input_list_2_list = [list(i) for i in list(input_list_2)]

    assigned_list = [[input_list_1_list[i],input_list_2_list[b]] for i, b in enumerate(assigment)]
    output_list_1 = [assigned_list[i][0] for i in range(len(assigned_list))]
    output_list_2 = [assigned_list[i][1] for i in range(len(assigned_list))]
    
    output_array_1 = np.array(output_list_1)
    output_array_2 = np.array(output_list_2)
    return np.array(output_array_1), np.array(output_array_2)



def hungarian_advanced_euc(input_list_1,input_list_2):
    """Combine advanced Euclidean matching with Hungarian re-assignment.

    Runs ``euclidean_distance_advanced`` first to equalise the number of
    points, then applies the Hungarian assignment to the resulting arrays.

    Args:
        input_list_1: array of points, shape ``(n, 2)``.
        input_list_2: array of points, shape ``(m, 2)``.

    Returns:
        Tuple of two aligned numpy arrays.

    Side effects:
        None.
    """
    #Hungarian advanced
    input_list_1_euc,input_list_2_euc = euclidean_distance_advanced(input_list_1,input_list_2)
    input_list_1_euc_hung,input_list_2_euc_hung = hungarian_zero_padded(input_list_1_euc,input_list_2_euc)
    return np.array(input_list_1_euc_hung),np.array(input_list_2_euc_hung)
   
    
#######################################################################################################
    
def euclidean_distance_zero_padded(input_list_1,input_list_2, num_pad, sim_technique=None):
    """Greedily pair points by ascending Euclidean distance.

    Every pair ``(i, j)`` is scored with ``distance.euclidean`` and sorted.
    Walking the sorted list, a pair is accepted only if neither endpoint has
    been used before. Unmatched points (including zero-padded rows) are
    dropped, so the outputs contain only the accepted pairs and are therefore
    shorter than the padded inputs.

    Args:
        input_list_1: array of points, shape ``(n, 2)`` (may contain zero rows).
        input_list_2: array of points, shape ``(m, 2)`` (may contain zero rows).
        num_pad: number of zero rows that were added by padding. Kept for API
            compatibility; it is not used by the (commented-out) padding logic.
        sim_technique: ``"ACD"`` triggers tiny random jitter on duplicate
            coordinates so repeated peaks can still be matched separately.

    Returns:
        Tuple ``(dataset_1, dataset_2)`` of paired points (numpy arrays).

    Side effects:
        May consume random numbers when ``sim_technique == "ACD"``.
    """
    
    ################### For duplicated simulated datapoints #################
    # do correction when peaks fall on the exact same spot in the simulated spectrum
    # Because otherwise when I do the matching later it will see that the peak is already in the list and will not consider it
    # maybe I will find a better logic...
    if sim_technique=="ACD":
        x = []
        y = []
        for i,j in input_list_1:
            if i not in x:
                x.append(i)
            else:
                rand_num = random.randint(0,100)/10000000000000
                x.append(i+rand_num)
            if j not in y:
                y.append(j)
            else:
                rand_num = random.randint(0,100)/10000000000000
                y.append(j+rand_num)

        input_list_1 = np.array([x,y]).transpose()

        x = []
        y = []
        # print(input_list_sim)
        for i,j in input_list_2:
            if i not in x:
                x.append(i)
            else:
                rand_num = random.randint(0,100)/10000000000000
                x.append(i+rand_num)
            if j not in y:
                y.append(j)
            else:
                rand_num = random.randint(0,100)/10000000000000
                y.append(j+rand_num)
        input_list_2 = np.array([x,y]).transpose()    
    ###########################################################################
    
    # calculate euclidean distance
    result = []
    for j in input_list_1:
        for i in input_list_2:
            dst = distance.euclidean(i, j)
            result.append([list(i),list(j),dst])
    
    # sort it for the lowest euclidean distance
    result = sorted(result, key=lambda x:x[-1], reverse=False)

    # This aligns the closest points with each other 
    # and compares the remaining with the zero padding
    dataset_1 = []
    dataset_2 = []
    ### Here I check if any of the combinations was already seen before
    ### Also the matches with the zero padding
    count = 0
    for i in result:
        if ((i[0] not in dataset_2) & (i[1] not in dataset_1)):
            dataset_2.append(i[0])
            dataset_1.append(i[1])
        # ### need to do that count because otherwise zero get more often assigned
        # elif (i[0] == [0,0]) and (i[1] not in dataset_1) and count > num_pad: 
        #     dataset_1.append(i[1])
        #     dataset_2.append([0,0])
        #     count+=1
        # elif (i[0] not in dataset_2) and (i[1] == [0,0]) and count > num_pad:
        #     dataset_2.append(i[0])
        #     dataset_1.append([0,0])
            # count+=1

    return np.array(dataset_1), np.array(dataset_2)
#### To really get the minimum it should minimize on all possible combinations to get the lowest error


def euclidean_distance_uneven(input_list_1,input_list_2, sim_technique=None):
    """Greedy Euclidean pairing of two sets with different sizes.

    Like ``euclidean_distance_zero_padded`` but without padding bookkeeping:
    it sorts all pairwise distances and accepts each pair whose endpoints are
    still unused. The returned arrays are equal length and contain only the
    matched pairs.

    Args:
        input_list_1: array of points, shape ``(n, 2)``.
        input_list_2: array of points, shape ``(m, 2)``.
        sim_technique: one of ``"ACD"``, ``"ML"``, ``"DFT"``, ``"DFT1"``,
            ``"DFT2"`` enables duplicate-coordinate jitter.

    Returns:
        Tuple of two matched numpy arrays.

    Side effects:
        May consume random numbers when jittering is enabled.
    """
    
    ################### For duplicated simulated datapoints #################
    # do correction when peaks fall on the exact same spot in the simulated spectrum
    # Because otherwise when I do the matching later it will see that the peak is already in the list and will not consider it
    # maybe I will find a better logic...
    if sim_technique in ["ACD", "ML", "DFT", "DFT1", "DFT2"]:
        x = []
        y = []
        for i,j in input_list_1:
            if i not in x:
                x.append(i)
            else:
                rand_num = random.randint(0,100)/10000000000000
                x.append(i+rand_num)
            if j not in y:
                y.append(j)
            else:
                rand_num = random.randint(0,100)/10000000000000
                y.append(j+rand_num)
        input_list_1 = np.array([x,y]).transpose()

        x = []
        y = []
        # print(input_list_sim)
        for i,j in input_list_2:
            if i not in x:
                x.append(i)
            else:
                rand_num = random.randint(0,100)/10000000000000
                x.append(i+rand_num)
            if j not in y:
                y.append(j)
            else:
                rand_num = random.randint(0,100)/10000000000000
                y.append(j+rand_num)
        input_list_2 = np.array([x,y]).transpose()    
    ###########################################################################
    
    # calculate euclidean distance
    result = []
    for j in input_list_1:
        for i in input_list_2:
            dst = distance.euclidean(i, j)
            result.append([list(i),list(j),dst])
    
    # sort it for the lowest euclidean distance
    result = sorted(result, key=lambda x:x[-1], reverse=False)
    
    # This aligns the closest points with each other 
    # and compares the remaining with the zero padding
    dataset_1 = []
    dataset_2 = []
    ### Here I check if any of the combinations was already seen before
    ### Also the matches with the zero padding
    for i in result:
        # print(i)
        if ((i[0] not in dataset_2) & (i[1] not in dataset_1)):
            dataset_2.append(i[0])
            dataset_1.append(i[1])
    return np.array(dataset_1), np.array(dataset_2)

def euclidean_distance_advanced(input_list_1,input_list_2, sim_technique=None):
    """Two-pass Euclidean matching for sets of unequal size.

    First matches as many points as possible with ``euclidean_distance_uneven``
    and then collects the points left unmatched in the larger set. Those
    leftovers are matched again against the full smaller set, and both passes
    are concatenated so every point of the larger set is paired exactly once
    and some points of the smaller set are reused.

    Args:
        input_list_1: array of points, shape ``(n, 2)``.
        input_list_2: array of points, shape ``(m, 2)``.
        sim_technique: jitter label for duplicate peaks (see
            ``euclidean_distance_uneven``).

    Returns:
        Tuple of two equal-length numpy arrays with the aligned points.

    Side effects:
        May consume random numbers when jittering is enabled.
    """
    ################### For duplicated simulated datapoints #################
    # do correction when peaks fall on the exact same spot in the simulated spectrum
    # Because otherwise when I do the matching later it will see that the peak is already in the list and will not consider it
    # maybe I will find a better logic...
    if sim_technique in ["ACD", "ML", "DFT", "DFT1", "DFT2"]:

        x = []
        y = []
        for i,j in input_list_1:
            if i not in x:
                x.append(i)
            else:
                rand_num = random.randint(0,100)/10000000000000
                x.append(i+rand_num)
            if j not in y:
                y.append(j)
            else:
                rand_num = random.randint(0,100)/10000000000000
                y.append(j+rand_num)
        input_list_1 = np.array([x,y]).transpose()

        x = []
        y = []
        # print(input_list_sim)
        for i,j in input_list_2:
            if i not in x:
                x.append(i)
            else:
                rand_num = random.randint(0,100)/10000000000000
                x.append(i+rand_num)
            if j not in y:
                y.append(j)
            else:
                rand_num = random.randint(0,100)/10000000000000
                y.append(j+rand_num)
        input_list_2 = np.array([x,y]).transpose()   


    ### first alignment of points with euclidean distance with different number of points in each set
    # input_list_1_pad, input_list_2_pad= padding_to_max_length(input_list_1,input_list_2)
    input_list_1_align, input_list_2_align = euclidean_distance_uneven(input_list_1, input_list_2, sim_technique=sim_technique)

    # # Select the remaining points that have not been matched in the first round for second alignment
    input_list_1_align_part_2 = []
    input_list_2_align_part_2 = []            
    if len(input_list_1)< len(input_list_2):
        for i in input_list_2 :
            if i not in input_list_2_align:
                input_list_2_align_part_2.append(i)

        # match them again with full number of points from other set
        input_list_1_align_part_2, input_list_2_align_part_2 = euclidean_distance_uneven(input_list_1,input_list_2_align_part_2, sim_technique=sim_technique)

    elif len(input_list_1)> len(input_list_2):
        for i in input_list_1:
            if i not in input_list_1_align:
                input_list_1_align_part_2.append(i)

        # match them again with full number of points from other set
        input_list_1_align_part_2, input_list_2_align_part_2 = euclidean_distance_uneven(input_list_1_align_part_2,input_list_2, sim_technique=sim_technique)

    # Combine both to final list
    input_list_1_align = list(input_list_1_align)
    input_list_2_align = list(input_list_2_align)
    input_list_1_align.extend(list(input_list_1_align_part_2))
    input_list_2_align.extend(list(input_list_2_align_part_2))
    return np.array(input_list_1_align), np.array(input_list_2_align)


def plot_all_similarities(min_sum_1, min_sum_2, \
                          min_sum_adv_1, min_sum_adv_2,\
                          euc_dist_zero_1, euc_dist_zero_2,\
                          euc_dist_advanced_1, euc_dist_advanced_2,\
                          hungarian_zero_1, hungarian_zero_2,\
                          hungarian_advanced_1, hungarian_advanced_2,\
                          similarities, label, value):
    """Plot all six alignment variants side-by-side with assignment lines.

    Builds a 3x2 grid of scatter plots (MinSum-Zero/NN, EucDist-Zero/NN,
    HungDist-Zero/NN), drawing a black line between each pair of matched
    points and annotating the score from ``similarities``.

    Args:
        min_sum_1/2: points from the min-sum-zero alignment.
        min_sum_adv_1/2: points from the min-sum nearest-neighbour alignment.
        euc_dist_zero_1/2: points from the Euclidean zero-padded alignment.
        euc_dist_advanced_1/2: points from the advanced Euclidean alignment.
        hungarian_zero_1/2: points from the Hungarian zero-padded alignment.
        hungarian_advanced_1/2: points from the Hungarian nearest-neighbour
            alignment.
        similarities: sequence of six score values indexed in the same order.
        label: two labels for the real and simulated point sets.
        value: molecule/spectrum identifier used in subplot titles.

    Returns:
        None.

    Side effects:
        Opens a matplotlib figure via ``plt.show()``.

    NOTE (known issue, left unchanged): a single ``assigment`` array based on
    ``len(min_sum_1)`` is reused for every subplot. When other alignment
    variants have a different number of points, indexing them with it either
    misaligns the lines or raises IndexError.
    """

    assigment = np.array(range(len(min_sum_1)))
    fig, axs = plt.subplots(3, 2)

    fig.set_figheight(20)
    fig.set_figwidth(15)

    axs[0, 0].plot(min_sum_1[:,0], min_sum_1[:,1],'bo', markersize = 10, label=label[0])
    axs[0, 0].plot(min_sum_2[:,0], min_sum_2[:,1],'rs',  markersize = 7, label=label[1])
    axs[0, 0].plot([5.5], [90.0],'gs',  markersize = 10, label="extra point")
    axs[0, 0].legend()
    N=len(min_sum_1)
    for p in range(N):
        axs[0, 0].plot([min_sum_1[p,0], min_sum_2[assigment[p],0]], [min_sum_1[p,1], min_sum_2[assigment[p],1]], 'k')
    axs[0, 0].set_title(f'{value} - MinSum-Zero: {similarities[0]}')
    axs[0, 0].invert_xaxis()
    axs[0, 0].invert_yaxis()

    axs[0, 1].plot(min_sum_adv_1[:,0], min_sum_adv_1[:,1],'bo', markersize = 10, label=label[0])
    axs[0, 1].plot(min_sum_adv_2[:,0], min_sum_adv_2[:,1],'rs',  markersize = 7, label=label[1])
    axs[0, 1].legend()
    N=len(min_sum_adv_1)
    for p in range(N):
        axs[0, 1].plot([min_sum_adv_1[p,0], min_sum_adv_2[assigment[p],0]], [min_sum_adv_1[p,1], min_sum_adv_2[assigment[p],1]], 'k')
    axs[0, 1].set_title(f'{value} - MinSum-NN: {similarities[1]}')
    axs[0, 1].invert_xaxis()
    axs[0, 1].invert_yaxis()

    axs[1, 0].plot(euc_dist_zero_1[:,0], euc_dist_zero_1[:,1],'bo', markersize = 10, label=label[0])
    axs[1, 0].plot(euc_dist_zero_2[:,0], euc_dist_zero_2[:,1],'rs',  markersize = 7, label=label[1])
    axs[1, 0].plot([5.5], [90.0],'gs',  markersize = 10, label="extra point")
    axs[1, 0].legend()
    N=len(euc_dist_zero_1)
    for p in range(N):
        axs[1, 0].plot([euc_dist_zero_1[p,0], euc_dist_zero_2[assigment[p],0]], [euc_dist_zero_1[p,1], euc_dist_zero_2[assigment[p],1]], 'k')
    axs[1, 0].set_title(f"{value} - EucDist-Zero: {similarities[2]}")
    axs[1, 0].invert_xaxis()
    axs[1, 0].invert_yaxis()
    
    axs[1, 1].plot(euc_dist_advanced_1[:,0], euc_dist_advanced_1[:,1],'bo', markersize = 10, label=label[0])
    axs[1, 1].plot(euc_dist_advanced_2[:,0], euc_dist_advanced_2[:,1],'rs',  markersize = 7, label=label[1])
    axs[1, 1].legend()
    N=len(euc_dist_advanced_1)
    for p in range(N):
        axs[1, 1].plot([euc_dist_advanced_1[p,0], euc_dist_advanced_2[assigment[p],0]], [euc_dist_advanced_1[p,1], euc_dist_advanced_2[assigment[p],1]], 'k')
    axs[1, 1].set_title(f'{value} - EucDist-NN: {similarities[3]}')
    axs[1, 1].invert_xaxis()
    axs[1, 1].invert_yaxis()
    
    axs[2, 0].plot(hungarian_zero_1[:,0], hungarian_zero_1[:,1],'bo', markersize = 10, label=label[0])
    axs[2, 0].plot(hungarian_zero_2[:,0], hungarian_zero_2[:,1],'rs',  markersize = 7, label=label[1])
    axs[2, 0].plot([5.5], [90.0],'gs',  markersize = 10, label="extra point")
    axs[2, 0].legend()
    N=len(hungarian_zero_1)
    for p in range(N):
        axs[2, 0].plot([hungarian_zero_1[p,0], hungarian_zero_2[assigment[p],0]], [hungarian_zero_1[p,1], hungarian_zero_2[assigment[p],1]], 'k')
    axs[2, 0].set_title(f'{value} - HungDist-Zero: {similarities[4]}')
    axs[2, 0].invert_xaxis()
    axs[2, 0].invert_yaxis()
    
    axs[2, 1].plot(hungarian_advanced_1[:,0], hungarian_advanced_1[:,1],'bo', markersize = 10, label=label[0])
    axs[2, 1].plot(hungarian_advanced_2[:,0], hungarian_advanced_2[:,1],'rs',  markersize = 7, label=label[1])
    axs[2, 1].legend()
    N=len(hungarian_advanced_1)
    for p in range(N):
        axs[2, 1].plot([hungarian_advanced_1[p,0], hungarian_advanced_2[assigment[p],0]], [hungarian_advanced_1[p,1], hungarian_advanced_2[assigment[p],1]], 'k')
    axs[2, 1].set_title(f'{value} - HungDist-NN: {similarities[5]}')
    axs[2, 1].invert_xaxis()
    axs[2, 1].invert_yaxis()
    
    for ax in axs.flat:
        ax.set(xlabel='1H-ppm', ylabel='13C-ppm')

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()
    plt.show()
    

    
def calculate_tanimoto_from_two_smiles(smi1, smi2, nbits, extra_info = True):
    """Compute (and optionally print) the Tanimoto similarity of two SMILES.

    Builds Morgan fingerprints (radius 2, ``nbits`` bits) for both molecules
    and returns a rounded Tanimoto coefficient together with a combined ID.

    Args:
        smi1: first SMILES string.
        smi2: second SMILES string.
        nbits: fingerprint length in bits.
        extra_info: when True, also prints the score (see NOTE below).

    Returns:
        Tuple ``(ids_compared, tan_sim)`` where ``ids_compared`` is
        ``smi1 + "_" + smi2`` and ``tan_sim`` is rounded to four decimals.

    Side effects:
        Prints a summary when ``extra_info`` is True.

    NOTE (known issue, left unchanged): both the print statement and the
    ``ids_compared`` expression reference ``selection_list``, which is not
    defined in this module. With ``extra_info == True`` the function raises
    NameError on the print line; with ``extra_info == False`` it raises
    NameError on the ID line. The function is therefore unusable as written
    and appears to be a leftover from a notebook global.
    """
    
    pattern1 = Chem.MolFromSmiles(smi1)
    pattern2 = Chem.MolFromSmiles(smi2)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(pattern1, 2, nBits=nbits)
    fp2 = AllChem.GetMorganFingerprintAsBitVect(pattern2, 2, nBits=nbits)

    tan_sim = DataStructs.TanimotoSimilarity(fp1, fp2)
    tan_sim = round(tan_sim,4)
    if extra_info:
        print(f"Smile 1: {selection_list[0]} \nSmile 2: {selection_list[1]} \nTanimoto score:{tan_sim}")
    
    ids_compared = selection_list[0]+"_"+selection_list[1]
    return (ids_compared,tan_sim)
    
    
def plot_correlation_plot_conf(all_H_data_non_zero_T, all_C_data_non_zero_T, conf_num, mode):
    """Plot H and C correlation scatter plots for a given conformer index.

    For both nuclei it scatters real vs simulated shifts, fits a linear
    regression, and annotates R² and MAE. ``conf_num`` is only used in the
    title; the plot order is fixed (H first, then C).

    Args:
        all_H_data_non_zero_T: two-row array ``[real_H, simulated_H]``.
        all_C_data_non_zero_T: two-row array ``[real_C, simulated_C]``.
        conf_num: conformer number shown in the titles.
        mode: alignment mode shown in the titles.

    Returns:
        None (the function falls through after the second ``plt.show()``).

    Side effects:
        Opens two matplotlib figures via ``plt.show()``.
    """
    plt.scatter(all_H_data_non_zero_T[0], all_H_data_non_zero_T[1], alpha=0.15)
    m, b = np.polyfit(all_H_data_non_zero_T[0], all_H_data_non_zero_T[1], 1)
    plt.plot(all_H_data_non_zero_T[0], m*all_H_data_non_zero_T[0]+b, color='red')
    plt.xlabel("ppm")
    plt.ylabel("ppm")
    plt.title(f"H correlation between real and {conf_num} generated spectrum with {mode} alignment")
    #calculate R value
    r_H = np.corrcoef(all_H_data_non_zero_T[0], all_H_data_non_zero_T[1])
    plt.text(0.5,9, f'$R^2$ value: {str(r_H[0][1])[:6]}', fontsize=10)
    #calculate MAE value         
    MAE_H = np.mean(np.absolute(all_H_data_non_zero_T[0]-all_H_data_non_zero_T[1]))
    plt.text(0.5,8.5, f'MAE value: {str(MAE_H)[:6]}', fontsize=10)
    plt.xticks(np.arange(0, 11,2))
    plt.yticks(np.arange(0, 11,2))
    plt.show()

    plt.scatter(all_C_data_non_zero_T[0], all_C_data_non_zero_T[1], alpha=0.15)
    m, b = np.polyfit(all_C_data_non_zero_T[0], all_C_data_non_zero_T[1], 1)
    plt.plot(all_C_data_non_zero_T[0], m*all_C_data_non_zero_T[0]+b, color='red')
    plt.xlabel("ppm")
    plt.ylabel("ppm")
    plt.title(f"C correlation between real and {conf_num} generated spectrum with {mode} alignment")
    #calculate R value
    r_C = np.corrcoef(all_C_data_non_zero_T[0], all_C_data_non_zero_T[1])
    plt.text(5,180, f'$R^2$ value: {str(r_C[0][1])[:6]}', fontsize=10)
    #calculate MAE value         
    MAE_C = np.mean(np.absolute(all_C_data_non_zero_T[0]-all_C_data_non_zero_T[1]))
    plt.text(5,170, f'MAE value: {str(MAE_C)[:6]}', fontsize=10)
    plt.xticks(np.arange(0, 201,25))
    plt.yticks(np.arange(0, 201,25))
    plt.show()
    
    
def plot_correlation_plot(all_H_data_non_zero_T, all_C_data_non_zero_T, sim_tech, mode):
    """Plot H/C correlation scatter plots for a simulation technique.

    Identical to ``plot_correlation_plot_conf`` but titles mention
    ``sim_tech`` instead of a conformer number, and the H/C MAE values are
    returned.

    Args:
        all_H_data_non_zero_T: two-row array ``[real_H, simulated_H]``.
        all_C_data_non_zero_T: two-row array ``[real_C, simulated_C]``.
        sim_tech: simulation technique label used in the titles.
        mode: alignment mode used in the titles.

    Returns:
        Tuple ``(MAE_H, MAE_C)`` of mean absolute errors.

    Side effects:
        Opens two matplotlib figures via ``plt.show()``.
    """
    plt.scatter(all_H_data_non_zero_T[0], all_H_data_non_zero_T[1], alpha=0.15)
    m, b = np.polyfit(all_H_data_non_zero_T[0], all_H_data_non_zero_T[1], 1)
    plt.plot(all_H_data_non_zero_T[0], m*all_H_data_non_zero_T[0]+b, color='red')
    plt.xlabel("ppm")
    plt.ylabel("ppm")
    plt.title(f"H correlation between real and {sim_tech} generated spectrum with {mode} alignment")
    #calculate R value
    r_H = np.corrcoef(all_H_data_non_zero_T[0], all_H_data_non_zero_T[1])
    plt.text(0.5,9, f'$R^2$ value: {str(r_H[0][1])[:6]}', fontsize=10)
    #calculate MAE value         
    MAE_H = np.mean(np.absolute(all_H_data_non_zero_T[0]-all_H_data_non_zero_T[1]))
    plt.text(0.5,8.5, f'MAE value: {str(MAE_H)[:6]}', fontsize=10)
    plt.xticks(np.arange(0, 11,2))
    plt.yticks(np.arange(0, 11,2))
    plt.show()

    plt.scatter(all_C_data_non_zero_T[0], all_C_data_non_zero_T[1], alpha=0.15)
    m, b = np.polyfit(all_C_data_non_zero_T[0], all_C_data_non_zero_T[1], 1)
    plt.plot(all_C_data_non_zero_T[0], m*all_C_data_non_zero_T[0]+b, color='red')
    plt.xlabel("ppm")
    plt.ylabel("ppm")
    plt.title(f"C correlation between real and {sim_tech} generated spectrum with {mode} alignment")
    r_C = np.corrcoef(all_C_data_non_zero_T[0], all_C_data_non_zero_T[1])
    plt.text(5,180, f'$R^2$ value: {str(r_C[0][1])[:6]}', fontsize=10)

    #calculate MAE value         
    MAE_C = np.mean(np.absolute(all_C_data_non_zero_T[0]-all_C_data_non_zero_T[1]))
    plt.text(5,170, f'MAE value: {str(MAE_C)[:6]}', fontsize=10)
    plt.xticks(np.arange(0, 201,25))
    plt.yticks(np.arange(0, 201,25))
    plt.show()
    return MAE_H, MAE_C


    
def plot_compare_scatter_plot_without_direction(df_orig, df_sim, name="Plot", transp=0.50):
    """Overlay two HSQC peak sets as scatter plots (real vs simulated).

    Axes are fixed to 1H 0-11 ppm and 13C 0-200 ppm and inverted so the usual
    NMR orientation is preserved.

    Args:
        df_orig: real spectrum DataFrame with ``F2 (ppm)``/``F1 (ppm)``.
        df_sim: simulated spectrum DataFrame with the same columns.
        name: plot title.
        transp: alpha (opacity) for the scatter points.

    Returns:
        None.

    Side effects:
        Opens a matplotlib figure via ``plt.show()``.
    """
    fig, ax = plt.subplots(figsize=(10,5), dpi=80)
    scatter_x_1 = list(np.array(df_orig['F2 (ppm)'].astype(float)))
    scatter_y_1 = list(np.array(df_orig['F1 (ppm)'].astype(float)))
    ax.scatter(scatter_x_1, scatter_y_1, label="Real",  alpha=transp, color="blue")

    scatter_x_2 = list(np.array(df_sim['F2 (ppm)'].astype(float)))
    scatter_y_2 = list(np.array(df_sim['F1 (ppm)'].astype(float)))
    ax.scatter(scatter_x_2, scatter_y_2, label="Simulated",  alpha=transp, color="red")

    ax.legend()
    plt.title(name)
    plt.xlim(xmin=0,xmax=11)
    plt.ylim(ymin=0,ymax=200)

    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()

    plt.grid()
    plt.show()

    
def plot_compare_scatter_plot_without_direction_all(df_real, df_acd, df_mnova, df_dft, df_ml, labels, transp=0.50):
    """Overlay real and four simulated HSQC peak sets in one scatter plot.

    Args:
        df_real: experimental spectrum DataFrame (``F2``/``F1`` ppm columns).
        df_acd: ACD-simulated spectrum DataFrame.
        df_mnova: MNova-simulated spectrum DataFrame.
        df_dft: DFT-simulated spectrum DataFrame.
        df_ml: ML-predicted spectrum DataFrame.
        labels: five legend labels ordered as real, ACD, MNova, DFT, ML.
        transp: alpha (opacity) for the simulated points.

    Returns:
        None.

    Side effects:
        Opens a matplotlib figure via ``plt.show()``.
    """
    fig, ax = plt.subplots(figsize=(10,5), dpi=80)
    scatter_x_1 = list(np.array(df_real['F2 (ppm)'].astype(float)))
    scatter_y_1 = list(np.array(df_real['F1 (ppm)'].astype(float)))
    ax.scatter(scatter_x_1, scatter_y_1, label=labels[0],  alpha=1, color="green", s=40, marker="s")

    scatter_x_2 = list(np.array(df_acd['F2 (ppm)'].astype(float)))
    scatter_y_2 = list(np.array(df_acd['F1 (ppm)'].astype(float)))
    ax.scatter(scatter_x_2, scatter_y_2, label=labels[1],  alpha=transp, color="blue")
    
    scatter_x_2 = list(np.array(df_mnova['F2 (ppm)'].astype(float)))
    scatter_y_2 = list(np.array(df_mnova['F1 (ppm)'].astype(float)))
    ax.scatter(scatter_x_2, scatter_y_2, label=labels[2],  alpha=transp, color="red")
    
    scatter_x_2 = list(np.array(df_dft['F2 (ppm)'].astype(float)))
    scatter_y_2 = list(np.array(df_dft['F1 (ppm)'].astype(float)))
    ax.scatter(scatter_x_2, scatter_y_2, label=labels[3],  alpha=transp, color="orange")
    
    scatter_x_2 = list(np.array(df_ml['F2 (ppm)'].astype(float)))
    scatter_y_2 = list(np.array(df_ml['F1 (ppm)'].astype(float)))
    ax.scatter(scatter_x_2, scatter_y_2, label=labels[4],  alpha=transp, color="m")
    
    ax.legend()
    plt.title("All spectra together")
    plt.xlim(xmin=0,xmax=11)
    plt.ylim(ymin=0,ymax=200)

    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()

    plt.grid()
    plt.show()

    
     
def correct_dft_calc_sdf_file(file_path):
    """Correct an SDF file containing DFT-calculated NMR data.

    Rewrites the ``atom.dprop.Isotropic sheilding`` property tag into a
    Python-attribute-safe name, strips surrounding brackets from the
    ``averaged_NMR_shifts`` (and shielding) properties, adds explicit
    hydrogens with coordinates, and writes the record back to the same file.

    This function opens an SDF file with DFT-calculated NMR data, replaces a specific line in the file, and
    saves the modified file. It also corrects the averaged NMR shifts property if necessary. Finally, it adds
    the H atoms to the SDF file from the corresponding .mol file and saves the modified SDF file under the
    same name.

    Parameters:
    -----------
    file_path: str
        The path to the SDF file to be corrected.

    Returns:
    --------
    None

    Side effects:
        Overwrites ``file_path`` in place. Reads and then rewrites the file
        twice (raw text edit, then SDF write).

    NOTE (known issue, left unchanged): this function uses ``PandasTools`` and
    ``rdkit.Chem.PandasTools`` but neither ``PandasTools`` nor the ``rdkit``
    package name is imported at module level (only ``from rdkit import Chem,
    ...``). At runtime it therefore raises NameError as soon as it reaches
    ``PandasTools.LoadSDF``.
    """
    
    # Open the SDF file
    with open(file_path, 'r') as f:
        content = f.read()
    # Replace the desired line
    content = content.replace('>  <atom.dprop.Isotropic sheilding>', '>  <atom_dprop_Isotropic_sheilding>')

    # Write the modified content back to the file
    with open(file_path, 'w') as f:
        f.write(content)
        
    data = PandasTools.LoadSDF(file_path)
    if data["averaged_NMR_shifts"][0].startswith("["):
        data["averaged_NMR_shifts"] = data["averaged_NMR_shifts"][0][1:-1]
    try:
        if data["atom_dprop_Isotropic_sheilding"][0].startswith("["):
            data["atom_dprop_Isotropic_sheilding"] = data["atom_dprop_Isotropic_sheilding"][0][1:-1]
    except:
        pass
    mol = data.ROMol.item()
    mol = Chem.AddHs(mol, addCoords=True)
    data.ROMol = mol
    rdkit.Chem.PandasTools.WriteSDF(data, file_path, molColName='ROMol', idName=None, properties=list(data.columns), allNumeric=False)


### TEST ###
#from utils.nmr_calculation_from_dft import run_chiral_and_symmetry_finder, get_molecule_data, get_c_h_connectivity, selecting_shifts, load_shifts_from_sdf
#from utils.nmr_calculation_from_dft import load_real_dataframe, get_similarity_comparison_variations, load_shifts_from_sdf_file
#from utils.nmr_calculation_from_dft import perform_deduplication_if_symmetric, generate_dft_dataframe, plot_spectra_together, load_acd_dataframe_from_file
#from utils.nmr_calculation_from_dft import check_num_peaks_in_acd, check_num_peaks_in_acd, load_mnova_dataframe, load_acd_dataframe, plot_spectra_together
#source_folder = "/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/12_HMDB/HMDB_DFT_ICOLOS_gaus_10_2"
#dft_files = glob.glob(source_folder+"/*/*/*")
#dft_files = [i for i in dft_files if ".sdf" in i]
#dft_files = [i for i in dft_files if "conf" in i]
#len(dft_files)
##dft_files = [i for i in dft_files if "NMR_H2O" in i]
#len(dft_files)
#### Perform corrections of the dft nmr files
#for file_path in tqdm(dft_files[:]):
#    print(file_path)
#    correct_dft_calc_sdf_file(file_path)
    
    


def copy_specific_files(src, dst):
    """Recursively copy selected NMR files from ``src`` into ``dst``.

    Creates ``dst`` (and each visited sub-directory) if needed, recurses into
    sub-folders except ``nmr``, ``e``, ``opt`` and ``conf``, and copies only
    files whose name starts with ``HMDB`` or starts with ``SEL_`` and ends with
    ``.sdf``.

    Parameters:

    src: str - The path to the source directory.
    dst: str - The path to the destination directory.
    The function creates the destination directory if it doesn't exist. Then it iterates through the subfolders and files in the source directory. If the item is a subfolder, it calls this function recursively. If the item is a file and matches the desired filename pattern (starting with "HMDB" or starting with "SEL_" and ending with ".sdf"), it copies the file to the corresponding location in the destination directory.

    Returns:
        None.

    Side effects:
        Creates directories and copies files with ``shutil.copy2`` (preserving
        metadata). ``nmr``, ``e``, ``opt`` and ``conf`` sub-trees are skipped.
    """
    
    # Create the destination directory if it doesn't exist
    if not os.path.exists(dst):
        os.makedirs(dst)

    # Iterate through the subfolders and files in the source directory
    for item in os.listdir(src):
        src_item = os.path.join(src, item)
        dst_item = os.path.join(dst, item)

        # If the item is a subfolder, call this function recursively
        if os.path.isdir(src_item):
            # Skip the subfolder if its name is in the exclusion list
            if item not in ["nmr", "e", "opt", "conf"]:
                copy_specific_files(src_item, dst_item)

        # If the item is a file and matches the desired filename pattern, copy it
        elif os.path.isfile(src_item) and (
            item.startswith('HMDB') or
            (item.startswith('SEL_') and item.endswith('.sdf'))
        ):
            shutil.copy2(src_item, dst_item)

# Define source and destination directories
#src_folder  = '/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/12_HMDB/HMDB_DFT_ICOLOS_gaus_10'
#dst_folder  = "/projects/cc/knlr326/1_NMR_project/1_NMR_data_AZ/12_HMDB/HMDB_DFT_ICOLOS_gaus_10_3"
#copy_specific_files(src_folder, dst_folder)
    

def convert_to_standardized_smiles(smiles):
    """
    Standardize smiles for Molformer.

    Parses without sanitisation, takes the charge parent, standardises with
    ``molvs`` and re-serialises with isomeric information retained. Explicit
    hydrogens ``[H]`` are removed by recursion.

    Args:
        smiles: A SMILES string.

    Returns:
        A standardised SMILES string with no explicit ``[H]``. Note that this
        variant keeps stereochemistry (``isomericSmiles=True``), unlike the
        copy in ``molformer_functions``.

    Side effects:
        None.
    """
    mol = MolFromSmiles(smiles, sanitize=False)
    mol = ChargeParent(mol)
    smol = Standardizer().standardize(mol)
    smi = MolToSmiles(smol, isomericSmiles=True)
    if '[H]' in smi:
        return convert_to_standardized_smiles(smi)
    else:
        return smi
    
    
