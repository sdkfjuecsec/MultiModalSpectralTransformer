"""COSY (2-D homonuclear correlation) spectrum reconstruction from 1-D shifts.

The MMST pipeline predicts one 13C shift array and one 1H shift array per
molecule. A COSY spectrum is then *derived* geometrically: any two hydrogens
that sit on neighbouring carbons (and are therefore 3J-coupled) produce a
cross peak at ``(H1, H2)`` and its mirror ``(H2, H1)``; a hydrogen also
produces a diagonal peak at ``(H1, H1)``.

Data model
----------
* ``heavy_atom_dict`` : ``{atom_idx: n_hydrogens}``  (not hydrogens itself)
* ``heavy_atom_hydrogen_shift_dict`` : ``{carbon_idx: [h_shift, ...]}``
* ``carbon_dict`` : ``{carbon_idx: [neighbour_carbon_idx, ...]}`` where both
  carbons carry at least one hydrogen (i.e. graph edges of H-bearing carbons)

The shift arrays produced by the network are laid out as
``[13C shifts for heavy atoms..., 1H shifts for attached hydrogens...]``; see
:func:`extract_symmetric_hydrogen_shifts` for how the two halves are split.

NOTE (known issue, left unchanged)
    1. :func:`extract_symmetric_hydrogen_shifts` 假设 ``shifts`` 的前
       ``len(heavy_atom_dict)`` 个是 13C，其余按“重原子索引倒序、每个原子取
       最后 n 个 H”来分配。这个顺序是模型输出约定，一旦数据集侧改动就会静默
       错位，没有断言保护。
    2. :func:`average_shifts` 中 ``shift_list.get(i, 0)[0]`` 在 key 缺失时会
       对 int ``0`` 取下标 → ``TypeError``。
    3. :func:`find_symmetric_positions` 用分子自身做子结构匹配
       (``GetSubstructMatches(mol, uniquify=False)``)，在最坏情况下是指数级
       开销；仅在存在对称原子（``CanonicalRankAtoms(breakTies=False)`` 出
       重复值）时才真正进入该分支。
"""

#import plotly.graph_objs as go
from IPython.display import display
from rdkit.Chem import MolFromSmiles, MolToSmiles
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG, display
import os
from tqdm import tqdm
from rdkit.Chem.EnumerateStereoisomers import GetStereoisomerCount,EnumerateStereoisomers
from rdkit.Chem import SDMolSupplier, Draw, MolFromSmiles, MolToSmiles
import pandas as pd
import numpy as np
from rdkit import Chem
import matplotlib.pyplot as plt

from rdkit.Chem import rdmolfiles
import collections

def find_chiral_centers(molecule):
    """Return atom indices of carbon atoms carrying a defined chiral tag.

    Only *specified* stereocentres count (``CHI_UNSPECIFIED`` is skipped), which
    means an unassigned stereocentre is treated as achiral -> its attached
    hydrogens are later collapsed into one peak.

    Args:
        molecule (rdkit.Chem.Mol): input molecule.

    Returns:
        List[int]: sorted-by-construction list of atom indices.
    """
    chiral_centers = []
    
    # Loop through all atoms in the molecule
    for atom in molecule.GetAtoms():
        
        # Get the atomic number and the index (atom number) of the atom
        atomic_num = atom.GetAtomicNum()
        atom_idx = atom.GetIdx()
        
        # Check for chiral tag
        chiral_tag = atom.GetChiralTag()
        
        # Carbon atoms with atomic number 6 are the most common chiral centers
        if atomic_num == 6 and chiral_tag != Chem.ChiralType.CHI_UNSPECIFIED:
            chiral_centers.append(atom_idx)
            
    return chiral_centers

def find_carbons_with_relevant_neighbors(molecule):
    """Build the H-bearing carbon adjacency map used to generate cross peaks.

    Args:
        molecule (rdkit.Chem.Mol): input molecule.

    Returns:
        Dict[int, List[int]]: ``{carbon_idx: [neighbour_carbon_idx, ...]}``.
        Only carbons with at least one hydrogen appear as keys, and only
        neighbouring carbons that also carry hydrogens appear in the lists
        (C-H ... no-H carbons cannot give a COSY cross peak).
    """
    carbon_dict = {}
    
    for atom in molecule.GetAtoms():
        atomic_num = atom.GetAtomicNum()
        atom_idx = atom.GetIdx()
        
        # Check if the atom is a carbon atom
        if atomic_num == 6:
            neighbor_carbons_with_hydrogens = []
            
            # Loop through the neighbors of the current atom
            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()
                neighbor_atomic_num = neighbor.GetAtomicNum()
                
                # Check if the neighbor is also a carbon atom and both have hydrogens
                if neighbor_atomic_num == 6 and neighbor.GetTotalNumHs() > 0 and atom.GetTotalNumHs() > 0:
                    neighbor_carbons_with_hydrogens.append(neighbor_idx)
            
            # Include carbons with hydrogens, even if they don't couple with any other
            if atom.GetTotalNumHs() > 0:
                carbon_dict[atom_idx] = neighbor_carbons_with_hydrogens
                
    return carbon_dict


def find_heavy_atoms_with_hydrogens(molecule):
    """Map every non-hydrogen atom to the number of attached hydrogens.

    Used to (a) know how many 1H shifts to consume and (b) split the network's
    combined shift vector into its 13C and 1H halves.

    Args:
        molecule (rdkit.Chem.Mol): input molecule.

    Returns:
        Dict[int, int]: ``{atom_idx: n_hydrogens}`` for atoms with >= 1 H.
    """
    heavy_atom_dict = {}
    
    # Loop through all atoms in the molecule
    for atom in molecule.GetAtoms():
        atom_idx = atom.GetIdx()
        atomic_num = atom.GetAtomicNum()
        
        # Check if the atom is a heavy atom (i.e., not hydrogen)
        if atomic_num != 1:
            num_hydrogens = atom.GetTotalNumHs()
            
            if num_hydrogens > 0:
                heavy_atom_dict[atom_idx] = num_hydrogens
                
    return heavy_atom_dict


def extract_symmetric_hydrogen_shifts(shifts, heavy_atom_dict):
    """Split the flat shift vector and attach the 1H shifts to their carbons.

    The network emits ``[13C_0, 13C_1, ..., 1H_0, 1H_1, ...]`` where the 13C
    part is indexed by heavy-atom order and the 1H part is allocated to heavy
    atoms in **reverse index order**, each atom taking the last
    ``n_hydrogens`` entries.

    Args:
        shifts (Sequence[Sequence[float]]): flat list of ``[shift]`` pairs as
            produced by ``generate_COSY_dataframe``'s input form.
        heavy_atom_dict (Dict[int, int]): ``{atom_idx: n_hydrogens}``.

    Returns:
        Dict[int, List[List[float]]]: ``{atom_idx: [h_shift_pair, ...]}``.

    Example:
        ``heavy_atom_dict = {0: 3, 2: 1}``, ``shifts`` has 2 C-shifts then 4
        H-shifts -> carbon 2 gets the last 1 H-shift, carbon 0 the preceding 3.

    NOTE (known issue, left unchanged)
        Relies on the reversed-index convention above; nothing here validates
        that the total number of 1H shifts equals ``sum(heavy_atom_dict.values())``.
    """
    num_heavy_atoms = len(heavy_atom_dict)
    
    # Splitting the shifts into heavy atom shifts and hydrogen shifts
    heavy_atom_shifts = shifts[:num_heavy_atoms]
    hydrogen_shifts = shifts[num_heavy_atoms:]
    
    # Initialize the dictionary to store the carbon number and the shift of the attached hydrogens
    carbon_hydrogen_shifts_dict = {}
    
    # Iterate over the heavy_atom_dict in reverse order to pick the hydrogen shifts
    # NOTE: ``heavy_atom_shifts`` is sliced but never used afterwards (dead value).
    for carbon, num_hydrogens in sorted(heavy_atom_dict.items(), key=lambda x: x[0], reverse=True):
        # Pick the last 'num_hydrogens' from the hydrogen_shifts list
        attached_hydrogen_shifts = hydrogen_shifts[-num_hydrogens:]
        
        # Remove the picked hydrogen shifts from the list
        hydrogen_shifts = hydrogen_shifts[:-num_hydrogens]
        
        # Store in the resulting dictionary
        carbon_hydrogen_shifts_dict[carbon] = attached_hydrogen_shifts
    
    return carbon_hydrogen_shifts_dict


def find_symmetric_positions(stereo_smi):
    """https://github.com/rdkit/rdkit/issues/1411

    Find atom-index groups that are symmetry-equivalent (e.g. the two meta
    positions of an aromatic ring). Equivalent atoms must share an averaged
    shift, otherwise the spectrum shows split peaks that do not exist.

    Args:
        stereo_smi (str): SMILES (usually with stereochemistry, so that
            stereocentres break the symmetry they should break).

    Returns:
        List[List[int]]: list of symmetry groups, each a list of atom indices.
        Empty list when the molecule has no symmetry.

    Implementation note:
        ``CanonicalRankAtoms(mol, breakTies=False)`` assigns identical ranks to
        symmetry-equivalent atoms; duplicate ranks therefore mark the groups.
        ``GetSubstructMatches(mol, uniquify=False)`` then enumerates *all*
        automorphisms of the molecular graph, and the first automorphism
        (``matches[0]``) is used to translate ranks into actual atom indices.

    NOTE (known issue, left unchanged)
        Enumerating all automorphisms is factorial in the worst case; for large
        symmetric molecules this is the slowest step of the COSY pipeline.
        Only reached when duplicate ranks *and* >1 automorphism exist.
    """

    mol = Chem.MolFromSmiles(stereo_smi)
    z=list(rdmolfiles.CanonicalRankAtoms(mol, breakTies=False))
    matches = mol.GetSubstructMatches(mol, uniquify=False)
    
    if len(z) != len(set(z)) and len(matches) > 1:
    # if len(matches) > 1:

        # # Get a list with all the duplicate numbers
        symmetric = [item for item, count in collections.Counter(z).items() if count > 1]
        
        # Get a list of lists with the positions of the duplicates in match list
        all_duplicates = []
        for j in symmetric:
            indices = [i for i, v in enumerate(z) if v == j]
            all_duplicates.append(indices)
        
        # Get a list of lists with the positions of the duplicates
        # ``matches[0]`` is the identity mapping, so ``example_match[i]`` maps
        # the *rank list position* i to the corresponding atom index.
        example_match = matches[0]
        sym_dupl_lists = []
        for sub_list in all_duplicates:
            indices_list = []
            for i in sub_list:
                position = example_match[i]
                indices_list.append(position)
            sym_dupl_lists.append(indices_list)
    
    else:
        # no symmetry -> nothing to average
        sym_dupl_lists = []
    return sym_dupl_lists

# Function to check if an atom has hydrogens attached to it
def has_hydrogens(mol, atom_idx):
    """True if the atom carries at least one explicit/implicit hydrogen."""
    atom = mol.GetAtomWithIdx(atom_idx)
    return atom.GetTotalNumHs() > 0


# Function to average shifts for symmetric carbons
def average_shifts(shift_list, sym_groups):
    """Average the shifts within each symmetry group and assign to all members.

    Args:
        shift_list (Dict[int, Sequence[float]]): ``{atom_idx: [shift]}``; the
            value is a *list* because the code always takes ``[0]``.
        sym_groups (List[List[int]]): output of :func:`find_symmetric_positions`.

    Returns:
        Dict[int, List[float]]: ``{atom_idx: [averaged_shift]}`` for every atom
        that belongs to a symmetry group.

    NOTE (known issue, left unchanged)
        ``shift_list.get(i, 0)[0]`` -> for a missing key the default is the
        integer ``0`` and subscripting it raises ``TypeError``. In practice all
        atoms are present, so this never triggers in the MMST flow.
    """
    avg_shifts = {}
    for sym_group in sym_groups:
        # NOTE: nested list over a single element -> ``avg_shift`` is ``[v]``
        avg_shift = [sum([shift_list.get(i, 0)[0] for i in sym_group]) / len(sym_group)]
        for i in sym_group:
            avg_shifts[i] = avg_shift
    return avg_shifts


# Function to update original shift list with averaged values for symmetric carbons
def update_shifts_with_averaged(original_shifts, averaged_shifts):
    """Return a copy of ``original_shifts`` with symmetry-averaged values applied.

    Args:
        original_shifts (Dict[int, Any]): ``{atom_idx: shift}``.
        averaged_shifts (Dict[int, Any]): output of :func:`average_shifts`.

    Returns:
        Dict[int, Any]: a **new** dict (the input is not mutated).
    """
    updated_shifts = original_shifts.copy()  # Make a copy of the original dictionary
    for carbon, avg_shift in averaged_shifts.items():
        updated_shifts[carbon] = avg_shift  # Update with the averaged value
    return updated_shifts


# Updated function to make plotting optional and return coordinates of plotted points
# Updated function to avoid duplicate peaks in the plot
def plot_and_save_cosy_spectrum_with_zoom_no_duplicates(heavy_atom_hydrogen_shift_dict, carbon_dict, chiral_centers, plot=True, xlim=None, ylim=None):
    """Compute (and optionally draw) the set of COSY peaks for one molecule.

    Peak generation rules:
        * diagonal peak ``(H1, H1)`` for every H-bearing carbon (first H only).
        * if either carbon of an edge is a (defined) stereocentre, every
          H1 x H2 combination is emitted in both orders (diastereotopic Hs
          become distinct signals).
        * otherwise only the first hydrogen of each carbon is used, mirrored.

    Args:
        heavy_atom_hydrogen_shift_dict (Dict[int, List[Sequence[float]]]):
            ``{carbon_idx: [h_shift_pair, ...]}``.
        carbon_dict (Dict[int, List[int]]): H-bearing carbon adjacency map.
        chiral_centers (List[int]): atom indices from :func:`find_chiral_centers`.
        plot (bool): draw with matplotlib when True.
        xlim (Optional[Tuple[float, float]]): 1H window.
        ylim (Optional[Tuple[float, float]]): 13C window.

    Returns:
        List[Tuple[float, float]]: unique peaks as ``(1H_ppm, 13C_ppm)`` pairs.
        A ``set`` is used internally, so duplicates (e.g. symmetric carbons
        giving the same shift) collapse automatically.

    NOTE (known issue, left unchanged)
        For non-chiral carbons only ``h1_shifts[0]`` / ``h2_shifts[0]`` is used,
        i.e. a CH2/CH3 group gives a single cross peak instead of one per
        hydrogen. This is a deliberate simplification of the original
        implementation, not an oversight - annotated only.
    """
    plotted_points = set()  # Using a set to automatically handle duplicates
    
    for carbon1, neighbors in carbon_dict.items():
        h1_shifts = heavy_atom_hydrogen_shift_dict.get(carbon1, [])
        if h1_shifts:
            # diagonal peak (only the first H of this carbon)
            plotted_points.add((h1_shifts[0], h1_shifts[0]))
            
        for carbon2 in neighbors:
            h2_shifts = heavy_atom_hydrogen_shift_dict.get(carbon2, [])
            if h1_shifts and h2_shifts:
                is_chiral1 = carbon1 in chiral_centers
                is_chiral2 = carbon2 in chiral_centers

                if is_chiral1 or is_chiral2:
                    # diastereotopic Hs -> every combination, both orders
                    for h1_shift in h1_shifts:
                        for h2_shift in h2_shifts:
                            plotted_points.add((h1_shift, h2_shift))
                            plotted_points.add((h2_shift, h1_shift))
                else:
                    plotted_points.add((h1_shifts[0], h2_shifts[0]))
                    plotted_points.add((h2_shifts[0], h1_shifts[0]))
                
    x_coords = [x for x, y in plotted_points]
    y_coords = [y for x, y in plotted_points]
    
    if plot:
        # NOTE: ``s``/``s`` above are shift *pairs* (lists), matplotlib accepts
        #       them through the implicit float conversion of a 1-element list.
        plt.scatter(x_coords, y_coords, c='blue', marker='o', label='Cross Peaks', alpha=0.5)
        plt.title('COSY Spectrum with All Diagonal Peaks')
        plt.xlabel('Chemical Shift (ppm)')
        plt.ylabel('Chemical Shift (ppm)')
        plt.grid(True)
        # NMR convention: origin at the top-right corner
        plt.gca().invert_xaxis()
        plt.gca().invert_yaxis()
        plt.legend()
        # Apply zooming window if provided
        if xlim:
            plt.xlim(xlim)
        if ylim:
            plt.ylim(ylim)
        plt.show()
    
    return list(plotted_points)  # Converting set back to list


def generate_COSY_dataframe(shifts):
    """Pack a list of ``[13C, 1H]`` shift pairs into the plotting dataframe.

    Args:
        shifts (Sequence[Sequence[float]]): pairs ``[c_shift, h_shift]`` (the
            ``dtype=object`` cast keeps ragged pairs intact).

    Returns:
        pandas.DataFrame: two float columns, ``F2 (ppm)`` = 1H and
        ``F1 (ppm)`` = 13C, i.e. the column names expected by
        ``utils_MMT/plotting_v15_4.py``.
    """
    # save shifts as dataframe
    df_dft = pd.DataFrame()
    # column 0 of each pair -> 13C, column 1 -> 1H
    c_shifts = np.array(shifts, dtype=object)[:,0] 
    h_shifts = np.array(shifts, dtype=object)[:,1]
    df_dft['F2 (ppm)'] = h_shifts
    df_dft['F1 (ppm)'] = c_shifts
    return df_dft
