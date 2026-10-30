##############################################################################
################################ SGNN ########################################
##############################################################################

# Standard library imports
import collections
import glob
import os
import random
import shutil
import time
from argparse import Namespace   # only used for type hints: config is an argparse.Namespace
from typing import Any, Dict, List, Optional, Tuple   # type hints used by the annotations below
from tqdm import tqdm
import collections
from IPython.display import display, SVG
from torch.utils.data import Dataset, DataLoader

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
#import plotly.graph_objs as go
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, MolFromSmiles, rdmolfiles, MolToSmiles, SDMolSupplier, Descriptors, PandasTools, AddHs, rdDepictor
from rdkit.Chem import EnumerateStereoisomers
from rdkit.Chem.Draw import rdMolDraw2D, rdDepictor
from rdkit.Chem.EnumerateStereoisomers import GetStereoisomerCount, EnumerateStereoisomers

# Local application/library specific imports
from utils_MMT.sgnn_code_pl_v15_4 import *
import utils_MMT.dataloaders_pl_v15_4 as dl
# import utils_MMT.clip_functions_v15_4 as cl
import utils_MMT.nmr_calculation_from_dft_v15_4 as ncfd



def contains_hydrogen_rdkit(smiles: str) -> bool:
    """Tell whether a SMILES string contains at least one hydrogen atom.

    Molecules without hydrogen cannot produce a 1H / HSQC spectrum, so they are
    dropped before the expensive SGNN inference (see :func:`run_sgnn`).

    Args:
        smiles: SMILES string, e.g. ``"c1ccccc1"`` (benzene).

    Returns:
        ``True`` when RDKit parses the SMILES *and* an ``H`` atom is present.

    Example:
        >>> contains_hydrogen_rdkit("c1ccccc1")
        True
        >>> contains_hydrogen_rdkit("ClC(Cl)(Cl)Cl")   # CCl4: no hydrogen
        False

    Note:
        Any RDKit failure (invalid SMILES) is swallowed by the ``except`` and
        reported as ``False``, which removes the row from the dataset.
    """
    try:
        # Parse SMILES into a molecule graph; invalid input yields None.
        mol: Optional[Chem.Mol] = Chem.MolFromSmiles(smiles)
        # Implicit hydrogens are not atoms in the graph, so make them explicit.
        mol = Chem.AddHs(mol)
        if mol:
            for atom in mol.GetAtoms():
                if atom.GetSymbol() == 'H':
                    return True
        return False
    except:
        return False

# Calculate molecular weight
def calculate_mw(smiles: str) -> Optional[float]:
    """Return the molecular weight of a SMILES string in Dalton.

    Args:
        smiles: SMILES string, e.g. ``"CC(=O)Oc1ccccc1C(=O)O"`` (aspirin).

    Returns:
        Molecular weight as ``float`` (aspirin -> 180.16), or ``None`` when the
        SMILES cannot be parsed.

    Example:
        >>> calculate_mw("c1ccccc1")      # benzene
        78.114
        >>> calculate_mw("###")           # invalid SMILES
        None
    """
    try:
        # Parse the SMILES; MolFromSmiles returns None for invalid input.
        mol: Optional[Chem.Mol] = Chem.MolFromSmiles(smiles)
        # Descriptors.MolWt requires a molecule, hence the None guard.
        return Descriptors.MolWt(mol) if mol is not None else None
    except:
        # Never let a bad row break the whole filtering pass.
        return None

tqdm.pandas()

# Function to check if a SMILES has disconnected structures
def has_disconnected_smiles(smiles_string: str) -> bool:
    """True when the SMILES encodes more than one disconnected fragment.

    RDKit separates fragments with a dot, so salts and mixtures are detected
    without parsing the molecule at all.

    Args:
        smiles_string: SMILES string, e.g. ``"CC(=O)[O-].[Na+]"``.

    Returns:
        ``True`` if a ``"."`` is present (multi-fragment structure).

    Example:
        >>> has_disconnected_smiles("CCO")
        False
        >>> has_disconnected_smiles("CC(=O)[O-].[Na+]")   # sodium acetate
        True
    """
    # A literal dot in the SMILES means "new fragment", no parsing required.
    return '.' in smiles_string



def run_sgnn(config: Namespace) -> pd.DataFrame:
    """Predict 13C / 1H NMR shifts for every molecule of a SMILES CSV (stage 1).

    Three filters are applied before any inference, which is why the output row
    count is usually much smaller than the input CSV:

    1. the molecule must contain at least one hydrogen;
    2. the molecule must be a single connected fragment (no ``"."``);
    3. the molecular weight must be ``<= config.SGNN_size_filter``.

    The SGNN is then run twice: once in batches of 128 (fast path) and once with
    ``batch_size = 1`` for the molecules that failed in the first pass (usually
    batching artefacts such as odd graph sizes).

    Args:
        config: namespace with ``SGNN_csv_gen_smi`` (input CSV),
            ``SGNN_gen_folder_path`` (output folder) and ``SGNN_size_filter``
            (molecular-weight cutoff in Da).

    Returns:
        DataFrame with one row per successfully processed molecule; it carries the
        SMILES, the predicted shifts and the ``sample-id`` used to join the
        per-modality CSVs later.

    Example:
        >>> config = Namespace(SGNN_csv_gen_smi="data/test_data/IBM_SMI_data_top20.csv",
        ...                    SGNN_gen_folder_path="experiments/SGNN_gen_folder",
        ...                    SGNN_size_filter=550)
        >>> combined_df = run_sgnn(config)
        >>> combined_df.shape          # 20 input molecules, a few dropped by the filters
        (17, 5)
    """
    # Graph representation expected by the pretrained SGNN checkpoint.
    graph_representation: str = "sparsified"

    # Normalisation statistics of the training set: the SGNN predicts z-scores,
    # so the mean/std must be the exact ones used during training.
    target: str = "13C"
    train_y_mean_C, train_y_std_C = load_std_mean(target, graph_representation)
    target = "1H"
    train_y_mean_H, train_y_std_H = load_std_mean(target, graph_representation)
    sgnn_means_stds: Tuple[float, float, float, float] = (
        train_y_mean_C, train_y_std_C, train_y_mean_H, train_y_std_H)

    # Batches of 128 for throughput; complicated molecules are retried one by one.
    batch_size: int = 128
    path_csv: str = config.SGNN_csv_gen_smi
    ML_save_folder: str = config.SGNN_gen_folder_path

    # Read the input table. It must contain a "SMILES" and a "sample-id" column.
    data_df: pd.DataFrame = pd.read_csv(path_csv)

    # Filter 1: drop molecules without hydrogen (no 1H signal to simulate).
    data_df = data_df[data_df['SMILES'].apply(contains_hydrogen_rdkit)]

    # Filter 2: drop multi-fragment SMILES such as salts ("CC(=O)[O-].[Na+]").
    # The leading "~" negates the boolean series.
    data_df = data_df[~data_df['SMILES'].apply(has_disconnected_smiles)]

    # Filter 3: molecular-weight cutoff, e.g. drop everything above 550 Da.
    data_df['Molecular_Weight'] = data_df['SMILES'].apply(calculate_mw)
    data_df_final: pd.DataFrame = data_df[data_df['Molecular_Weight'] <= config.SGNN_size_filter]

    # Create the output folder on first run (SDF files are written into it).
    if not os.path.exists(ML_save_folder):
        os.mkdir(ML_save_folder)

    # Pass 1: batched inference. Returns the successful rows + the ids that failed.
    batch_data_1, failed_ids_1 = main_execute(data_df_final, sgnn_means_stds, ML_save_folder, batch_size)

    # Number of molecules that failed in pass 1 (usually a few percent).
    print(len(failed_ids_1))

    # Pass 2: rerun only the failed molecules, one at a time, to avoid
    # batch-dependent graph errors.
    batch_size = 1
    data_df_final = data_df_final[data_df_final['sample-id'].isin(failed_ids_1)]
    batch_data_2, failed_ids_2 = main_execute(data_df_final, sgnn_means_stds, ML_save_folder, batch_size)

    # Concatenate both passes; ignore_index renumbers 0..n-1.
    combined_df: pd.DataFrame = pd.concat([batch_data_1, batch_data_2], axis=0, ignore_index=True)
    return combined_df



##############################################################################
################################ 1H ##########################################
##############################################################################

# Add this function to read shifts from the SDF file
def read_shifts_from_sdf(file_path: str) -> Dict[int, float]:
    """Read per-atom NMR shifts from the first molecule of an SDF file.

    The SGNN stage stores the predicted shift of every atom in the custom SDF
    property ``_Shift``; this helper just re-reads it into a dictionary.

    Args:
        file_path: path to an SDF file written by the SGNN step, e.g.
            ``".../NMR_1234567.sdf"``.

    Returns:
        Mapping ``atom index -> shift in ppm``, e.g. ``{0: 128.4, 1: 7.26, ...}``.

    Example:
        >>> read_shifts_from_sdf("experiments/SGNN_gen_folder/NMR_1234567.sdf")
        {0: 128.4, 1: 7.26, 2: 7.26}

    Note:
        Only the first record is read (``supplier[0]``), because this pipeline
        writes one molecule per SDF file.
    """
    # Lazy SDF reader: molecules are parsed on access, not upfront.
    supplier: SDMolSupplier = SDMolSupplier(file_path)
    # One molecule per file, so index 0 is the only record (None if unparsable).
    sdf_mol = supplier[0]
    shifts: Dict[int, float] = {}
    # Walk the atom list in canonical order; the index is what the caller keys on.
    for atom in sdf_mol.GetAtoms():
        atom_idx: int = atom.GetIdx()
        # "_Shift" is a string property in SDF, so cast it to float.
        atom_shift: str = atom.GetProp("_Shift")
        shifts[atom_idx] = float(atom_shift)
    return shifts

def lorentzian(x: np.ndarray, x0: float, gamma: float) -> np.ndarray:
    """Lorentzian line shape, the profile of a single NMR resonance.

    The function is normalised so that its integral over ``x`` equals 1 and its
    FWHM (full width at half maximum) equals ``gamma``.

    Args:
        x: chemical-shift axis in ppm, e.g. ``np.linspace(6.0, 8.0, 1000)``.
        x0: centre of the peak in ppm, e.g. ``7.26`` (residual CHCl3).
        gamma: full width at half maximum in ppm, e.g. ``0.01``.

    Returns:
        Intensity at every point of ``x``; the maximum is ``2 / (pi * gamma)``.

    Example:
        >>> x = np.array([7.26])
        >>> lorentzian(x, x0=7.26, gamma=0.01)
        array([63.66])     # = 2 / (pi * 0.01)
    """
    # (1/pi) * (gamma/2) / ((x-x0)^2 + (gamma/2)^2) -- peak height 2/(pi*gamma)
    return (1 / np.pi) * (0.5 * gamma) / ((x - x0) ** 2 + (0.5 * gamma) ** 2)


def simulate_splitting(shifts: np.ndarray, coupling_patterns: List[List[Tuple[float, float]]],
                       gamma: float, spectrometer_frequency: float) -> Tuple[np.ndarray, np.ndarray]:
    """Sum Lorentzian peaks for a whole molecule, one peak per multiplet line.

    Each proton has a coupling pattern: a list of ``(J, intensity)`` pairs where
    ``J`` is the coupling constant in Hz and ``intensity`` is the relative
    intensity of that line. The line position in ppm is
    ``shift + J / spectrometer_frequency``, which converts Hz into ppm.

    Args:
        shifts: chemical shift of every proton in ppm, e.g. ``np.array([7.26, 1.20])``.
        coupling_patterns: one pattern per proton, e.g.
            ``[[(-8.0, 0.25), (0.0, 0.5), (8.0, 0.25)]]`` for a triplet.
        gamma: line width (FWHM) in ppm, e.g. ``0.01``.
        spectrometer_frequency: instrument frequency in MHz, e.g. ``400``.

    Returns:
        ``(x, y)``: the ppm axis and the simulated intensity.

    Example:
        A triplet at 7.26 ppm with J = 8 Hz on a 400 MHz instrument gives three
        lines at ``7.26 - 0.02``, ``7.26`` and ``7.26 + 0.02`` ppm.
    """
    # Build a ppm axis spanning the full shift range plus 1 ppm margins.
    x: np.ndarray = np.linspace(shifts.min() - 1, shifts.max() + 1, 1000)
    # Accumulate the spectrum of every proton into one array.
    y: np.ndarray = np.zeros_like(x)
    for shift, coupling_pattern in zip(shifts, coupling_patterns):
        # Spectrum of a single proton: the sum of its multiplet lines.
        peak: np.ndarray = np.zeros_like(x)
        for J, intensity in coupling_pattern:
            # J is in Hz; J/frequency converts it to ppm before drawing the peak.
            peak += intensity * lorentzian(x, shift + J / spectrometer_frequency, gamma)
        # Note: no scaling by intensity normalisation here, patterns already sum to 1.
        y += peak
    return x, y

def get_adjacent_aromatic_hydrogens(atom: Chem.Atom) -> List[Tuple[Chem.Atom, int]]:
    """Collect the hydrogens attached to the aromatic neighbours of ``atom``.

    These are the *ortho* protons of an aromatic ring; their number determines
    the multiplet shape of an aromatic signal (e.g. two ortho protons -> triplet).

    Args:
        atom: an aromatic carbon carrying (at least) one hydrogen.

    Returns:
        List of ``(hydrogen atom, atom index)`` tuples.

    Example:
        For toluene, the ring carbon C2 has the C3 neighbour whose H is returned,
        so ``len(...)`` is 1 and the C2 signal is simulated as a doublet.
    """
    # Only aromatic neighbours contribute ortho coupling in this model.
    aromatic_neighbors: List[Chem.Atom] = [neighbor for neighbor in atom.GetNeighbors() if neighbor.GetIsAromatic()]
    aromatic_hydrogens: List[Tuple[Chem.Atom, int]] = []
    for aromatic_neighbor in aromatic_neighbors:
        # get_surrounding_hydrogens returns the H atoms bound to that neighbour.
        aromatic_hydrogens.extend(get_surrounding_hydrogens(aromatic_neighbor))
    return aromatic_hydrogens

def get_surrounding_hydrogens(atom: Chem.Atom) -> List[Tuple[Chem.Atom, int]]:
    """Return every hydrogen directly bonded to ``atom``.

    Args:
        atom: any RDKit atom (typically a carbon).

    Returns:
        List of ``(hydrogen atom, hydrogen index)`` tuples; empty when the atom
        carries no hydrogen (e.g. a quaternary carbon).

    Example:
        >>> mol = Chem.AddHs(Chem.MolFromSmiles("C"))   # methane
        >>> len(get_surrounding_hydrogens(mol.GetAtomWithIdx(0)))
        4
    """
    neighboring_hydrogens: List[Tuple[Chem.Atom, int]] = []
    for neighbor in atom.GetNeighbors():
        if neighbor.GetSymbol() == 'H':
            # Keep the index as well: the manifest index is what later code uses.
            neighboring_hydrogens.append((neighbor, neighbor.GetIdx()))
    return neighboring_hydrogens

# Updated analyze_molecule function to combine symmetric hydrogen shifts
def analyze_molecule(mol: Chem.Mol) -> Tuple[List[Dict[str, Any]], Dict[Tuple[str, int], float], Chem.Mol]:
    """Collect one record per hydrogen atom, grouped by its parent heavy atom.

    Every entry describes one proton: which atom it is, whether its parent is
    aromatic, how many *other* protons sit on the same parent (that number
    defines the multiplet, 3 -> quartet, 2 -> triplet, ...), a human-readable
    label such as ``"C3H3"`` (a methyl group) and the parent key used to group
    equivalent protons.

    Args:
        mol: RDKit molecule **with explicit hydrogens** (``Chem.AddHs``) and a
            ``_Shift`` property on every atom, as produced by
            :func:`load_mol_and_assign_shifts`.

    Returns:
        ``(nmr_data, assigned_shifts, mol)`` where

        * ``nmr_data`` -- list of dicts with keys ``atom``, ``aromatic``,
          ``neighbors``, ``label``, ``group_key``;
        * ``assigned_shifts`` -- ``{(element, parent index): shift}``;
        * ``mol`` -- the input molecule (returned unchanged for convenience).

    Example:
        Acetone ``CC(=O)C`` has two methyl groups; each carbon yields three
        entries with ``label="C1H3"`` / ``"C3H3"`` and ``neighbors=2``.

    Note:
        ``assigned_shifts`` keeps a *single* value per parent atom: walking the
        hydrogens of one carbon overwrites the same key, so the last H wins.
    """
    # Every explicit hydrogen in the molecule (implicit H's do not exist here).
    hydrogens: List[Chem.Atom] = [atom for atom in mol.GetAtoms() if atom.GetSymbol() == 'H']
    nmr_data: List[Dict[str, Any]] = []
    assigned_shifts: Dict[Tuple[str, int], float] = {}

    for hydrogen in hydrogens:
        # The first (only) neighbour of an explicit H is the heavy atom it hangs on.
        parent_atom: Chem.Atom = hydrogen.GetNeighbors()[0]
        # Aromatic protons couple differently (ortho coupling) than aliphatic ones.
        is_aromatic: bool = parent_atom.GetIsAromatic()
        # Grouping key: element symbol + atom index, e.g. ("C", 3).
        group_key: Tuple[str, int] = (parent_atom.GetSymbol(), parent_atom.GetIdx())

        # All protons on the same parent atom; "-1" excludes this very hydrogen.
        surrounding_hydrogens: List[Tuple[Chem.Atom, int]] = get_surrounding_hydrogens(parent_atom)
        num_h_neighbors: int = len(surrounding_hydrogens) - 1

        # Label = "<element><index>H<n>", e.g. "C3H3" for a methyl carbon.
        # NOTE: `surrounding_hydrogens.count(hydrogen)` compares an (atom, idx)
        # tuple with a bare atom, so it is always 0; the suffix therefore equals
        # the total number of hydrogens on this parent atom.
        hydrogen_label: str = f"{parent_atom.GetSymbol()}{parent_atom.GetIdx()}H{num_h_neighbors + 1 - surrounding_hydrogens.count(hydrogen)}"

        # One record per hydrogen (duplicates for CH3 are intentional: they are
        # de-duplicated later with `set` in run_1H_generation).
        nmr_data.append({
            'atom': hydrogen,
            'aromatic': is_aromatic,
            'neighbors': num_h_neighbors,
            'label': hydrogen_label,
            'group_key': group_key
        })

        # Shift predicted by the SGNN and stored as an SDF property; the value is
        # a string, so cast to float. Same key -> overwritten by each sibling H.
        assigned_shifts[group_key] = float(hydrogen.GetProp("_Shift"))
    
    return nmr_data, assigned_shifts, mol


def pascals_triangle(n: int) -> List[int]:
    """Return row ``n`` of Pascal's triangle.

    Used to build multiplet intensities: the binomial coefficients of row ``n``
    describe the relative intensities of an ``n+1``-line multiplet.

    Args:
        n: row index, i.e. number of equivalent neighbouring protons (0-based).

    Returns:
        List of ``n+1`` binomial coefficients.

    Example:
        >>> pascals_triangle(0)
        [1]
        >>> pascals_triangle(1)
        [1, 1]
        >>> pascals_triangle(2)
        [1, 2, 1]        # triplet 1:2:1
        >>> pascals_triangle(3)
        [1, 3, 3, 1]     # quartet 1:3:3:1
    """
    if n == 0:
        # Row 0 is the seed.
        return [1]
    else:
        # Build row n from row n-1: each inner element is the sum of its two parents.
        previous_row: List[int] = pascals_triangle(n - 1)
        current_row: List[int] = [1]
        for i in range(len(previous_row) - 1):
            current_row.append(previous_row[i] + previous_row[i + 1])
        # Every row ends with 1.
        current_row.append(1)
        return current_row

def generate_nmr_coupling_pattern(n_neighbors: int, J: float) -> List[Tuple[float, float]]:
    """Multiplet of a proton split by ``n_neighbors`` equivalent protons.

    The ``n+1`` lines follow Pascal's triangle and are normalised so their
    intensities sum to 1; the offsets are multiples of the coupling constant ``J``
    (in Hz), centred on the proton's own shift.

    Args:
        n_neighbors: number of equivalent neighbouring protons (n >= 1).
        J: coupling constant in Hz, e.g. ``7.0``.

    Returns:
        List of ``(offset_in_Hz, relative_intensity)`` pairs.

    Example:
        >>> generate_nmr_coupling_pattern(2, 7.0)     # triplet
        [(-7.0, 0.25), (0.0, 0.5), (7.0, 0.25)]
        >>> generate_nmr_coupling_pattern(3, 7.0)     # quartet
        [(-14.0, 0.125), (-7.0, 0.375), (0.0, 0.375), (7.0, 0.125)]

    Note:
        The offsets come from ``range(-n//2, n//2 + 1)``, which is only
        symmetric for even ``n``. Measured output for ``n = 1`` is
        ``[(-J, 0.5), (0, 0.5)]`` instead of the physically expected
        ``(-J/2, +J/2)``, and for ``n = 3`` it is ``(-2J, -J, 0, J)`` instead of
        ``(-1.5J, -0.5J, +0.5J, +1.5J)``: odd multiplets are shifted by half a
        coupling constant.
    """
    # Binomial coefficients of row n -> relative line intensities.
    coefficients: List[int] = pascals_triangle(n_neighbors)
    # Normalise by 2^n so the intensities sum to 1 (e.g. 1:2:1 -> .25:.5:.25).
    intensities: List[float] = [coef / (2 ** n_neighbors) for coef in coefficients]
    # Offsets in Hz: integer multiples of J, from -n//2 to n//2.
    Js: List[float] = [i * J for i in range(-n_neighbors // 2, n_neighbors // 2 + 1)]
    # Pair every offset with its intensity, e.g. [(-7.0, 0.25), (0.0, 0.5), ...].
    return list(zip(Js, intensities))


def load_mol_and_assign_shifts(file_path: str) -> Chem.Mol:
    """Load one SDF record and attach the SGNN shifts as an atom property.

    The SDF stores the predicted shifts in the property ``averaged_NMR_shifts``
    (space-separated strings). This helper re-attaches them to the molecule as
    ``_Shift`` so the simulation code can read them per atom.

    Args:
        file_path: SDF file written by the SGNN stage, one molecule per file.

    Returns:
        RDKit molecule with explicit hydrogens and a ``_Shift`` property on each
        atom, ready for :func:`analyze_molecule`.

    Example:
        >>> mol = load_mol_and_assign_shifts("experiments/SGNN_gen_folder/NMR_123.sdf")
        >>> mol.GetAtomWithIdx(0).GetProp("_Shift")
        '128.4'
    """
    # PandasTools.LoadSDF turns each record into a row; properties become columns.
    data: pd.DataFrame = PandasTools.LoadSDF(file_path)
    # Only one molecule per file, so take the single element out of the series.
    mol: Chem.Mol = data["ROMol"].item()
    # Add explicit hydrogens (with 3D-ish coordinates) so they can carry shifts too.
    mol = AddHs(mol, addCoords=True)

    # Space-separated shift list, e.g. "128.4 137.2 7.26 7.26".
    str_shifts: str = data["averaged_NMR_shifts"].item()
    shifts: List[float] = [float(i) for i in str_shifts.split()]

    # Assign one shift per atom, in SDF atom order.
    atoms: List[Chem.Atom] = list(mol.GetAtoms())
    i: int = 0
    for idx, atom in enumerate(atoms):
        # Raises IndexError when the property lists fewer shifts than atoms; the
        # caller (run_1H_generation) catches it and skips that molecule.
        atom.SetProp("_Shift", str(shifts[idx]))
    # AddHs is called again (already done above), kept as in the original code;
    # with the same molecule this is a no-op.
    mol = AddHs(mol, addCoords=False)

    return mol

def add_shifts_to_data(nmr_data: List[Dict[str, Any]],
                       assigned_shifts: Dict[Tuple[str, int], float]) -> List[Dict[str, Any]]:
    """Attach a ``shift`` value to every hydrogen record.

    All protons sharing one parent carbon are meant to share one averaged shift
    (a methyl group has a single chemical shift, not three). The function groups
    the records by ``group_key``, averages the available shifts and writes the
    result back into each record.

    Args:
        nmr_data: records produced by :func:`analyze_molecule`.
        assigned_shifts: ``{(element, parent index): shift}``.

    Returns:
        The same list, mutated in place and also returned, with a new ``shift``
        key on every record that belongs to a known group.

    Example:
        Two records with ``group_key=("C", 1)`` and one shift of 2.1 both end up
        with ``shift=2.1``.

    Note:
        Because ``assigned_shifts`` holds only one value per parent atom, the
        group list contains that same value repeated, so the mean equals the
        single value. The averaging only becomes meaningful if per-hydrogen
        shifts differ.
    """
    ### Calculate the average shift to the nmr_data dictionary 
    ### where multiple H are attached to one Carbon
    grouped_shifts: Dict[Tuple[str, int], List[float]] = {}
    for atom_data in nmr_data:
        # Group records by the heavy atom they belong to.
        group_key: Tuple[str, int] = atom_data['group_key']
        if group_key not in grouped_shifts:
            grouped_shifts[group_key] = []
        if group_key in assigned_shifts:  
            # Append the group's shift once per hydrogen in the group.
            grouped_shifts[group_key].append(assigned_shifts[group_key])
    
    # Average per group, ignoring non-finite values (NaN from failed predictions).
    avg_shifts: Dict[Tuple[str, int], float] = {
        group_key: np.mean([shift for shift in shifts if np.isfinite(shift)])
        for group_key, shifts in grouped_shifts.items()
    }

    # Write the group shift back into every record of that group.
    for atom_data in nmr_data:
        group_key = atom_data['group_key']
        if group_key in avg_shifts:  
            atom_data['shift'] = avg_shifts[group_key]
            ### This creates duplicates entries for the hydrogens that are on the same Carbon
    return nmr_data

def calculate_couplings_constants(nmr_data: List[Dict[str, Any]]
                                  ) -> Tuple[List[List[Tuple[float, float]]], List[str], List[float], List[int]]:
    """Build the proton multiplet pattern of every unique proton (stage 2 core).

    For each proton record produced by :func:`analyze_molecule` this function
    returns a coupling pattern: a list of ``(offset_in_Hz, relative_intensity)``
    pairs that :func:`create_shift_intensity_label_data` later converts into ppm
    peaks.

    Two very different code paths are used:

    * **aromatic protons** -- the multiplet only depends on the number of
      *ortho* hydrogens (``get_adjacent_aromatic_hydrogens``) and a fixed
      ``J_aromatic = 8.0 Hz``;
    * **aliphatic protons** -- a hand-written rule table. Its key is the number
      of hydrogens carried by each carbon neighbour (``hydrogen_counts``)
      combined with the bond order of those C-C bonds, e.g.
      ``[1]`` + a double bond -> the CH=CH2 "roof effect" 16/10 Hz pattern.

    Args:
        nmr_data: proton records from :func:`analyze_molecule` /
            :func:`add_shifts_to_data`; each record has ``atom``, ``label``,
            ``aromatic``, ``neighbors`` and ``shift``.

    Returns:
        ``(coupling_patterns, atoms_done, shifts, hydrogen_num)`` where

        * ``coupling_patterns`` -- one pattern per unique proton label;
        * ``atoms_done`` -- the labels, used later for a second de-duplication;
        * ``shifts`` -- the ppm shift of each unique proton;
        * ``hydrogen_num`` -- number of equivalent hydrogens per group (collected
          but not used downstream).

    Example:
        Ethanol ``CCO``: the OH proton is skipped (label contains "O"), the
        CH2 gives a quartet ``[(-10.5, 2/6), (-3.5, 4/6), (3.5, 4/6), (10.5, 2/6)]``
        (J = 7 Hz) and the CH3 a triplet.

    Note:
        1. Symmetric protons are de-duplicated through ``atoms_done``: a CH3
           yields three identical records (``label="C1H3"``) but only one pattern.
        2. ``coupling_pattern`` is **not** reset at the start of each iteration,
           so when no rule matches the variable is either undefined (caught by
           the bare ``except`` below) or still holds the pattern of the previous
           proton, which is then appended silently.
        3. The rule table mixes ``if`` and ``elif``: earlier blocks (single
           hydrogen neighbours) are independent ``if`` chains and can be
           overwritten by later rules. Order therefore matters.
    """

    # Coupling constant used for every aromatic proton (typical ortho coupling).
    J_aromatic: float = 8.0

    ### Version 2
    # Calculate coupling patterns using the average shifts
    atoms_done: List[str] = []                       # labels already processed (de-duplication)
    coupling_patterns: List[List[Tuple[float, float]]] = []   # one multiplet per unique proton
    hydrogen_num: List[int] = []                     # equivalent H count (unused downstream)
    shifts: List[float] = []                         # ppm shift per unique proton
    hydrogen_counts: Optional[List[int]] = None      # H counts on the carbon neighbours (rule key)
    
    # Walk every proton record; symmetric protons appear several times and are
    # filtered by the `label not in atoms_done` checks below.
    for atom_data in nmr_data:
        # Heteroatom-bound protons (N-H, O-H) are not simulated: their exchange
        # rate makes them unreliable, and the model was trained without them.
        # Example: ethanol's OH has label "O1H1" -> skipped, CH2/CH3 are kept.
        if ("N" in atom_data["label"] or "O" in atom_data["label"]) :
            continue
        # The single neighbour of an explicit H is the heavy atom carrying it.
        parent_atom: Chem.Atom = atom_data['atom'].GetNeighbors()[0]
        # ---- Aromatic protons ------------------------------------------------
        # The label check de-duplicates symmetric protons: benzene yields six
        # "C0H1"/"C1H1"... labels but each label is processed once.
        if (atom_data['aromatic'] and atom_data['label'] not in atoms_done):
            # Number of other protons on the same carbon (0 for aromatic CH).
            n_neighbors: int = atom_data['neighbors']
            # Hydrogens sitting on the *ortho* (adjacent aromatic) carbons.
            adjacent_aromatic_hydrogens: List[Tuple[Chem.Atom, int]] = get_adjacent_aromatic_hydrogens(parent_atom)
            # Their count decides the multiplet: 1 -> doublet, 2 -> triplet, ...
            arom_n_neighbors: int = len(adjacent_aromatic_hydrogens)
            if arom_n_neighbors == 0:
                # Isolated aromatic proton. NOTE: the single line is placed at
                # +J_aromatic (8 Hz = 0.02 ppm on 400 MHz) instead of at 0 Hz,
                # so this "singlet" is slightly shifted to the right.
                coupling_patterns.append([(J_aromatic, 1)])
            else:
                # 1 -> [(-4, .5), (4, .5)] (doublet), 2 -> triplet 1:2:1 at J = 8 Hz.
                coupling_patterns.append(generate_nmr_coupling_pattern(arom_n_neighbors, J_aromatic))
            # Store the shift of this unique proton (ppm).
            shifts.append(atom_data['shift'])
            # Mark the label as processed so the remaining symmetric H's are skipped.
            atoms_done.append(atom_data['label'])
            # Total number of hydrogens on the group (kept for compatibility).
            hydrogen_num.append(atom_data['neighbors']+1)
        # ---- Aliphatic protons -----------------------------------------------
        # Below this line the multiplet comes from a hand-written lookup table:
        # the key is (hydrogen_counts, bond orders), see the comment block right
        # before the first rule.
        elif atom_data['label'] not in atoms_done:

            bond_types = [bond.GetBondType() for bond in parent_atom.GetBonds() if bond.GetOtherAtom(parent_atom).GetSymbol() == 'C']        

            # H count on the parent atom itself (0 for a CH, 3 for a CH3, ...).
            n_neighbors = atom_data['neighbors']

            # Carbon atoms directly bonded to the parent atom: these are the
            # coupling partners whose hydrogens split this proton's signal.
            carbon_neighbors: List[Chem.Atom] = [neighbor for neighbor in parent_atom.GetNeighbors() if neighbor.GetSymbol() == 'C']

            # How many H sit on each carbon neighbour -> the shape of the multiplet.
            # Example: propane CH3-CH2-CH3, proton on C2 -> [3, 3] -> septet;
            #          ethanol CH3-CH2-OH, proton on CH3 -> [2] -> triplet.
            hydrogen_counts: List[int] = [sum(1 for neighbor in carbon_neighbor.GetNeighbors() if neighbor.GetSymbol() == 'H') for carbon_neighbor in carbon_neighbors]
            # Rule-based coupling pattern generation.
            #
            # HOW TO READ THE TABLE BELOW
            # ----------------------------
            # * `hydrogen_counts` = H count of every carbon neighbour, in atom order.
            #      [3]      -> one CH3 neighbour      -> quartet (3 neighbours split into 4)
            #      [1, 1]   -> two CH neighbours      -> more complex multiplet
            #      []       -> no carbon neighbour at all (e.g. N-CH3, CCl3-CH<)
            # * `bond_types` = bond order of each of those C-C bonds, so the same
            #      `hydrogen_counts` produces a different pattern for single vs
            #      double bonds (large J = 13-18 Hz for cis/trans alkenes).
            # * `n_neighbors` = H count on the parent atom itself.
            # * Intensities use a common denominator that sums to 2, e.g.
            #      1/6 + 2/6 + 2/6 + 1/6 = 1   (approximated quartet)
            #      2/6 + 4/6 + 4/6 + 2/6 = 2   (CH3-CH2 coupling, counts the 6 H)
            #   so they are relative weights, not normalised probabilities.
            # * Offsets are in Hz and get divided by the spectrometer frequency
            #   later (8 Hz = 0.02 ppm on a 400 MHz instrument).
            # * J values are empirical approximations, not quantum calculations.
            if hydrogen_counts == [] and n_neighbors == 2:
                #N-CH3
                coupling_pattern = [(0, 3)]   

            if hydrogen_counts == [] and n_neighbors == 1:
                #CCl2=CH2
                coupling_pattern = [(0, 2)]  
                
            if hydrogen_counts == [] and n_neighbors == 0:
                #(CCl2)3-CH
                coupling_pattern = [(0, 0)]  

            if hydrogen_counts == [0] and n_neighbors == 2:
                coupling_pattern = [(0, 3)]

            if hydrogen_counts == [0] and n_neighbors == 1:
                coupling_pattern = [(0, 2)]   

            if hydrogen_counts == [0] and n_neighbors == 0:
                coupling_pattern = [(0, 1)]  


            if hydrogen_counts == [0, 0] and n_neighbors == 0:
                coupling_pattern = [(0, 1)]  
            
            if hydrogen_counts == [0, 0] and n_neighbors == 1:
                coupling_pattern = [(0, 2)]   

            # CH=CH2 case: one neighbouring CH carbon + one double bond.
            # The proton couples to cis (10 Hz) and trans (16 Hz) neighbours at
            # once, producing a 4-line "roof" pattern instead of a clean triplet.
            # Offsets below are the four possible J combinations of the two
            # couplings; intensities are equal (1/2 each, total 2).
            ### Approximation
            if hydrogen_counts == [1] and Chem.rdchem.BondType.DOUBLE in bond_types and n_neighbors == 1:
                J_doublet_1 = 16  
                J_doublet_2 = 10 
                coupling_pattern = [(-0.5*J_doublet_1-0.5*J_doublet_2, 1/2), 
                                    (-0.5*J_doublet_1+0.5*J_doublet_2, 1/2),
                                    (0.5*J_doublet_1+0.5*J_doublet_2, 1/2), 
                                    (0.5*J_doublet_1-0.5*J_doublet_2, 1/2)]

            elif hydrogen_counts == [1] and Chem.rdchem.BondType.SINGLE in bond_types and n_neighbors == 1:
                # CH-CH2-Cl case J=5.9
                J_doublet = 5.9  # Coupling constant for the single bond between the CH hydrogens
                coupling_pattern = [(-0.5*J_doublet, 1), 
                                      (0.5*J_doublet, 1)]

            elif hydrogen_counts == [1] and Chem.rdchem.BondType.SINGLE in bond_types and n_neighbors == 2:
                # CH-CH3 case J = 6.1
                J_doublet = 6.1  # Coupling constant for the single bond between the CH hydrogens
                coupling_pattern = [(-0.5*J_doublet, 1.5), 
                                      (0.5*J_doublet, 1.5)]
                
            if hydrogen_counts == [1] and n_neighbors == 0:
                #(CCl2)2-CH-CHCl2                
                J_doublet = 6.1
                coupling_pattern = [(-0.5*J_doublet, 0.5),
                                    (0.5*J_doublet, 0.5)]  
                
            #elif hydrogen_counts == [2] and Chem.rdchem.BondType.DOUBLE in bond_types:
            #    # CH2=CH2 case
            #    J_triplet = J_double_bond  # Coupling constant for the double bond between the CH2=CH2 hydrogens
            #    coupling_pattern = [(-J_triplet, 1/2), 
            #                          (0, 2/2), 
            #                          (J_triplet, 1/2)]

            elif hydrogen_counts == [2] and Chem.rdchem.BondType.SINGLE in bond_types:
                # CH2-CH2 case J = 6.3
                J_triplet = 6.3  # Coupling constant for the single bond between the CH2-CH2 hydrogens
                coupling_pattern = [(-J_triplet, 1/2), 
                                      (0, 2/2), 
                                      (J_triplet, 1/2)]

            elif hydrogen_counts == [3] and Chem.rdchem.BondType.SINGLE in bond_types:
                # CH3-CH2 case J = 7
                J_quartet =   7.0
                coupling_pattern = [(-1.5*J_quartet, 2/6), 
                                      (-0.5*J_quartet, 4/6), 
                                      (0.5*J_quartet, 4/6), 
                                      (1.5*J_quartet, 2/6)]

            elif (hydrogen_counts == [1,0] or hydrogen_counts == [0,1]):
                # CH-CH2-CO case J = 7
                J_doublet = 6.9  
                coupling_pattern = [(-0.5*J_doublet, 1), 
                                      (0.5*J_doublet, 1)]            


            elif (hydrogen_counts == [2,0] or hydrogen_counts == [0,2]) and Chem.rdchem.BondType.DOUBLE in bond_types:
                # CH2=CH-CO case J = 7
                J_doublet_1 = 18  # Coupling constant for the double bond between the CH=CH2 hydrogens
                J_doublet_2 = 10  # Coupling constant for the double bond between the CH=CH2 hydrogens
                coupling_pattern = [(-0.5*J_doublet_1-0.5*J_doublet_2, 1/4), 
                                    (-0.5*J_doublet_1+0.5*J_doublet_2, 1/4),
                                    (0.5*J_doublet_1-0.5*J_doublet_2, 1/4), 
                                    (0.5*J_doublet_1+0.5*J_doublet_2, 1/4)]  

            elif (hydrogen_counts == [2,0] or hydrogen_counts == [0,2]) and not Chem.rdchem.BondType.DOUBLE in bond_types:
                # CH2-CH2-CO case J = 7
                J_triplet = 6.7  # Coupling constant for the double bond between the CH=CH2 hydrogens
                coupling_pattern = [(-J_triplet, 1/2), 
                                      (0, 2/2), 
                                      (J_triplet, 1/2)]   

            elif hydrogen_counts == [3,0]  or hydrogen_counts == [0,3]:
                # CH3-CHCl-CO case J = 7
                J_quartet = 7  
                coupling_pattern = [(-1.5*J_quartet, 1/6), 
                                      (-0.5*J_quartet, 2/6), 
                                      (0.5*J_quartet, 2/6), 
                                      (1.5*J_quartet, 1/6)]
                
            elif hydrogen_counts == [1, 1] and Chem.rdchem.BondType.DOUBLE in bond_types:
                # CH-CH=CH non-aromatic case  13/6.06
                J_doublet_1 = 6.06  # Coupling constant for the single bond between CH hydrogens
                J_doublet_2 = 13 # Coupling constant for the double bond between the CH=CH hydrogens
                coupling_pattern = [(-0.5*J_doublet_1-0.5*J_doublet_2, 1/4), 
                                    (-0.5*J_doublet_1+0.5*J_doublet_2, 1/4),
                                    (0.5*J_doublet_1-0.5*J_doublet_2, 1/4), 
                                    (0.5*J_doublet_1+0.5*J_doublet_2, 1/4)]

            elif hydrogen_counts == [1, 1] and not Chem.rdchem.BondType.DOUBLE in bond_types:
                # CH-CH2-CH non-aromatic case  13/6.06
                J_triplet = 6.0  # Coupling constant for the double bond between the CH=CH2 hydrogens
                coupling_pattern = [(-J_triplet, 1/2), 
                                      (0, 2/2), 
                                      (J_triplet, 1/2)]   

            elif (hydrogen_counts == [1, 2] or hydrogen_counts == [2, 1]) and bond_types.count(Chem.rdchem.BondType.SINGLE) == 1 and bond_types.count(Chem.rdchem.BondType.DOUBLE) == 1:  
                # CH=CH-CH2 case  J = 7.4 
                J_quartet = 7.4  # Coupling constant for the single bond between the CH3-CH3 hydrogens
                coupling_pattern = [(-1.5*J_quartet, 1/6), 
                                      (-0.5*J_quartet, 2/6), 
                                      (0.5*J_quartet, 2/6), 
                                      (1.5*J_quartet, 1/6)]

            elif (hydrogen_counts == [1, 2] or hydrogen_counts == [2, 1]) and bond_types.count(Chem.rdchem.BondType.SINGLE) == 2:  
                # CH-CH2-CH2 case  J = 7.4 
                # an approximation
                J_quartet = 7.4   # Coupling constant for the single bond between the CH3-CH3 hydrogens
                coupling_pattern = [(-1.5*J_quartet, 2/6), 
                                      (-0.5*J_quartet, 4/6), 
                                      (0.5*J_quartet, 4/6), 
                                      (1.5*J_quartet, 2/6)]

            elif hydrogen_counts == [2, 2] and bond_types.count(Chem.rdchem.BondType.SINGLE) == 2:  
                # CH2-CH2-CH2 case (quintet)  J=6.57
                J_quintet = 6.57
                coupling_pattern = [(-2 * J_quintet, 2/9), 
                                    (-J_quintet, 4/9), 
                                    (0, 6/9), 
                                    (J_quintet, 4/9), 
                                    (2 * J_quintet, 2/9)]

            elif (hydrogen_counts == [3, 1] or hydrogen_counts == [1, 3]) and Chem.rdchem.BondType.DOUBLE in bond_types:
                # CH3-CH=CH case or CH=CH-CH3 J = 7 (Douplet of quartet)
                # an approximation
                J_quintet = 7
                coupling_pattern = [(-2 * J_quintet, 1/9), 
                                    (-J_quintet, 2/9), 
                                    (0, 3/9), 
                                    (J_quintet, 2/9), 
                                    (2 * J_quintet, 1/9)]

            elif (hydrogen_counts == [3, 1] or hydrogen_counts == [1, 3]) and not Chem.rdchem.BondType.DOUBLE in bond_types:
                # CH3-CH-CHCl case or CH=CH-CH3 J = 7 (Douplet of quartet)
                # an approximation
                J_octet = 3.5
                coupling_pattern = [(-3.5*J_octet, 1/12), 
                                   (-2.5*J_octet, 1/12),  
                                  (-1.5*J_octet, 2/12), 
                                  (-0.5*J_octet, 2/12), 
                                  (0.5*J_octet, 2/12), 
                                  (1.5*J_octet, 2/12), 
                                  (2.5*J_octet, 1/12),
                                  (3.5*J_octet, 1/12)]


            elif (hydrogen_counts == [3, 2] or hydrogen_counts == [2, 3]) and bond_types.count(Chem.rdchem.BondType.SINGLE) == 2:  
                # CH3-CH2-CH2 case (Quartet of Triplets) most likely like a sextet
                # an approximation
                J_sixtet = 7
                coupling_pattern = [(-2.5*J_sixtet, 2/12), 
                                      (-1.5*J_sixtet, 4/12), 
                                      (-0.5*J_sixtet, 6/12), 
                                      (0.5*J_sixtet, 6/12),
                                      (1.5*J_sixtet, 4/12), 
                                      (2.5*J_sixtet, 2/12)]

            elif hydrogen_counts == [3, 3]:
                # CH3-CH-CH3 with another connection to CH with CH3 J= 6.4
                J_septet = 6.4  # Coupling constant between the central CH hydrogen and the CH3 hydrogens
                coupling_pattern = [(-3*J_septet, 1/16), 
                                   (-2*J_septet, 2/16),  
                                  (-1*J_septet, 3/16), 
                                  (0*J_septet, 4/16), 
                                  (1*J_septet, 3/16), 
                                  (2*J_septet, 2/16),
                                  (3*J_septet, 1/16)]
            elif hydrogen_counts == [0, 0, 0]:
                coupling_pattern = [(0, 1)]  

            elif hydrogen_counts == [0, 0, 1] or hydrogen_counts == [0, 1, 0] or hydrogen_counts == [1, 0, 0]:
                # (CCl3)2-CH-CHCl2
                J_doublet = 6.1 
                coupling_pattern = [(-0.5*J_doublet, 0.5), 
                                  (0.5*J_doublet, 0.5)]  
                
            elif (hydrogen_counts == [0, 1, 1] or hydrogen_counts == [1,1,0] or hydrogen_counts == [1,0,1]):
                #COCl-CH-(CHCl)2
                J_triplet = 7  # Coupling constant for the double bond between the CH=CH2 hydrogens
                coupling_pattern = [(-J_triplet, 1/4), 
                                      (0, 2/4), 
                                      (J_triplet, 1/4)]   
                
            elif (hydrogen_counts == [0,2,2] or hydrogen_counts == [2,2,0] or hydrogen_counts == [2,0,2]):
                #COCl-CH-(CH2)
                J_quintet = 7.5
                coupling_pattern = [(-2 * J_quintet, 1/9), 
                                    (-J_quintet, 2/9), 
                                    (0, 3/9), 
                                    (J_quintet, 2/9), 
                                    (2 * J_quintet, 1/9)]
            elif (hydrogen_counts ==  [0, 2, 0] 
                   or hydrogen_counts ==  [0, 0, 2] 
                   or hydrogen_counts ==  [2, 0, 0]):
                # Approximation dd ->t
                J_triplet = 7  # Coupling constant for the double bond between the CH=CH2 hydrogens
                coupling_pattern = [(-J_triplet, 1/4), 
                                      (0, 2/4), 
                                      (J_triplet, 1/4)]   
            
            elif (hydrogen_counts ==  [0, 2, 1] 
                   or hydrogen_counts ==  [0, 1, 2] 
                   or hydrogen_counts ==  [1, 2, 0]
                   or hydrogen_counts ==  [1, 0, 2]
                   or hydrogen_counts ==  [2, 0, 1]
                   or hydrogen_counts ==  [2, 1, 0]):
                #COCl-CH-(CH2)(CH) ddd
                # an approximation
                J_quartet = 7.0   # Coupling constant for the single bond between the CH3-CH3 hydrogens
                coupling_pattern = [(-1.5*J_quartet, 1/6), 
                                      (-0.5*J_quartet, 2/6), 
                                      (0.5*J_quartet, 2/6), 
                                      (1.5*J_quartet, 1/6)]

            elif (hydrogen_counts == [1,2,2]
                   or hydrogen_counts ==  [2, 1, 2] 
                   or hydrogen_counts ==  [2, 2, 1]):
                    #CH-CH-(CH2)2  ttd
                J_septet = 6.2  # Coupling constant between the central CH hydrogen and the CH3 hydrogens
                J_12 = 3
                coupling_pattern = [(-5.5*J_12, 1/42), 
                                   (-4.5*J_12, 2/42),  
                                  (-3.5*J_12, 3/42), 
                                  (-2.5*J_12, 4/42), 
                                  (-1.5*J_12, 5/42), 
                                  (-0.5*J_12, 6/42),
                                  (0.5*J_12, 6/42), 
                                  (1.5*J_12, 5/42), 
                                  (2.5*J_12, 4/42),
                                  (3.5*J_12, 3/42), 
                                  (4.5*J_12, 2/42), 
                                  (5.5*J_12, 1/42)]   
                
            elif hydrogen_counts == [2,2,2]:
                #CH2-CH-(CH2)2
                J_septet = 6.2  # Coupling constant between the central CH hydrogen and the CH3 hydrogens
                coupling_pattern = [(-3*J_septet, 1/16), 
                                   (-2*J_septet, 2/16),  
                                  (-1*J_septet, 3/16), 
                                  (0*J_septet, 4/16), 
                                  (1*J_septet, 3/16), 
                                  (2*J_septet, 2/16),
                                  (3*J_septet, 1/16)]            

            try:
                # `hydrogen_counts` is only None before the first assignment, so in
                # practice this always appends -- unless no rule matched, in which
                # case `coupling_pattern` may be undefined (UnboundLocalError) or
                # stale from the previous proton; both are handled below.
                if hydrogen_counts != None:
                    # Register the multiplet of this unique proton.
                    coupling_patterns.append(coupling_pattern)
                    # Mark the label so symmetric protons are skipped.
                    atoms_done.append(atom_data['label'])
                    # Store the ppm shift used when the multiplet is drawn.
                    shifts.append(atom_data['shift'])
                    # Total H count of the group (unused downstream).
                    hydrogen_num.append(atom_data['neighbors']+1)
                else:
                    continue
            except:
                # No rule matched (or the first proton has no rule at all):
                # print the key so a missing rule can be added to the table.
                # Example output: "[1, 2] 3" -> no branch for this combination.
                print(hydrogen_counts, n_neighbors)

        #if atom_data["label"] =="C1H1":
        #    break
    # Returned in parallel lists of equal length: pattern i belongs to label i.
    return coupling_patterns, atoms_done, shifts, hydrogen_num


def create_plot_NMR(shifts, coupling_patterns, gamma, spectrometer_frequency):
    x, y = simulate_splitting(np.array(shifts), coupling_patterns, gamma, spectrometer_frequency)
    plt.plot(x, y)
    plt.xlabel('Chemical shift (ppm)')
    plt.ylabel('Intensity')
    for shift, label in zip(shifts, atoms_done):
        plt.text(shift, np.max(lorentzian(x, shift, gamma)), label, ha='center', va='bottom', fontsize=8, rotation=45)

    plt.gca().invert_xaxis()
    plt.show()
    
def create_plot_NMR_interactiv(shift_intensity_label_data):
    
    # Create an interactive plot using Plotly
    fig = go.Figure()

    for shift, intensity, label in shift_intensity_label_data:
        fig.add_trace(
            go.Scatter(
                x=[shift, shift],  # Use two points to create a vertical line
                y=[0, intensity],
                mode="lines",
                line=dict(color="black", width=1.5),
                hoverinfo="none",  # Disable hover info
            )
        )

    # Find the maximum intensity for each shift
    max_intensities_dict = {}
    for item in shift_intensity_label_data:
        _, intensity, label = item
        if label not in max_intensities_dict:
            max_intensities_dict[label] = intensity
        else:
            max_intensities_dict[label] = max(max_intensities_dict[label], intensity)
    max_intensities = list(max_intensities_dict.values())


    # Add a separate trace for the labels
    fig.add_trace(
        go.Scatter(
            x=shifts,
            y=max_intensities,
            mode="text",
            text=atoms_done,
            textposition="top center",
            hoverinfo="none",  # Disable hover info
        )
    )

    fig.update_layout(
        xaxis=dict(title="Chemical shift (ppm)", range=[11, 0]),  # Set x-axis range
        yaxis=dict(title="Intensity"),
        showlegend=False,  # Remove the legend from the plot
    )

    multiplicity_labels = []
    for coupling_pattern in coupling_patterns:
        if len(coupling_pattern) == 1:
            multiplicity_labels.append("Singlet")
        else:
            n_split = len(coupling_pattern)
            if n_split == 2:
                multiplicity_labels.append("Doublet")
            elif n_split == 3:
                multiplicity_labels.append("Triplet")
            elif n_split == 4:
                multiplicity_labels.append("Quartet")
            elif n_split == 5:
                multiplicity_labels.append("Quintet")
            elif n_split>5:
                multiplicity_labels.append(f"{n_split} peaks")

    fig.add_trace(
        go.Scatter(
            x=shifts,
            y=[intensity - 0.1 * intensity for intensity in max_intensities],
            mode="text",
            text=multiplicity_labels,
            textposition="bottom center",
            textfont=dict(color="red"),  # Use a different color for multiplicity labels
            hoverinfo="none",  # Disable hover info
        )
    )
    fig.show()
    
def create_labeled_structure(mol,assigned_shifts):


    # Generate a 2D depiction of the molecule to better fit the drawing canvas
    rdDepictor.Compute2DCoords(mol)

    # Create a MolDraw2DSVG object to draw the molecule as an SVG
    # You can adjust the width and height values to better fit the molecule
    drawer = rdMolDraw2D.MolDraw2DSVG(600, 200)
    opts = drawer.drawOptions()

    # Set atom labels based on the assigned_shifts dictionary
    for (atom_type, atom_idx) in assigned_shifts.keys():
        atom = mol.GetAtomWithIdx(atom_idx)
        opts.atomLabels[atom_idx] = f"{atom_type}{atom_idx}"

    # Draw the molecule
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()

    # Display the SVG in the notebook
    svg = SVG(drawer.GetDrawingText())
    return svg                     
                             
def create_shift_intensity_label_data(shifts: List[float], coupling_patterns: List[List[Tuple[float, float]]],
                                      atoms_done: List[str], spectrometer_frequency: float
                                      ) -> Tuple[List[Tuple[float, float, str]], List[Tuple[float, float]]]:
    """Expand multiplets into flat ``(ppm, intensity)`` peak lists.

    This is the step that converts "one proton at 7.26 ppm split into a triplet"
    into three concrete peaks, ready to be stored in the CSV.

    Args:
        shifts: chemical shift of every proton in ppm.
        coupling_patterns: one ``[(J_Hz, intensity), ...]`` list per proton.
        atoms_done: label of every proton, e.g. ``"C3H3"`` (only used in the
            labelled output).
        spectrometer_frequency: instrument frequency in MHz, used to convert
            ``J`` from Hz to ppm (``J / frequency``).

    Returns:
        ``(shift_intensity_label_data, shift_intensity_data)`` where

        * ``shift_intensity_label_data`` -- ``[(ppm, intensity, label), ...]``;
        * ``shift_intensity_data`` -- ``[(ppm, intensity), ...]`` (what the CSV
          actually stores).

    Example:
        One proton at 7.26 ppm, triplet with J = 8 Hz, 400 MHz:
        ``shift_intensity_data == [(7.24, .25), (7.26, .5), (7.28, .25)]``.
    """
    shift_intensity_label_data: List[Tuple[float, float, str]] = []
    shift_intensity_data: List[Tuple[float, float]] = []
    # Walk the three parallel lists (shift, pattern, label) of every proton.
    for shift, coupling_pattern, label in zip(shifts, coupling_patterns, atoms_done):
        if shift!=0.0:   ### Because of ACD labs sometimes there are 0.0 that we used for padding to the right number. these should be ignored
            # One (ppm, intensity) pair per multiplet line.
            for J, intensity in coupling_pattern:
                if len(coupling_pattern) > 1:
                    # Hz -> ppm conversion: 8 Hz on a 400 MHz instrument = 0.02 ppm.
                    shift1: float = shift + (J / spectrometer_frequency)
                else:
                    # A singlet has one line exactly at the proton's shift.
                    shift1 = shift
                shift_intensity_label_data.append((shift1, intensity, label))
                shift_intensity_data.append((shift1, intensity))
    return shift_intensity_label_data, shift_intensity_data


def run_1H_generation(config: Namespace) -> Tuple[pd.DataFrame, str]:
    """Simulate 1H spectra for every SDF file written by :func:`run_sgnn` (stage 2).

    For each molecule the function loads the SGNN shifts, builds per-proton
    records, derives coupling patterns and flattens them into a list of
    ``(ppm, intensity)`` peaks. The result is written as
    ``data_1H_<ran_num>.csv`` with columns ``SMILES``, ``shifts``, ``sample-id``.

    Args:
        config: namespace with ``SGNN_gen_folder_path`` (folder holding
            ``NMR_*.sdf``) and ``SGNN_csv_save_folder`` (output folder), plus
            ``ran_num`` used in the output file name.

    Returns:
        ``(data, csv_1H_path)``: the DataFrame and the path of the CSV written.

    Example:
        >>> data, path = run_1H_generation(config)
        >>> data.columns.tolist()
        ['SMILES', 'shifts', 'sample-id']
        >>> data.iloc[0]['shifts'][:2]        # first two peaks of molecule 1
        [(1.24, 0.5), (1.26, 0.25)]

    Note:
        The loop has a bare ``except`` that prints the offending file and moves
        on, so molecules whose SDF is inconsistent are silently skipped.
    """
    # Folder that holds one SDF per molecule, written by run_sgnn.
    folder_path: str = config.SGNN_gen_folder_path
    # Collect every file and keep only the NMR_*.sdf ones ("NMR_" is the SGNN prefix).
    nmr_files: List[str] = glob.glob(folder_path+"/*")
    nmr_files = [file for file in nmr_files if "NMR_" in file.split("/")[-1] ]
    nmr_files = [file for file in nmr_files if not ".mol" in file]
    # Deterministic order so two runs produce identical CSVs.
    nmr_files = sorted(nmr_files, reverse=False)


    ### Settings
    # 400 MHz instrument: J (Hz) / 400 -> ppm separation between multiplet lines.
    spectrometer_frequency: float = 400  # Example spectrometer frequency in MHz
    # Peak width (FWHM) in ppm used by the Lorentzian profile.
    gamma: float = 0.01
    # Debug switches; all off for batch generation (they open figures inline).
    plot_NMR: bool = False
    plot_NMR_interactiv: bool = False
    show_labeled_structure: bool = False

    ### Stores
    smiles_list: List[str] = []
    data_list: List[List[Tuple[float, float]]] = []
    sample_id_list: List[str] = []

    for file_path in nmr_files[:]:
        try:
            # Molecule + per-atom shifts read back from the SDF.
            mol: Chem.Mol = load_mol_and_assign_shifts(file_path)

            # sample id = the part of the file name after "NMR_", e.g. "1234567".
            file_name: str = os.path.basename(file_path)
            sample_id: str = os.path.splitext(file_name)[0].split('NMR_')[-1]

            # One record per proton: parent atom, aromaticity, label, shift.
            nmr_data: List[Dict[str, Any]]
            assigned_shifts: Dict[Tuple[str, int], float]
            nmr_data, assigned_shifts, mol = analyze_molecule(mol)
            # Attach the (group) shift to every proton record.
            nmr_data = add_shifts_to_data(nmr_data, assigned_shifts)
            # Build the multiplet of every proton; also returns the flat lists.
            coupling_patterns, atoms_done, shifts, hydrogen_num = calculate_couplings_constants(nmr_data)
            if plot_NMR:
                create_plot_NMR(shifts, coupling_patterns, gamma, spectrometer_frequency)
            # Expand each multiplet into concrete (ppm, intensity) peaks.
            shift_intensity_label_data, shift_intensity_data = create_shift_intensity_label_data(shifts, coupling_patterns, atoms_done, spectrometer_frequency)
            if plot_NMR_interactiv:
                create_plot_NMR_interactiv(shift_intensity_label_data)
            if show_labeled_structure:
                svg_output = create_labeled_structure(mol,assigned_shifts)
                display(svg_output)
            if len(shift_intensity_data)!=0:
                # Store a canonical SMILES (hydrogens removed) next to the peaks.
                mol = Chem.RemoveHs(mol)
                smi: str = MolToSmiles(mol)            
                smiles_list.append(smi)
                ### Use a set because symmetric H on C will come for every atom the same
                # A methyl group yields 3 identical peak lists; the set keeps one,
                # sorted() restores a deterministic order for the CSV.
                data_shifts_ints: List[Tuple[float, float]] = sorted(list(set(shift_intensity_data)), reverse=False)
                data_list.append(data_shifts_ints)
                sample_id_list.append(sample_id)
            else:
                # No peaks at all -> nothing to save; report for traceability.
                print(sample_id)
        except:
            # Any failure (bad SDF, shift/atom mismatch, RDKit error) skips the
            # molecule; the file name is printed so it can be investigated.
            print(file_path)


    # Create a DataFrame with the lists as columns
    data: pd.DataFrame = pd.DataFrame({
        'SMILES': smiles_list,
        'shifts': data_list,
        'sample-id': sample_id_list,
    })

    # Renumber rows 0..n-1 after the skipped molecules.
    data.reset_index(drop=True, inplace=True)
    # Output file carries the run id so different runs never overwrite each other.
    csv_1H_path: str = os.path.join(config.SGNN_csv_save_folder, f"data_1H_{config.ran_num}.csv")
    data.to_csv(csv_1H_path, index=False)
    return data, csv_1H_path


##############################################################################
################################ C13 #########################################
##############################################################################


def find_symmetric_positions(stereo_smi: str) -> List[List[int]]:
    """Find groups of symmetry-equivalent atoms in a molecule.

    Adapted from https://github.com/rdkit/rdkit/issues/1411 . Two atoms belong
    to the same group when RDKit's canonical atom ranking gives them the same
    value with tie breaking disabled (``breakTies=False``), i.e. when they are
    topologically interchangeable (both ring halves of benzene, both methyls
    of acetone, ...).

    Args:
        stereo_smi: SMILES string **including stereochemistry**, e.g. ``"CC"``.

    Returns:
        List of atom-index groups, e.g. ``[[0, 1]]`` for ethane; an empty list
        when the molecule has no symmetric positions at all.

    Example:
        >>> find_symmetric_positions("CC")            # ethane: 2 equal carbons
        [[0, 1]]
        >>> find_symmetric_positions("CCl")           # chloromethane: none
        []
    """

    mol = Chem.MolFromSmiles(stereo_smi)
    # Canonical rank per atom; equal ranks (ties kept) mean equivalent atoms.
    z=list(rdmolfiles.CanonicalRankAtoms(mol, breakTies=False))
    # Self-matches of the molecule onto itself: one per automorphism.
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
        example_match = matches[0]
        sym_dupl_lists = []
        for sub_list in all_duplicates:
            indices_list = []
            for i in sub_list:
                position = example_match[i]
                indices_list.append(position)
            sym_dupl_lists.append(indices_list)
    
    else:
        sym_dupl_lists = []
    return sym_dupl_lists


# Function to average the symmetric peaks and update the list
def consolidate_peaks(averaged_shifts: List[float], symmetric_positions: List[List[int]]) -> List[float]:
    """Give every symmetric atom the mean shift of its group.

    Symmetry-equivalent atoms must produce a single 13C signal, so their
    individual (slightly different) predicted shifts are replaced by the group
    average before plotting / writing.

    Args:
        averaged_shifts: per-atom shifts, indexed by atom position.
        symmetric_positions: groups of equivalent atom indices, as returned by
            :func:`find_symmetric_positions`, e.g. ``[[0, 1]]``.

    Returns:
        A new list of shifts (the input list is not modified).

    Example:
        >>> consolidate_peaks([10.0, 12.0], [[0, 1]])
        [11.0, 11.0]
    """
    # Make a copy of the original list to avoid modifying it in place
    consolidated_shifts = averaged_shifts.copy()

    for positions in symmetric_positions:
        # Calculate the average for the symmetric positions
        # e.g. positions [0, 1] with shifts [10.0, 12.0] -> 11.0
        avg_value = sum(averaged_shifts[i] for i in positions) / len(positions)
        
        # Update the peaks at these positions with the average value
        for i in positions:
            consolidated_shifts[i] = avg_value

    return consolidated_shifts


def run_13C_generation(config):
    
    folder_path = config.SGNN_gen_folder_path
    nmr_files = glob.glob(folder_path+"/*")
    nmr_files = [file for file in nmr_files if "NMR_" in file.split("/")[-1] ]
    nmr_files = [file for file in nmr_files if not ".mol" in file]
    nmr_files = sorted(nmr_files, reverse=False)

    ### Stores
    smiles_list = []
    data_list = []
    sample_id_list = []


    for file_path in nmr_files[:]:
        try:
            #mol = load_mol_and_assign_shifts(file_path)
            mol = SDMolSupplier(file_path)[0]
            isomers = tuple(EnumerateStereoisomers(mol))
            stereo_smi = Chem.MolToSmiles(isomers[0],isomericSmiles=True)

            averaged_nmr_shifts = mol.GetProp('averaged_NMR_shifts')
            sample_shifts = list(map(float, averaged_nmr_shifts.split()))


            # Extract the sample_id from the path
            file_name = os.path.basename(file_path)
            sample_id = os.path.splitext(file_name)[0].split('NMR_')[-1]

            # Remove symmetric C
            sym_dupl_lists = find_symmetric_positions(stereo_smi)
            sym_corr_nmr_shifts = consolidate_peaks(sample_shifts, sym_dupl_lists)

            # Initialize counter for non-hydrogen atoms
            non_hydrogen_count = 0
            # Loop through each atom and count non-hydrogen atoms
            for atom in mol.GetAtoms():
                if atom.GetSymbol() != 'H':
                    non_hydrogen_count += 1

            heavy_atoms_shifts = sym_corr_nmr_shifts[:non_hydrogen_count]
            # Remove zeros
            C_atoms_shifts = [x for x in heavy_atoms_shifts if x != 0]

            # Remove symmetric peaks
            C_atoms_shifts = sorted(list(set(C_atoms_shifts)), reverse=False)

            mol = Chem.RemoveHs(mol)
            smi = MolToSmiles(mol)     

            smiles_list.append(smi)
            data_list.append(C_atoms_shifts)
            sample_id_list.append(sample_id)
        except:
            print(file_path)


    # Create a DataFrame with the lists as columns
    data = pd.DataFrame({
        'SMILES': smiles_list,
        'shifts': data_list,
        'sample-id': sample_id_list,
    })

    csv_13C_path = os.path.join(config.SGNN_csv_save_folder, f"data_13C_{config.ran_num}.csv")
    data.to_csv(csv_13C_path, index=False)
    return data, csv_13C_path



##############################################################################
################################ COSY ########################################
##############################################################################

def find_chiral_centers(molecule):
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

from rdkit import Chem
def find_carbons_with_relevant_neighbors(molecule):
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
    num_heavy_atoms = len(heavy_atom_dict)
    
    # Splitting the shifts into heavy atom shifts and hydrogen shifts
    heavy_atom_shifts = shifts[:num_heavy_atoms]
    hydrogen_shifts = shifts[num_heavy_atoms:]
    
    # Initialize the dictionary to store the carbon number and the shift of the attached hydrogens
    carbon_hydrogen_shifts_dict = {}
    
    # Iterate over the heavy_atom_dict in reverse order to pick the hydrogen shifts
    for carbon, num_hydrogens in sorted(heavy_atom_dict.items(), key=lambda x: x[0], reverse=True):
        # Pick the last 'num_hydrogens' from the hydrogen_shifts list
        attached_hydrogen_shifts = hydrogen_shifts[-num_hydrogens:]
        
        # Remove the picked hydrogen shifts from the list
        hydrogen_shifts = hydrogen_shifts[:-num_hydrogens]
        
        # Store in the resulting dictionary
        carbon_hydrogen_shifts_dict[carbon] = attached_hydrogen_shifts
    
    return carbon_hydrogen_shifts_dict


def find_symmetric_positions(stereo_smi: str) -> List[List[int]]:
    """Find groups of symmetry-equivalent atoms in a molecule.

    Adapted from https://github.com/rdkit/rdkit/issues/1411 . Two atoms belong
    to the same group when RDKit's canonical atom ranking gives them the same
    value with tie breaking disabled (``breakTies=False``), i.e. when they are
    topologically interchangeable (both ring halves of benzene, both methyls
    of acetone, ...).

    Args:
        stereo_smi: SMILES string **including stereochemistry**, e.g. ``"CC"``.

    Returns:
        List of atom-index groups, e.g. ``[[0, 1]]`` for ethane; an empty list
        when the molecule has no symmetric positions at all.

    Example:
        >>> find_symmetric_positions("CC")            # ethane: 2 equal carbons
        [[0, 1]]
        >>> find_symmetric_positions("CCl")           # chloromethane: none
        []
    """

    mol = Chem.MolFromSmiles(stereo_smi)
    # Canonical rank per atom; equal ranks (ties kept) mean equivalent atoms.
    z=list(rdmolfiles.CanonicalRankAtoms(mol, breakTies=False))
    # Self-matches of the molecule onto itself: one per automorphism.
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
        example_match = matches[0]
        sym_dupl_lists = []
        for sub_list in all_duplicates:
            indices_list = []
            for i in sub_list:
                position = example_match[i]
                indices_list.append(position)
            sym_dupl_lists.append(indices_list)
    
    else:
        sym_dupl_lists = []
    return sym_dupl_lists

# Function to check if an atom has hydrogens attached to it
def has_hydrogens(mol, atom_idx):
    atom = mol.GetAtomWithIdx(atom_idx)
    return atom.GetTotalNumHs() > 0



# Function to average shifts for symmetric carbons
def average_shifts(shift_list, sym_groups):
    avg_shifts = {}
    for sym_group in sym_groups:
        avg_shift = [sum([shift_list.get(i, 0)[0] for i in sym_group]) / len(sym_group)]
        for i in sym_group:
            avg_shifts[i] = avg_shift
    return avg_shifts


# Function to update original shift list with averaged values for symmetric carbons
def update_shifts_with_averaged(original_shifts, averaged_shifts):
    updated_shifts = original_shifts.copy()  # Make a copy of the original dictionary
    for carbon, avg_shift in averaged_shifts.items():
        updated_shifts[carbon] = avg_shift  # Update with the averaged value
    return updated_shifts


# Updated function to make plotting optional and return coordinates of plotted points
# Updated function to avoid duplicate peaks in the plot
def plot_and_save_cosy_spectrum_with_zoom_no_duplicates(heavy_atom_hydrogen_shift_dict, carbon_dict, chiral_centers, plot=True, xlim=None, ylim=None):
    plotted_points = set()  # Using a set to automatically handle duplicates
    
    for carbon1, neighbors in carbon_dict.items():
        h1_shifts = heavy_atom_hydrogen_shift_dict.get(carbon1, [])
        if h1_shifts:
            plotted_points.add((h1_shifts[0], h1_shifts[0]))
            
        for carbon2 in neighbors:
            h2_shifts = heavy_atom_hydrogen_shift_dict.get(carbon2, [])
            if h1_shifts and h2_shifts:
                is_chiral1 = carbon1 in chiral_centers
                is_chiral2 = carbon2 in chiral_centers

                if is_chiral1 or is_chiral2:
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
        plt.scatter(x_coords, y_coords, c='blue', marker='o', label='Cross Peaks', alpha=0.5)
        plt.title('COSY Spectrum with All Diagonal Peaks')
        plt.xlabel('Chemical Shift (ppm)')
        plt.ylabel('Chemical Shift (ppm)')
        plt.grid(True)
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




def run_COSY_generation(config):
    folder_path = config.SGNN_gen_folder_path
    nmr_files = glob.glob(folder_path+"/*")
    nmr_files = [file for file in nmr_files if "NMR_" in file.split("/")[-1] ]
    nmr_files = [file for file in nmr_files if not ".mol" in file]
    nmr_files = sorted(nmr_files, reverse=False)

    ### Stores
    smiles_list = []
    data_list = []
    sample_id_list = []

    for file_path in nmr_files[:]:
        try:
            mol = SDMolSupplier(file_path)[0]
            isomers = tuple(EnumerateStereoisomers(mol))
            stereo_smi = Chem.MolToSmiles(isomers[0],isomericSmiles=True)

            # Get NMR shifts
            averaged_nmr_shifts = mol.GetProp('averaged_NMR_shifts')
            sample_shifts = list(map(float, averaged_nmr_shifts.split()))

            # Extract the sample_id from the path
            file_name = os.path.basename(file_path)
            sample_id = os.path.splitext(file_name)[0].split('NMR_')[-1]

            # Find chiral centers
            chiral_centers = find_chiral_centers(mol)

            carbon_dict = find_carbons_with_relevant_neighbors(mol)

            heavy_atom_dict = find_heavy_atoms_with_hydrogens(mol)

            heavy_atom_hydrogen_shift_dict = extract_symmetric_hydrogen_shifts(sample_shifts, heavy_atom_dict)

            sym_dupl_lists = find_symmetric_positions(stereo_smi)

            # Remove symmetric positions that don't have hydrogens
            sym_dupl_lists = [positions for positions in sym_dupl_lists if all(has_hydrogens(mol, idx) for idx in positions)]

            averaged_shifts = average_shifts(heavy_atom_hydrogen_shift_dict, sym_dupl_lists)

            updated_heavy_atom_hydrogen_shift_dict = update_shifts_with_averaged(heavy_atom_hydrogen_shift_dict, averaged_shifts)

            COSY_shifts = plot_and_save_cosy_spectrum_with_zoom_no_duplicates(heavy_atom_hydrogen_shift_dict, carbon_dict, chiral_centers, plot=False, xlim=None, ylim=None)

            COSY_shifts = sorted(COSY_shifts, key=lambda x: x[0])

            mol = Chem.RemoveHs(mol)
            smi = MolToSmiles(mol)     

            smiles_list.append(smi)
            data_list.append(COSY_shifts)
            sample_id_list.append(sample_id)
        except:
            print(file_path)

    # Create a DataFrame with the lists as columns
    data = pd.DataFrame({
        'SMILES': smiles_list,
        'shifts': data_list,
        'sample-id': sample_id_list,
    })

    data.reset_index(drop=True, inplace=True)
    csv_COSY_path = os.path.join(config.SGNN_csv_save_folder, f"data_COSY_{config.ran_num}.csv")
    data.to_csv(csv_COSY_path, index=False)
    return data, csv_COSY_path


##############################################################################
################################ HSQC ########################################
##############################################################################



def run_HSQC_generation(config):
    
    folder_path = config.SGNN_gen_folder_path
    nmr_files = glob.glob(folder_path+"/*")
    nmr_files = [file for file in nmr_files if "NMR_" in file.split("/")[-1] ]
    nmr_files = [file for file in nmr_files if not ".mol" in file]
    nmr_files = sorted(nmr_files, reverse=False)
    ### Stores
    smiles_list = []
    data_list = []
    sample_id_list = []


    for file_path in nmr_files[:]:
        try:
            sample_df = ncfd.load_dft_dft_comparison(file_path)

            # Extract the sample_id from the path
            file_name = os.path.basename(file_path)
            sample_id = os.path.splitext(file_name)[0].split('NMR_')[-1]

            # Use the apply method to create a list of lists
            HSQC_shifts = sample_df.apply(lambda row: [row['F2 (ppm)'], row['F1 (ppm)']], axis=1).tolist()
            HSQC_shifts = sorted(HSQC_shifts, key=lambda x: x[0])

            mol = SDMolSupplier(file_path)[0]   
            mol = Chem.RemoveHs(mol)
            smi = MolToSmiles(mol)   

            smiles_list.append(smi)
            data_list.append(HSQC_shifts)
            sample_id_list.append(sample_id)
        except:
            print(file_path)


    # Create a DataFrame with the lists as columns
    data = pd.DataFrame({
        'SMILES': smiles_list,
        'shifts': data_list,
        'sample-id': sample_id_list,
    })

    data.reset_index(drop=True, inplace=True)
    csv_HSQC_path = os.path.join(config.SGNN_csv_save_folder, f"data_HSQC_{config.ran_num}.csv")
    data.to_csv(csv_HSQC_path, index=False)
    return data, csv_HSQC_path

##############################################################################
################################ MAIN ########################################
##############################################################################


def cleanup_files(config):
    """Move NMR SDF files to a dedicated subfolder and clean up other files"""
    # Ensure we have a random number
    if not hasattr(config, 'ran_num'):
        config.ran_num = random.randint(1, 100000)
    
    # Create a subfolder for SDF files
    sdf_subfolder = os.path.join(config.SGNN_gen_folder_path, f"sdf_{config.ran_num}")
    if not os.path.exists(sdf_subfolder):
        os.makedirs(sdf_subfolder, exist_ok=True)
    
    # Find all files in the main folder
    main_folder = config.SGNN_gen_folder_path
    all_files = os.listdir(main_folder)
    
    # Separate files by type
    nmr_sdf_files = []
    non_nmr_sdf_files = []
    mol_files = []
    
    for f in all_files:
        if not os.path.isfile(os.path.join(main_folder, f)):
            continue
            
        if f.endswith('.sdf'):
            if f.startswith('NMR_'):
                nmr_sdf_files.append(f)
            else:
                non_nmr_sdf_files.append(f)
        elif f.endswith('.mol'):
            mol_files.append(f)
    
    print(f"Found {len(nmr_sdf_files)} NMR SDF files to keep")
    print(f"Found {len(non_nmr_sdf_files)} non-NMR SDF files to delete")
    print(f"Found {len(mol_files)} MOL files to delete")
    
    # Move NMR SDF files to the subfolder
    for sdf_file in nmr_sdf_files:
        src_path = os.path.join(main_folder, sdf_file)
        dst_path = os.path.join(sdf_subfolder, sdf_file)
        
        try:
            shutil.copy2(src_path, dst_path)  # Copy with metadata
            os.remove(src_path)  # Remove the original
            # print(f"Moved {sdf_file} to {sdf_subfolder}")
        except Exception as e:
            print(f"Error moving {sdf_file}: {str(e)}")
    
    # Delete non-NMR SDF files
    for sdf_file in non_nmr_sdf_files:
        try:
            os.remove(os.path.join(main_folder, sdf_file))
            # print(f"Deleted non-NMR SDF file: {sdf_file}")
        except Exception as e:
            print(f"Error deleting {sdf_file}: {str(e)}")
    
    # Delete MOL files
    for mol_file in mol_files:
        try:
            os.remove(os.path.join(main_folder, mol_file))
            # print(f"Deleted MOL file: {mol_file}")
        except Exception as e:
            print(f"Error deleting {mol_file}: {str(e)}")
    
    return sdf_subfolder

def main_run_data_generation(config):
    # Ensure necessary directories exist
    if not os.path.exists(config.SGNN_gen_folder_path):
        os.makedirs(config.SGNN_gen_folder_path, exist_ok=True)
    
    if not os.path.exists(config.SGNN_csv_save_folder):
        os.makedirs(config.SGNN_csv_save_folder, exist_ok=True)
    
    # Ensure we have a random number
    if not hasattr(config, 'ran_num'):
        config.ran_num = random.randint(1, 100000)
        print(f"Setting random ran_num: {config.ran_num}")
    else:
        print(f"Using existing ran_num: {config.ran_num}")
    
    # Run the pipeline
    combined_df = run_sgnn(config)
    print("\033[1m\033[33m run_sgnn: DONE\033[0m")
    
    data_1H, csv_1H_path = run_1H_generation(config)
    print("\033[1m\033[33m run_1H_generation: DONE\033[0m")
    print(f"1H CSV saved to: {csv_1H_path}")
    
    data_13C, csv_13C_path = run_13C_generation(config)
    print("\033[1m\033[33m run_13C_generation: DONE\033[0m")
    print(f"13C CSV saved to: {csv_13C_path}")
    
    data_COSY, csv_COSY_path = run_COSY_generation(config)
    print("\033[1m\033[33m run_COSY_generation: DONE\033[0m")
    print(f"COSY CSV saved to: {csv_COSY_path}")
    
    data_HSQC, csv_HSQC_path = run_HSQC_generation(config)
    print("\033[1m\033[33m run_HSQC_generation: DONE\033[0m")
    print(f"HSQC CSV saved to: {csv_HSQC_path}")
    
    # Print data shapes for verification
    print(f"1H data shape: {data_1H.shape}")
    print(f"13C data shape: {data_13C.shape}")
    print(f"COSY data shape: {data_COSY.shape}")
    print(f"HSQC data shape: {data_HSQC.shape}")
    
    # Clean up files at the very end
    print("\033[1m\033[33m Starting final cleanup...\033[0m")
    sdf_subfolder = cleanup_files(config)
    print(f"Organized SDF files into: {sdf_subfolder}")
    print("\033[1m\033[33m Cleanup complete\033[0m")
    
    return combined_df, data_1H, data_13C, data_COSY, data_HSQC, csv_1H_path, csv_13C_path, csv_COSY_path, csv_HSQC_path



def create_CLIP_dataloaders(config, stoi, stoi_MF):
    #NUM_WORKERS = os.cpu_count()
    batch_size = config.CLIP_batch_size

    train_dataset = dl.MultimodalData(config, 
                                   stoi, 
                                   stoi_MF, 
                                   mode="train")
    
    train_sampler = cf.WeightSortedBatchSampler(train_dataset, 
                                             batch_size, 
                                             drop_last=False)

    test_dataset = dl.MultimodalData(config, 
                                  stoi, 
                                  stoi_MF, 
                                  mode="test")

    test_sampler = cf.WeightSortedBatchSampler(test_dataset, 
                                            batch_size, 
                                            drop_last=False)
    
    val_dataset = dl.MultimodalData(config, 
                                stoi, 
                                stoi_MF, 
                                mode="val")

    #val_sampler = cf.WeightSortedBatchSampler(val_dataset, 
    #                                        1, 
    #                                        drop_last=False)
        
    # Create DataLoaders
    train_dataloader = DataLoader(train_dataset, 
                                  batch_sampler=train_sampler,
                                  #shuffle=True, 
                                  collate_fn=dl.collate_fn, 
                                  num_workers=config.num_workers, 
                                  drop_last=False)
    
    test_dataloader = DataLoader(test_dataset, 
                                  batch_sampler=test_sampler,
                                  #shuffle=False, 
                                  collate_fn=dl.collate_fn, 
                                  num_workers=config.num_workers, 
                                  drop_last=False)

    val_dataloader = DataLoader(val_dataset, 
                                batch_size=1,  # because will run multimodal in parallel on many samples
                                  shuffle=False, 
                                  collate_fn=dl.collate_fn, 
                                  num_workers=config.num_workers, 
                                  drop_last=False)

    dataloaders = {"train":train_dataloader, "test":test_dataloader, "val":val_dataloader} #, "all":all_dataloader}
    return dataloaders

