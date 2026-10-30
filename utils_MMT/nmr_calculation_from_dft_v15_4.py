"""NMR shift calculation from DFT conformer ensembles (v15.4).

This module turns quantum-chemistry output (DFT isotropic shieldings per
conformer) into predicted 1H/13C chemical shifts and, finally, into the
2D HSQC-style peak tables used by the rest of the project.

Pipeline overview
-----------------
1. ``get_solvent_ref_shifts`` / ``get_molecule_data`` parse the reference
   (solvent) and compound SD/MOL files: atom symbols, connectivity, raw text
   lines and an RDKit molecule.
2. ``get_shielding_for_conformers`` -> ``calc_rel_en`` ->
   ``calculate_boltzmann_factors`` -> ``perform_boltzmann_averaging`` convert
   per-conformer isotropic shieldings into one Boltzmann-weighted shielding
   per atom (298.15 K, kcal/mol).
3. ``calculate_nmr_shifts_from_shieldings*`` convert shieldings to shifts,
   either via a solvent reference (TMS/DMSO) or via the linear scaling
   recommended by Cheshire NMR.
4. ``save_results_sdf_file`` stores the shifts in the SDF property
   ``averaged_NMR_shifts``; the ``load_shifts_from_sdf*`` helpers read it back.
5. Chirality/symmetry helpers (``get_chiral_carbons`` ...
   ``find_symmetric_positions``) decide which carbons are split into separate
   peaks or are symmetry-equivalent; ``selecting_shifts`` and
   ``perform_deduplication_if_symmetric`` assemble the peak list and
   ``generate_dft_dataframe`` wraps it into the canonical
   ``F2 (ppm)`` (1H) / ``F1 (ppm)`` (13C) dataframe layout.
6. ``load_*`` helpers read the ACD, real, MNova, ML and DFT sources, and
   ``get_similarity_comparison_variations`` scores two spectra against each
   other with several assignment strategies.

Coordinate conventions: axis ``F1`` = 13C, axis ``F2`` = 1H; all values are in
ppm. Paths are handled as POSIX strings (``/``) throughout, and many loaders
assume the SDF property names produced by this module.
"""

import ast
import collections
import copy
import glob
import math
import os
import pandas as pd
import sys
import time
from IPython.display import SVG, display
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, MolFromSmiles, MolToSmiles, PandasTools, rdmolfiles, rdMolDescriptors
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions
from rdkit.Chem.EnumerateStereoisomers import EnumerateStereoisomers, GetStereoisomerCount

# Custom utilities
# sys.path.insert(0, "/projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/1_Dataexploration/1.1_paper_code")
from utils_MMT.plotting_v15_4 import plot_compare_scatter_plot, plot_compare_scatter_plot_without_direction
from utils_MMT.functions_HSQC_sim_v15_4 import similarity_calculations
from utils_MMT.helper_functions_pl_v15_4 import smile2SDF

# Miscellaneous settings
IPythonConsole.ipython_useSVG = True

# Get the indices of the atoms for calculating the reference shift for carbon and hydrogen
def get_solvent_ref_shifts(solvent_path):
    """Calculate the mean reference shieldings of the solvent.

    The solvent (e.g. TMS or DMSO) is computed in the same DFT protocol as the
    compound, so its mean shieldings act as the internal reference used later
    to convert shieldings into chemical shifts.

    Args:
        solvent_path (str): Path to the solvent SDF file containing an
            ``atom.dprop.Isotropic sheilding`` property block (the misspelling
            is inherited from the source files).

    Returns:
        tuple[float, float]: ``(C13_shield_calc, H1_shield_calc)`` - arithmetic
        mean isotropic shielding over all C atoms and all H atoms, in ppm.

    Side effects:
        Calls ``get_molecule_data``, which writes a ``.mol`` file next to
        ``solvent_path``, and reads the SDF text twice.

    NOTE (known issue, left unchanged):
        Element detection uses a plain substring test (``if "C" in i`` /
        ``if "H" in i``), so any line containing the letter C or H is counted,
        e.g. ``Cl`` atoms, ``M  CHG`` charge lines or the ``$$$$`` separator.
        This shifts the atom-to-shielding zip and can silently pair a carbon
        shielding with a hydrogen atom.
    """
    atom_list, connectivity_list, docline_list, name, mol = get_molecule_data(solvent_path)

    index_list_C =[]
    index_list_H =[]
    start = False

    with open(solvent_path) as f:
        for i in f:

            if "V2000" in i:
                start = True
                counter = -1   #because first round already adds one count without really matchin anything'

            if start:
                if "C" in i:
                    index_list_C.append(counter) 
                if "H" in i:
                    index_list_H.append(counter)
                if "END" in i:
                    break
                counter += 1

    # Get the shielding values from the file
    ticker_shielding = False
    float_shielding_list = []
    with open(solvent_path) as f:
        for idx, i in enumerate(f):
          # get the value streight after the keyword
            if ticker_shielding:
                # To get the same structure as for the Gaussian example save the list in the same way [name, conformer_energy, shifts]
                count = idx               # start the counting of lines from there -> in case the shieldings go over multiple lines
                while ">" not in docline_list[count]:
                    shielding_values = docline_list[count]
                    if shielding_values != "\n":    # This is the separation to the next section
                        float_shielding_list = float_shielding_list + [float(i) for i in shielding_values[:-2].split()]  # convert string to float values and save them in the dict
                    count += 1
                ticker_shielding = False
            if "atom.dprop.Isotropic sheilding" in i:
                ticker_shielding = True

    C_shieldings = []
    H_shieldings = []
    for atom, shielding in zip(atom_list, float_shielding_list):
        if atom == "C":
            C_shieldings.append(shielding)
        if atom == "H":
            H_shieldings.append(shielding)      

    C13_shield_calc = sum(C_shieldings)/len(C_shieldings)
    H1_shield_calc = sum(H_shieldings)/len(H_shieldings)
    return C13_shield_calc, H1_shield_calc


# # Get the indices of the atoms for calculating the reference shift for carbon and hydrogen
# ### ALSO IN NMR SHIFT RECONSTRUCTION
def get_molecule_data(compound_path):
    # Purpose: read an SDF/MOL file into plain Python structures.
    # Args: compound_path (str) - path to the compound .sdf file.
    # Returns: (atom_list, connectivity_list, docline_list, name, mol):
    #   atom_list        - element symbol of every atom in file order,
    #   connectivity_list - parsed bond rows (atom1, atom2, bond type, ...),
    #   docline_list     - every raw line of the file as a string,
    #   name             - file stem used to label conformers,
    #   mol              - the RDKit molecule built from a truncated .mol copy.
    # Side effect: writes "<path without .sdf>.mol" next to the input file, because
    #   RDKit cannot parse the SDF properties and only the MOL block is copied.
    # NOTE (known issue, left unchanged): the stem is taken with
    #   compound_path.split("/")[-1], so a Windows path such as
    #   "C:\\data\\x.sdf" yields a name that still contains the path; any later
    #   filename-based filtering or matching against "/"-separated lists fails.
    """ This function returns the list of atoms of the molecules 
    and a list of strings with each line of the compound document as one string 
    because reading the SDF files with pandas causes an error"""
    
    ################ atom_list ################
    index_list_C =[]
    index_list_H =[]
    docline_list = []
    start = False
    name = compound_path.split("/")[-1].split(".")[0]

    # save each string line of the document in a list
    with open(compound_path) as f:
        for i in f:
            docline_list.append(i)

    # get index location of C and H and atom list
    atom_list = []
    stop_list = ["1","0","2","3","4","5","6","7","8","9","10","11","12","13","14","15"]
    for i in docline_list:

        if start:
            if "C" in i:
                index_list_C.append(counter) 
            if "H" in i:
                index_list_H.append(counter)
            if (i.split()[0] in stop_list):   # to find the end of the atom defining list
                break
            atom = i.split()[3]
            if atom != "0":
                atom_list.append(atom)
                counter += 1

        if "V2000" in i:   # select when to start looking
            start = True
            counter = 0    
            
    ################ connectivity_list ################
    # # To get the reference right add one empty string to the list -> need to check that there is something wrong
    # atom_list_ = [" "] + atom_list
    atom_list_ = atom_list
    start_line = len(atom_list_)+4
    end_line = len(atom_list_)+4 + len(atom_list_)
    connectivity_list = []

    for idx, i in enumerate(docline_list):
        if idx >= start_line and "M  END" not in i:
            add_list = i.split()
            ### if there are more than 100 connection the first and second columns are connected
            ### therefore I need to manually split them
            if len(add_list) ==3:
                part_1 = str(int(i[:3]))
                part_2 = str(int(i[3:6]))
                part_3 = str(int(i[6:9]))
                part_4 = str(int(i[9:12]))
                add_list =[part_1,part_2,part_3,part_4]

            ### For some reason sometimes it is too long and sometimes it is too short
            if add_list[0]=="M" or add_list[0]==">" or add_list[0]=="$$$$":
                pass
            else:
                connectivity_list.append(add_list)
        if  "M  END" in i:
            break  
    
    ################ mol ################
    # save each string line of the document in a list
    with open(compound_path[:-4]+".mol", "w",  encoding='utf-8', errors='ignore') as output:  # Path to broken file
        with open(compound_path) as f:
            for element in f:
                if "END" in element:
                    output.write(element)
                    break
                else:
                    output.write(element)

    mol = Chem.MolFromMolFile(compound_path[:-4]+".mol")
    return atom_list, connectivity_list, docline_list, name, mol

def get_conformer_energy_list(docline_list):
    """Collect the SCF energies of all conformers from the raw DFT text lines.

    Args:
        docline_list (list[str]): Raw lines of the DFT output file as returned
            by ``get_molecule_data``.

    Returns:
        list[float]: One energy (in whatever unit the DFT output used, usually
        Hartree) per conformer, in file order, taken from the line following
        each ``conformer_energy`` marker.

    Side effects:
        None; the input list is only read.
    """
    ticker_shielding = False
    conformer_energy_list = []
    for idx, i in enumerate(docline_list):
        # get the value streight after the keyword
        if ticker_shielding:
            conformer_energy_list.append(float(i))
            ticker_shielding = False
        if "conformer_energy" in i:
            ticker_shielding = True
    return conformer_energy_list


def get_shielding_for_conformers(docline_list, name):
    # Purpose: extract the per-conformer isotropic shielding vectors.
    # Args: docline_list (list[str]) - raw DFT output lines; name (str) - label
    #   prefixed to each conformer (becomes "<name>_<conformer_index>").
    # Returns: (shielding_tensors_list, conformer_energy_list):
    #   shielding_tensors_list - list of rows [label, energy, sh_1, sh_2, ...],
    #     sorted ascending by energy, one row per conformer,
    #   conformer_energy_list - energies in original file order.
    # Side effects: none (pure parsing).
    # NOTE (known issue, left unchanged): the shielding reader repeatedly calls
    #   get_conformer_energy_list(docline_list), which rescans the whole file once
    #   per conformer (quadratic in file length); harmless for correctness but
    #   slow on files with many conformers.
    """ this Function reads out the energies from the provided file (saved in the docline_list) and 
    extracts all the shielding tensors and saves them in from of a list"""
    ticker_shielding = False
    shielding_values_list = []
    conformer = 0
    shielding_tensors_list = []
    float_shielding_list = [name+"_"+str(conformer)] 

    for idx, i in enumerate(docline_list):
        # get the value streight after the keyword
        if ticker_shielding:
            # To get the same structure as for the Gaussian example save the list in the same way [name, conformer_energy, shifts]
            conformer_energy_list = get_conformer_energy_list(docline_list)
            float_shielding_list = float_shielding_list + [conformer_energy_list[conformer]]
            count = idx               # start the counting of lines from there -> in case the shieldings go over multiple lines
            while ">" not in docline_list[count]:
                shielding_values = docline_list[count]
                if shielding_values != "\n":    # This is the separation to the next section
                    float_shielding_list = float_shielding_list + [float(i) for i in shielding_values[:-2].split()]  # convert string to float values and save them in the dict
                count += 1
            shielding_tensors_list.append(float_shielding_list)
            ticker_shielding = False
            conformer += 1
            float_shielding_list = [name+"_"+str(conformer)] 
            
        # find the right line in the document when to start searching for the shielding values
        if "atom.dprop.Isotropic sheilding" in i:  
            ticker_shielding = True
    
    # sort from smallest to biggest scf energy
    shielding_tensors_list.sort(key = lambda x: x[1])
    
    return shielding_tensors_list, conformer_energy_list

def calc_rel_en(shielding_tensors_list):
    """Calculate the energy of each conformer relative to the lowest one.

    Args:
        shielding_tensors_list (list[list]): Rows ``[label, energy, ...]`` as
            produced by ``get_shielding_for_conformers`` (already sorted by
            energy).

    Returns:
        list[float]: Relative energies in the same order as the input, with
        ``0`` for the (sorted) first/lowest conformer.
    """
    minimum_val = shielding_tensors_list[0][1]
    list_min_val = [0]   # First item is zero
    for i in shielding_tensors_list[1:]:
        list_min_val.append(i[1]-minimum_val)
    return list_min_val



def calculate_boltzmann_factors(list_min_val):
    """Compute Boltzmann populations at 298.15 K for the relative energies.

    Args:
        list_min_val (list[float]): Relative conformer energies (kcal/mol), as
            returned by ``calc_rel_en``.

    Returns:
        tuple[numpy.ndarray, float, list[float]]:
        ``(boltzmann_perc, partition_function_Q, list_boltzmann_factors)`` where
        ``boltzmann_perc`` are the normalised populations, ``Q`` the partition
        function (sum of the raw factors) and ``list_boltzmann_factors`` the
        unnormalised ``exp(-E/kT)`` values.

    Note:
        The constants (k = 0.001987204 kcal/(mol*K), T = 298.15 K) assume that
        the energies passed in are in kcal/mol.
    """
    # Variables
    T = 298.15
    k = 0.001987204
    kT = k*T
    
    list_boltzmann_factors = []
    for i in list_min_val:
        val = math.exp(-i/kT)
        list_boltzmann_factors.append(val)

    partition_function_Q = sum(list_boltzmann_factors)
    # boltzmann_perc = list_boltzmann_factors/partition_function_Q
    boltzmann_perc = np.array(list_boltzmann_factors)/partition_function_Q

    return boltzmann_perc, partition_function_Q, list_boltzmann_factors

def perform_boltzmann_averaging(shielding_tensors_list,list_boltzmann_factors,partition_function_Q):
    """Average the shielding values of all conformers per atom.

    Args:
        shielding_tensors_list (list[list]): Rows ``[label, energy, sh_1, ...]``
            as produced by ``get_shielding_for_conformers``.
        list_boltzmann_factors (list[float]): Unnormalised Boltzmann factors
            from ``calculate_boltzmann_factors``.
        partition_function_Q (float): Normalisation constant (sum of factors).

    Returns:
        list[float]: Boltzmann-weighted average shielding for every atom,
        ordered exactly like the per-atom values inside each row.

    Side effects:
        None; builds temporary lists only.
    """

    master_shielding_list = []
    for idx, shielding_list in enumerate(shielding_tensors_list):
        new_shielding_list = []
        for j in shielding_list[2:]:
            val =  list_boltzmann_factors[idx]/ partition_function_Q *j
            new_shielding_list.append(val)
        master_shielding_list.append(new_shielding_list)
        
    # Add all the contribution together
    boltzman_avg_shifts = [sum(i) for i in zip(*master_shielding_list)]
    return boltzman_avg_shifts


def calculate_nmr_shifts_from_shieldings(atom_list,boltzman_avg_shifts,C13_shift_exp,C13_shield_calc,H1_shield_calc,H1_shift_exp):
    # Purpose: convert averaged shieldings to shifts using the solvent reference.
    # Args: atom_list - element symbol per atom (same order as the shifts);
    #   boltzman_avg_shifts - averaged shielding per atom (ppm);
    #   C13_shift_exp / H1_shift_exp - experimental reference shift of the solvent;
    #   C13_shield_calc / H1_shield_calc - computed mean shielding of the solvent.
    # Returns: list[float] copy of the shifts converted to ppm, leaving atoms whose
    #   symbol is not "C"/"H" at their original shielding value.
    # Side effects: none - the input list is deep-copied, never mutated.
    """ This function corrects the carbon and hydrogen shifts based on the standards of 1H and 13C solvent DFT calculation to correct the 
    shift and saved the corrected shifts inplace in boltzman_avg_shifts """
    import copy
    boltzman_avg = copy.deepcopy(boltzman_avg_shifts)
    for idx, (shieldings,atom) in enumerate(zip(boltzman_avg, atom_list)):
        if atom == "C":
            c_shift = C13_shield_calc-shieldings+C13_shift_exp
            boltzman_avg[idx] = c_shift
            
        if atom == "H":
            h_shift = H1_shield_calc-shieldings+H1_shift_exp
            boltzman_avg[idx] = h_shift
    return boltzman_avg

# Parameters from this website: http://cheshirenmr.info/Recommendations.htm
# http://cheshirenmr.info/ScalingFactors.htm#table5dimethylsulfoxideheading
def calculate_nmr_shifts_from_shieldings_slope_intercept(atom_list,boltzman_avg_shifts, slope_H, intercept_H, slope_C, intercept_C):
    # Purpose: convert averaged shieldings to shifts with a linear scaling model.
    # Args: atom_list - element symbol per atom; boltzman_avg_shifts - averaged
    #   shieldings (ppm); slope_H/intercept_H/slope_C/intercept_C - linear
    #   regression parameters of the 1H and 13C referencing (Cheshire NMR tables).
    # Returns: list[float] copy of the shifts in ppm, with atoms that are neither
    #   "C" nor "H" left at their original shielding value.
    # Side effects: none - the input list is deep-copied.
    """ This function corrects the carbon and hydrogen shifts based on the standards of 1H and 13C solvent DFT calculation to correct the 
    shift and saved the corrected shifts inplace in boltzman_avg_shifts 
    based on: http://cheshirenmr.info/Instructions.htm"""
    import copy
    boltzman_avg = copy.deepcopy(boltzman_avg_shifts)

    for idx, (atom, shielding) in enumerate(zip(atom_list,boltzman_avg)):
        if atom =="C":
            shift_C = (intercept_C-shielding)/-slope_C
            boltzman_avg[idx] = shift_C
        if atom == "H":
            shift_H = (intercept_H-shielding)/-slope_H
            boltzman_avg[idx] = shift_H
    return boltzman_avg

def save_results_sdf_file(conformer_energy_list, docline_list, boltzman_avg_shifts_corr_2, compound_path):
    """Write the predicted shifts into an SDF built from the lowest-energy conformer.

    Args:
        conformer_energy_list (list[float]): Energies in the order the conformers
            appear in the source file (used to locate the minimum).
        docline_list (list[str]): Raw source-file lines; the block belonging to the
            lowest-energy conformer is extracted via its ``0:0:<index>`` marker.
        boltzman_avg_shifts_corr_2 (list[float]): Final shifts, one per atom.
        compound_path (str): Source file path; the output is written next to it.

    Returns:
        str: Path of the written ``*_shifts.sdf`` file.

    Side effects:
        Creates/overwrites ``<compound_path without extension>_shifts.sdf`` and
        appends the ``averaged_NMR_shifts`` property plus its conformer index.
    """
    lowest_conformer_index = conformer_energy_list.index(min(conformer_energy_list))
    
    ### Select the lowest energy conformer
    selected_confirmer = []
    ticker = False
    for i in docline_list:

        if  f"0:0:{lowest_conformer_index}\n" in i:
            ticker = True

        if f"0:0:{lowest_conformer_index+1}\n" in i:
            break

        if ticker == True:
            selected_confirmer.append(i)
    
    ### generate string that will be added in the file (with all the calculated shifts
    string = ""
    for i in boltzman_avg_shifts_corr_2:
        string = string + " " + str(i)
    final_string = string[1:]+"\n"
    final_string
    
    ### add NMR shift data at the end before the name of the compound
    selected_confirmer.insert(-1, f">  <averaged_NMR_shifts>  ({lowest_conformer_index+1}) \n")
    selected_confirmer.insert(-1, final_string)
    selected_confirmer.insert(-1, "\n")

    ### Save file in selected location
    save_path = compound_path.split(".")[0]+"_shifts.sdf"
    with open(os.path.join(save_path), 'w') as f:
        for i in selected_confirmer:
            f.write(i)
    return save_path

# from nmr_calculation_from_dft import save_results_sdf_file, get_solvent_ref_shifts, get_molecule_data, get_shielding_for_conformers, calc_rel_en, calculate_boltzmann_factors, perform_boltzmann_averaging, calculate_nmr_shifts_from_shieldings, calculate_nmr_shifts_from_shieldings_slope_intercept
# Calculate ref values from solvent
def run_script(solvent_path, compound_path, C13_shift_exp, H1_shift_exp, \
               slope_H, intercept_H, slope_C, intercept_C, extra_info=True):
    """End-to-end shift prediction for a single compound.

    Runs the whole DFT post-processing chain: solvent reference -> molecule
    parsing -> per-conformer shieldings -> relative energies -> Boltzmann
    weights -> averaging -> shift conversion -> SDF export.

    Args:
        solvent_path (str): SDF file of the solvent used as DFT reference.
        compound_path (str): SDF file of the compound to process.
        C13_shift_exp (float): Experimental 13C shift of the solvent reference.
        H1_shift_exp (float): Experimental 1H shift of the solvent reference.
        slope_H, intercept_H, slope_C, intercept_C (float): Linear scaling
            parameters used by the slope/intercept referencing.
        extra_info (bool): If True, print every intermediate result.

    Returns:
        tuple[str, list[float]]: ``(save_path, boltzman_avg_shifts_corr_1)`` -
        the path of the written SDF and the final per-atom shifts (ppm).

    Side effects:
        Writes the ``*_shifts.sdf`` file (through ``save_results_sdf_file``),
        creates the intermediate ``.mol`` file and prints to stdout.
    """
    
    C13_shield_calc, H1_shield_calc = get_solvent_ref_shifts(solvent_path)

    
    atom_list, connectivity_list, docline_list, name, mol = get_molecule_data(compound_path)

    
    shielding_tensors_list, conformer_energy_list = get_shielding_for_conformers(docline_list, name)


    list_min_val = calc_rel_en(shielding_tensors_list)


    boltzmann_perc, partition_function_Q, list_boltzmann_factors = calculate_boltzmann_factors(list_min_val)


    boltzman_avg_shifts = perform_boltzmann_averaging(shielding_tensors_list,list_boltzmann_factors,partition_function_Q)

    # #DMSO as reference
    # boltzman_avg_shifts_corr_1 = calculate_nmr_shifts_from_shieldings(atom_list,boltzman_avg_shifts,C13_shift_exp,C13_shield_calc,H1_shield_calc,H1_shift_exp)
    # # print(boltzman_avg_shifts_corr_1)

    # TMS as reference
    boltzman_avg_shifts_corr_1 =  calculate_nmr_shifts_from_shieldings_slope_intercept(atom_list,boltzman_avg_shifts, slope_H, intercept_H, slope_C, intercept_C)
    print(boltzman_avg_shifts_corr_1)

    save_path = save_results_sdf_file(conformer_energy_list, docline_list, boltzman_avg_shifts_corr_1, compound_path)
    if extra_info:
        print(C13_shield_calc, H1_shield_calc)
        print("---------------------------------------------")
        print(atom_list, name)
        print("---------------------------------------------")
        print(shielding_tensors_list, conformer_energy_list)
        print("---------------------------------------------")
        print(list_min_val)
        print("---------------------------------------------")
        print(boltzmann_perc,partition_function_Q, list_boltzmann_factors)
        print("---------------------------------------------")
        print(boltzman_avg_shifts)
        print("---------------------------------------------")
        print(boltzman_avg_shifts_corr_1)
        print("---------------------------------------------")    
        # print(boltzman_avg_shifts_corr_2)
        print("---------------------------------------------")
        print(save_path)
        print("---------------------------------------------")
    return save_path, boltzman_avg_shifts_corr_1
# save_path, boltzman_avg_shifts_corr_2 = run_script(solvent_path, compound_path, C13_shift_exp, H1_shift_exp, slope_H, intercept_H, slope_C, intercept_C, extra_info=False)

###############################################################################################

def get_HSQC_info_data(mol=None, smiles="", file_path="", extra_info = False):
    # Purpose: build the C->H connectivity used to assemble HSQC peaks.
    # Args: exactly one of mol (RDKit Mol), smiles (str) or file_path (str);
    #   extra_info (bool) only triggers a debug display of the H-suppressed mol.
    # Returns: (count_list, m, stereo_smi, am, elements):
    #   count_list - {carbon_atom_index: [hydrogen_atom_index, ...]},
    #   m          - molecule with explicit H and atom-map numbers,
    #   stereo_smi - isomeric SMILES of the first stereoisomer,
    #   am         - adjacency matrix (numpy array, H included),
    #   elements   - [(element_symbol, atom_index), ...] in atom order.
    # Side effects: when extra_info is True, renders and prints via IPython.
    # NOTE (known issue, left unchanged): neither argument is validated, so calling
    #   with all three empty returns "unbound local variable m" (the branch chain
    #   silently falls through).
    """ This function takes a smiles and calculates the carbons that are connected
    to H that will be displayed in an HSQC spectrum and it returns a dict that
    has the labels of the C with connections to which H
    does not consider chiral compound peak splits yet
    it just takes SMILES or the path to an SDF file
    """
    if mol!=None:
        m = mol_with_atom_index(mol,True)
        am = Chem.GetAdjacencyMatrix(m)
        
    elif smiles != "":
        mol = MolFromSmiles(smiles)
        m = mol_with_atom_index(mol,True)
        am = Chem.GetAdjacencyMatrix(m)
        
    elif file_path!="":
        mol = Chem.MolFromMolFile(file_path)
        m = mol_with_atom_index(mol,True)
        am = Chem.GetAdjacencyMatrix(m)
        
    elements = [(atom.GetSymbol(),idx) for idx, atom in enumerate(m.GetAtoms())]

    if extra_info:
        m = Chem.RemoveAllHs(m)
        smi = MolToSmiles(m)
        m = MolFromSmiles(smi)
        display(m)
        print(elements)
    
    # check for C-H neighbors
    count_list = {}
    for atom, nr in elements:
        if atom == "C":
            for idx, connector in enumerate(am[nr]):

                if connector == 1 and elements[idx][0] =="H":
                    try:
                        # if there is no carbon in the dict yet it will fail and go to except
                        type(count_list[nr]) == list
                        count_list[nr]+=[elements[idx][1]]
                    except:
                        count_list[nr]=[elements[idx][1]]
                        
    # get stereo smile
    # num_iso = GetStereoisomerCount(mol)

    try:
        isomers = tuple(EnumerateStereoisomers(mol))
        stereo_smi = Chem.MolToSmiles(isomers[0],isomericSmiles=True)
    except:
        stereo_smi = Chem.MolToSmiles(mol,isomericSmiles=True)

    return count_list, m, stereo_smi, am, elements


# from utils.functions import mol_with_atom_index
def mol_with_atom_index(mol, include_H=True):
    # Purpose: make atom indices visible when the molecule is rendered.
    # Args: mol (RDKit Mol); include_H (bool) - add explicit hydrogens first.
    # Returns: a new RDKit Mol where every atom carries the property
    #   "molAtomMapNumber" equal to its index (used for drawing/labels only).
    # Side effects: adds H atoms and coordinates to the molecule when include_H.
    """ 
    Takes a mol as input and adds H to them
    Visualizes the number which is assigned to which atom as a mol
    if include_H is True then H will be added and also labeled in the mol file
    """
    # mol = MolFromSmiles(smiles)
    if include_H:
        mol = Chem.AddHs(mol, addCoords=True)

    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol


### Get the chiral carbons from the compound
def get_chiral_carbons(stereo_smi):
    # Purpose: locate stereocentres in a SMILES string.
    # Args: stereo_smi (str) - isomeric SMILES.
    # Returns: list[int] - 0-based "heteroatom counter" positions of the atoms that
    #   are immediately followed by "@" (carbon or nitrogen stereocentres).
    # Side effects: none.
    # NOTE (known issue, left unchanged): the counter skips only the characters in a
    #   fixed ignore list; two-letter elements such as "Cl"/"Br" count as one atom,
    #   and ring-closure digits above 8 are counted as atoms, so the returned
    #   positions can be offset relative to the RDKit atom indices.
    """ this function gives the chiral carbons based on the stereo-smiles provided and returns the number of the chiral carbon
    based on the occurence of that hetero-atom within the smiles string"""
    chiral_centers = []
    heteroatom_count = 0
    for idx, atom in enumerate(stereo_smi):
        try:
            if (atom =="C" or atom =="c") and stereo_smi[idx+1]=="@":
                chiral_centers.append(heteroatom_count)

            # Catch NH+ compounds that have isomeric centers
            if atom =="N" and stereo_smi[idx+1]=="@":  
                chiral_centers.append(heteroatom_count)

            if atom not in ["H","@","+","-","(",")","[","]","1","2","3","4","5","6","7","8"]:   # just defining the symbols that are not counted in the heteroatom count
                heteroatom_count +=1
        except:
            pass
    return chiral_centers


def check_terminal_chiral_center(chiral_C, am, elements):
    """Decide whether a stereocentre sits at the end of a chain.

    A centre is treated as "terminal" when it has exactly three distinct
    non-carbon neighbours (e.g. H, halogen, O, N), i.e. only one carbon
    substituent.

    Args:
        chiral_C (int): Atom index of the stereocentre.
        am (numpy.ndarray): Adjacency matrix including hydrogens.
        elements (list[tuple[str, int]]): ``(symbol, index)`` per atom.

    Returns:
        bool: True if the centre has three distinct non-carbon substituents.
    """
    surrounding_list = []
    for idx, j in enumerate(am[chiral_C]):
        #not refering to itself and no C because otherwise it is not a terminal steroecenter
        if j == 1  and elements[idx][1] != chiral_C and elements[idx][0] != "C": 
            surrounding_list.append(elements[idx][0])
    # print(surrounding_list)
    # check if there are 3 different substituents but no C because otherwise it is not a terminal stereocenter
    if len(set(surrounding_list))==3:
        return True
    else:
        return False  

    
# get ID from direct C neighbor
# and check if it is a terminal chiral center and save it in a separate dict
from collections import defaultdict
def chiral_center_type(chiral_centers, elements, am):
    # Purpose: split stereocentres into "terminal" and "chain" centres and collect
    #   the carbon neighbours whose peaks are split by the stereocentre.
    # Args: chiral_centers (list[int]) - positions from get_chiral_carbons;
    #   elements - (symbol, index) pairs; am - adjacency matrix.
    # Returns: (direct_neighbor_list, direct_neighbor_dict,
    #           terminal_neighbor_list, terminal_neighbor_dict), where the lists
    #   flatten the dicts and the dicts map centre -> carbon neighbour indices.
    # Side effects: none.
    """This function checks if the chiral center is at the end of a chain (terminal) 
    or somewhere in the middle of the molecule"""
    direct_neighbor_dict = defaultdict(list)
    terminal_neighbor_dict = defaultdict(list)
    direct_neighbor_list = []
    terminal_neighbor_list = []
    # chiral_C = 1   # for now just on one

    for chiral_center in chiral_centers:

        #check if chiral center is terminal
        terminal_neighbor = check_terminal_chiral_center(chiral_center, am, elements)
        if terminal_neighbor:
            for idx,i in enumerate(am[chiral_center]):
                if i == 1 and elements[idx][0]=="C":
                    terminal_neighbor_list.append(idx)
                    terminal_neighbor_dict[chiral_center].append(idx)
        else:
            for idx,i in enumerate(am[chiral_center]):
                if i == 1 and elements[idx][0]=="C":
                    direct_neighbor_list.append(idx)
                    direct_neighbor_dict[chiral_center].append(idx)

    return direct_neighbor_list, direct_neighbor_dict, terminal_neighbor_list, terminal_neighbor_dict


def get_bond_type(idx,ring_bound_list):
    """Return the bond type of the first ring bond that contains ``idx``.

    Args:
        idx (int): Atom index to look up.
        ring_bound_list (list[list]): ``[atom_i, atom_j, bond_type_string]`` rows
            for consecutive atoms of every ring.

    Returns:
        str: ``"AROMATIC"`` or ``"SINGLE"``.

    NOTE (known issue, left unchanged):
        The loop always exits after the first row (``break`` is unconditional), so
        only the very first ring bond in the list is ever inspected. Every ring
        bond after it is classified as ``"SINGLE"`` even when it is aromatic,
        which can wrongly mark chiral centres as "not next to an aromatic ring".
    """
    for i in ring_bound_list:
        if idx in i and i[2] == "AROMATIC":
            bond_type = "AROMATIC"
        else:
            bond_type = "SINGLE"
        break
    return bond_type


# check if chiral center is next to aromatic ring then don't split the second layer distance
# because if chiral center is next to aromatic ring the second layer doesn't split
def get_exclude_second_degree_list(mol, chiral_centers, elements, am):
    # Purpose: detect stereocentres adjacent to an aromatic ring, where the
    #   second-layer splitting does not occur in real spectra.
    # Args: mol (RDKit Mol) - needed for ring perception;
    #   chiral_centers (list[int]); elements - (symbol, index) pairs; am - adjacency.
    # Returns: (exclude_second_degree_list, ring_bound_list): the centre indices to
    #   exclude and the ring bond rows [atom_i, atom_j, bond_type].
    # Side effects: none.
    """This function checks if chiral centers are next to aromatic rings 
    in that case no second layer gets split"""
    ringinfo = mol.GetRingInfo()
    rings = ringinfo.AtomRings()
    ring_atoms = [i for ring in rings for i in ring]
    ring_bound_list = []
    for ring_atoms in rings:
        for i in range(len(ring_atoms)-1):
            bound_type = str(mol.GetBondBetweenAtoms(ring_atoms[i], ring_atoms[i+1]).GetBondType())
            ring_bound_list.append([ring_atoms[i], ring_atoms[i+1], bound_type])

    exclude_second_degree_list = []
    if len(chiral_centers) !=0:
        for i in chiral_centers:
            # print(i)
            for idx, j in enumerate(am[i]):
                if (j == 1) and elements[idx][0] == "C" and idx in ring_atoms:
                    bond_type = get_bond_type(idx,ring_bound_list)
                    if bond_type  == "AROMATIC":
                        exclude_second_degree_list.append(i)
    return exclude_second_degree_list, ring_bound_list



# remove the reference that is from the chiral center that is next to the ring
def get_second_degree_neighbor(direct_neighbor_dict, exclude_second_degree_list, am, elements, direct_neighbor_list, chiral_centers, terminal_neighbor_list):
    # Purpose: extend split positions from direct neighbours to second-layer carbons.
    # Args: direct_neighbor_dict - centre -> direct carbon neighbours;
    #   exclude_second_degree_list - centres next to aromatic rings (ignored);
    #   am/elements - graph data; direct_neighbor_list/terminal_neighbor_list -
    #   already known positions that must not be added again; chiral_centers.
    # Returns: (sec_direct_neighbor, direct_neighbor_dict): the newly found second
    #   layer indices and the updated mapping.
    # Side effects: appends to the *caller's* dict in place (the deep copy is only
    #   used to drop excluded centres); the caller therefore sees the mutation.
    """This function removes the chiral centers next to aromatic rings and addes the positions that
    with a distance of two to the list"""
    import copy
    direct_neighbor_dict_ = copy.deepcopy(direct_neighbor_dict)

    if len(direct_neighbor_dict_) !=0:
        for i in exclude_second_degree_list:
            del direct_neighbor_dict_[i]

    # get second degree neighbor
    sec_direct_neighbor = []

    for key,values in direct_neighbor_dict_.items():
        # print(key,values)
        for neighbor in values:
            # print(neighbor)
            for idx, i in enumerate(am[neighbor]):
                if (i == 1 and elements[idx][0]=="C") and (elements[idx][1] not in list(chiral_centers)) and (elements[idx][1] not in direct_neighbor_list) and (idx not in terminal_neighbor_list):
                    direct_neighbor_dict[key].append(idx)
                    sec_direct_neighbor.append(idx)
    return sec_direct_neighbor, direct_neighbor_dict


def get_all_chiral_split_positions(direct_neighbor_dict, terminal_neighbor_dict):
    """Flatten both neighbour dictionaries into one list of split positions.

    Args:
        direct_neighbor_dict (dict[int, list[int]]): Chain-centre mapping.
        terminal_neighbor_dict (dict[int, list[int]]): Terminal-centre mapping.

    Returns:
        list[int]: All carbon atom indices whose peaks are split by a stereocentre.
    """
    all_split_positions = []
    for k,v in direct_neighbor_dict.items():
        for i in v:
            all_split_positions.append(i)

    for k,v in terminal_neighbor_dict.items():
        for i in v:
            all_split_positions.append(i)
    return all_split_positions


# Check if those neighbors have symmetric substitutes - if so pop them because will not be duplicated
def check_for_symmetric_substituents(all_split_positions, chiral_centers, elements, am):
    """Drop split positions whose substituents are all identical (e.g. CH3).

    Args:
        all_split_positions (list[int]): Candidate positions (mutated in place).
        chiral_centers (list[int]): Stereocentre indices.
        elements (list[tuple[str, int]]): ``(symbol, index)`` per atom.
        am (numpy.ndarray): Adjacency matrix.

    Returns:
        list[int]: Positions that were removed because they are symmetrical.

    Side effects:
        Pops symmetric entries from ``all_split_positions`` in place.

    NOTE (known issue, left unchanged):
        ``elements[idx][1] != chiral_centers`` compares an int with a *list*, so it
        is always True. The intended test was "not in chiral_centers"; as written,
        carbon neighbours that are themselves stereocentres are not excluded, so
        a symmetric assignment can be produced for those positions.
    """
    remove_neighbor_list = []
    for i in all_split_positions:
        surrounding_list=[]
        for idx, j in enumerate(am[i]):

            if j == 1  and elements[idx][1] != chiral_centers and elements[idx][1] not in all_split_positions: 
                surrounding_list.append(elements[idx][0])

        ### Remove the neighbor if all substituents are the same or if there are 3 equivalent attoms attached to that carbon (e.g. 3xH)
        if len(set(surrounding_list)) == 1 and len(surrounding_list) ==3:
            index_to_delete = all_split_positions.index(i)
            sym_terminal_neighbor = all_split_positions.pop(index_to_delete)
            
            # to catch the case if it is a ring with 2 C as connection - the I don't want the direct neighbor being removed
            if len(surrounding_list)!= 2:
                remove_neighbor_list.append(sym_terminal_neighbor)
    return remove_neighbor_list


#################### this is a weird exception #########################
# maybe do something about polar compound structures
########################################################################
# # these index carbons will see a split in their spectrum because of the chiral center close by
# # to catch this case: smile_aa = "Fc1c(cccc1CC(Br)(F)OC)CC(O)CC"
# # the second last C in the chain is also not split for some reason in ACD labs?!
# # this has to do with the electronegativity next to the chiral center if polar F or benzo ring then no split
def remove_symmetric_centers(all_split_positions, remove_neighbor_list, sec_direct_neighbor, elements, am):
    """Remove second-layer positions that are equivalent to a CH3-like carbon.

    Args:
        all_split_positions (list[int]): Candidate split positions (mutated).
        remove_neighbor_list (list[int]): Positions flagged as symmetric.
        sec_direct_neighbor (list[int]): Second-layer positions found earlier.
        elements (list[tuple[str, int]]): ``(symbol, index)`` per atom.
        am (numpy.ndarray): Adjacency matrix.

    Returns:
        list[int]: The pruned ``all_split_positions`` list.

    Side effects:
        Pops entries from ``all_split_positions`` in place.
    """
    for i in remove_neighbor_list:
        if (i in sec_direct_neighbor) and (i in all_split_positions):
            for idx, j in enumerate(am[i]):
                if j == 1  and elements[idx][0] =="C" and idx != i:
                    # just delete if there is no other "CH2" spacer
                    # h_num = check_number_of_h(am, jdx)
                    # if h_num ==2:
                    if idx in all_split_positions: #otherwise crashes for some compounds
                        index_to_delete = all_split_positions.index(idx)
                        all_split_positions.pop(index_to_delete)
    return all_split_positions


# check if those carbon positions have 2 hydrogens attached which can be distinguished
# if not pop them as well
# need to do a copy otherwise it doesn't loop over everything
def check_for_two_H(all_split_positions, elements, am):
    """Keep only split positions that carry exactly two hydrogens.

    A CH2 group is the only case in which two distinguishable protons of the same
    carbon produce separate HSQC peaks; CH, CH3 or quaternary carbons are dropped.

    Args:
        all_split_positions (list[int]): Candidate positions (mutated).
        elements (list[tuple[str, int]]): ``(symbol, index)`` per atom.
        am (numpy.ndarray): Adjacency matrix.

    Returns:
        list[int]: The pruned ``all_split_positions`` list.

    Side effects:
        Pops entries from ``all_split_positions`` in place.
    """
    all_split_positions_ = all_split_positions.copy()
    for i in all_split_positions_:
        num_h = 0
        for idx, j in enumerate(am[i]):
            if j == 1  and elements[idx][0] == "H":
                num_h += 1
        if num_h != 2:

            index_to_delete = all_split_positions.index(i)
            all_split_positions.pop(index_to_delete)
    return all_split_positions


def find_symmetric_positions(stereo_smi):
    """Find groups of symmetry-equivalent atoms in a molecule.

    Uses RDKit canonical ranking with ties kept (``breakTies=False``) and
    self-substructure matches to group atoms that are equivalent under the
    molecular symmetry.

    Args:
        stereo_smi (str): Isomeric SMILES of the compound.

    Returns:
        list[list[int]]: One inner list per group of equivalent atom indices;
        empty when the molecule has no symmetry.

    Side effects:
        None; builds temporary RDKit molecules.
    """

    mol = Chem.MolFromSmiles(stereo_smi)
    isomers = tuple(EnumerateStereoisomers(mol))
    stereo_smi_ = Chem.MolToSmiles(isomers[0], isomericSmiles=True, canonical=True)
    mol = Chem.MolFromSmiles(stereo_smi_) 

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



def run_chiral_and_symmetry_finder(smiles="", compound_path="", extra_info=False):
    """Run the full chirality/symmetry analysis for one compound.

    Either builds a molecule from ``smiles`` or reads ``compound_path`` (exactly
    one of the two should be given). The function chains all stereocentre,
    second-layer and symmetry helpers and returns the two lists needed to build
    the final peak table.

    Args:
        smiles (str): SMILES of the compound (mutually exclusive with path).
        compound_path (str): Path to an SDF/MOL file.
        extra_info (bool): Print every intermediate result to stdout.

    Returns:
        tuple[list[list[int]], list[int], Chem.Mol, str]:
        ``(sym_dupl_lists, all_split_positions, mol, compound_path)``.

    Side effects:
        When ``smiles`` is given, writes a temporary SDF via ``smile2SDF`` and
        prints debug output if ``extra_info`` is True.
    """

    if compound_path != "":
        mol = Chem.MolFromMolFile(compound_path)
        mol = Chem.AddHs(mol, addCoords=True)
    elif smiles != "":
        mol = MolFromSmiles(smiles)
        mol = Chem.AddHs(mol, addCoords=True)
        compound_path = smile2SDF(smiles)
        
    count_list, m, stereo_smi, am, elements = get_HSQC_info_data(mol=mol, smiles="", file_path="", extra_info = False)
    chiral_centers = get_chiral_carbons(stereo_smi)
    direct_neighbor_list, direct_neighbor_dict, terminal_neighbor_list, terminal_neighbor_dict = chiral_center_type(chiral_centers, elements, am)
    
    exclude_second_degree_list, ring_bound_list = get_exclude_second_degree_list(mol, chiral_centers, elements, am)
    # print(exclude_second_degree_list,  direct_neighbor_dict)
    
    sec_direct_neighbor, direct_neighbor_dict = get_second_degree_neighbor(direct_neighbor_dict, exclude_second_degree_list, am, elements, direct_neighbor_list, chiral_centers, terminal_neighbor_list)
    
    all_split_positions = get_all_chiral_split_positions(direct_neighbor_dict, terminal_neighbor_dict)
    
    remove_neighbor_list = check_for_symmetric_substituents(all_split_positions, chiral_centers, elements, am)
    
    all_split_positions = remove_symmetric_centers(all_split_positions, remove_neighbor_list, sec_direct_neighbor, elements, am)
    
    all_split_positions = check_for_two_H(all_split_positions, elements, am)
    
    sym_dupl_lists = find_symmetric_positions(stereo_smi)

    if extra_info:
        print(count_list, stereo_smi, am, elements)
        print("---------------------------------------------")
        print(chiral_centers)
        print("---------------------------------------------")
        print(direct_neighbor_dict,terminal_neighbor_dict)
        print("---------------------------------------------")
        print(exclude_second_degree_list)
        print("---------------------------------------------")
        print(sec_direct_neighbor, direct_neighbor_dict)
        print("---------------------------------------------")
        print(all_split_positions)
        print("---------------------------------------------")
        print(all_split_positions)
        print("---------------------------------------------")
        print(all_split_positions)
        print("---------------------------------------------")
        print(all_split_positions)
        print("---------------------------------------------")
        print(sym_dupl_lists)
        print("---------------------------------------------")
    return sym_dupl_lists, all_split_positions, mol, compound_path

# sym_dupl_lists, all_split_positions, mol, compound_path = run_chiral_and_symmetry_finder(compound_path=compound_path)
# print(sym_dupl_lists, all_split_positions)



# Get C-H connectivity dict with index starting from 0 to correct it to python convention
def get_c_h_connectivity(connectivity_list, atom_list):
    # Purpose: build the carbon -> hydrogens mapping from the SDF bond block.
    # Args: connectivity_list (list[list[str]]) - bond rows [atom1, atom2, type, ...]
    #   (1-based as in the file); atom_list (list[str]) - element per atom.
    # Returns: dict[int, list[int]] - 0-based carbon index -> 0-based H indices.
    # Side effects: none.
    # Note: the "type(...) == list" line is not an assert; it deliberately raises
    #   KeyError for a carbon that is not yet in the dict, which is caught to
    #   initialise the list. It works, but reads like dead code.
    """ This function checks the connectifity list and creates a dictionary with all the 
    carbons that are connected to hydrogens with their labelled numbers"""
    c_h_connectivity_dict = {}
    # print(connectivity_list)
    for i in connectivity_list:
        selected_atom_nr = int(i[0])-1
        selected_connection_nr = int(i[1])-1
        atom = atom_list[selected_atom_nr]
        connection = atom_list[selected_connection_nr]
        num_connection = atom_list[int(i[2])]
        # check atom C-H bonds and add them to dictionary
        if atom =="C" and connection == "H":
            found_H_nr = [selected_connection_nr]
            found_C_nr = selected_atom_nr
            try:
                # if there is no carbon in the dict yet it will fail and go to except
                type(c_h_connectivity_dict[found_C_nr]) == list
                c_h_connectivity_dict[found_C_nr]+=found_H_nr
            except:
                c_h_connectivity_dict[found_C_nr]=found_H_nr
        # check atom H-C bonds and add them to dictionary
        if atom =="H" and connection == "C":
            found_C_nr = selected_connection_nr
            found_H_nr = [selected_atom_nr]
            try:
                # if there is no carbon in the dict yet it will fail and go to except
                type(c_h_connectivity_dict[found_C_nr]) == list
                c_h_connectivity_dict[found_C_nr]+=found_H_nr
            except:
                c_h_connectivity_dict[found_C_nr]=found_H_nr
    return c_h_connectivity_dict


def plot_dft_spectrum(shifts, amplifyer=850, transp=0.50):
    """Plot a quick 2D scatter view of the predicted peaks.

    Args:
        shifts (list[list]): Rows ``[C_shift, H_shift, intensity]``.
        amplifyer (int): Kept for API compatibility; not used for sizing here.
        transp (float): Marker transparency.

    Returns:
        None; the figure is shown via ``plt.show()``.

    Side effects:
        Creates a matplotlib figure and opens an interactive plot window.

    NOTE (known issue, left unchanged):
        ``match`` is computed once from ``group[0]`` and reused for every group,
        so both the positive and the negative intensities are drawn from the same
        subset of points and the legend colours therefore do not reflect the
        actual sign of each peak.
    """
    import matplotlib.pyplot as plt

    scatter_x_2 = np.array(shifts)[:,1]
    scatter_y_2 = np.array(shifts)[:,0]
    intensity =np.array(shifts)[:,2]
    group = [ '+ve' if i>0 else '-ve' for i in intensity]

    fig, ax = plt.subplots(figsize=(10,5), dpi=80)
    match = [True if i == group[0] else False for i in group]

    for g in np.unique(group):
        # print(scatter_x_2[i], scatter_y_2[i],abs(intensity[i])*amplifyer, label=g,  alpha=transp)
        ax.scatter(scatter_x_2[match], scatter_y_2[match], label=g,  alpha=transp)
    ax.legend()
    plt.xlim(xmin=0,xmax=11)
    plt.ylim(ymin=0,ymax=200)

    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()

    plt.grid()
    plt.title("Plot")
    plt.show()


    # here we calculate the averaged H shifts and match them with the carbon shifts 
# generating the HSQC spectrum including directions of the intensity

def selecting_shifts(c_h_connectivity_dict, all_split_positions, boltzman_avg_shifts_corr_2):
    # Purpose: turn per-atom shifts into HSQC peak rows.
    # Args: c_h_connectivity_dict - carbon -> hydrogen indices;
    #   all_split_positions - carbons whose two protons are resolved separately;
    #   boltzman_avg_shifts_corr_2 - final per-atom shifts (ppm).
    # Returns: list[list] rows [C_shift, H_shift, direction, carbon_index,
    #   hydrogen_index_list], where direction is +1 for odd H counts (CH/CH3) and
    #   -1 for even ones (CH2, phase inversion in HSQC-DEPT). A CH2 carbon in
    #   all_split_positions produces one row per proton.
    # Side effects: none.
    """This functions takes the connectifity list and the symmetry and chirality information and
    reconstructs the shifts including their splittings and returns it as a final list
    including the direction of the peak based on the number of H bound to C"""
    shifts = []
    # iterate over all items of dict
    for cab_pos,hyd_pos_list in c_h_connectivity_dict.items():
        hyd_val = 0
        # average the hydrogen values
        if len(hyd_pos_list)==3 or (len(hyd_pos_list)==2 and cab_pos not in all_split_positions):
            for hyd_pos in hyd_pos_list:
                hyd_val += boltzman_avg_shifts_corr_2[hyd_pos]
            hyd_val_avg =  hyd_val/len(hyd_pos_list)
        else:
            hyd_val_avg = boltzman_avg_shifts_corr_2[hyd_pos_list[0]]
        # For HSQC-DEPT add the directionsSS
        if len(hyd_pos_list)%2 != 0:
            direction = 1
        else:
            direction = -1

        if len(hyd_pos_list)==2 and cab_pos in all_split_positions:
            for hyd_pos in hyd_pos_list:
                hyd_val_avg = boltzman_avg_shifts_corr_2[hyd_pos]
                shifts.append([boltzman_avg_shifts_corr_2[cab_pos],hyd_val_avg, direction, cab_pos, hyd_pos_list])
        else:
            shifts.append([boltzman_avg_shifts_corr_2[cab_pos],hyd_val_avg, direction, cab_pos, hyd_pos_list])
    return shifts


def perform_deduplication_if_symmetric(shifts, sym_dupl_lists):
    # Purpose: average the shifts of symmetry-equivalent carbons and merge
    #   duplicate rows, so that equivalent atoms yield a single peak.
    # Args: shifts (list[list]) - rows from selecting_shifts;
    #   sym_dupl_lists (list[list[int]]) - groups of equivalent atom indices.
    # Returns: list[list] - deduplicated rows in the same format as the input.
    # Side effects: none; new lists are built.
    # Note: when no symmetry is found the original list object is returned (not a
    #   copy), so the caller and this function would share the same list.
    """ This function performs deduplication based on the symmetries found in the compound
    and returns the new corrected shifts"""
    #shifts_arr = np.array(shifts)
    shifts_arr = np.array(shifts, dtype=object)
    C_nums = shifts_arr[:,3]
    if len(sym_dupl_lists) != 0:
        new_shifts = []
        c_done = []
        # iterate over the lists of dublicates
        for carbons in sym_dupl_lists:
            carb_sum = 0
            hyd_sum = 0
            carb_pos = []
            hyd_pos = []
            counter = 0
            # iterate over the single carbon positions of the duplicates
            for carb_num in carbons:
                if carb_num in C_nums:
                    # iterate over all shifts in shifts
                    for C,H,direction,c_pos,h_pos in shifts:
                        # check if the carb number of the shift corresponds to the carb dupl list
                        if carb_num == c_pos:
                            carb_sum+=C
                            carb_pos.append(c_pos)
                            hyd_sum+=H
                            hyd_pos.append(h_pos)
                            counter+=1
                            c_done.append(c_pos)
                else:
                    pass
            # that is necessary for compounds with symmetric N 
            # then counter does not go up and leads to zero division error
            if counter!=0:
                carb_avg = carb_sum/counter
                hyd_avg = hyd_sum/counter
                new_shifts.append([carb_avg,hyd_avg, direction, carb_pos, hyd_pos])

        ### perform averaging of the shifts that come occure more often
        grouped_data = {}
        for row in shifts:
            key = row[-2]
            if key not in grouped_data:
                grouped_data[key] = {'count': 0, 'sum': 0, 'row': row}
            grouped_data[key]['count'] += 1
            grouped_data[key]['sum'] += row[1]

        averaged_shifts = []
        for key in grouped_data:
            avg = grouped_data[key]['sum'] / grouped_data[key]['count']
            new_row = grouped_data[key]['row'][:1] + [avg] + grouped_data[key]['row'][2:]
            averaged_shifts.append(new_row)

        # add all other carb peaks that have not been added
        for C,H,direction,c_pos,h_pos in averaged_shifts:
            # check if the carb number of the hift corresponds to the carb dupl list
            if c_pos not in c_done and c_pos in C_nums:
                new_shifts.append([C,H,direction,[c_pos],h_pos])
                c_done.append(c_pos)
    else:
        new_shifts = shifts

    return new_shifts



def generate_dft_dataframe(shifts):
    """Wrap predicted peaks into the canonical dataframe layout.

    Args:
        shifts (list[list]): Rows ``[C_shift, H_shift, direction, atom_index,
            hydrogen_index_list]`` as produced by ``selecting_shifts`` /
            ``perform_deduplication_if_symmetric``.

    Returns:
        pandas.DataFrame: Columns ``F2 (ppm)`` (1H), ``F1 (ppm)`` (13C),
        ``direction`` and ``atom_index`` (underscore-joined indices).
    """
    # save shifts as dataframe
    df_dft = pd.DataFrame()
    c_shifts = np.array(shifts, dtype=object)[:,0]
    h_shifts = np.array(shifts, dtype=object)[:,1]
    directions = np.array(shifts, dtype=object)[:,2]
    atom_index = list(np.array(shifts, dtype=object)[:,3])
    #atom_index = ['_'.join(map(str, sublist)) for sublist in atom_index]
    atom_index = ['_'.join(map(str, sublist if isinstance(sublist, list) else [sublist])) for sublist in atom_index]

    #c_shifts = np.array(shifts)[:,0]
    #h_shifts = np.array(shifts)[:,1]
    #directions = np.array(shifts)[:,2]
    df_dft['F2 (ppm)'] = h_shifts
    df_dft['F1 (ppm)'] = c_shifts
    df_dft["direction"] = directions
    df_dft["atom_index"] = atom_index
    return df_dft
# df_dft = generate_dft_dataframe(shifts)


# sample_id = "HMDB0000159"
# acd_files = glob.glob("/projects/cc/knlr326/5_Downloads/hmdb_data/hmdb/*")


def check_num_peaks_in_acd(df_acd):
    """Count the distinct (1H, 13C) peak pairs of an ACD spectrum.

    Args:
        df_acd (pandas.DataFrame): ACD dataframe with ``F2 (ppm)`` and
            ``F1 (ppm)`` columns.

    Returns:
        int: Number of unique peak pairs.

    NOTE (known issue, left unchanged):
        Coordinates are concatenated as plain strings without a separator
        (``str(F2) + str(F1)``), so different pairs can collide, e.g.
        ("1.0", "23.0") and ("1.02", "3.0") both become "1.023.0" and are
        counted as one peak. The count can therefore be too small.
    """
    list_strings = []
    for i,j in zip(df_acd["F2 (ppm)"], df_acd["F1 (ppm)"]):
        string = str(i)+str(j)
        list_strings.append(string)
        num_peaks = len(set(list_strings))
    return num_peaks

def load_acd_dataframe(sample_id, acd_files):
    """Load the ACD-predicted HSQC spectrum for a sample.

    Args:
        sample_id (str): Substring identifying the sample in the file list.
        acd_files (list[str]): Candidate ACD SDF file paths.

    Returns:
        tuple[pandas.DataFrame, list[str]]: The deduplicated spectrum
        (``F2 (ppm)``, ``F1 (ppm)``, ``direction``) and the matched path list.

    Side effects:
        None beyond reading the SDF file.

    NOTE (known issue, left unchanged):
        ``[i for i in acd_files if sample_id in i][0]`` raises ``IndexError`` when
        no file matches the sample id.
    """
    acd_file_path = [i for i in acd_files if sample_id in i ][0]
    data = PandasTools.LoadSDF(acd_file_path)#, smilesName='SMILES',molColName='Mol',includeFingerprints=False)
    data["HSQC_13C-1H"]
    string_data = str(data['HSQC_13C-1H'][0])
    procd = [d.split(';') for d in string_data.split('\n')]
    procd_2 = [i[0].split("\t") for i in procd]
    df_acd = pd.DataFrame(procd_2[1:],columns=procd_2[0])
    direction = [-1 if "<" in i else 1 for i in df_acd["F2 Atom"]]
    df_acd["direction"] = direction
    
    #deduplicate double entrences
    df_acd_dedup = df_acd.drop_duplicates(
      subset = ['F2 (ppm)', "F1 (ppm)"],
      keep = 'last').reset_index(drop = True)
    return df_acd_dedup, acd_file_path
# df_acd, acd_file_path = load_acd_dataframe(sample_id, acd_files)

def load_acd_dataframe_from_file(acd_file_path):
    """Parse an ACD SDF file into a deduplicated HSQC dataframe.

    The SDF property ``HSQC_13C-1H`` holds a semicolon/tab separated table; the
    first row is used as the column header, and rows marked with ``<`` are
    given a negative direction (DEPT phase).

    Args:
        acd_file_path (str): Path to the ACD SDF file.

    Returns:
        pandas.DataFrame: ``F2 (ppm)``, ``F1 (ppm)``, ``direction``, deduplicated
        on the (1H, 13C) pair.
    """
    data = PandasTools.LoadSDF(acd_file_path)#, smilesName='SMILES',molColName='Mol',includeFingerprints=False)
    data["HSQC_13C-1H"]
    string_data = str(data['HSQC_13C-1H'][0])
    procd = [d.split(';') for d in string_data.split('\n')]
    procd_2 = [i[0].split("\t") for i in procd]
    df_acd = pd.DataFrame(procd_2[1:],columns=procd_2[0])
    direction = [-1 if "<" in i else 1 for i in df_acd["F2 Atom"]]
    df_acd["direction"] = direction
    
    #deduplicate double entrences
    df_acd_dedup = df_acd.drop_duplicates(
      subset = ['F2 (ppm)', "F1 (ppm)"],
      keep = 'last').reset_index(drop = True)
    return df_acd_dedup

def load_real_dataframe_from_file(real_file_path):
    """Load an experimental HSQC table from a whitespace/tab separated text file.

    Args:
        real_file_path (str): Text file with ``F2ppm`` and ``F1ppm`` columns.

    Returns:
        pandas.DataFrame: Same data with columns renamed to ``F2 (ppm)`` and
        ``F1 (ppm)`` so it matches the canonical layout.
    """

    df_real = pd.read_csv(real_file_path, sep="\t|\s+")
    df_real = df_real.rename(columns={"F2ppm": "F2 (ppm)", "F1ppm": "F1 (ppm)"})
    return df_real

def load_real_dataframe(sample_id, real_files):
    """Load the experimental HSQC table of a sample from a candidate file list.

    Args:
        sample_id (str): Substring identifying the sample.
        real_files (list[str]): Candidate file paths.

    Returns:
        pandas.DataFrame: Columns renamed to ``F2 (ppm)`` / ``F1 (ppm)``.

    NOTE (known issue, left unchanged):
        Only files whose name also contains "two" are considered, and the first
        match is used; a sample without such a file raises ``IndexError``.
    """
    filter_id = [i for i in real_files if sample_id in i]
    real_file_path = [i for i in filter_id if "two" in i][0]
    df_real = pd.read_csv(real_file_path, sep="\t|\s+")
    df_real = df_real.rename(columns={"F2ppm": "F2 (ppm)", "F1ppm": "F1 (ppm)"})
    return df_real

def load_mnova_dataframe(sample_id, mnova_files):
    """Load an MNova peak list for a sample from a candidate file list.

    Args:
        sample_id (str): Substring identifying the sample.
        mnova_files (list[str]): Candidate MNova text file paths.

    Returns:
        tuple[pandas.DataFrame, str]: Peak table with columns
        ``id, F2 (ppm), F1 (ppm), Intensity`` and the matched path.

    NOTE (known issue, left unchanged):
        Raises ``IndexError`` when no candidate matches the sample id.
    """
    
    file_path = [i for i in mnova_files if sample_id in i][0]
    df_mnova = pd.read_csv(file_path, sep="\t|\s+", names=["id", "F2 (ppm)",  "F1 (ppm)", 'Intensity'])

    return df_mnova, file_path

def load_mnova_dataframe_from_file(file_path):
    """Load an MNova peak list directly from a given path.

    Args:
        file_path (str): MNova text file (tab/space separated, no header).

    Returns:
        pandas.DataFrame: Columns ``id, F2 (ppm), F1 (ppm), Intensity``.
    """
    
    df_mnova = pd.read_csv(file_path, sep="\t|\s+", names=["id", "F2 (ppm)",  "F1 (ppm)", 'Intensity'])
    return df_mnova


def load_sarotti_data(file_path, mode=["exp","calc1","calc2"], nmr_type=["HSQC", "1H", "13C"]):
    # Purpose: read the Sarotti benchmark CSV and select a spectrum variant.
    # Args: file_path (str) - CSV with exp_*/calc*_* ppm columns;
    #   mode (str) - "exp", "calc1" or "calc2" (note: the default is a list, but
    #   the body compares against strings, so the default call returns None);
    #   nmr_type (str) - currently only "HSQC" is implemented.
    # Returns: pandas.DataFrame with "F2 (ppm)" / "F1 (ppm)" for the requested
    #   variant, or None for unsupported combinations.
    # Side effects: none.
    # NOTE (known issue, left unchanged): unsupported mode/nmr_type values fall
    #   through without a return, so the caller receives None; and the
    #   "1H"/"13C" branches advertised in the docstring are not implemented.
    """This function loads the HSQC data from the sarotti files and it can choose between experimental results and the calculated ones.
        To get the 1H and 13C still needs to be implemented"""   
    
    data = pd.read_csv(file_path)
    data_load = data[["exp_F2ppm","exp_F1ppm","calc1_F2ppm","calc1_F1ppm","calc2_F2ppm","calc2_F1ppm"]]
    if mode == "exp":
        if nmr_type == "HSQC":
            df = data_load[["exp_F2ppm","exp_F1ppm"]]
            df = df.rename(columns={"exp_F2ppm": "F2 (ppm)", "exp_F1ppm": "F1 (ppm)"})
            df = df.dropna()
            return df
            
    ################# From the website:
    ########## http://cheshirenmr.info/ScalingFactors.htm#table1aheading
    # slope_H = -1.0936
    # intercept_H = 31.8018
    # slope_C = -1.0533
    # intercept_C = 186.5242
    
    if mode == "calc1":
        if nmr_type == "HSQC":
            #### From DP4 File /projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/5_git_repos/DP5/TMSdata 
            ## Gas phase
            slope_H = -1
            intercept_H =  31.6828083333
            slope_C = -1
            intercept_C = 195.8016

            df = data_load[["calc1_F2ppm","calc1_F1ppm"]]
            df = df.dropna()
            df["calc1_F1ppm"] = df["calc1_F1ppm"].apply(lambda shielding: (intercept_C-shielding)/-slope_C)
            df["calc1_F2ppm"] = df["calc1_F2ppm"].apply(lambda shielding: (intercept_H-shielding)/-slope_H)
            df = df.rename(columns={"calc1_F2ppm": "F2 (ppm)", "calc1_F1ppm": "F1 (ppm)"})
            return df
    if mode == "calc2":
        if nmr_type == "HSQC":
                ################ STILL NEED TO CORRECT THE SHIELDING ##################
            #### From DP4 File /projects/cc/knlr326/1_NMR_project/2_Notebooks/nmr_project/5_git_repos/DP5/TMSdata 
            ## Chloroform phase
            slope_H = -1
            intercept_H =  31.668625
            slope_C = -1
            intercept_C = 196.1301
            df = data_load[["calc2_F2ppm","calc2_F1ppm"]]
            df = df.dropna()
            df["calc2_F1ppm"] = df["calc2_F1ppm"].apply(lambda shielding: (intercept_C-shielding)/-slope_C)
            df["calc2_F2ppm"] = df["calc2_F2ppm"].apply(lambda shielding: (intercept_H-shielding)/-slope_H)
            df = df.rename(columns={"calc2_F2ppm": "F2 (ppm)", "calc2_F1ppm": "F1 (ppm)"})
            return df


def plot_spectra_together(df_list, labels, sample_id):
    # NOTE (known issue, left unchanged): acd_num_peaks is only assigned when
    #   "acd" appears in labels, but the else-branch below references it for every
    #   non-acd dataframe. Calling this without an "acd" spectrum therefore raises
    #   NameError. The first "if len(df_list)==2" is also not an "elif", so the
    #   unpacking chain falls through for two dataframes (harmless, df_1/df_2 are
    #   still bound).
    """
    Plot multiple 2D NMR spectra together, correcting for points that fall together in ACD Labs spectra.

    Parameters:
    -----------
    df_list: list of pandas.DataFrame
        List of dataframes with the spectra to be plotted.
    labels: list of str
        List of labels to be used for each spectrum.
    sample_id: str
        Identifier for the sample.

    Returns:
    --------
    point_numbers: dict
        Dictionary with the number of data points for each spectrum (excluding ACD Labs spectrum).
    """
    if len(df_list)==2:
        [df_1,df_2] = df_list
    if len(df_list)==3:
        [df_1,df_2,df_3] = df_list
    elif len(df_list)==4:
        [df_1,df_2,df_3,df_4] = df_list
    elif len(df_list)==5:
        [df_1,df_2,df_3,df_4,df_5] = df_list
    elif len(df_list)==6:
        [df_1,df_2,df_3,df_4, df_5, df_6] = df_list
    
    if "acd" in labels:
        acd_idx = labels.index("acd")
        df_acd = df_list[acd_idx]
        acd_num_peaks = check_num_peaks_in_acd(df_acd)

    colors = ["green", "red", "blue", "orange", '#17becf', "#9467bd", '#17becf']
    colors = colors[:len(df_list)]
    fig, ax = plt.subplots(figsize=(10,5), dpi=80)
    
    point_numbers = {}
    for idx, df in enumerate(df_list):
        print(idx)
        scatter_x_2 = np.array(df['F2 (ppm)'].astype(float))
        scatter_y_2 = np.array(df['F1 (ppm)'].astype(float))
        ax.scatter(scatter_x_2, scatter_y_2, label=labels[idx], color=colors[idx],  alpha=0.5)
        if labels[idx] != "acd":
            point_numbers[labels[idx]] = len(scatter_x_2)
        else:
            point_numbers[labels[idx]] = acd_num_peaks


    ax.legend()
    plt.gca().invert_yaxis()
    plt.gca().invert_xaxis()
    plt.grid()
    plt.title(sample_id)
    plt.show()
    return point_numbers

    
# check_num_peaks_in_acd(df_acd["F2 (ppm)"], df_acd["F1 (ppm)"])


def get_similarity_comparison_variations(df_1, df_2, sample_id, similarity_type=["euclidean", "cosine_similarity", "pearson_similarity"], error=["sum","avg"], display_img=False, sim_technique=None):
    # Purpose: score two spectra under nine different assignment strategies.
    # Args: df_1/df_2 - dataframes with "F2 (ppm)"/"F1 (ppm)";
    #   sample_id - label used in plots; similarity_type - list of point
    #   similarity measures passed to similarity_calculations; error - error
    #   aggregation mode; display_img - draw the overlaid spectra; sim_technique -
    #   optional assignment algorithm override.
    # Returns: (result, input_dfs) where result is a 9-element list ordered as
    #   [min_sum_zero, euc_dist_zero, hung_dist_zero, min_sum_trunc,
    #    euc_dist_trunc, hung_dist_trunc, min_sum_nn, euc_dist_nn, hung_dist_nn]
    #   and input_dfs maps each mode name to the [df_1, df_2] pair of aligned peak
    #   sets that produced the score.
    # Side effects: optional plotting; no printing.
#"min_sum","euc_dist_zero","euc_dist_advanced","hungarian_zero","hungarian_advanced"
    """ This function calculates the similarities of two HSQC dataframes and returns all 4 different
    similarity measures in from of a list [minSum, euclidean, hungarian, adv_hungarian"""
    input_dfs = {}

    # Plot 2 spectra on top of each other
    if display_img:
        try:
            plot_compare_scatter_plot(df_1, df_2, name = sample_id, transp=0.50, style="both",  direction=False)
        except:
            plot_compare_scatter_plot_without_direction(df_1, df_2, name = sample_id, transp=0.50)

    # calculate Cosine Similarity based on spectra
    sim_min_sum_zero, input_list_1, input_list_2 = similarity_calculations(df_1, df_2, mode="min_sum_zero", similarity_type=similarity_type, error=error, assignment_plot=display_img, sim_technique=sim_technique)
    df1_msz = pd.DataFrame(input_list_1, columns=['F2 (ppm)', 'F1 (ppm)'])
    df2_msz = pd.DataFrame(input_list_2, columns=['F2 (ppm)', 'F1 (ppm)'])
    input_dfs["min_sum_zero"] = [df1_msz, df2_msz]
    
    sim_euc_dist_zero, input_list_1, input_list_2 = similarity_calculations(df_1, df_2, mode="euc_dist_zero", similarity_type=similarity_type, error=error, assignment_plot=display_img, sim_technique=sim_technique)
    df1_edz = pd.DataFrame(input_list_1, columns=['F2 (ppm)', 'F1 (ppm)'])
    df2_edz = pd.DataFrame(input_list_2, columns=['F2 (ppm)', 'F1 (ppm)'])
    input_dfs["euc_dist_zero"] = [df1_edz, df2_edz]
    
    sim_hung_dist_zero, input_list_1, input_list_2 = similarity_calculations(df_1, df_2, mode="hung_dist_zero", similarity_type=similarity_type, error=error, assignment_plot=display_img, sim_technique=sim_technique)
    df1_hdz = pd.DataFrame(input_list_1, columns=['F2 (ppm)', 'F1 (ppm)'])
    df2_hdz = pd.DataFrame(input_list_2, columns=['F2 (ppm)', 'F1 (ppm)'])
    input_dfs["hung_dist_zero"] = [df1_hdz, df2_hdz]
    
    sim_min_sum_trunc, input_list_1, input_list_2 = similarity_calculations(df_1, df_2, mode="min_sum_trunc", similarity_type=similarity_type, error=error, assignment_plot=display_img, sim_technique=sim_technique)
    df1_mst = pd.DataFrame(input_list_1, columns=['F2 (ppm)', 'F1 (ppm)'])
    df2_mst = pd.DataFrame(input_list_2, columns=['F2 (ppm)', 'F1 (ppm)'])
    input_dfs["min_sum_trunc"] = [df1_mst, df2_mst]
    
    sim_euc_dist_trunc, input_list_1, input_list_2 = similarity_calculations(df_1, df_2, mode="euc_dist_trunc", similarity_type=similarity_type, error=error, assignment_plot=display_img, sim_technique=sim_technique)
    df1_edt = pd.DataFrame(input_list_1, columns=['F2 (ppm)', 'F1 (ppm)'])
    df2_edt = pd.DataFrame(input_list_2, columns=['F2 (ppm)', 'F1 (ppm)'])
    input_dfs["euc_dist_trunc"] = [df1_edt, df2_edt]
    
    sim_hung_dist_trunc, input_list_1, input_list_2 = similarity_calculations(df_1, df_2, mode="hung_dist_trunc", similarity_type=similarity_type, error=error, assignment_plot=display_img, sim_technique=sim_technique)
    df1_hdt = pd.DataFrame(input_list_1, columns=['F2 (ppm)', 'F1 (ppm)'])
    df2_hdt = pd.DataFrame(input_list_2, columns=['F2 (ppm)', 'F1 (ppm)'])    
    input_dfs["hung_dist_trunc"] = [df1_hdt, df2_hdt]
    
    sim_min_sum_nn, input_list_1, input_list_2 = similarity_calculations(df_1, df_2, mode="min_sum_nn", similarity_type=similarity_type, error=error, assignment_plot=display_img, sim_technique=sim_technique)
    df1_msn = pd.DataFrame(input_list_1, columns=['F2 (ppm)', 'F1 (ppm)'])
    df2_msn = pd.DataFrame(input_list_2, columns=['F2 (ppm)', 'F1 (ppm)'])
    input_dfs["min_sum_nn"] = [df1_msn, df2_msn]
    
    sim_euc_dist_nn, input_list_1, input_list_2 = similarity_calculations(df_1, df_2, mode="euc_dist_nn", similarity_type=similarity_type, error=error, assignment_plot=display_img, sim_technique=sim_technique)
    df1_edn = pd.DataFrame(input_list_1, columns=['F2 (ppm)', 'F1 (ppm)'])
    df2_edn = pd.DataFrame(input_list_2, columns=['F2 (ppm)', 'F1 (ppm)'])
    input_dfs["euc_dist_nn"] = [df1_edn, df2_edn]
    
    sim_hung_dist_nn, input_list_1, input_list_2 = similarity_calculations(df_1, df_2, mode="hung_dist_nn", similarity_type=similarity_type, error=error, assignment_plot=display_img, sim_technique=sim_technique)
    df1_hdn = pd.DataFrame(input_list_1, columns=['F2 (ppm)', 'F1 (ppm)'])
    df2_hdn = pd.DataFrame(input_list_2, columns=['F2 (ppm)', 'F1 (ppm)'])
    input_dfs["hung_dist_nn"] = [df1_hdn, df2_hdn]

    result = [sim_min_sum_zero, sim_euc_dist_zero, sim_hung_dist_zero, sim_min_sum_trunc, sim_euc_dist_trunc, sim_hung_dist_trunc, sim_min_sum_nn, sim_euc_dist_nn, sim_hung_dist_nn]
    return result, input_dfs


def load_shifts_from_sdf(sample_id, files):
    """Load the ``averaged_NMR_shifts`` property from a matching shift-SDF file.

    Args:
        sample_id (str): Substring identifying the sample.
        files (list[str]): Candidate SDF paths.

    Returns:
        tuple[list[float], str]: Per-atom shifts (ppm) and the matched file path.
    """
    file_path = [i for i in files if sample_id in i][0]
    atom_list, connectivity_list, docline_list, name, mol = get_molecule_data(file_path)
    for idx, line in enumerate(docline_list):
        if ">  <averaged_NMR_shifts>" in line:
            boltzman_avg_shifts_corr_2  = docline_list[idx+1]
            boltzman_avg_shifts_corr_2  = [float(i) for i in boltzman_avg_shifts_corr_2.split()]
            break
    return boltzman_avg_shifts_corr_2, file_path

def load_shifts_from_sdf_file(file_path):
    """Load the ``averaged_NMR_shifts`` property from a given SDF file.

    Args:
        file_path (str): Path to the SDF file.

    Returns:
        list[float]: Per-atom shifts (ppm). Both comma- and space-separated
        encodings of the property are accepted.

    NOTE (known issue, left unchanged):
        ``data["averaged_NMR_shifts"].item()`` raises ValueError when the SDF
        contains more than one record (the item() call requires exactly one).
    """
    # file_path = [i for i in files if sample_id in i][0]
    data = PandasTools.LoadSDF(file_path)
    str_shifts = data["averaged_NMR_shifts"].item()
    try:
        boltzman_avg_shifts_corr_2  = [float(i) for i in str_shifts.split(",")]
    except:
        boltzman_avg_shifts_corr_2  = [float(i) for i in str_shifts.split()]
        
    """
    atom_list, connectivity_list, docline_list, name, mol = get_molecule_data(file_path)
        for idx, line in enumerate(docline_list):
            if ">  <averaged_NMR_shifts>" in line:
                boltzman_avg_shifts_corr_2  = docline_list[idx+1]
                boltzman_avg_shifts_corr_2  = [float(i) for i in boltzman_avg_shifts_corr_2.split()]
                break"""
    return boltzman_avg_shifts_corr_2



def load_ml_dataframe(sample_id, ml_files):
    # Side effects: none besides reading the SDF; the ML molecule is re-parsed by
    #   run_chiral_and_symmetry_finder and get_molecule_data.
    # NOTE (known issue, left unchanged): the SDF is loaded with
    #   data["averaged_NMR_shifts"].item(), which raises ValueError if the matched
    #   file contains more than one record.
    """
    Load an ML SDF file into a Pandas DataFrame.

    Parameters:
    -----------
    sample_id: str
        The ID of the sample to load.
    ml_files: List[str]
        A list of file paths to the ML SDF files.

    Returns:
    --------
    df_ml: pd.DataFrame
        A Pandas DataFrame containing the ML data.
    """
    
    ml_file_path = [i for i in ml_files if sample_id in i][0]

    data = PandasTools.LoadSDF(ml_file_path)
    str_shifts = data["averaged_NMR_shifts"].item()
    boltzman_avg_shifts_corr_2  = [float(i) for i in str_shifts.split()]

    sym_dupl_lists, all_split_positions, mol, compound_path = run_chiral_and_symmetry_finder(compound_path=ml_file_path)
    atom_list, connectivity_list, docline_list, name, mol = get_molecule_data(ml_file_path)
    c_h_connectivity_dict = get_c_h_connectivity(connectivity_list, atom_list) 
    shifts = selecting_shifts(c_h_connectivity_dict, all_split_positions, boltzman_avg_shifts_corr_2)
    shifts = perform_deduplication_if_symmetric(shifts, sym_dupl_lists)
    df_ml = generate_dft_dataframe(shifts)
    return df_ml


def load_ml_dataframe_from_file(ml_file_path):
    """Load an ML-predicted shift SDF into a peak dataframe.

    Args:
        ml_file_path (str): Path to the ML SDF file. The property
            ``averaged_NMR_shifts`` must contain space-separated per-atom shifts.

    Returns:
        pandas.DataFrame: Peak table with ``F2 (ppm)``, ``F1 (ppm)``,
        ``direction`` and ``atom_index`` columns.
    """
    
    data = PandasTools.LoadSDF(ml_file_path)
    str_shifts = data["averaged_NMR_shifts"].item()
    boltzman_avg_shifts_corr_2  = [float(i) for i in str_shifts.split()]

    sym_dupl_lists, all_split_positions, mol, compound_path = run_chiral_and_symmetry_finder(compound_path=ml_file_path)
    atom_list, connectivity_list, docline_list, name, mol = get_molecule_data(ml_file_path)
    c_h_connectivity_dict = get_c_h_connectivity(connectivity_list, atom_list) 
    shifts = selecting_shifts(c_h_connectivity_dict, all_split_positions, boltzman_avg_shifts_corr_2)
    shifts = perform_deduplication_if_symmetric(shifts, sym_dupl_lists)
    df_ml = generate_dft_dataframe(shifts)
    return df_ml


def load_dft_dft_comparison(dft_file_path):
    """Build a peak dataframe from a DFT-computed shift SDF.

    Args:
        dft_file_path (str): Path to the DFT SDF file.

    Returns:
        pandas.DataFrame: Peak table (``F2``/``F1``/``direction``/``atom_index``)
        produced by the same chirality/symmetry/peak assembly chain used for ML
        data.

    Note:
        The local variable ``dft_num_peaks`` is computed but never returned; the
        number of peaks is simply ``len(df_dft)``.
    """
    
#     files = glob.glob('/projects/cc/knlr326/1_NMR_project/6_Paper_1_analysis/1_HMDB_conformer_analysis/1_HMDB_ICOLOS_DFT/*/*')

#     files = [i for i in files if "shift" in i]
#     files = [i for i in files if not "NMR2_" in i]
#     files = [i for i in files if sample_id in i]
#     files = [i for i in files if not "mol" in i]
#     dft_file_path = [i for i in files if f"CONF_{conf_num}_" in i][0]
    #import IPython; IPython.embed();

    boltzman_avg_shifts_corr_2 = load_shifts_from_sdf_file(dft_file_path)
    sym_dupl_lists, all_split_positions, mol, compound_path = run_chiral_and_symmetry_finder(compound_path=dft_file_path)
    atom_list, connectivity_list, docline_list, name, mol = get_molecule_data(dft_file_path)
    c_h_connectivity_dict = get_c_h_connectivity(connectivity_list, atom_list) 
    shifts = selecting_shifts(c_h_connectivity_dict, all_split_positions, boltzman_avg_shifts_corr_2)
    shifts = perform_deduplication_if_symmetric(shifts, sym_dupl_lists)
    df_dft = generate_dft_dataframe(shifts)
    dft_num_peaks = len(df_dft)
    return df_dft

def load_sdf_comparison(sdf_file_path):
    """Build a peak dataframe from any shift SDF file.

    Args:
        sdf_file_path (str): Path to the SDF file.

    Returns:
        pandas.DataFrame: Peak table in the canonical layout, assembled via
        ``run_chiral_and_symmetry_finder``, ``selecting_shifts`` and
        ``perform_deduplication_if_symmetric``.
    """
    
#     files = glob.glob('/projects/cc/knlr326/1_NMR_project/6_Paper_1_analysis/1_HMDB_conformer_analysis/1_HMDB_ICOLOS_DFT/*/*')

#     files = [i for i in files if "shift" in i]
#     files = [i for i in files if not "NMR2_" in i]
#     files = [i for i in files if sample_id in i]
#     files = [i for i in files if not "mol" in i]
#     dft_file_path = [i for i in files if f"CONF_{conf_num}_" in i][0]

    boltzman_avg_shifts_corr_2 = load_shifts_from_sdf_file(sdf_file_path)
    sym_dupl_lists, all_split_positions, mol, compound_path = run_chiral_and_symmetry_finder(compound_path=sdf_file_path)
    atom_list, connectivity_list, docline_list, name, mol = get_molecule_data(sdf_file_path)
    c_h_connectivity_dict = get_c_h_connectivity(connectivity_list, atom_list) 
    shifts = selecting_shifts(c_h_connectivity_dict, all_split_positions, boltzman_avg_shifts_corr_2)
    shifts = perform_deduplication_if_symmetric(shifts, sym_dupl_lists)
    df_sdf = generate_dft_dataframe(shifts)
    return df_sdf

# ##### NOT sure where i used that one
def load_real_dataframe_from_path(real_file_path):
    """Load an experimental HMDB text file without renaming any columns.

    Args:
        real_file_path (str): Path to the tab/space separated file.

    Returns:
        pandas.DataFrame: Raw table as read (typically ``F2ppm``/``F1ppm``).
    """
    df_real = pd.read_csv(real_file_path, sep="\t|\s+")
    # df_real = pd.read_csv(real_file_path, sep="\t|\s+", names=["id", "F2 (ppm)",  "F1 (ppm)", 'Intensity'])

    return df_real


def load_real_df_from_file(path_txt):
    """Load an experimental peak list and derive the canonical ppm columns.

    Args:
        path_txt (str): Text file with ``F2ppm``/``F1ppm`` columns.

    Returns:
        tuple[pandas.DataFrame, str]: The dataframe (with added ``F2 (ppm)`` /
        ``F1 (ppm)`` columns) and the first 11 characters of the file name.

    Note:
        If parsing with the mixed tab/space separator fails, the function retries
        with a tab-only separator.
    """
    try:
        df_real = pd.read_csv(path_txt, sep="\t|\s+")  
    except:
        df_real = pd.read_csv(path_txt, sep="\t")  

    df_real['F2 (ppm)'] = list(df_real['F2ppm'])
    df_real['F1 (ppm)'] = list(df_real['F1ppm'])
    name = path_txt.split("/")[-1][:11]

    return df_real, name


def load_shifts_from_dp_sdf_file(sdf_file_dp):
    """Load the shift list stored in a DP4/DP5-style SDF file.

    Args:
        sdf_file_dp (str): SDF file whose ``averaged_NMR_shifts`` property is a
            Python literal (e.g. ``"[1.2, 3.4, ...]"``).

    Returns:
        list[float]: Per-atom shifts parsed with ``ast.literal_eval``.

    NOTE (known issue, left unchanged):
        ``ast.literal_eval`` only accepts a syntactically valid Python literal.
        A property written as space-separated numbers ("1.2 3.4 5.6") raises
        SyntaxError/ValueError, so only DP files written with list syntax load.
    """
    data = PandasTools.LoadSDF(sdf_file_dp)
    chemical_shifts = ast.literal_eval(data["averaged_NMR_shifts"][0])
    return chemical_shifts


def load_dft_dp_comparison(dft_file_path):
    """Build a peak dataframe from a DP5-style DFT shift SDF.

    Args:
        dft_file_path (str): Path to the DP SDF file.

    Returns:
        pandas.DataFrame: Peak table in the canonical layout.
    """
    boltzman_avg_shifts_corr_2 = load_shifts_from_dp_sdf_file(dft_file_path)
    sym_dupl_lists, all_split_positions, mol, compound_path = run_chiral_and_symmetry_finder(compound_path=dft_file_path)
    atom_list, connectivity_list, docline_list, name, mol = get_molecule_data(dft_file_path)
    c_h_connectivity_dict = get_c_h_connectivity(connectivity_list, atom_list) 
    shifts = selecting_shifts(c_h_connectivity_dict, all_split_positions, boltzman_avg_shifts_corr_2)
    shifts = perform_deduplication_if_symmetric(shifts, sym_dupl_lists)
    df_dft = generate_dft_dataframe(shifts)
    dft_num_peaks = len(df_dft)
    return df_dft

def load_dft_dp_dataframe(dft_file_path):
    # This function duplicates load_dft_dp_comparison (same body, same result);
    # kept as a separate name for callers that use the "dataframe" naming.
    # Side effects: none besides reading the SDF.
    """Load a DFT (density functional theory) chemical shift file in SD format to a pandas DataFrame.

    Args:
        dft_file_path (str): The path to the DFT file in SD format.

    Returns:
        pandas.DataFrame: A DataFrame containing the chemical shifts and corresponding atomic positions for the DFT file.
    """
    boltzman_avg_shifts_corr_2 = load_shifts_from_dp_sdf_file(dft_file_path)
    sym_dupl_lists, all_split_positions, mol, compound_path = run_chiral_and_symmetry_finder(compound_path=dft_file_path)
    atom_list, connectivity_list, docline_list, name, mol = get_molecule_data(dft_file_path)
    c_h_connectivity_dict = get_c_h_connectivity(connectivity_list, atom_list) 
    shifts = selecting_shifts(c_h_connectivity_dict, all_split_positions, boltzman_avg_shifts_corr_2)
    shifts = perform_deduplication_if_symmetric(shifts, sym_dupl_lists)
    df_dft = generate_dft_dataframe(shifts)
    dft_num_peaks = len(df_dft)
    return df_dft




def load_all_df_from_id(sample_id, acd_files, real_files, mnova_files, ml_files, dft_files):
    """Load every available spectrum source for one sample id.

    Args:
        sample_id (str): Identifier used as a substring filter in all lists.
        acd_files, real_files, mnova_files, ml_files, dft_files (list[str]):
            Candidate paths for the five data sources.

    Returns:
        tuple[pandas.DataFrame, ...]: ``(df_real, df_acd, df_mnova, df_dft,
        df_ml)`` in that order (note: not the order of the arguments).

    Side effects:
        Reads files only. Each source is resolved with the first substring match;
        missing sources raise ``IndexError``.
    """
    
    # Get acd df
    acd_file_path = [i for i in acd_files if sample_id in i][0]
    df_acd = load_acd_dataframe_from_file(acd_file_path)
    acd_num_peaks = check_num_peaks_in_acd(df_acd)
    
    # Get real df
    filter_id = [i for i in real_files if sample_id in i]
    real_file_path = [i for i in filter_id if "two" in i][0]
    df_real = load_real_dataframe_from_file(real_file_path)
    real_num_peaks = len(df_real)
    
    # Get mnova df
    mnova_files = [i for i in mnova_files if not "mol" in i]
    file_path_mnova = [i for i in mnova_files if sample_id in i][0]
    df_mnova = load_mnova_dataframe_from_file(file_path_mnova)
    mnova_num_peaks = len(df_mnova)
    
    # Get ML df  
    ml_file_path = [i for i in ml_files if sample_id in i][0]
    df_ml = load_ml_dataframe_from_file(ml_file_path)
    ml_num_peaks = len(df_ml)

    # Get dft df           
    ## Gaussian DP DFT results
    dft_file_path = [i for i in dft_files if sample_id in i][0]    
    df_dft = load_dft_dp_comparison(dft_file_path)
    dft_num_peaks = len(df_dft)

    
    return df_real, df_acd, df_mnova, df_dft, df_ml


def correct_dft_calc_sdf_file(file_path):
    # Extra notes for the long docstring below:
    # - The SDF property name is rewritten because the original spelling
    #   ("sheilding") and the spaces in the tag are not accepted by RDKit writers.
    # - NOTE (known issue, left unchanged): sdf_filename/movl_filename are only
    #   assigned inside the "filename.startswith('NMR_SEL_')" loop; if the folder
    #   contains no such file they stay unbound and the later open() raises
    #   NameError. The code also calls rdkit.Chem.PandasTools while only
    #   "from rdkit.Chem import PandasTools" is imported, which can raise
    #   AttributeError/NameError depending on the import side effects.
    """Corrects an SDF file with DFT-calculated NMR data.
    
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
    """
    
    ### First make sure that the mol file with H cooridinated is there
    mol = Chem.SDMolSupplier(file_path)[0]

    # Add H atoms to the molecule
    mol_with_h = Chem.AddHs(mol,addCoords=True)

    # Save the molecule to a MOL file with the same name
    mol_filename = file_path.replace('.sdf', '.mol')
    with open(mol_filename, 'w') as f:  
        f.write(Chem.MolToMolBlock(mol_with_h))
    
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
    rdkit.Chem.PandasTools.WriteSDF(data, file_path, molColName='ROMol', idName=None, properties=list(data.columns), allNumeric=False)

    ########################################################
    ### add the H to the SDF file again from the mol file###
    folder = os.path.dirname(file_path)
    # Load the .sdf file
    for filename in os.listdir(folder):
        if filename.startswith('NMR_SEL_') and filename.endswith('.sdf'):
            # This is an .sdf file starting with "NMR_SEL_"
            sdf_filename = os.path.join(folder, filename)
            mol_filename = os.path.join(folder, filename.replace('.sdf', '.mol'))
    # Load the .sdf file
    with open(sdf_filename, 'rb') as f:
        lines = f.readlines()
        lines = [line.decode('utf-8') for line in lines]
    molblock = ''.join(lines)

    # Load the .mol file and append lines from the .sdf file after "M  END"
    with open(mol_filename, 'r') as f:
        molblock_with_h = f.read()
        
    ## add extra lines to 3D block from Mol files
    for line in lines[lines.index('M  END\n')+1:]:
        molblock_with_h += line
    # save as corrected sdf file under the same name
    with open(sdf_filename, 'w') as f:
        f.write(molblock_with_h)