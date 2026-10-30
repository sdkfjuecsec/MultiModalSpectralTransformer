"""Inference helpers for the NMR message-passing graph neural network (SGNN).

The module converts RDKit molecules into the sparsified DGL graphs expected by
the pretrained ``nmr_mpnn_PROPOSED`` models, runs 13C and 1H shift prediction,
and writes the predicted shifts back into SDF files.

High-level pipeline (``main_execute``):
    1. ``save_as_npy_for_13C`` / ``save_as_npy_for_1H`` featurise each molecule
       and dump an intermediate ``.npz`` file in the system temp directory.
    2. ``load_model`` loads the matching pretrained checkpoint per nucleus.
    3. ``inference_C`` / ``inference_H`` run the forward pass.
    4. ``create_shift_list`` re-assembles per-atom shifts and
       ``save_results_sdf_file`` writes them into a ``NMR_<id>.sdf`` record.

Import-time side effects: RDKit logging is disabled, the chemical feature
factory is built, DGL/RDKit keys are registered and a fixed random seed
constant is defined. Global hyper-parameters at module level are defaults only;
the ``save_as_npy_*`` functions re-declare their own local copies.
"""

# Standard library imports
import argparse
import ast
import math
import os
import random
import sys
import tempfile
import time

# Third-party imports
import cairosvg
import numpy as np
from collections import defaultdict
from dgl.convert import graph
from IPython.display import SVG, display
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import AllChem, Draw, rdDepictor
from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D
from tqdm import tqdm
import torch


from nmr_sgnn_norm.gnn_models.mpnn_proposed import nmr_mpnn_PROPOSED
from nmr_sgnn_norm.gnn_models.mpnn_baseline import nmr_mpnn_BASELINE

import numpy as np
import os
from rdkit import Chem, RDConfig, rdBase
from rdkit.Chem import AllChem, ChemicalFeatures
import argparse
import ast
# Update import path to use local repository
import sys
from pathlib import Path

# Add the project root to the path to ensure imports work correctly
project_root = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from nmr_sgnn_norm.dataset import GraphDataset

from dgllife.utils import RandomSplitter
from dgl.data.utils import split_dataset
from nmr_sgnn_norm.util import collate_reaction_graphs
from nmr_sgnn_norm.model import training, inference
from torch.utils.data import DataLoader, Dataset
from rdkit import Chem
import pandas as pd


# RDKit specific settings
IPythonConsole.drawOptions.addAtomIndices = True
rdBase.rdkitVersion


########################################## Some Parameter ##########################################


atom_list = ['H','Li','B','C','N','O','F','Na','Mg','Al','Si','P','S','Cl','K','Ti','Zn','Ge','As','Se','Br','Pd','Ag','Sn','Sb','Te','I','Hg','Tl','Pb','Bi','Ga']
charge_list = [1, 2, 3, -1, -2, -3, 0]
degree_list = [1, 2, 3, 4, 5, 6, 0]
valence_list = [1, 2, 3, 4, 5, 6, 0]
hybridization_list = ['SP','SP2','SP3','SP3D','SP3D2','S','UNSPECIFIED']
hydrogen_list = [1, 2, 3, 4, 0]
ringsize_list = [3, 4, 5, 6, 7, 8]

bond_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
max_graph_distance = 20

rdBase.DisableLog('rdApp.error')
rdBase.DisableLog('rdApp.warning')
chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
mol_dict = {'n_node': [],
                'n_edge': [],
                'node_attr': [],
                'edge_attr': [],
                'src': [],
                'dst': [],
                'shift': [],
                'mask': [],
                'smi': [],
                'h_c_connectivity': []}


message_passing_mode = "proposed"
readout_mode = "proposed"
graph_representation = "sparsified"
memo = ""
fold_seed = 0
data_split = [0.95, 0.05]
batch_size = 128
random_seed = 27407
node_embedding_dim = 256
node_hidden_dim = 512
readout_n_hidden_dim = 512





def mol_with_atom_index(mol, include_H=True):
    """
    Label every atom with its index so the mapping can be drawn.

    Takes a mol as input and adds H to them
    Visualizes the number which is assigned to which atom as a mol
    if include_H is True then H will be added and also labeled in the mol file

    Args:
        mol: RDKit ``Mol`` instance.
        include_H: add explicit hydrogens (with coordinates) before labelling.

    Returns:
        The same ``Mol`` object with a ``molAtomMapNumber`` property on every
        atom, suitable for ``moltosvg`` / RDKit drawing.

    Side effects:
        Mutates the passed molecule in place when ``include_H`` is False; when
        True a new hydrogen-added copy is returned.
    """
    # mol = MolFromSmiles(smiles)
    if include_H:
        mol = Chem.AddHs(mol,addCoords=True)

    atoms = mol.GetNumAtoms()
    for idx in range( atoms ):
        mol.GetAtomWithIdx( idx ).SetProp( 'molAtomMapNumber', str( mol.GetAtomWithIdx( idx ).GetIdx() ) )
    return mol


def moltosvg(mol, molSize = (300,300), kekulize = True):
    """Render a molecule as an inline SVG string for notebook display.

    Args:
        mol: RDKit ``Mol`` to draw.
        molSize: ``(width, height)`` of the SVG canvas in pixels.
        kekulize: attempt aromatic kekulisation before drawing; if it fails the
            original (possibly aromatic) molecule is used.

    Returns:
        SVG markup as a ``str`` with the ``svg:`` namespace prefixes removed
        so it can be embedded directly in HTML/Jupyter output.

    Side effects:
        None; 2D coordinates are computed on an internal copy only.
    """
    mc = Chem.Mol(mol.ToBinary())
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        rdDepictor.Compute2DCoords(mc)
    drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    return svg.replace('svg:','')


def mol2SDF(mol, folder=None, name=None):
    """Write one molecule to ``<folder>/<name>.sdf``.

    The molecule itself is written unchanged; callers are responsible for
    adding explicit hydrogens beforehand (required by the ICOLOS workflow).
    The default folder is ``../../trash`` relative to the script, created if
    missing.

    Args:
        mol: RDKit ``Mol`` instance.
        folder: destination directory; defaults to ``<script>/../../trash``.
        name: file stem; a random 4-digit number when omitted.

    Returns:
        Full path of the written ``.sdf`` file.

    Side effects:
        Creates the destination directory (``exist_ok=True``) and writes the
        file.
    """
    #print("mol2SDF")
    #import IPython; IPython.embed();
    
    if name==None:
        rand_num = random.randint(0,1000)
        name = "%04d" % (rand_num)
    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)
    
    # Build the default folder path relative to the script's location
    if folder is None:
        folder = os.path.abspath(os.path.join(script_dir, '../../trash'))
    os.makedirs(folder, exist_ok=True)
    
    name = name+".sdf"
    save_path = os.path.join(folder,name)

    writer = Chem.SDWriter(save_path)
    writer.write(mol)
    writer.close()
    return save_path

    # Get the atom numbers that have hydrogens attached to it


def get_molecule_data(compound_path):
    """Parse an SDF file manually and reconstruct the RDKit molecule.

    This function returns the list of atoms of the molecules
    and a list of strings with each line of the compound document as one string
    because reading the SDF files with pandas causes an error

    The atom block and the bond/connectivity block are parsed as plain text
    (rather than with RDKit/pandas), the atom list is written out as a
    companion ``.mol`` file, and the molecule is then loaded from it.

    Args:
        compound_path: path to a ``.sdf`` file. The extension is assumed to be
            exactly three characters because ``[:-4]`` is used to build the
            companion ``.mol`` path.

    Returns:
        Tuple ``(atom_list, connectivity_list, docline_list, name, mol)``:
        element symbols per atom, raw bond rows, all file lines, the file stem,
        and the parsed RDKit molecule.

    Side effects:
        Writes ``<compound_path without extension>.mol`` next to the input.

    NOTE (known issue, left unchanged): ``name`` is derived with
    ``split("/")[-1]``, so on Windows (paths use ``\\``) the "name" becomes the
    whole path. Also ``counter`` is only initialised when a ``V2000`` line is
    found; absence of that marker leaves the atom listing empty instead of
    raising a clear error.
    """

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


# Get C-H connectivity dict with index starting from 0 to correct it to python convention
def get_x_h_connectivity(connectivity_list, atom_list):
    """ Build a mapping of heavy atoms to the hydrogens bonded to them.

    This function checks the connectifity list and creates a dictionary with all the
    X atoms that are connected to hydrogens with their labelled numbers

    Args:
        connectivity_list: rows of the SDF bond block, each a list/tuple of
            atom indices (1-based) plus bond metadata.
        atom_list: element symbol per atom, indexed from 0.

    Returns:
        Dict ``{heavy_atom_index: [hydrogen_indices...]}`` for every X-H bond
        where X is C, N, O, P or S. Indices are converted to 0-based.

    Side effects:
        None.

    Note: the ``type(...) == list`` statements are no-ops; the code relies on
    the KeyError of the first access to initialise the list, which works but
    is fragile.
    """
    x_h_connectivity_dict = {}
    for i in connectivity_list:
        selected_atom_nr = int(i[0])-1
        selected_connection_nr = int(i[1])-1
        atom = atom_list[selected_atom_nr]
        connection = atom_list[selected_connection_nr]
        num_connection = atom_list[int(i[2])]
        # check atom X-H bonds and add them to dictionary
        if (atom =="O" or atom =="S" or atom =="N" or atom =="P" or atom == "C") and connection == "H":
            found_H_nr = [selected_connection_nr]
            found_X_nr = selected_atom_nr
            try:
                # if there is no carbon in the dict yet it will fail and go to except
                type(x_h_connectivity_dict[found_X_nr]) == list
                x_h_connectivity_dict[found_X_nr]+=found_H_nr
            except:
                x_h_connectivity_dict[found_X_nr]=found_H_nr
        # check atom X-H bonds and add them to dictionary
        if atom =="H" and (connection =="O" or connection =="S" or connection =="N" or connection =="P" or connection =="C"):
            found_X_nr = selected_connection_nr
            found_H_nr = [selected_atom_nr]
            try:
                # if there is no carbon in the dict yet it will fail and go to except
                type(x_h_connectivity_dict[found_X_nr]) == list
                x_h_connectivity_dict[found_X_nr]+=found_H_nr
            except:
                x_h_connectivity_dict[found_X_nr]=found_H_nr

    return x_h_connectivity_dict


def _DA(mol):
    """Return the donor and acceptor atom indices of a molecule.

    Uses RDKit's ``BaseFeatures.fdef`` chemical feature factory: atoms of
    family ``Donor`` and ``Acceptor`` are collected.

    Args:
        mol: RDKit ``Mol``.

    Returns:
        Tuple ``(D_list, A_list)`` of integer atom indices.
    """
    D_list, A_list = [], []
    for feat in chem_feature_factory.GetFeaturesForMol(mol):
        if feat.GetFamily() == 'Donor': D_list.append(feat.GetAtomIds()[0])
        if feat.GetFamily() == 'Acceptor': A_list.append(feat.GetAtomIds()[0])

    return D_list, A_list

def _chirality(atom):
    """Encode tetrahedral chirality of one atom as two boolean flags.

    Args:
        atom: RDKit ``Atom``.

    Returns:
        ``[is_Tet_CW, is_Tet_CCW]`` when a ``Chirality`` property exists,
        otherwise ``[0, 0]``.
    """
    if atom.HasProp('Chirality'):
        c_list = [(atom.GetProp('Chirality') == 'Tet_CW'), (atom.GetProp('Chirality') == 'Tet_CCW')]
    else:
        c_list = [0, 0]

    return c_list


def _stereochemistry(bond):
    """Encode double-bond cis/trans stereochemistry as two boolean flags.

    Args:
        bond: RDKit ``Bond``.

    Returns:
        ``[is_Bond_Cis, is_Bond_Trans]`` when a ``Stereochemistry`` property
        exists, otherwise ``[0, 0]``.
    """
    if bond.HasProp('Stereochemistry'):
        s_list = [(bond.GetProp('Stereochemistry') == 'Bond_Cis'), (bond.GetProp('Stereochemistry') == 'Bond_Trans')]
    else:
        s_list = [0, 0]

    return s_list

def add_mol_sparsified_graph(mol_dict, mol, x_h_connectivity_dict=None):
    """Append one molecule's graph features to an accumulating dict.

    Builds the node feature matrix by one-hot encoding element, charge,
    degree, hybridisation, hydrogen count, valence, donor/acceptor status,
    chirality, ring sizes and aromaticity, and (when bonds exist) the doubled
    edge list/edge features for the sparsified graph representation. Shift
    values and masks are read from atom properties.

    Args:
        mol_dict: accumulator dict of lists (see ``save_as_npy_*``).
        mol: RDKit ``Mol`` whose atoms carry ``shift``/``mask`` properties.
        x_h_connectivity_dict: heavy-atom-to-hydrogen mapping for this
            molecule, stored for the 1H loader.

    Returns:
        The same ``mol_dict`` with one entry appended to each list.

    Side effects:
        Mutates ``mol_dict`` in place.

    NOTE (known issue, left unchanged): the module-level ``mol_dict`` declares
    the key ``'h_c_connectivity'`` whereas this function appends to
    ``'h_x_connectivity'``. Using that module-level dict raises KeyError; the
    ``save_as_npy_*`` functions avoid it by declaring the correct key locally.
    """
    n_node = mol.GetNumAtoms()
    n_edge = mol.GetNumBonds() * 2

    D_list, A_list = _DA(mol)

    atom_fea1 = np.eye(len(atom_list), dtype = bool)[[atom_list.index(a.GetSymbol()) for a in mol.GetAtoms()]]
    atom_fea2 = np.eye(len(charge_list), dtype = bool)[[charge_list.index(a.GetFormalCharge()) for a in mol.GetAtoms()]][:,:-1]
    atom_fea3 = np.eye(len(degree_list), dtype = bool)[[degree_list.index(a.GetDegree()) for a in mol.GetAtoms()]][:,:-1]
    atom_fea4 = np.eye(len(hybridization_list), dtype = bool)[[hybridization_list.index(str(a.GetHybridization())) for a in mol.GetAtoms()]][:,:-2]
    atom_fea5 = np.eye(len(hydrogen_list), dtype = bool)[[hydrogen_list.index(a.GetTotalNumHs(includeNeighbors=True)) for a in mol.GetAtoms()]][:,:-1]
    atom_fea6 = np.eye(len(valence_list), dtype = bool)[[valence_list.index(a.GetTotalValence()) for a in mol.GetAtoms()]][:,:-1]
    atom_fea7 = np.array([[(j in D_list), (j in A_list)] for j in range(mol.GetNumAtoms())], dtype = bool)
    atom_fea8 = np.array([_chirality(a) for a in mol.GetAtoms()], dtype = bool)
    atom_fea9 = np.array([[a.IsInRingSize(s) for s in ringsize_list] for a in mol.GetAtoms()], dtype = bool)
    atom_fea10 = np.array([[a.GetIsAromatic(), a.IsInRing()] for a in mol.GetAtoms()], dtype = bool)
    node_attr = np.concatenate([atom_fea1, atom_fea2, atom_fea3, atom_fea4, atom_fea5, atom_fea6, atom_fea7, atom_fea8, atom_fea9, atom_fea10], 1)

    shift = np.array([ast.literal_eval(atom.GetProp('shift')) for atom in mol.GetAtoms()])
    mask = np.array([atom.GetBoolProp('mask') for atom in mol.GetAtoms()])


    mol_dict['n_node'].append(n_node)
    mol_dict['n_edge'].append(n_edge)
    mol_dict['node_attr'].append(node_attr)

    mol_dict['shift'].append(shift)
    mol_dict['mask'].append(mask)
    mol_dict['smi'].append(Chem.MolToSmiles(mol))
    mol_dict['h_x_connectivity'].append(x_h_connectivity_dict)

    if n_edge > 0:

        bond_fea1 = np.eye(len(bond_list), dtype = bool)[[bond_list.index(str(b.GetBondType())) for b in mol.GetBonds()]]
        bond_fea2 = np.array([_stereochemistry(b) for b in mol.GetBonds()], dtype = bool)
        bond_fea3 = [[b.IsInRing(), b.GetIsConjugated()] for b in mol.GetBonds()]

        edge_attr = np.array(np.concatenate([bond_fea1, bond_fea2, bond_fea3], 1), dtype = bool)
        edge_attr = np.vstack([edge_attr, edge_attr])

        bond_loc = np.array([[b.GetBeginAtomIdx(), b.GetEndAtomIdx()] for b in mol.GetBonds()], dtype = int)
        src = np.hstack([bond_loc[:,0], bond_loc[:,1]])
        dst = np.hstack([bond_loc[:,1], bond_loc[:,0]])

        mol_dict['edge_attr'].append(edge_attr)
        mol_dict['src'].append(src)
        mol_dict['dst'].append(dst)

    return mol_dict

# Python code to merge dict using a single
# expression
def Merge(dict1, dict2):
    """Merge two dictionaries into a new one (keys of ``dict2`` win).

    Args:
        dict1: base mapping.
        dict2: mapping overriding/extending ``dict1``.

    Returns:
        A new dict combining both; inputs are left unchanged.
    """
    res = {**dict1, **dict2}
    return res

########################################## Dataset Class ##########################################
class GraphDataset_sample():
    """Single (or batched) sample dataset backed by a saved ``.npz`` graph dict.

    Used for inference only: it loads the intermediate file produced by
    ``save_as_npy_for_13C``/``save_as_npy_for_1H`` and reconstructs one DGL
    graph per molecule on demand.
    """

    def __init__(self, target, file_path):
        """Load the graph dictionary from an ``.npz`` file.

        Args:
            target: ``"13C"`` or ``"1H"``; selects how shifts are returned.
            file_path: path to the ``.npz`` created by ``save_as_npy_*``.

        Side effects:
            Reads the file eagerly via :meth:`load`.
        """

        self.target = target
        # self.graph_representation = graph_representation
        self.split = None
        self.file_path = file_path
        self.load()


    def load(self):
        """Read the ``.npz`` file and split concatenated arrays per molecule.

        Populates node/edge features, bond endpoints, shifts, masks and the
        H-connectivity list, and precomputes cumulative node/edge offsets
        (``n_csum``/``e_csum``) used to slice per-molecule views.

        Returns:
            None.

        Side effects:
            Reads ``self.file_path`` and sets instance attributes. For 1H the
            connectivity list is also stored as ``self.h_x_connectivity``.
        """
        if self.target == "13C":
            [mol_dict] = np.load(self.file_path, allow_pickle=True)['data']
        elif self.target == "1H":
            [mol_dict] = np.load(self.file_path, allow_pickle=True)['data']
            self.h_x_connectivity = mol_dict['h_x_connectivity']

        self.n_node = mol_dict['n_node']
        self.n_edge = mol_dict['n_edge']
        self.node_attr = mol_dict['node_attr']
        self.edge_attr = mol_dict['edge_attr']
        self.src = mol_dict['src']
        self.dst = mol_dict['dst']

        self.shift = mol_dict['shift']
        self.mask = mol_dict['mask']
        self.smi = mol_dict['smi']
        self.h_x_connectivity_list = mol_dict['h_x_connectivity']


        self.n_csum = np.concatenate([[0], np.cumsum(self.n_node)])
        self.e_csum = np.concatenate([[0], np.cumsum(self.n_edge)])


    def __getitem__(self, idx):
        """Build the DGL graph and target tensors for molecule ``idx``.

        Args:
            idx: molecule index within the loaded sample.

        Returns:
            Tuple ``(g, n_node, numHshifts, shift_test, shift, mask)``:
            the DGL graph with ``node_attr``/``edge_attr``, node count, number
            of hydrogens per heavy atom, the zero placeholder shifts used for
            inference, the stored shift values and the boolean mask.

        Side effects:
            None (tensors are created per call).

        Note for 1H: ``shift``/``shift_test`` are filled with zeros because
        the model predicts new shifts rather than reusing stored ones.
        """
        g = graph((self.src[self.e_csum[idx]:self.e_csum[idx+1]], self.dst[self.e_csum[idx]:self.e_csum[idx+1]]), num_nodes = self.n_node[idx])
        g.ndata['node_attr'] = torch.from_numpy(self.node_attr[self.n_csum[idx]:self.n_csum[idx+1]]).float()
        g.edata['edge_attr'] = torch.from_numpy(self.edge_attr[self.e_csum[idx]:self.e_csum[idx+1]]).float()

        n_node = self.n_node[idx:idx+1].astype(int)
        numHshifts = np.zeros(n_node)
        shift = self.shift[self.n_csum[idx]:self.n_csum[idx+1]]#.astype(float)
        shift_test = shift
        mask = self.mask[self.n_csum[idx]:self.n_csum[idx+1]].astype(bool)

        ### Fill with all zeros for inference on new sample
        if self.target == '1H':
                shift = np.hstack([0.0 for s in self.shift[idx]])
                numHshifts = np.hstack([len(s) for s in self.h_x_connectivity_list[idx].values()])
                shift_test = np.hstack([np.hstack([0.0 for i in range(len(s))]) for s in self.h_x_connectivity_list[idx].values() if len(s) > 0])

                # shift_test = np.append(shift_test,[0.])
        return g, n_node, numHshifts, shift_test, shift, mask


    def __len__(self):
        """Return the number of molecules stored in the sample."""
        return self.n_node.shape[0]



########################################## General ##########################################
# Load Model

def load_std_mean(target,graph_representation):
    """Return the training-set mean and standard deviation of shifts.

    This functions returns the train_y_mean, train_y_std of the train dataset for either H or C

    Args:
        target: ``"1H"`` or ``"13C"``; selects the graph dataset.
        graph_representation: graph representation name passed to
            ``GraphDataset`` (e.g. ``"sparsified"``).

    Returns:
        Tuple ``(train_y_mean, train_y_std)`` computed over all training
        shifts (numpy scalars).

    Side effects:
        Reads the training dataset from disk and builds a temporary DataLoader.
    """
    #Load Train data (to get the train_y_mean and std)
    batch_size = 128
    data = GraphDataset(target, graph_representation)
    all_train_data_loader_C = DataLoader(dataset=data, batch_size=batch_size, shuffle=True, collate_fn=collate_reaction_graphs, drop_last=True)
    train_y = np.hstack([inst[-2][inst[-1]] for inst in iter(all_train_data_loader_C.dataset)])
    train_y_mean, train_y_std = np.mean(train_y), np.std(train_y)
    return train_y_mean, train_y_std

def load_model(target, save_path):
    """Instantiate and load the pretrained SGNN model for a nucleus.

    The input dimensions are inferred from a sample file, then the matching
    ``<target>_sparsified_proposed_proposed_1.pt`` checkpoint is located
    (with several alternative directories tried) and loaded into
    ``nmr_mpnn_PROPOSED``.

    Args:
        target: ``"1H"`` or ``"13C"``.
        save_path: path to a sample file readable by ``GraphDataset_sample``,
            used only to determine node/edge feature dimensions.

    Returns:
        The loaded ``nmr_mpnn_PROPOSED`` network (in eval-agnostic train mode).

    Side effects:
        Reads files from disk and prints path resolution messages.

    NOTE (known issue, left unchanged): the network is moved with ``.cuda()``
    unconditionally, so this fails on CPU-only machines. ``torch.load`` is
    also called without ``map_location``, so a checkpoint saved on GPU cannot
    be loaded on CPU.
    """
    node_embedding_dim = 256
    node_hidden_dim = 512
    readout_n_hidden_dim = 512
    readout_mode = "proposed"

    data_sample = GraphDataset_sample(target, save_path)

    node_dim = data_sample.node_attr.shape[1]
    edge_dim = data_sample.edge_attr.shape[1]
    net = nmr_mpnn_PROPOSED(node_dim, edge_dim, readout_mode, node_embedding_dim, readout_n_hidden_dim).cuda()

    # Get the directory of the current script
    script_dir = os.path.dirname(__file__)

    # Build the base path to the models directory relative to the script's location
    # Updated to use the new models/sgnn directory structure
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_base_path = os.path.join(project_root, 'models', 'sgnn', 'model')
    
    # print(f"Looking for models in: {models_base_path}")

    # Define model paths based on the target
    if target == "1H":
        model_path = os.path.join(models_base_path, '1H_sparsified_proposed_proposed_1.pt')
    elif target == "13C":
        model_path = os.path.join(models_base_path, '13C_sparsified_proposed_proposed_1.pt')
    else:
        raise ValueError(f"Unsupported target: {target}")
        
    # Check if the model file exists
    if not os.path.exists(model_path):
        # Try alternative paths
        alternative_paths = [
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nmr_sgnn_norm', 'model'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'sgnn_norm', 'model'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models', 'sgnn'),
            os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'nmr_sgnn_norm'),
        ]
        
        found = False
        for alt_base_path in alternative_paths:
            alt_model_path = os.path.join(alt_base_path, f"{target}_sparsified_proposed_proposed_1.pt")
            print(f"Trying alternative path: {alt_model_path}")
            if os.path.exists(alt_model_path):
                model_path = alt_model_path
                found = True
                print(f"Found model at alternative path: {model_path}")
                break
        
        if not found:
            raise FileNotFoundError(f"Model file not found: {model_path}. Please make sure the model files are in the correct location.")
    
    # print(f"Loading model from: {model_path}")
    net.load_state_dict(torch.load(model_path))
    return net

########################################## C related ##########################################

def save_as_npy_for_13C(mol_list):
    """Featurise molecules for 13C inference and save an intermediate ``.npz``.

    For each molecule: add hydrogens, write a temporary SDF, parse it back to
    recover the explicit atom/bond ordering, derive C-H connectivity, restore
    stereochemistry properties, set a zero shift and a mask of 1 for every
    carbon (mask 0 otherwise), and accumulate the sparsified graph features.
    The concatenated arrays are then written to
    ``<tempdir>/sample_<rand>_13C.npz``.

    Args:
        mol_list: list of RDKit ``Mol`` objects (usually from SMILES).

    Returns:
        Path to the written ``.npz`` file.

    Side effects:
        Creates temporary SDF files (via ``mol2SDF``) and the ``.npz`` file;
        prints nothing but relies on RDKit logging being disabled.

    NOTE (known issue, left unchanged): ``np.vstack`` is used for
    ``edge_attr`` unconditionally. A molecule with no bonds never appends an
    edge array (see ``add_mol_sparsified_graph``), so such input makes the
    stack fail with a ValueError.
    """

    x_h_connectivity_dict_list = []
    target = "13C" #"1H"
    atom_list = ['H','Li','B','C','N','O','F','Na','Mg','Al','Si','P','S','Cl','K','Ti','Zn','Ge','As','Se','Br','Pd','Ag','Sn','Sb','Te','I','Hg','Tl','Pb','Bi','Ga']
    charge_list = [1, 2, 3, -1, -2, -3, 0]
    degree_list = [1, 2, 3, 4, 5, 6, 0]
    valence_list = [1, 2, 3, 4, 5, 6, 0]
    hybridization_list = ['SP','SP2','SP3','SP3D','SP3D2','S','UNSPECIFIED']
    hydrogen_list = [1, 2, 3, 4, 0]
    ringsize_list = [3, 4, 5, 6, 7, 8]
    bond_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
    max_graph_distance = 20
    rdBase.DisableLog('rdApp.error')
    rdBase.DisableLog('rdApp.warning')
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
    message_passing_mode = "proposed"
    readout_mode = "proposed"
    graph_representation = "sparsified"
    target = "13C"
    memo = ""
    fold_seed = 0
    data_split = [0.95, 0.05]
    batch_size = 128
    random_seed = 27407
    node_embedding_dim = 256
    node_hidden_dim = 512
    readout_n_hidden_dim = 512

    mol_dict = {'n_node': [],
                    'n_edge': [],
                    'node_attr': [],
                    'edge_attr': [],
                    'src': [],
                    'dst': [],
                    'shift': [],
                    'mask': [],
                    'smi': [],
                    'h_x_connectivity': []}

    ### Loop over each mol
    for i, mol in enumerate(mol_list):

        ### Generate SDF File
        mol = Chem.AddHs(mol, addCoords=True)
        sdf_path = mol2SDF(mol)

        ### Generate connectivity list
        atom_list, connectivity_list, docline_list, name, mol = get_molecule_data(sdf_path)
        # c_h_connectivity_dict = get_c_h_connectivity(connectivity_list, atom_list)
        x_h_connectivity_dict = get_x_h_connectivity(connectivity_list, atom_list)
        #x_h_connectivity_dict = check_for_CH3_CH_CH3_structures(mol,x_h_connectivity_dict)

        x_h_connectivity_dict_list.append(x_h_connectivity_dict)

        ### Check on Mol
        mol = Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
        si = Chem.FindPotentialStereo(mol)
        for element in si:
            if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified':
                mol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
            elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified':
                mol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))
        assert '.' not in Chem.MolToSmiles(mol)

        # Write dummy shift (0.0) into the shift property and set mask for carbons True
        atom_selection_list = ["C"]#"H"
        for j, atom in enumerate(mol.GetAtoms()):
            if atom.GetSymbol() in atom_selection_list:
                atom.SetProp('shift', str(0.0))
                atom.SetBoolProp('mask', 1)
            else:
                atom.SetProp('shift', str(0.0))
                atom.SetBoolProp('mask', 0)

        mol = Chem.RemoveHs(mol)
        mol_dict = add_mol_sparsified_graph(mol_dict, mol, x_h_connectivity_dict)

    mol_dict['n_node'] = np.array(mol_dict['n_node']).astype(int)
    mol_dict['n_edge'] = np.array(mol_dict['n_edge']).astype(int)
    mol_dict['node_attr'] = np.vstack(mol_dict['node_attr']).astype(bool)
    mol_dict['edge_attr'] = np.vstack(mol_dict['edge_attr']).astype(bool)
    mol_dict['src'] = np.hstack(mol_dict['src']).astype(int)
    mol_dict['dst'] = np.hstack(mol_dict['dst']).astype(int)
    if target == '13C': mol_dict['shift'] = np.hstack(mol_dict['shift'])
    mol_dict['mask'] = np.hstack(mol_dict['mask']).astype(bool)
    mol_dict['smi'] = np.array(mol_dict['smi'])
    mol_dict['h_x_connectivity'] = np.array(mol_dict['h_x_connectivity'])

    #save the paramenters
    # This file is only an intermediate for the inference loader and is deleted
    # by main_execute, so keep it in the temp dir: writing it to the caller's
    # working directory leaves stray sample_*.npz files in the repository
    # whenever a batch raises before the cleanup lines run.
    folder = tempfile.gettempdir()
    rand_num = random.randint(0,100000)
    name = f"sample_{rand_num}_{target}.npz"
    save_path = os.path.join(folder, name)
    np.savez_compressed(save_path, data = [mol_dict])
    return save_path



def inference_C(net, save_path, train_y_mean_C, train_y_std_C):
    """Run 13C shift inference for the sample stored at ``save_path``.

    Args:
        net: loaded 13C ``nmr_mpnn_PROPOSED`` network.
        save_path: ``.npz`` sample produced by ``save_as_npy_for_13C``.
        train_y_mean_C: training mean used to un-standardise predictions.
        train_y_std_C: training std used to un-standardise predictions.

    Returns:
        Array of predicted 13C shifts (one per carbon, in graph order).

    Side effects:
        Builds a DataLoader (batch size from the module-level ``batch_size``)
        and runs a forward pass. Timing is produced by ``inference`` but only
        the predictions are returned.
    """
    # Load data sample
    target = "13C"

    data_sample = GraphDataset_sample(target, save_path)
    test_sample_loader = DataLoader(dataset=data_sample, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)

    # inference
    test_y_pred_C, time_per_mol_C = inference(net, test_sample_loader, train_y_mean_C, train_y_std_C)
    return test_y_pred_C


########################################## H related ##########################################

def save_as_npy_for_1H(mol_list):
    """Featurise molecules for 1H inference and save an intermediate ``.npz``.

    Mirrors ``save_as_npy_for_13C`` but selects atoms that carry hydrogens
    (via the X-H connectivity dict) as masked prediction targets, and stores
    the connectivity dicts so shifts can be reassembled later.

    Args:
        mol_list: list of RDKit ``Mol`` objects.

    Returns:
        Tuple ``(save_path, x_h_connectivity_dict_list)``: the ``.npz`` path
        and one connectivity dict per molecule.

    Side effects:
        Creates temporary SDF files and the ``.npz`` file.

    NOTE (known issue, left unchanged): like the 13C variant, ``np.vstack``
    over ``edge_attr`` fails for a bond-less molecule because no edge array
    was appended in that case.
    """
    length = len(mol_list)

    x_h_connectivity_dict_list = []

    target = "1H" #"13C"

    atom_list = ['H','Li','B','C','N','O','F','Na','Mg','Al','Si','P','S','Cl','K','Ti','Zn','Ge','As','Se','Br','Pd','Ag','Sn','Sb','Te','I','Hg','Tl','Pb','Bi','Ga']
    charge_list = [1, 2, 3, -1, -2, -3, 0]
    degree_list = [1, 2, 3, 4, 5, 6, 0]
    valence_list = [1, 2, 3, 4, 5, 6, 0]
    hybridization_list = ['SP','SP2','SP3','SP3D','SP3D2','S','UNSPECIFIED']
    hydrogen_list = [1, 2, 3, 4, 0]
    ringsize_list = [3, 4, 5, 6, 7, 8]
    bond_list = ['SINGLE', 'DOUBLE', 'TRIPLE', 'AROMATIC']
    max_graph_distance = 20
    rdBase.DisableLog('rdApp.error')
    rdBase.DisableLog('rdApp.warning')
    chem_feature_factory = ChemicalFeatures.BuildFeatureFactory(os.path.join(RDConfig.RDDataDir, 'BaseFeatures.fdef'))
    message_passing_mode = "proposed"
    readout_mode = "proposed"
    graph_representation = "sparsified"
    memo = ""
    fold_seed = 0
    data_split = [0.95, 0.05]
    batch_size = 128
    random_seed = 27407
    node_embedding_dim = 256
    node_hidden_dim = 512
    readout_n_hidden_dim = 512

    mol_dict = {'n_node': [],
                'n_edge': [],
                'node_attr': [],
                'edge_attr': [],
                'src': [],
                'dst': [],
                'shift': [],
                'mask': [],
                'smi': [],
                'h_x_connectivity': []}


    ### Loop over each mol
    for i, mol in enumerate(mol_list):

        ### Generate SDF File
        mol = Chem.AddHs(mol, addCoords=True)
        sdf_path = mol2SDF(mol)

        ### Generate connectivity list
        atom_list, connectivity_list, docline_list, name, mol = get_molecule_data(sdf_path)
        # c_h_connectivity_dict = get_c_h_connectivity(connectivity_list, atom_list)
        x_h_connectivity_dict = get_x_h_connectivity(connectivity_list, atom_list)
        #x_h_connectivity_dict = check_for_CH3_CH_CH3_structures(mol,x_h_connectivity_dict)
        x_h_connectivity_dict_list.append(x_h_connectivity_dict)

        ### Check on Mol
        mol = Chem.RemoveHs(mol)
        Chem.SanitizeMol(mol)
        si = Chem.FindPotentialStereo(mol)
        for element in si:
            if str(element.type) == 'Atom_Tetrahedral' and str(element.specified) == 'Specified':
                mol.GetAtomWithIdx(element.centeredOn).SetProp('Chirality', str(element.descriptor))
            elif str(element.type) == 'Bond_Double' and str(element.specified) == 'Specified':
                mol.GetBondWithIdx(element.centeredOn).SetProp('Stereochemistry', str(element.descriptor))
        assert '.' not in Chem.MolToSmiles(mol)

        # Filter to get just the atoms that are connected to hydrogens
        H_ID_list = [k for k,v in x_h_connectivity_dict.items() if v !=[] ]

        # Set shift = 0.0 for all C where H is connected
        for j, atom in enumerate(mol.GetAtoms()):
            if atom.GetIdx() in H_ID_list:
                atom.SetProp('shift', str(0.0))
                atom.SetBoolProp('mask', 1)
            else:
                atom.SetProp('shift', str(0.0))
                atom.SetBoolProp('mask', 0)
        mol_dict = add_mol_sparsified_graph(mol_dict, mol, x_h_connectivity_dict)


    # generate Mol_dict with all the information
    mol_dict['n_node'] = np.array(mol_dict['n_node']).astype(int)
    mol_dict['n_edge'] = np.array(mol_dict['n_edge']).astype(int)
    mol_dict['node_attr'] = np.vstack(mol_dict['node_attr']).astype(bool)
    mol_dict['edge_attr'] = np.vstack(mol_dict['edge_attr']).astype(bool)
    mol_dict['src'] = np.hstack(mol_dict['src']).astype(int)
    mol_dict['dst'] = np.hstack(mol_dict['dst']).astype(int)
    if target == '1H': mol_dict['shift'] = np.array(mol_dict['shift'])#, dtype=object)
    mol_dict['mask'] = np.hstack(mol_dict['mask']).astype(bool)
    mol_dict['smi'] = np.array(mol_dict['smi'])
    mol_dict['h_x_connectivity'] = np.array(mol_dict['h_x_connectivity'])

    #save the paramenters
    # This file is only an intermediate for the inference loader and is deleted
    # by main_execute, so keep it in the temp dir: writing it to the caller's
    # working directory leaves stray sample_*.npz files in the repository
    # whenever a batch raises before the cleanup lines run.
    folder = tempfile.gettempdir()
    rand_num = random.randint(0,100000)
    name = f"sample_{rand_num}_{target}.npz"
    save_path = os.path.join(folder, name)
    np.savez_compressed(save_path, data = [mol_dict])
    return save_path, x_h_connectivity_dict_list
    #save_path = os.path.join(folder, "sample_%s.npz"%(target))
    #np.savez_compressed('./sample_%s.npz'%(target), data = [mol_dict])
    #return save_path, x_h_connectivity_dict_list


def inference_H(net, save_path, train_y_mean_H, train_y_std_H):
    """Run 1H shift inference for the sample stored at ``save_path``.

    Args:
        net: loaded 1H ``nmr_mpnn_PROPOSED`` network.
        save_path: ``.npz`` sample produced by ``save_as_npy_for_1H``.
        train_y_mean_H: training mean used to un-standardise predictions.
        train_y_std_H: training std used to un-standardise predictions.

    Returns:
        Array of predicted 1H shifts.

    Side effects:
        Builds a DataLoader and runs a forward pass; only predictions are
        returned.
    """
    # Load data sample
    target = "1H" #"13C"
    data_sample = GraphDataset_sample(target, save_path)
    test_sample_loader = DataLoader(dataset=data_sample, batch_size=batch_size, shuffle=False, collate_fn=collate_reaction_graphs)

    # inference

    test_y_pred_H, time_per_mol_H = inference(net, test_sample_loader, train_y_mean_H, train_y_std_H)
    return test_y_pred_H

########################################## Reconstruct ##########################################

def create_shift_list(mol, x_h_connectivity_dict, test_y_pred_C, test_y_pred_H):
    """Re-assemble a per-atom shift list from C and H predictions.

    This function recreates the shift list based on the index labelling of the molecule (including H atoms)

    Walks the (hydrogen-added) molecule in atom order: carbons consume the
    next 13C prediction, hydrogens consume one 1H prediction per bonded heavy
    atom group, and all other elements get 0.

    Args:
        mol: RDKit ``Mol`` including explicit hydrogens.
        x_h_connectivity_dict: heavy-atom-to-hydrogen index mapping.
        test_y_pred_C: predicted 13C shifts, ordered by carbon.
        test_y_pred_H: predicted 1H shifts.

    Returns:
        List of shifts, one per atom (index-aligned with ``mol.GetAtoms()``).

    Side effects:
        Prints diagnostic dumps on exception (bare ``except``) and returns the
        partially built ``final_list`` in that case.

    Note: all hydrogens of one heavy atom receive the same prediction
    (``test_y_pred_H[h_count]``) and ``h_count`` advances once per heavy atom.
    """
    try:
        final_list = []
        c_count = 0
        h_count = 0
        h_done = []
        for idx, atom in enumerate(mol.GetAtoms()):

            if atom.GetSymbol() == "C":
                final_list.append(test_y_pred_C[c_count])
                c_count += 1

            elif atom.GetSymbol() == "H":
                # Check if the ID is in the connectivity list of the
                for k,v in x_h_connectivity_dict.items():
                    if idx in v and idx not in h_done:
                        for i in v:           # iterate over the number of H and add them to list and also to done list
                            final_list.append(test_y_pred_H[h_count])
                            h_done.append(i)
                        h_count += 1
            else:
                 final_list.append(0)
        return final_list
    except:
        print("x_h_connectivity_dict")
        print(x_h_connectivity_dict)
        print("test_y_pred_C")
        print(test_y_pred_C)
        print("test_y_pred_H")
        print(test_y_pred_H)
        print("mol")
        print(Chem.MolToSmiles(mol))
        return final_list



def get_shift_string(assigned_shifts):
    """Convert a shift list into the single space-separated SDF property line.

    Args:
        assigned_shifts: iterable of numeric shifts (one per atom).

    Returns:
        A string of the form ``"a b c"`` followed by a newline, ready to be
        written as the value of ``averaged_NMR_shifts``.
    """
    string = ""
    for idx,shift in enumerate(assigned_shifts):
        string = string + " " + str(shift)
    final_string = string[1:]+"\n"
    return final_string


def save_results_sdf_file(mol, save_folder, ID, final_list):
    """Write an SDF record with the predicted shifts as an ``NMR_<ID>.sdf``.

    This function saves the final shift predictions into a SDF files with the lowest energy conformer

    The molecule is first written with ``mol2SDF`` and then copied line by
    line into ``NMR_<ID>.sdf``; after the ``M  END`` marker an
    ``averaged_NMR_shifts`` data field is inserted.

    Args:
        mol: RDKit ``Mol`` (typically with explicit hydrogens).
        save_folder: destination directory.
        ID: sample identifier used in the output filename and record.
        final_list: per-atom shifts produced by ``create_shift_list``.

    Returns:
        Path to the written ``NMR_<ID>.sdf``.

    Side effects:
        Writes two SDF files (the intermediate ``<ID>.sdf`` and the final
        ``NMR_<ID>.sdf``).

    NOTE (known issue, left unchanged): the shift block is only inserted when
    a line is exactly ``'M  END\n'``. On Windows-style files terminated with
    ``'M  END\r\n'`` the comparison fails, so the shifts are silently omitted
    from the output SDF.
    """

    sdf_path = mol2SDF(mol, save_folder, ID)

    name = "NMR_" + str(ID)+".sdf"
    output_sdf = os.path.join(save_folder,name)
    with open(output_sdf, "w",  encoding='utf-8', errors='ignore') as output:
        with open(sdf_path) as g:
            for i in g:
                output.write(i)
                if i == 'M  END\n':
                    output.write("\n")
                    output.write(f">  <averaged_NMR_shifts>  ({1}) \n")
                    shifts_string = get_shift_string(final_list)
                    output.write(shifts_string)
                    output.write("\n")
    g.close()
    output.close()
    return output_sdf


def split_predictions(test_y_pred, n_shifts):
    """Split a flat prediction array into per-molecule chunks.

    This function splits up the predictions of the batch into a list of lists that contain just the predictions for each molecule in separate lists

    Args:
        test_y_pred: flat array of predictions for the whole batch.
        n_shifts: iterable giving the number of predictions for each molecule,
            in batch order.

    Returns:
        List of sub-arrays, one per molecule.
    """
    split_atom_pred = []
    start_index = 0

    for n in n_shifts:
        split_atom_pred.append(test_y_pred[start_index:start_index+n])
        start_index += n

    return split_atom_pred


def count_atoms(mol_list, mol_dict):
    """Count carbons and hydrogen-bearing heavy atoms per molecule.

    Args:
        mol_list: list of RDKit ``Mol`` objects (hydrogen-free input molecules).
        mol_dict: graph dict containing ``h_x_connectivity`` per molecule.

    Returns:
        Tuple ``(c_list, h_list)``: carbon counts and counts of heavy atoms
        bonded to at least one hydrogen, both in input order. ``h_list`` is
        used to split the 1H predictions.

    Side effects:
        None; ``mol_list`` entries are copied by ``Chem.AddHs``.
    """
    c_list = []  # list to store carbon counts
    h_list = []  # list to store hydrogen counts

    #### Count Carbons
    for mol in mol_list:
        c_count = 0  # count of carbons in the current molecule
        h_count = 0  # count of hydrogens in the current molecule
        mol = Chem.AddHs(mol)

        for atom in mol.GetAtoms():
            if atom.GetSymbol() == 'C':
                c_count += 1

        c_list.append(c_count)

    #### Count Hydrogens
    h_list = []
    for mol, x_h_connectivity_dict in zip(mol_list,mol_dict["h_x_connectivity"]):
        H_ID_list = [k for k,v in x_h_connectivity_dict.items() if v !=[] ]
        # Set shift = 0.0 for all C where H is connected
        h_count = 0
        for j, atom in enumerate(mol.GetAtoms()):
            if atom.GetIdx() in H_ID_list:
                h_count+=1
        h_list.append(h_count)

    return c_list, h_list

class MoleculeDataset(Dataset):
    """Minimal torch Dataset wrapping a ``SMILES``/``sample-id`` DataFrame."""

    def __init__(self, data_df):
        """Store the source DataFrame.

        Args:
            data_df: DataFrame with ``SMILES`` and ``sample-id`` columns.
        """
        self.data_df = data_df

    def __len__(self):
        """Return the number of molecules in the DataFrame."""
        return len(self.data_df)

    def __getitem__(self, idx):
        """Return ``(SMILES, sample_id)`` for row ``idx`` as strings.

        Args:
            idx: positional row index (``iloc``).
        """
        return self.data_df.iloc[idx]["SMILES"], str(self.data_df.iloc[idx]["sample-id"])

########################################## Main Execution Function ##########################################

def main_execute(data_df, means_stds, ML_save_folder, batch_size):
    """Predict 13C/1H shifts for a batch and write SDF files.

    Given a pandas dataframe with columns SMILES and sample-id it calculates
    all the sdf files including the nmr shifts for every C and H atom

    For each DataLoader batch the function featurises molecules, loads the C
    and H models, runs inference, splits predictions per molecule, rebuilds
    per-atom shifts and writes one SDF per molecule. Individual batch failures
    are caught, reported and their sample ids returned.

    Args:
        data_df: DataFrame with ``SMILES`` and ``sample-id``.
        means_stds: tuple ``(mean_C, std_C, mean_H, std_H)`` from
            ``load_std_mean``.
        ML_save_folder: destination directory for the SDF files.
        batch_size: DataLoader batch size.

    Returns:
        Tuple ``(batch_data, failed_ids)``: a DataFrame with columns
        ``sample-id``/``sdf_path``/``SMILES`` for successful molecules, and a
        list of ids whose batch failed.

    Side effects:
        Writes SDF files and removes the temporary ``.npz`` files after each
        successful batch.

    NOTE (known issue, left unchanged): the ``.npz`` cleanup runs only on the
    success path. If an exception occurs after ``save_as_npy_*`` created the
    files, they are left behind in the temp directory. The exception handler
    also references ``IDs``, which could be unbound if the failure happens
    before the batch is unpacked.
    """
    ############## General ##############
    (train_y_mean_C, train_y_std_C, train_y_mean_H, train_y_std_H) = means_stds
    SMILES_list = data_df["SMILES"]
    sample_id = data_df["sample-id"]

    dataset = MoleculeDataset(data_df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    result_list = []
    failed_ids = []
    for idx, batch in enumerate(dataloader):

        try:
            start_time = time.time()

            SMILES_list, IDs = batch
            IDs = [str(ID) for ID in IDs]
            mol_list = [Chem.MolFromSmiles(smi) for smi in SMILES_list]
            # ############## For 13C ##############
            target = "13C"
            save_path_C = save_as_npy_for_13C(mol_list)
            net_C = load_model(target, save_path_C)
            test_y_pred_C = inference_C(net_C, save_path_C, train_y_mean_C, train_y_std_C)
            [mol_dict_C] = np.load(save_path_C, allow_pickle=True)['data']
            c_list, h_list = count_atoms(mol_list, mol_dict_C)

            split_C_pred = split_predictions(test_y_pred_C, c_list)

            ############## For 1H ##############
            target = "1H"
            save_path_H, x_h_connectivity_dict_list = save_as_npy_for_1H(mol_list)
            net_H = load_model(target, save_path_H)
            test_y_pred_H = inference_H(net_H, save_path_H, train_y_mean_H, train_y_std_H)

            [mol_dict_H] = np.load(save_path_H, allow_pickle=True)['data']
            split_H_pred = split_predictions(test_y_pred_H, h_list)

            ############### Reconstruction ##############
            for (mol, test_y_pred_C, test_y_pred_H, x_h_connectivity_dict, ID) in zip(mol_list, split_C_pred, split_H_pred, x_h_connectivity_dict_list, IDs):
                    mol = Chem.AddHs(mol)
                    smi = Chem.MolToSmiles(mol)
                    final_list = create_shift_list(mol, x_h_connectivity_dict, test_y_pred_C, test_y_pred_H)

                    output_sdf = save_results_sdf_file(mol, ML_save_folder, ID, final_list)
                    mol = Chem.RemoveHs(mol)
                    smi = Chem.MolToSmiles(mol)               
                    result_list.append([ID, output_sdf, smi])

            time_taken = time.time() - start_time
            if os.path.isfile(save_path_H):
                os.remove(save_path_H)
            if os.path.isfile(save_path_C):
                os.remove(save_path_C)

        except Exception as e:
            print(f"Error processing batch: {str(e)}")
            import traceback
            traceback.print_exc()
            print(f"Failed IDs: {IDs}")
            print(f"Number of failed IDs: {len(IDs)}")
            failed_ids.extend(IDs)  # Add the failed IDs to the list           
    # Convert the results to a DataFrame
    batch_data = pd.DataFrame(result_list, columns=['sample-id', 'sdf_path', 'SMILES'])
    #print("main end")
 
    return batch_data, failed_ids