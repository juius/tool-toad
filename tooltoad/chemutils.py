import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import List

import networkx as nx
import numpy as np
from hide_warnings import hide_warnings
from rdkit import Chem
from rdkit.Chem import (
    ResonanceMolSupplier,
    rdDetermineBonds,
    rdFMCS,
    rdMolAlign,
    rdMolDescriptors,
)
from rdkit.ML.Cluster import Butina

from tooltoad.utils import stream

logger = logging.getLogger(__name__)


class SymmetryMapper:
    def __init__(self, mol):
        """Initialize the SymmetryMapper with an RDKit Mol object.

        Generates resonance structures and calculates the atom rankings.
        """
        self.mol = mol
        self.resonance_structures = list(
            ResonanceMolSupplier(mol, Chem.ResonanceFlags.ALLOW_INCOMPLETE_OCTETS)
        )
        self.equivalence_map = self._calculate_equivalence_map()

    def _calculate_equivalence_map(self):
        """Calculate the mapping of symmetry-equivalent atoms based on
        canonical ranks across all resonance structures.

        Returns a dictionary with canonical rank as keys and atom
        indices as values.
        """
        # Initialize an array to hold rank counts for each atom across all resonance structures
        atom_count = len(self.mol.GetAtoms())
        rank_matrix = np.zeros((len(self.resonance_structures), atom_count), dtype=int)

        # Get ranks for each resonance structure and store them in the matrix
        for i, res_mol in enumerate(self.resonance_structures):
            ranks = Chem.CanonicalRankAtoms(res_mol, breakTies=False)
            rank_matrix[i] = ranks

        # Calculate the average rank for each atom across resonance forms
        average_ranks = np.mean(rank_matrix, axis=0)

        # Create an equivalence map based on average ranks
        equivalence_map = {}
        for idx, avg_rank in enumerate(average_ranks):
            rank = int(avg_rank)  # Use integer representation for mapping
            if rank not in equivalence_map:
                equivalence_map[rank] = []
            equivalence_map[rank].append(idx)

        # Filter out entries with only one atom (not symmetric)
        equivalence_map = {
            rank: indices
            for rank, indices in equivalence_map.items()
            if len(indices) > 1
        }

        return equivalence_map

    def get_equivalent_atoms(self, atom_id):
        """Given an atom ID, return a list of equivalent atom IDs."""
        for rank, indices in self.equivalence_map.items():
            if atom_id in indices:
                return [atom for atom in indices if atom != atom_id]
        return []

    def get_canonical_ranks(self):
        """Return a list of canonical ranks averaged across all resonance
        structures."""
        atom_count = len(self.mol.GetAtoms())
        rank_matrix = np.zeros((len(self.resonance_structures), atom_count), dtype=int)

        for i, res_mol in enumerate(self.resonance_structures):
            ranks = Chem.CanonicalRankAtoms(res_mol, breakTies=False)
            rank_matrix[i] = ranks

        average_ranks = np.mean(rank_matrix, axis=0)
        return average_ranks.astype(int).tolist()


def hartree2kcalmol(hartree: float) -> float:
    """Converts Hartree to kcal/mol."""
    return hartree * 627.509474


def get_num_confs(mol: Chem.Mol, conf_rule: str = "3x+3,max10") -> int:
    """Calculate the number of conformers based on the supplied rule and the
    number of rotatable bonds.

    Args:
        mol (Chem.Mol): RDKit molecule object.
        conf_rule (str, optional): Expression with which to calculate the number of conformers based on the number of rotatable bonds. Defaults to "3x+3".

    Returns:
        int: _description_
    """
    for c in conf_rule:
        assert c in [
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
            "6",
            "7",
            "8",
            "9",
            "x",
            "+",
            ",",
            "m",
            "a",
        ], f"`conf_rule` must be of the form '3x+3,max10' but got {conf_rule}"
    fragments = re.split(r"\+|,", conf_rule)
    constant_term = 0
    linear_term = 0
    max_term = float("inf")
    for fragment in fragments:
        if "max" in fragment:
            max_term = int(fragment.lstrip("max"))
        elif "x" in fragment:
            linear_term = int(fragment.rstrip("x"))
        else:
            constant_term = int(fragment)
    x = rdMolDescriptors.CalcNumRotatableBonds(mol)
    return min([linear_term * x + constant_term, max_term])


def filter_conformers(
    mol: Chem.Mol,
    rmsdThreshold: float = 1.0,
    numThreads: int = -1,
    onlyHeavyAtoms: bool = True,
) -> Chem.Mol:
    """Filter conformers of a molecule based on RMSD clustering.

    Args:
        mol (Chem.Mol): Molecule to filter
        rmsdThreshold (float, optional): Threshold for clustering. Defaults to 1.0.
        numThreads (int, optional): Number of cores to use. Defaults to -1.
        onlyHeavyAtoms (bool, optional): Only consider heavy atoms. Defaults to True.

    Returns:
        Chem.Mol: Molecule with filtered conformers
    """
    eval_mol = Chem.RemoveHs(mol) if onlyHeavyAtoms else mol
    distance_matrix = rdMolAlign.GetAllConformerBestRMS(
        eval_mol, numThreads=numThreads, symmetrizeConjugatedTerminalGroups=True
    )

    clusters = Butina.ClusterData(
        data=distance_matrix,
        nPts=mol.GetNumConformers(),
        distThresh=rmsdThreshold,
        isDistData=True,
        reordering=True,
    )

    retained_confs = [mol.GetConformer(i[0]) for i in clusters]
    new_mol = Chem.Mol(mol)
    new_mol.RemoveAllConformers()
    for conf in retained_confs:
        new_mol.AddConformer(conf, assignId=True)
    return new_mol


def get_atom_map(mol1, mol2):
    """Get mapping between atom indices based on connectivity ! Mappings are
    not unique due to symmetry!

    Args:
        mol1 (_type_): _description_
        mol2 (_type_): _description_

    Returns:
        _type_: _description_
    """
    mcs = Chem.MolFromSmarts(
        rdFMCS.FindMCS(
            [mol1, mol2], bondCompare=Chem.rdFMCS.BondCompare.CompareAny
        ).smartsString
    )
    match1 = list(mol1.GetSubstructMatch(mcs))
    match2 = list(mol2.GetSubstructMatch(mcs))

    amap = {}
    for i1, i2 in zip(match1, match2):
        amap[i1] = i2
    return amap


def determine_connectivity_xtb(mol: Chem.Mol) -> Chem.Mol:
    """Determine connectivity based on GFNFF-xTB implementation."""
    calc_dir = tempfile.TemporaryDirectory()
    tmp_file = Path(calc_dir.name) / "input.xyz"
    Chem.MolToXYZFile(mol, str(tmp_file))
    CMD = f"xtb --gfnff {str(tmp_file)} --norestart --wrtopo nb"
    output = list(stream(CMD, cwd=calc_dir.name))
    logger.debug("".join(output))
    with open(Path(calc_dir.name) / "gfnff_lists.json", "r") as f:
        data_dict = json.load(f)
    calc_dir.cleanup()

    # set connectivity in rdkit mol object
    nb = data_dict["nb"]
    emol = Chem.EditableMol(mol)
    for i, a in enumerate(mol.GetAtoms()):
        for j in nb[i]:
            if j == 0:
                break
            elif i > j - 1:
                continue
            emol.AddBond(i, j - 1, Chem.BondType.SINGLE)
    return emol.GetMol()


@hide_warnings
def _determineConnectivity(mol, usextb=False, **kwargs):
    """Determine bonds in molecule."""
    if usextb:
        mol = determine_connectivity_xtb(mol)
    else:
        try:
            rdDetermineBonds.DetermineConnectivity(mol, **kwargs)
        finally:
            # cleanup extended hueckel files
            try:
                os.remove("nul")
                os.remove("run.out")
            except FileNotFoundError:
                pass
    return mol


def xyz2mol(xyzblock: str, useHueckel=True, **kwargs):
    """Converts atom symbols and coordinates to RDKit molecule."""
    rdkit_mol = Chem.MolFromXYZBlock(xyzblock)
    Chem.SanitizeMol(rdkit_mol)
    _determineConnectivity(rdkit_mol, useHueckel=useHueckel, **kwargs)
    return rdkit_mol


def xyz2ac(xyzblock: str):
    """Converts atom symbols and coordinates to xyz string."""
    lines = xyzblock.split("\n")
    atoms = []
    coords = []
    for line in lines[2:]:
        line = line.strip()
        if len(line) > 0:
            atom, x, y, z = line.split()
            atoms.append(atom)
            coords.append([float(x), float(y), float(z)])
        else:
            break
    return atoms, coords


def ac2xyz(atoms: List[str], coords: List[list]):
    """Converts atom symbols and coordinates to xyz string."""
    xyz = f"{len(atoms)}\n\n"
    for atom, coord in zip(atoms, coords):
        xyz += f"{atom} {coord[0]:.8f} {coord[1]:.8f} {coord[2]:.8f}\n"
    return xyz


def ac2mol(
    atoms: List[str],
    coords: List[list],
    charge: int = 0,
    perceive_connectivity: bool = True,
    sanitize: bool = True,
):
    """Converts atom symbols and coordinates to RDKit molecule."""
    xyz = ac2xyz(atoms, coords)
    rdkit_mol = Chem.MolFromXYZBlock(xyz)
    if sanitize:
        Chem.SanitizeMol(rdkit_mol)
    if charge != 0:
        rdkit_mol.GetAtomWithIdx(0).SetFormalCharge(charge)
    if perceive_connectivity:
        _determineConnectivity(rdkit_mol)
    return rdkit_mol


def iteratively_determine_bonds(mol, linspace=np.linspace(0.3, 0.1, 30)):
    """Iteratively determine bonds until the molecule is connected."""
    for threshold in linspace:
        _determineConnectivity(mol, useHueckel=True, overlapThreshold=threshold)
        adjacency = Chem.GetAdjacencyMatrix(mol, force=True)
        graph = nx.from_numpy_array(adjacency)
        if nx.is_connected(graph):
            break
    if not nx.is_connected(graph):
        raise ValueError("Molecule contains disconnected fragments")
