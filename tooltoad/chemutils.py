import os
import re
from typing import List

import networkx as nx
import numpy as np
from hide_warnings import hide_warnings
from rdkit import Chem
from rdkit.Chem import rdFMCS, rdMolDescriptors

try:
    from rdkit.Chem import rdDetermineBonds
except ImportError:
    print("Needs rdkit >= 2020.09.1")


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


@hide_warnings
def _determineConnectivity(mol, **kwargs):
    """Determine bonds in molecule."""
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


def ac2mol(atoms: List[str], coords: List[list], useHueckel=True, **kwargs):
    """Converts atom symbols and coordinates to RDKit molecule."""
    xyz = ac2xyz(atoms, coords)
    rdkit_mol = Chem.MolFromXYZBlock(xyz)
    Chem.SanitizeMol(rdkit_mol)
    _determineConnectivity(rdkit_mol, useHueckel=useHueckel, **kwargs)
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
