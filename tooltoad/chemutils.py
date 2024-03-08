import os
from typing import List

import networkx as nx
import numpy as np
from hide_warnings import hide_warnings
from rdkit import Chem
from rdkit.Chem import rdFMCS

try:
    from rdkit.Chem import rdDetermineBonds
except ImportError:
    print("Needs rdkit >= 2020.09.1")


def get_atom_map(mol1, mol2):
    """Get mapping between atom indices based on connectivity
    ! Mappings are not unique due to symmetry!

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
