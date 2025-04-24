import itertools
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
    rdmolops,
)
from rdkit.ML.Cluster import Butina
from sklearn.cluster import DBSCAN

from tooltoad.utils import stream

logger = logging.getLogger(__name__)

COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
}


VDW_RADII = {
    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "H": 1.2,
    "S": 1.8,
    # Add more as needed
}


class Constraint:
    def __init__(self, ids: list[int], value: float):
        self.ids = ids
        self.value = value
        assert all(isinstance(i, int) for i in ids)

    def __repr__(self):
        return f"{self.xtb_type.capitalize()} Constraint at {self.xtb_value} for ({','.join([str(x) for x in self.ids])})"

    @property
    def xtb_type(self):
        if len(self.ids) == 2:
            return "distance"
        elif len(self.ids) == 3:
            return "angle"
        elif len(self.ids) == 4:
            return "dihedral"
        else:
            raise ValueError

    @property
    def xtb_ids(self):
        return [i + 1 for i in self.ids]

    @property
    def xtb_value(self):
        return self.value if self.value else "auto"

    @property
    def xtb(self):
        return f"{self.xtb_type}: {', '.join([str(x) for x in self.xtb_ids])}, {self.xtb_value}"

    @property
    def orca_type(self):
        if len(self.ids) == 2:
            return "B"
        elif len(self.ids) == 3:
            return "A"
        elif len(self.ids) == 4:
            return "D"
        else:
            raise ValueError

    @property
    def orca_ids(self):
        return self.ids

    @property
    def orca_value(self):
        return self.value if self.value else ""

    @property
    def orca(self):
        return f"{{ {self.orca_type} {' '.join([str(x) for x in self.orca_ids])} {self.orca_value} C }}"


class EnsembleCluster:
    def __init__(
        self,
        atoms: list[str],
        ensemble_coords: list[list[list[float]]],
        energies: list[float] | None = None,
    ):
        self.atoms = atoms
        self.ensemble_coords = np.asarray(ensemble_coords)
        self.n_conformers = len(ensemble_coords)
        self.energies = np.asarray(energies) if energies else None

        mols = [ac2mol(atoms, c) for c in self.ensemble_coords]
        smis = [Chem.MolToSmiles(mol) for mol in mols]
        if len(set(smis)) > 1:
            raise ValueError("Change in connectivity in ensemble")
            # TODO: deal with that
        mol = mols[0]
        # add conformers to mol
        for c in mols[1:]:
            mol.AddConformer(c.GetConformer(), assignId=True)
        self.mol = mol

    @classmethod
    def from_goat(cls, goat_results: dict):
        assert (
            "goat" in goat_results
        ), f"No GOAT results found: {list(goat_results.keys())}"
        return cls(
            atoms=goat_results["goat"]["ensemble"]["atoms"],
            ensemble_coords=goat_results["goat"]["ensemble"]["coords"],
            energies=goat_results["goat"]["ensemble"]["energies"],
        )

    def __call__(
        self, eps: float = 1.0, min_samples: int = 1, enantio_selective: bool = False
    ):
        self._calc_rmsd_matrix()
        self._cluster_conformers(eps=eps, min_samples=min_samples)
        logger.info(f"Found {len(set(self.labels))} clusters")
        if not enantio_selective:
            self._join_enantio_clusters()
            logger.info(
                f"Retained {len(set(self.labels))} clusters after joining enantiomer clusters"
            )
        if self.energies is not None:
            return self._select_best_conformers()
        else:
            logger.warning("No energies provided, returning labels only")
            return self.labels

    def _join_enantio_clusters(self):
        def _pairs2groups(pairs):
            G = nx.Graph()
            G.add_edges_from(pairs)
            groups = list(nx.connected_components(G))

            return groups

        def pairwise_distance(coords):
            return np.sqrt(
                np.sum(
                    (coords[:, np.newaxis, :] - coords[np.newaxis, :, :]) ** 2,
                    axis=-1,
                )
            )

        pairs = []
        sm = SymmetryMapper(self.mol)
        equivalent_atoms = sm.equivalence_map.values()
        for i, j in itertools.combinations(range(self.rmsd_matrix.shape[0]), 2):
            di = pairwise_distance(self.ensemble_coords[i])
            dj = pairwise_distance(self.ensemble_coords[j])
            for atoms in equivalent_atoms:
                di[atoms] = di[atoms].mean(axis=0)
                di[:, atoms] = di[:, atoms].mean(axis=1)[:, np.newaxis]
                dj[atoms] = dj[atoms].mean(axis=0)
                dj[:, atoms] = dj[:, atoms].mean(axis=1)[:, np.newaxis]
            if np.allclose(di, dj, atol=0.2):
                pairs.append(set([i, j]))

        enantio_groups = _pairs2groups(pairs)
        for g in enantio_groups:
            self.labels[list(g)] = max(self.labels) + 1

    def _calc_rmsd_matrix(self):
        rmsds = np.zeros((self.n_conformers, self.n_conformers))
        rmsds[np.tril_indices(self.n_conformers, k=-1)] = np.asarray(
            Chem.rdMolAlign.GetAllConformerBestRMS(self.mol, numThreads=-1)
        )
        rmsds += rmsds.T
        self.rmsd_matrix = rmsds

    def _cluster_conformers(self, eps: float = 1.0, min_samples: int = 1):
        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels = dbscan.fit_predict(self.rmsd_matrix)

    def _select_best_conformers(self):
        # Find the structure with the lowest energy in each cluster
        clustered_structures = []
        for cluster_label in set(self.labels):
            if cluster_label == -1:  # Skip noise points (-1)
                continue
            cluster_indices = np.where(self.labels == cluster_label)[0]
            best_point_idx = cluster_indices[np.argmin(self.energies[cluster_indices])]
            clustered_structures.append(self.ensemble_coords[best_point_idx])
        return clustered_structures


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
        rank_matrix = np.zeros(
            (max(len(self.resonance_structures), 1), atom_count), dtype=int
        )

        if len(Chem.DetectChemistryProblems(self.mol)) == 0:
            # Get ranks for each resonance structure and store them in the matrix
            for i, res_mol in enumerate(self.resonance_structures):
                ranks = Chem.CanonicalRankAtoms(res_mol, breakTies=False)
                rank_matrix[i] = ranks
        else:
            rank_matrix[0] = Chem.CanonicalRankAtoms(self.mol, breakTies=False)

        new_ranks = np.sum(rank_matrix, axis=0)

        # Create an equivalence map based on new ranks
        equivalence_map = {}
        for idx, rank in enumerate(new_ranks):
            rank = int(rank)  # Use integer representation for mapping
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

    def __call__(self, structure):
        ranks = self.get_canonical_ranks()
        if isinstance(structure, int):
            # Map integer to its rank
            return ranks[structure]
        elif isinstance(structure, np.ndarray):
            if structure.size == 0:
                return structure
            # Convert numpy array element-wise using ranks
            return np.vectorize(lambda x: ranks[x])(structure)
        elif isinstance(structure, list):
            # Recursively process each item in the list
            return [self(s) for s in structure]
        else:
            raise ValueError(f"Unknown type {type(structure)}")


def hartree2kcalmol(hartree: float) -> float:
    """Converts Hartree to kcal/mol."""
    return hartree * 627.509474


def read_multi_xyz(
    xyz_traj_file: str, extract_property_function: None | (str) = None, n_skip: int = 0
) -> tuple:
    """Reads a multi-frame XYZ trajectory file and returns a list of
    coordinates and optionally properties."""
    with open(xyz_traj_file, "r") as f:
        lines = f.readlines()

    n_atoms = int(lines[0])
    frame_size = n_atoms + 2 + n_skip  # Atoms + comment + atom count line
    n_frames = len(lines) // frame_size

    atoms = []
    coords = []
    property = []
    for i in range(n_frames):
        start = i * frame_size + 2  # Skip atom count and metadata lines
        if extract_property_function:
            property.append(extract_property_function(lines[start - 1]))
        frame_coords = [
            list(map(float, line.split()[1:]))
            for line in lines[start : start + n_atoms]
        ]
        coords.append(frame_coords)
        atoms.append([line.split()[0] for line in lines[start : start + n_atoms]])
    if extract_property_function:
        return atoms, coords, property
    return atoms, coords


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
    charge = rdmolops.GetFormalCharge(mol)
    CMD = f"xtb --gfnff {str(tmp_file)} -c {int(charge)} --norestart --wrtopo nb"
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
    use_xtb: bool = False,
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
        _determineConnectivity(rdkit_mol, usextb=use_xtb)
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
