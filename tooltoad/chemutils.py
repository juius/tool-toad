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
from joblib import Parallel, delayed
from rdkit import Chem
from rdkit.Chem import AllChem  # noqa: F401
from rdkit.Chem import (
    ResonanceMolSupplier,
    rdDetermineBonds,
    rdFMCS,
    rdMolAlign,
    rdMolDescriptors,
    rdmolops,
)
from rdkit.Chem.rdchem import Mol
from rdkit.ML.Cluster import Butina
from sklearn.cluster import DBSCAN

from tooltoad.utils import stream

logger = logging.getLogger(__name__)

COVALENT_RADII = {
    "H": 0.31,
    "C": 0.76,
    "N": 0.71,
    "O": 0.66,
    "F": 0.57,
}


VDW_RADII = {
    "C": 1.7,
    "N": 1.55,
    "O": 1.52,
    "H": 1.2,
    "S": 1.8,
    "F": 1.47,
    # Add more as needed
}

# convert solvent names to canonical names
# at least in xtb 6.7.1 the DCM name doesn't work
CANONICAL_SOLVENT_NAMES = {"xtb": {"dcm": "ch2cl2"}, "orca": {}}


def get_bond_change(reactant, product):
    # cleanup stereo and bond order
    reactant = ac2mol(
        [a.GetSymbol() for a in reactant.GetAtoms()],
        reactant.GetConformer().GetPositions(),
    )
    product = ac2mol(
        [a.GetSymbol() for a in product.GetAtoms()],
        product.GetConformer().GetPositions(),
    )
    # get bond diff
    ac1 = rdmolops.GetAdjacencyMatrix(reactant)
    ac2 = rdmolops.GetAdjacencyMatrix(product)
    diff = ac2 - ac1

    pairs = list(zip(*np.where(np.triu(diff) != 0)))
    bond_changes = [(diff[p], p) for p in pairs]

    return bond_changes


def get_connectivity_smiles(mol):
    m = Chem.Mol(mol)
    Chem.RemoveStereochemistry(m)
    try:
        Chem.Kekulize(m, clearAromaticFlags=True)
    except Exception:
        for a in m.GetAtoms():
            a.SetIsAromatic(False)
        for b in m.GetBonds():
            b.SetIsAromatic(False)

    for a in m.GetAtoms():
        a.SetIsotope(0)
        a.SetAtomMapNum(0)
        a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        a.SetIsAromatic(False)
        a.SetFormalCharge(0)

    for b in m.GetBonds():
        b.SetBondType(Chem.BondType.SINGLE)
        b.SetStereo(Chem.BondStereo.STEREONONE)
        b.SetIsAromatic(False)

    return Chem.MolToSmiles(m, canonical=True, isomericSmiles=False)


def canonicalize_resonance(mol):
    try:
        mol = ResonanceMolSupplier(mol).__next__()
    except StopIteration:
        pass
    return mol


def canonicalize_solvent(solvent: str, qm: str):
    assert qm.lower() in ["xtb", "orca"], "QM must be either xtb or orca"
    if solvent:
        return CANONICAL_SOLVENT_NAMES[qm.lower()].get(solvent.lower(), solvent)


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


class ConformerCalculator:
    """Calculator for parallel QM calculations on conformers of a molecule."""

    qm_functions = ["xtb_calculate", "orca_calculate"]

    def __init__(
        self,
        qm_function,
        options: dict,
        scr: str = ".",
        check_connectivity: bool = True,
    ):
        assert (
            qm_function.__name__ in self.qm_functions
        ), f"QM function {qm_function} not supported."
        self.qm_function = qm_function
        self.options = options
        self.scr = scr
        self.check_connectivity = check_connectivity

    def __call__(
        self,
        mol: Mol,
        multiplicity: None | int = None,
        n_cores: int = 1,
        memory: int = 4,
        constraints: None | list[Constraint] = None,
        xtb_detailed_input_str: None | str = None,
    ):
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        coords = [conf.GetPositions() for conf in mol.GetConformers()]
        charge = Chem.GetFormalCharge(mol)
        if not multiplicity:
            multiplicity = (
                sum([a.GetAtomicNum() for a in mol.GetAtoms()]) - charge
            ) % 2 + 1
        kwargs = {}
        if constraints:
            if self.qm_function.__name__ == "xtb_calculate":
                joined_constraints = "\n".join(c.xtb for c in constraints)
                kwargs[
                    "detailed_input_str"
                ] = f"""$constrain
   {joined_constraints}
$end"""
            elif self.qm_function.__name__ == "orca_calculate":
                joined_constraints = "\n".join(c.orca for c in constraints)
                kwargs[
                    "xtra_inp_str"
                ] = f"""%geom Constraints
        {joined_constraints}
        end
      end"""
            else:
                raise ValueError("QM function not supported")
        if xtb_detailed_input_str:
            kwargs["detailed_input_str"] += xtb_detailed_input_str
        results = Parallel(n_jobs=n_cores)(
            delayed(self.qm_function)(
                atoms=atoms,
                coords=c,
                charge=charge,
                multiplicity=multiplicity,
                options=self.options,
                scr=self.scr,
                # memory=memory,
                n_cores=1,
                **kwargs,
            )
            for c in coords
        )
        new_mol = Chem.Mol(mol)
        new_mol.RemoveAllConformers()
        # sort results by electronic_energy if results["normal_termination"] else np.inf
        results = sorted(
            results,
            key=lambda x: x["electronic_energy"] if x["normal_termination"] else np.inf,
        )
        ac_diffs = []
        for result in results:
            if not result["normal_termination"]:
                logger.warning("Error in QM calculation")
            else:
                if "opt" in self.options:
                    coords = result["opt_coords"]
                    if self.check_connectivity:
                        # check for change in connectivity
                        connectivity_check, ac_diff = same_connectivity(
                            mol,
                            atoms,
                            result["opt_coords"],
                            charge,
                            multiplicity,
                            self.scr,
                        )
                        if not connectivity_check:
                            ac_diffs.append(ac_diff)
                            logger.debug(
                                "Change in connectivity during optimization, skipping conformer."
                            )
                            continue
                else:
                    coords = result["coords"]
                conf = Chem.Conformer(mol.GetNumAtoms())
                conf.SetDoubleProp("electronic_energy", result["electronic_energy"])
                conf.SetPositions(np.array(coords))
                _ = new_mol.AddConformer(conf, assignId=True)
        if len(new_mol.GetConformers()) == 0:
            logger.warning("No conformers found")
            return new_mol, ac_diffs
        return new_mol, None


class MolCalculator:
    """Calculator for parallel QM calculations on list/dict of a molecules."""

    qm_functions = ["xtb_calculate", "orca_calculate"]

    def __init__(
        self,
        qm_function: str,
        options: dict,
        scr: str = ".",
    ):
        assert (
            qm_function.__name__ in self.qm_functions
        ), f"QM function {qm_function} not supported."
        self.qm_function = qm_function
        self.options = options
        self.scr = scr

    def __call__(
        self,
        mols: dict[str, Mol] | list[Mol],
        multiplicities: list[int] = None,
        n_cores: int = 1,
        memory: int = 4,
        xtb_detailed_input_str: None | str = None,
    ):
        dtype = "list"
        if isinstance(mols, dict):
            dtype = "dict"
            labels = list(mols.keys())
            mols = list(mols.values())
        elif not isinstance(mols, list):
            raise ValueError("mols must be a list or dict")
        for mol in mols:
            assert mol.GetNumConformers() == 1, "Molecules must have only one conformer"

        def wrap(
            mol: Mol,
            qm_function: str,
            scr: str,
            multiplicity: int,
            n_cores: int,
        ):
            atoms = [a.GetSymbol() for a in mol.GetAtoms()]
            coords = mol.GetConformer().GetPositions()
            charge = Chem.GetFormalCharge(mol)
            if not multiplicity:
                multiplicity = (
                    sum([a.GetAtomicNum() for a in mol.GetAtoms()]) - charge
                ) % 2 + 1
            results = qm_function(
                atoms=atoms,
                coords=coords,
                charge=charge,
                multiplicity=multiplicity,
                options=self.options,
                scr=scr,
                # memory=memory,
                n_cores=n_cores,
            )
            if results["normal_termination"]:
                mol.SetDoubleProp("electronic_energy", results["electronic_energy"])
            else:
                logger.warning("Error in QM calculation")
                logger.warning(results["log"])
            return results

        results = Parallel(n_jobs=n_cores)(
            delayed(wrap)(
                mol=mol,
                qm_function=self.qm_function,
                scr=self.scr,
                multiplicity=multiplicities[i] if multiplicities else None,
                n_cores=1,
                # memory=memory,
            )
            for i, mol in enumerate(mols)
        )
        for res, mol in zip(results, mols):
            if res["normal_termination"]:
                mol.SetDoubleProp("electronic_energy", res["electronic_energy"])
            else:
                logger.warning("Error in QM calculation")
                logger.warning(res["log"])
                mol.SetDoubleProp("electronic_energy", np.inf)
        if dtype == "dict":
            return {labels[i]: mol for i, mol in enumerate(mols)}
        else:
            return mols


def same_connectivity(
    mol: Mol,
    atoms: list[str],
    opt_coords: list[list[float]],
    charge: int,
    multiplicity: int,
    scr: str,
) -> tuple[bool, None | np.ndarray]:
    """Check if the connectivity of the molecule is the same before and after
    optimization."""
    ac1 = rdmolops.GetAdjacencyMatrix(mol)
    new_mol = ac2mol(atoms, opt_coords, charge, multiplicity, scr, sanitize=False)
    ac2 = rdmolops.GetAdjacencyMatrix(new_mol)
    if (ac1 == ac2).all():
        return True, None
    else:
        logger.debug("Connectivity changed")
        return False, ac2 - ac1


def reorder_product_atoms(mol: Mol) -> Mol:
    """Reorder product atom and bond indices to match reactant indices."""
    # Collect (reactant_idx, product_idx) pairs
    product2reactant = []
    idx_counter = mol.GetNumAtoms()
    for atom in mol.GetAtoms():
        idx = atom.GetIdx()
        if atom.HasProp("react_atom_idx"):
            reactant_idx = atom.GetIntProp("react_atom_idx")
        else:
            reactant_idx = idx_counter
            idx_counter += 1
        product2reactant.append((idx, reactant_idx))

    new_order = [idx for idx, _ in sorted(product2reactant, key=lambda x: x[1])]
    renumbered = Chem.RenumberAtoms(mol, new_order)

    for prop_name in mol.GetPropNames():
        renumbered.SetProp(prop_name, mol.GetProp(prop_name))

    for conf in mol.GetConformers():
        for prop_name in conf.GetPropNames():
            renumbered.GetConformer(conf.GetId()).SetProp(
                prop_name, conf.GetProp(prop_name)
            )

    for old_idx, new_idx in enumerate(new_order):
        old_atom = mol.GetAtomWithIdx(old_idx)
        new_atom = renumbered.GetAtomWithIdx(new_idx)
        for prop_name in old_atom.GetPropNames():
            new_atom.SetProp(prop_name, old_atom.GetProp(prop_name))

    for bond in mol.GetBonds():
        old_begin = bond.GetBeginAtomIdx()
        old_end = bond.GetEndAtomIdx()
        new_begin = new_order.index(old_begin)
        new_end = new_order.index(old_end)

        new_bond = renumbered.GetBondBetweenAtoms(new_begin, new_end)
        if new_bond is not None:
            for prop_name in bond.GetPropNames():
                new_bond.SetProp(prop_name, bond.GetProp(prop_name))

    return renumbered


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


def energy_filter_conformer(mol: Mol, cutoff_kcalmol: float = 5.0) -> Mol:
    """Only retain conformers with energy within `cutoff_kcalmol` of lowest energy conformer.
    Requires conformers to have 'electronic_energy' property set."""
    # make sure conformers are sorted by energy
    cIds = [conf.GetId() for conf in mol.GetConformers()]
    energies = [conf.GetDoubleProp("electronic_energy") for conf in mol.GetConformers()]
    relative_energies = [
        hartree2kcalmol(e - min(energies)) for e in energies
    ]  # convert to kcal/mol
    # sort conformers by energy
    sorted_cIds = [x for _, x in sorted(zip(relative_energies, cIds))]
    sorted_energies = sorted(relative_energies)
    new_mol = Chem.Mol(mol)
    new_mol.RemoveAllConformers()
    for cId, energy in zip(sorted_cIds, sorted_energies):
        if energy <= cutoff_kcalmol:
            conf = mol.GetConformer(cId)
            new_conf = Chem.Conformer(mol.GetNumAtoms())
            new_conf.SetDoubleProp(
                "electronic_energy", conf.GetDoubleProp("electronic_energy")
            )
            new_conf.SetPositions(conf.GetPositions())
            new_mol.AddConformer(new_conf, assignId=True)
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


# @hide_warnings
def _determineConnectivity(mol, usextb=False, **kwargs):
    """Determine bonds in molecule."""
    if usextb:
        atoms = [a.GetSymbol() for a in mol.GetAtoms()]
        coords = mol.GetConformer().GetPositions()
        charge = rdmolops.GetFormalCharge(mol)
        multiplicity = kwargs.get("multiplicity", 1)
        scr = kwargs.get("scr", ".")
        adj = gfnff_connectivity(atoms, coords, charge, multiplicity, scr)
        emol = Chem.EditableMol(mol)
        for i, j in np.argwhere(adj):
            if i > j:
                emol.AddBond(int(i), int(j), Chem.BondType.SINGLE)
        mol = emol.GetMol()
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
    multiplicity: int = 1,
    scr: str = ".",
    perceive_connectivity: bool = True,
    use_xtb: bool = True,
    sanitize: bool = False,
):
    """Converts atom symbols and coordinates to RDKit molecule."""
    xyz = ac2xyz(atoms, coords)
    rdkit_mol = Chem.MolFromXYZBlock(xyz)
    if sanitize:
        Chem.SanitizeMol(rdkit_mol)
    if perceive_connectivity:
        rdkit_mol = _determineConnectivity(
            rdkit_mol, usextb=use_xtb, charge=charge, multiplicity=multiplicity, scr=scr
        )
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


def gfnff_connectivity(atoms, coords, charge, multiplicity, scr):
    # Determine connectivity based on GFNFF-xTB implementation
    calc_dir = tempfile.TemporaryDirectory(dir=scr)
    tmp_file = Path(calc_dir.name) / "input.xyz"
    with open(tmp_file, "w") as f:
        f.write(ac2xyz(atoms, coords))
    CMD = f"xtb --gfnff {str(tmp_file.name)} --chrg {charge} --uhf {multiplicity-1} --norestart --wrtopo blist"
    _ = list(stream(CMD, cwd=calc_dir.name))
    with open(Path(calc_dir.name) / "gfnff_lists.json", "r") as f:
        data_dict = json.load(f)
    calc_dir.cleanup()
    blist = data_dict["blist"]
    adj = np.zeros((len(atoms), len(atoms)), dtype=int)
    for i, j, _ in blist:
        adj[i - 1, j - 1] = 1
        adj[j - 1, i - 1] = 1
    return adj
