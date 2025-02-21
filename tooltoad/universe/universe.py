import json
import logging
from dataclasses import asdict, dataclass, field

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdMolTransforms

from tooltoad.chemutils import (
    _determineConnectivity,
    ac2mol,
    ac2xyz,
    hartree2kcalmol,
    xyz2ac,
)
from tooltoad.utils import WorkingDir, stream
from tooltoad.vis import draw3d
from tooltoad.xtb import xtb_calculate

from .utils import fibonacci_sphere, get_rotation_matrix

_logger = logging.getLogger(__name__)


@dataclass
class Universe:
    atoms: list[str]
    coords: np.ndarray
    init_topology: None | np.ndarray = None
    traj_topology: None | np.ndarray = None
    charge: int = 0
    multiplicity: int = 1
    cavity_radius: None | float = None
    settings: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.coords.ndim == 2:
            self.coords = self.coords[np.newaxis, np.newaxis, :, :]
        assert (
            len(self.atoms) == self.coords.shape[2]
        ), "Missmatch between number of atoms and coordinates"
        assert (
            self.coords.ndim == 4
        ), "Coordinates should be 2D array with shape (n_frames, n_conformers, n_atoms, 3)"
        if self.init_topology is None:
            self.determine_topology()
        self.frag_ids = [np.asarray(ids) for ids in Chem.GetMolFrags(self.to_rdkit())]

    def __repr__(self):
        return f"Universe with {len(self.atoms)} atoms"

    @property
    def n_frames(self):
        return self.coords.shape[0]

    @property
    def n_conformers(self):
        return self.coords.shape[1]

    @property
    def n_atoms(self):
        return self.coords.shape[2]

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, "r") as f:
            data = json.load(f)

        # Convert lists back to numpy arrays for the appropriate fields
        for key in ["coords", "init_topology", "traj_topology"]:
            if key in data and data[key] is not None:
                data[key] = np.array(data[key])

        return cls(**data)

    @classmethod
    def from_dict(cls, results_dict: dict) -> "Universe":
        assert results_dict[
            "atoms"
        ], f"Dict is missing 'atoms' key: {list(results_dict.keys())}"
        assert results_dict[
            "coords"
        ], f"Dict is missing 'coords' key: {list(results_dict.keys())}"
        return cls(
            atoms=results_dict["atoms"],
            coords=results_dict["coords"],
            cavity_radius=results_dict["cavity_radius"],
            settings=results_dict["settings"],
        )

    @classmethod
    def from_rdkit(cls, mol: Chem.Mol, cId: int = 0):
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        coords = mol.GetConformer(cId).GetPositions()
        charge = Chem.GetFormalCharge(mol)
        ac = Chem.GetAdjacencyMatrix(mol)
        return cls(atoms=atoms, coords=coords, charge=charge, init_topology=ac)

    @classmethod
    def from_smiles(
        cls,
        smiles_list: list[str],
        radius: float = 5.0,
        random_seed: int = -1,
        xtb_optimize: bool = True,
        xtb_options: dict = {},
    ) -> "Universe":
        """Initializes a Universe from a list of SMILES strings by embedding
        them on a sphere.

        Parameters:
            smiles_list (list[str]): List of SMILES strings.
            radius (float): Radius of the sphere on which to embed the molecules.
            random_seed (int): Random seed for reproducibility.
            xtb_optimize (bool): Whether to optimize the molecules with xTB.
            xtb_options (dict): Options to pass to xTB.

        Returns:
            Universe: Initialized Universe object with embedded molecules.
        """
        molecules = []
        charges = []
        for smi in smiles_list:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smi}")
            mol = Chem.AddHs(mol)
            rdDistGeom.EmbedMolecule(mol, randomSeed=random_seed)
            if xtb_optimize:
                atoms = [a.GetSymbol() for a in mol.GetAtoms()]
                coords = mol.GetConformer().GetPositions()
                charge = Chem.GetFormalCharge(mol)
                charges.append(charge)
                xtb_options.setdefault("opt", None)
                opt_results = xtb_calculate(
                    atoms=atoms, coords=coords, charge=charge, options=xtb_options
                )
                mol.GetConformer().SetPositions(
                    np.asarray(opt_results["opt_coords"], dtype=np.double)
                )
            molecules.append(mol)

        all_atoms, all_coords = cls.position_fragments(molecules, radius, random_seed)
        # TODO: check for distances between molecules too small
        return cls(atoms=all_atoms, coords=all_coords, charge=sum(charges))

    def save(self, file_path: str):
        with open(file_path, "w") as f:
            json.dump(
                {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in asdict(self).items()
                },
                f,
            )

    def to_rdkit(self):
        return ac2mol(self.atoms, self.coords[0][0], charge=self.charge)

    @staticmethod
    def position_fragments(
        fragments: list[Chem.Mol], radius: float = 5.0, random_seed: int = -1
    ):
        # Generate uniformly distributed points on the sphere
        if len(fragments) == 1:
            positions = np.array([[0, 0, 0]])
        else:
            positions = fibonacci_sphere(len(fragments), radius)
        rng = np.random.default_rng(random_seed if random_seed > 0 else None)
        rng.shuffle(positions)

        all_atoms = []
        all_coords = []
        for mol, pos in zip(fragments, positions):
            mol_coords = (
                mol.GetConformer().GetPositions()
                - rdMolTransforms.ComputeCentroid(mol.GetConformer())
            )
            # Apply random rotation
            angles = rng.random(3) * 2 * np.pi
            rotation_matrix = get_rotation_matrix(angles)
            rotated_coords = mol_coords @ rotation_matrix.T
            all_coords.append(rotated_coords + pos[np.newaxis, :])
            all_atoms.extend([atom.GetSymbol() for atom in mol.GetAtoms()])
        return np.asarray(all_atoms), np.concatenate(all_coords)

    def determine_topology(self):
        self.init_topology = Chem.GetAdjacencyMatrix(
            _determineConnectivity(self.to_rdkit())
        )

    def relax(self, options: dict = {}, **xtb_kwargs):
        options.setdefault("opt", None)
        mol = self.to_rdkit()
        atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
        coords = mol.GetConformer().GetPositions()
        charge = Chem.GetFormalCharge(mol)
        results = xtb_calculate(
            atoms=atoms, coords=coords, charge=charge, options=options, **xtb_kwargs
        )
        self.coords = np.asarray(results["opt_coords"], dtype=np.double)[
            np.newaxis, np.newaxis, :, :
        ]

    def find_ncis(
        self,
        n_cores: int = 1,
        energy_threshold: float = 5.0,
        scr: str = ".",
        **crest_kwargs,
    ):
        working_dir = WorkingDir(root=scr)
        # run crest with nci
        with open(working_dir / "universe.xyz", "w") as f:
            f.write(ac2xyz(self.atoms, self.coords[0][0]))
        cmd = f"crest universe.xyz --nci --chrg {self.charge} --T {n_cores} "
        for k, v in crest_kwargs.items():
            cmd += f"--{k} {v} "
        cmd += " | tee crest.log"
        generator = stream(cmd, cwd=str(working_dir))
        lines = []
        for line in generator:
            _logger.debug(line.rstrip("\n"))
        # TODO: check for normal termination, etc

        with open(working_dir / "crest_conformers.xyz", "r") as f:
            lines = f.readlines()
        n_atoms = int(lines[0].strip())
        xyzs = [lines[i : i + n_atoms + 2] for i in range(0, len(lines), n_atoms + 2)]
        coords = np.array([xyz2ac("".join(xyz))[1] for xyz in xyzs])
        energies = np.array([float(line.strip()) for line in lines[1 :: n_atoms + 2]])
        relative_energies = hartree2kcalmol(energies - np.min(energies))
        relevant_coords = coords[relative_energies <= energy_threshold]
        _logger.info(
            f"CREST found {len(relevant_coords)} NCI conformers within {energy_threshold} kcal/mol"
        )
        self.coords = relevant_coords[np.newaxis, :, :, :]
        working_dir.cleanup()

    def show(self, conf_id: int = 0, frame_id: int = 0, **draw3d_kwargs):
        draw3d_kwargs.setdefault("width", 500)
        draw3d_kwargs.setdefault("height", 500)
        view = draw3d(
            ac2mol(self.atoms, self.coords[frame_id][conf_id]), **draw3d_kwargs
        )
        if self.cavity_radius:
            view.addSphere(
                {
                    "center": {"x": 0, "y": 0, "z": 0},
                    "radius": self.cavity_radius,
                    "color": "blue",
                    "alpha": 0.4,
                    "wireframe": True,
                }
            )
        view.zoomTo()
        return view
