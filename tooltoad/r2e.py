import argparse
import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import (
    rdChemReactions,
    rdDistGeom,
    rdForceFieldHelpers,
    rdMolDescriptors,
)
from rdkit.Chem.rdchem import Bond, Mol
from rdkit.Geometry import Point3D

from tooltoad.chemutils import (
    ConformerCalculator,
    Constraint,
    MolCalculator,
    canonicalize_solvent,
    energy_filter_conformer,
    filter_conformers,
    hartree2kcalmol,
    reorder_product_atoms,
)
from tooltoad.orca import orca_calculate
from tooltoad.xtb import xtb_calculate

_logger = logging.getLogger(__name__)


def get_all_ring_bonds(mol: Mol, bond_idx: int) -> list[Bond]:
    """Retrieves all bonds in rings that include the specified bond index."""
    all_ring_bonds = []
    for ring in Chem.GetSymmSSSR(mol):
        ring_atoms = list(ring)
        ring_bonds = []

        # Loop over pairs of consecutive atoms in the ring (including wraparound)
        for i in range(len(ring_atoms)):
            a1 = ring_atoms[i]
            a2 = ring_atoms[(i + 1) % len(ring_atoms)]
            b = mol.GetBondBetweenAtoms(a1, a2)
            if b is not None:
                ring_bonds.append(b)

        # Check if the input bond is part of this ring
        if any(b.GetIdx() == bond_idx for b in ring_bonds):
            all_ring_bonds.extend(ring_bonds)
    return all_ring_bonds


def get_bonds_to_constrain(mol: Mol, ac_diff: list[list[int]]) -> list[list[int]]:
    """Get bonds that broke during optimization and need to be constrained."""
    ac_diff = np.asarray(ac_diff)
    # bond constraints to prevent bond breaking
    indices = np.argwhere(ac_diff < 0)[:, 1:]
    unique_bonds_breaking = set([tuple(sorted(li)) for li in indices.tolist()])
    additional_bonds = []
    for b in unique_bonds_breaking:
        _logger.info(f"Bond between {b[0]} and {b[1]} broke during optimization")
        if mol.GetBondBetweenAtoms(*b).IsInRing():
            ring_bonds = get_all_ring_bonds(mol, mol.GetBondBetweenAtoms(*b).GetIdx())
            _logger.info(
                f"Bond between {b[0]} and {b[1]} is part of {len(ring_bonds)} membered ring"
            )
            _logger.info("therefore also constraining bonds between atoms")
            for rbond in ring_bonds:
                _logger.info(f"{rbond.GetBeginAtomIdx()} and {rbond.GetEndAtomIdx()}")
                additional_bonds.append(
                    tuple(sorted([rbond.GetBeginAtomIdx(), rbond.GetEndAtomIdx()]))
                )
    unique_additional_bonds = set(additional_bonds)
    unique_bonds_breaking = unique_bonds_breaking.union(unique_additional_bonds)
    return [list(b) for b in unique_bonds_breaking]


def generate_reactant(
    reactant_smi: str, n_cores: int = 4, xtb_options: dict[str, None | bool | str] = {}
) -> Mol:
    """Generate a 3D conformer for a reactant from its SMILES string."""
    reactant = Chem.MolFromSmiles(reactant_smi)
    n_rot_bonds = rdMolDescriptors.CalcNumRotatableBonds(reactant)
    reactant3d = Chem.AddHs(reactant)
    ps = rdDistGeom.ETKDGv3()
    ps.trackFailures = True
    ps.useRandomCoords = True
    ps.pruneRmsThresh = 0.1
    ps.numThreads = n_cores
    ps.randomSeed = 42
    cIds = rdDistGeom.EmbedMultipleConfs(
        reactant3d,
        numConfs=min(100, 3 * n_rot_bonds + 3),
    )
    if len(cIds) == 0:
        for i, k in enumerate(rdDistGeom.EmbedFailureCauses.names):
            _logger.debug(k, ps.GetFailureCounts()[i])
    if rdForceFieldHelpers.MMFFHasAllMoleculeParams(reactant3d):
        _ = rdForceFieldHelpers.MMFFOptimizeMoleculeConfs(
            reactant3d, numThreads=n_cores
        )
    elif rdForceFieldHelpers.UFFHasAllMoleculeParams(reactant3d):
        _ = rdForceFieldHelpers.UFFOptimizeMoleculeConfs(reactant3d, numThreads=n_cores)
    else:
        raise ValueError("No force field available for optimization")
    reactant3d = filter_conformers(reactant3d, numThreads=n_cores, rmsdThreshold=0.1)
    xtb_ff = ConformerCalculator(
        xtb_calculate, {"opt": True, "gfn": "ff"} | xtb_options, scr="."
    )
    reactant3d, _ = xtb_ff(reactant3d, n_cores=n_cores)
    reactant3d = filter_conformers(reactant3d, numThreads=n_cores, rmsdThreshold=0.1)
    reactant3d = energy_filter_conformer(reactant3d, cutoff_kcalmol=5)
    xtb_gfn2 = ConformerCalculator(
        xtb_calculate, {"opt": True, "gfn": "2"} | xtb_options, scr="."
    )

    reactant3d, _ = xtb_gfn2(reactant3d, n_cores=n_cores)
    reactant3d = energy_filter_conformer(
        reactant3d, cutoff_kcalmol=0
    )  # only retain minimum energy structure
    reactant3d.SetDoubleProp(
        "electronic_energy",
        reactant3d.GetConformer().GetDoubleProp("electronic_energy"),
    )  # set conformers energy as molecules energy
    return reactant3d


def generate_products_from_reactant(
    reactant3d: Mol,
    reaction_smarts: str,
    n_cores: int = 4,
    xtb_options: dict[str, None | bool | str] = {},
    rmsd_pruning_threshold: float = 0.1,
    conf_search: bool = False,
) -> dict[int, Mol]:
    """Generate 3D conformers for products from a reactant and a reaction
    SMARTS.

    Returns a dictionary of products with the reaction center atom index
    as key. The reaction center is defined as the atom with the
    atomMapNum 1 in the reaction SMARTS.
    """
    rxn = rdChemReactions.ReactionFromSmarts(reaction_smarts)
    xtb_ff = ConformerCalculator(
        xtb_calculate, {"opt": True, "gfn": "ff"} | xtb_options, scr="."
    )
    xtb_gfn2 = ConformerCalculator(
        xtb_calculate, {"opt": True, "gfn": "2"} | xtb_options, scr="."
    )
    results = {}

    products = rxn.RunReactants(
        (reactant3d,),
    )
    unique_products = []
    visited = set()
    for product in products:
        _logger.debug("Processing product")
        product = product[0]
        Chem.SanitizeMol(product)
        smi = Chem.MolToSmiles(product)
        if smi not in visited:
            visited.add(smi)
            reaction_center = [
                a.GetIntProp("react_atom_idx")
                for a in product.GetAtoms()
                if a.HasProp("old_mapno") and a.GetIntProp("old_mapno") == 1
            ][0]
            product.SetIntProp("reaction_center", reaction_center)
            unique_products.append(product)
        else:
            continue

    for product in unique_products:
        if Chem.DetectChemistryProblems(product):
            _logger.debug("Chemistry problems detected in product")
            _logger.debug(Chem.MolToSmiles(product))
            continue

        Chem.SanitizeMol(product)
        product = reorder_product_atoms(product)
        if conf_search:
            coord_map = {}
            freeze_ids = []
            numConfs = 250
        else:
            numConfs = 50
            prc = product.GetIntProp("reaction_center")
            prc_neighbors = [
                a.GetIdx() for a in product.GetAtomWithIdx(prc).GetNeighbors()
            ]
            freeze_exclude = [prc] + prc_neighbors
            freeze_ids = [
                a.GetIdx()
                for a in product.GetAtoms()
                if a.HasProp("react_atom_idx") and a.GetIdx() not in freeze_exclude
            ]
            # constrained
            coords = reactant3d.GetConformer().GetPositions()
            coord_map = {}
            for idx in freeze_ids:
                coord_map[idx] = Point3D(*coords[idx])

        product.RemoveAllConformers()
        try:
            ps = rdDistGeom.ETKDGv3()
            ps.trackFailures = True
            ps.useRandomCoords = True
            ps.pruneRmsThresh = rmsd_pruning_threshold
            ps.numThreads = n_cores
            ps.randomSeed = 42
            ps.forceTransAmides = False
            ps.SetCoordMap(coord_map)
            cIds = rdDistGeom.EmbedMultipleConfs(
                product,
                numConfs=numConfs,
            )
        except RuntimeError as e:
            _logger.debug("Error in embedding conformers:", e)
            continue
        if len(cIds) == 0:
            _logger.debug("No conformers generated")
            for i, k in enumerate(rdDistGeom.EmbedFailureCauses.names):
                _logger.debug(k, ps.GetFailureCounts()[i])
        if rdForceFieldHelpers.MMFFHasAllMoleculeParams(reactant3d):
            mp = rdForceFieldHelpers.MMFFGetMoleculeProperties(product)
            ff = rdForceFieldHelpers.MMFFGetMoleculeForceField(product, mp)
            for i in freeze_ids:
                ff.MMFFAddPositionConstraint(i, 0, 1.0e4)
        elif rdForceFieldHelpers.UFFHasAllMoleculeParams(reactant3d):
            ff = rdForceFieldHelpers.UFFGetMoleculeForceField(product)
            for i in freeze_ids:
                ff.UFFAddDistanceConstraint(i, 0, 1.0e4)
        else:
            raise ValueError("No force field available for optimization")

        _ = rdForceFieldHelpers.OptimizeMoleculeConfs(
            product, ff, numThreads=n_cores, maxIters=1000
        )
        product = filter_conformers(product, numThreads=n_cores, rmsdThreshold=0.1)
        calc_product, ac_diff_ff = xtb_ff(
            product,
            n_cores=n_cores,
        )
        if calc_product.GetNumConformers() == 0:
            _logger.info("GFN-FF optimization failed for all conformers.")
            _logger.info("Trying optimization with constraints on bond")
            bonds2constrain = get_bonds_to_constrain(product, ac_diff_ff)
            constraints = [Constraint(bond, None) for bond in bonds2constrain]
            # try again with constraints
            calc_product, ac_diff_gfnff = xtb_ff(
                product, n_cores=n_cores, constraints=constraints
            )
            if (np.asarray(ac_diff_ff) == 0).sum() > 0:
                _logger.info(
                    "Still, things changed when constraining, skipping product"
                )
                continue
        product = filter_conformers(calc_product, numThreads=n_cores, rmsdThreshold=0.1)
        product = energy_filter_conformer(product, cutoff_kcalmol=5)
        calc_product, ac_diff_gfn2 = xtb_gfn2(
            product,
            n_cores=n_cores,
        )
        if calc_product.GetNumConformers() == 0:
            _logger.info("GFN-2 optimization failed for all conformers.")
            _logger.info("Trying optimization with constraints on bond")
            bonds2constrain = get_bonds_to_constrain(product, ac_diff_gfn2)
            constraints = [Constraint(bond, None) for bond in bonds2constrain]
            # try again with constraints
            calc_product, ac_diff_gfn2 = xtb_gfn2(
                product, n_cores=n_cores, constraints=constraints
            )
            if (np.asarray(ac_diff_gfn2) == 0).sum() > 0:
                _logger.info(
                    "Still, things changed when constraining, skipping product"
                )
                continue
        product = energy_filter_conformer(
            calc_product, cutoff_kcalmol=0
        )  # only retain minimum energy structure
        product.SetDoubleProp(
            "electronic_energy",
            product.GetConformer().GetDoubleProp("electronic_energy"),
        )  # set conformers energy as molecules energy
        results[product.GetIntProp("reaction_center")] = product
    return results


def calculate_reaction_energies(
    reactant_and_products: dict[str, Mol],
    offset: float = 0.0,
    orca_options: dict[str, None | bool | str] = {"r2scan-3c": None},
    n_cores: int = 4,
) -> dict[str, float]:
    """Performs ORCA calculations on all structures in dict and calculates
    reaction energies.

    The dictionary needs to contain a key 'reactant'.
    """
    assert "reactant" in reactant_and_products, "'reactant' must be in the input dict"
    mc = MolCalculator(orca_calculate, options=orca_options, scr=".")

    annotated = mc(reactant_and_products, n_cores=n_cores)

    energies = {k: v.GetDoubleProp("electronic_energy") for k, v in annotated.items()}
    reactant_energy = energies.pop("reactant")
    reaction_energies = {
        k: hartree2kcalmol(v - (reactant_energy + offset)) for k, v in energies.items()
    }
    return reaction_energies


if __name__ == "__main__":
    _logger = logging.getLogger("r2e")
    logging.basicConfig(level=logging.INFO, format="%(name)s: %(message)s")

    parser = argparse.ArgumentParser(description="Run r2e calculations.")
    parser.add_argument(
        "--smi",
        type=str,
        required=True,
        help="SMILES string of the reactant.",
    )
    parser.add_argument(
        "--rxn-smarts",
        type=str,
        required=True,
        help="SMARTS string of the reaction. Reaction center atom should have atomMapNum 1.",
    )
    parser.add_argument(
        "--solvent",
        type=str,
        help="Solvent for the reaction, used in xTB and ORCA calculations (ALPB/SMD).",
    )
    parser.add_argument(
        "--n-cores",
        type=int,
        default=4,
        help="Number of cores to use for calculations.",
    )
    parser.add_argument(
        "--only-xtb",
        action="store_true",
        help="If set, only use xtb for calculations, no ORCA required.",
    )

    args = parser.parse_args()
    smi = args.smi
    rxn_smarts = args.rxn_smarts
    solvent = args.solvent
    n_cores = args.n_cores
    only_xtb = args.only_xtb

    xtb_solvent = canonicalize_solvent(solvent, "xtb")
    orca_solvent = canonicalize_solvent(solvent, "orca")

    assert (
        ":1]" in rxn_smarts
    ), 'Reaction SMARTS must have reaction center atom marked with atomMapNum 1, f.x. "[#7:1]>>[#7+:1]-[O-]".'

    print(
        f"Running relative reaction energy calculations for {smi} with reaction {rxn_smarts} in {solvent if solvent else 'gas phase'}."
    )
    reactant3d = generate_reactant(smi, n_cores=n_cores)
    products = generate_products_from_reactant(
        reactant3d,
        rxn_smarts,
        xtb_options={"alpb": xtb_solvent} if xtb_solvent else {},
        n_cores=n_cores,
    )
    if only_xtb:
        reaction_energies = {}
        for idx, product in products.items():
            reaction_energies[idx] = hartree2kcalmol(
                product.GetDoubleProp("electronic_energy")
                - reactant3d.GetDoubleProp("electronic_energy")
            )
    else:
        reaction_energies = calculate_reaction_energies(
            {"reactant": reactant3d, **products},
            orca_options={"r2scan-3c": None}
            | ({"SMD": orca_solvent} if orca_solvent else {}),
            n_cores=n_cores,
        )
    print("\nRelative reaction energies (reaction site: energy):")
    values = list(reaction_energies.values())
    for k, v in reaction_energies.items():
        print(f"{k}: {v - min(values):.2f} kcal/mol")
