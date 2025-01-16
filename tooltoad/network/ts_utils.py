import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import rmsd as rmsdlib
from rdkit.ML.Cluster import Butina

from tooltoad.orca import orca_calculate

_logger = logging.getLogger(__name__)


def preoptimize(
    atoms, coords, charge, interactions, orca_options={"XTB2": None}, **kwargs
):
    constraint_str = "%geom\n  Constraints\n"
    for i in interactions:
        constraint_str += f"    {{ B {' '.join([str(idx) for idx in i])} C }} \n"
    constraint_str += "  end\nend"

    PREOPT = {"OPT": None}
    PREOPT.update(orca_options)
    preopt_results = orca_calculate(
        atoms,
        coords,
        charge=charge,
        options=PREOPT,
        xtra_inp_str=constraint_str,
        **kwargs,
    )
    return preopt_results


def locate_ts(
    atoms,
    coords,
    interaction_indices,
    orca_options={"XTB2": None},
    hybrid_hess: bool = False,
    max_iter: int = 250,
    **kwargs,
):
    internal_coord_setup = "\n".join(
        [f"modify_internal {{ B {i[0]} {i[1]} A }} end" for i in interaction_indices]
    )
    active_atoms = " ".join(
        [str(item) for sublist in interaction_indices for item in sublist]
    )

    if hybrid_hess:
        hh_str = f"Hybrid_Hess {{ {active_atoms} }} end"
    else:
        hh_str = ""
    TS = {"OPTTS": None, "freq": None}
    TS.update(orca_options)
    ts_results = orca_calculate(
        atoms,
        coords,
        options=TS,
        xtra_inp_str=f"""%geom
    {internal_coord_setup}
    TS_Active_Atoms {{ {active_atoms} }} end
    TS_Active_Atoms_Factor 1.5
    TS_Mode {{ B {int(interaction_indices[0][0])} {int(interaction_indices[0][1])} }} end
    {hh_str}
    Recalc_Hess 5
    MaxIter {max_iter}
    end
    """,
        **kwargs,
    )
    return ts_results


def run_irc(
    atoms,
    coords,
    orca_options={"XTB2": None},
    max_iter=100,
    **kwargs,
):
    IRC = {
        "IRC": None,
    }  # potentially a bug, need to calc analytical hessian before IRC and read in
    IRC.update(orca_options)
    irc_results = orca_calculate(
        atoms,
        coords,
        options=IRC,
        xtra_inp_str=f"""%irc
    MaxIter    {max_iter}
    end
    """,
        **kwargs,
    )
    return irc_results


def get_ts_active_bonds(coords, mode, distance_cutoff=2.75, projection_threshold=0.5):
    coords = np.asarray(coords)
    mode = np.asarray(mode)
    n_atoms = mode.shape[0]
    mode_flat = mode.reshape(-1, 3)
    atom_pairs = np.array([list(pair) for pair in combinations(range(n_atoms), 2)])
    vectors = coords[atom_pairs[:, 1]] - coords[atom_pairs[:, 0]]
    norms = np.linalg.norm(vectors, axis=1, ord=2)
    valid_mask = (1e-8 < norms) & (norms < distance_cutoff)
    unit_vectors = vectors[valid_mask] / norms[valid_mask, np.newaxis]
    mode_vectors = (
        mode_flat[atom_pairs[valid_mask, 1]] - mode_flat[atom_pairs[valid_mask, 0]]
    )
    # Compute projections using element-wise dot products
    projections = np.einsum("ij,ij->i", mode_vectors, unit_vectors)
    significant_indices = np.abs(projections) > projection_threshold
    return (
        atom_pairs[valid_mask][significant_indices].tolist(),
        projections[significant_indices].tolist(),
    )


def check_ts(coords, mode, interaction_ids):
    interaction_ids = set(frozenset(x) for x in interaction_ids)
    pairs, _ = get_ts_active_bonds(coords, mode)
    significant_pairs = {frozenset(pair) for pair in pairs}
    overlap = interaction_ids & significant_pairs
    return len(overlap) > 0, [list(pair) for pair in overlap]


def wrap_rmsd(atoms1, coords1, atoms2, coords2):
    p_coord_sub = np.array(coords1)
    q_coord_sub = np.array(coords2)
    p_atoms_sub = np.array([rmsdlib.NAMES_ELEMENT[s] for s in atoms1])
    q_atoms_sub = np.array([rmsdlib.NAMES_ELEMENT[s] for s in atoms2])

    # assert np.array_equal(np.sort(p_atoms_sub), np.sort(q_atoms_sub))
    # Recenter to centroid
    p_cent_sub = rmsdlib.centroid(p_coord_sub)
    q_cent_sub = rmsdlib.centroid(q_coord_sub)
    p_coord_sub -= p_cent_sub
    q_coord_sub -= q_cent_sub

    result_rmsd1, q_swap, q_reflection, q_review1 = rmsdlib.check_reflections(
        p_atoms_sub,
        q_atoms_sub,
        p_coord_sub,
        q_coord_sub,
        reorder_method=rmsdlib.reorder_inertia_hungarian,
        rmsd_method=rmsdlib.quaternion_rmsd,
    )
    result_rmsd2, q_swap, q_reflection, q_review2 = rmsdlib.check_reflections(
        p_atoms_sub,
        q_atoms_sub,
        p_coord_sub,
        q_coord_sub,
        reorder_method=rmsdlib.reorder_hungarian,
        rmsd_method=rmsdlib.quaternion_rmsd,
    )
    return min([result_rmsd1, result_rmsd2])


def cluster_conformers(goat_dict, original_ts_coords=None, threshold=0.5):
    atoms = goat_dict["ensemble"]["atoms"]
    coords = list(goat_dict["ensemble"]["coords"])
    # insert coords of original ts at index 0
    if original_ts_coords:
        coords = [original_ts_coords] + coords
    n_confs = len(coords)
    dists = []
    # TODO: parallelize this
    for i in range(n_confs):
        for j in range(i):
            dists.append(wrap_rmsd(atoms, coords[i], atoms, coords[j]))

    clusts = Butina.ClusterData(
        dists, n_confs, threshold, isDistData=True, reordering=True
    )

    return clusts


def ts_conf_search(
    atoms: list[str],
    coords: list[list[float]],
    ts_mode: list[list[float]],
    charge: int = 0,
    multiplicity: int = 1,
    distance_cutoff: float = 2.75,
    n_cores: int = 4,
    memory: int = 16,
    goat_options: dict = {"xtb2": None, "alpb": "water", "goat": None},
    orca_options: dict = {"r2scan-3c": None, "smd": "water"},
    scr: str = ".",
    save_dir: str | None = None,
):
    """Run GOAT conformer search in ORCA with `goat_options`. Bonds
    that are active in the `ts_mode` are constrained. Obtained conformers
    are clustered and re-ranked based on calculations with `orca_options`.

    Args:
        atoms (list[str]): _description_
        coords (list[list[float]]): _description_
        charge (int): _description_
        multiplicity (int): _description_
        ts_mode (list[list[float]]): _description_
        distance_cutoff (float, optional): _description_. Defaults to 2.75.
        n_cores (int, optional): _description_. Defaults to 8.
        memory (int, optional): _description_. Defaults to 24.
        goat_options (_type_, optional): _description_. Defaults to {"xtb2": None, "alpb": "water", "goat": None}.
        orca_options (_type_, optional): _description_. Defaults to {"r2scan-3c": None, "smd": "water"}.
        scr (str, optional): _description_. Defaults to ".".

    Returns:
        list: _description_
    """

    if isinstance(scr, str):
        scr = Path(scr)
        if not scr.exists():
            scr.mkdir(parents=True)

    if save_dir:
        save_dir = Path(save_dir)
        if not save_dir.exists():
            save_dir.mkdir(parents=True)
    active_pairs, _ = get_ts_active_bonds(
        coords, ts_mode, distance_cutoff=distance_cutoff
    )

    def format_orca_constraints(bond_pair_ids: list[tuple[int]]):
        cstr = "%GEOM\n  Constraints\n"
        for pair in bond_pair_ids:
            cstr += f"    {{B {int(pair[0])} {int(pair[1])} C}}\n"
        cstr += "  end\nend"

        return cstr

    assert "goat" in [option.lower() for option in goat_options.keys()]
    _logger.info("Running GOAT calculation")
    goat = orca_calculate(
        atoms,
        coords,
        charge=charge,
        multiplicity=multiplicity,
        xtra_inp_str=format_orca_constraints(active_pairs),
        n_cores=n_cores,
        memory=memory,
        options=goat_options,
        scr=scr,
    )
    if save_dir:
        with open(save_dir / "goat_results.json", "w") as f:
            json.dump(goat, f)

    _logger.info(f"Obtained {len(goat['goat']['ensemble']['coords'])} conformers")
    clusts = cluster_conformers(goat["goat"], original_ts_coords=coords)

    _logger.info(f"Retained {len(clusts)} conformers after RMSD clustering")

    max_tries = 3
    ts_ensemble = []
    for clust_idx, clust in enumerate(clusts):
        if 0 in clust:
            # skip this cluster bc it contains the original TS
            continue

        for i in range(min(max_tries, len(clust))):
            _logger.info(f"Locating TS nr. {i} from cluster {clust_idx}")
            coord_idx = clust[i] - 1  # -1 bc we inserted the original ts at index 0
            atoms, coords = (
                goat["goat"]["ensemble"]["atoms"],
                goat["goat"]["ensemble"]["coords"][coord_idx],
            )

            preopt = preoptimize(
                atoms,
                coords,
                charge=charge,
                multiplicity=multiplicity,
                interactions=active_pairs,
                orca_options=orca_options,
                n_cores=n_cores,
                memory=memory,
                scr=scr,
            )
            if save_dir:
                with open(
                    save_dir / f"preopt_results-{clust_idx}-{coord_idx}.json", "w"
                ) as f:
                    json.dump(preopt, f)
            ts = locate_ts(
                atoms,
                preopt["opt_coords"],
                charge=charge,
                multiplicity=multiplicity,
                max_iter=25,
                interaction_indices=active_pairs,
                orca_options=orca_options,
                n_cores=n_cores,
                memory=memory,
                scr=scr,
            )
            if save_dir:
                with open(
                    save_dir / f"ts_results-{clust_idx}-{coord_idx}.json", "w"
                ) as f:
                    json.dump(ts, f)
            if not ts["normal_termination"] or check_ts(
                ts["opt_coords"], ts["vibs"][0]["mode"], active_pairs
            ):
                _logger.info(f"Found a correct TS in cluster {clust_idx}")
                ts_ensemble.append(ts)
                break
            else:
                _logger.info(f"TS nr. {i} is incorrect, trying next conformer")
        if i == min(max_tries, len(clust)) - 1 and not ts["normal_termination"]:
            _logger.warning(
                f"Did not locate a correct TS in cluster {clust_idx} within {i+1} tries"
            )
    if save_dir:
        with open(save_dir / "ensemble_results.json", "w") as f:
            json.dump(ts_ensemble, f)
    return ts_ensemble
