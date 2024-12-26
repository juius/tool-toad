from itertools import combinations

import numpy as np

from tooltoad.orca import orca_calculate


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
    atoms, coords, interaction_indices, orca_options={"XTB2": None}, **kwargs
):
    internal_coord_setup = "\n".join(
        [f"modify_internal {{ B {i[0]} {i[1]} A }} end" for i in interaction_indices]
    )
    active_atoms = " ".join(
        [str(item) for sublist in interaction_indices for item in sublist]
    )

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
    Recalc_Hess 5
    MaxIter 250
    end
    """,
        **kwargs,
    )
    return ts_results


def run_irc(atoms, coords, interaction_indices, orca_options={"XTB2": None}, **kwargs):
    IRC = {"IRC": None}
    IRC.update(orca_options)
    irc_results = orca_calculate(
        atoms,
        coords,
        options=IRC,
        calc_dir="ts_irc",
        xtra_inp_str="""%irc
    MaxIter    100
    InitHess calc_anfreq
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
    pairs, _ = get_ts_active_bonds(coords, mode)
    significant_pairs = {frozenset(pair) for pair in pairs}
    overlap = interaction_ids & significant_pairs
    return len(overlap) > 0, [list(pair) for pair in overlap]
