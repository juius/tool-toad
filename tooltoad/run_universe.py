from pathlib import Path

from joblib import Parallel, delayed

from tooltoad.ndscan import PotentialEnergySurface, ScanCoord
from tooltoad.network import Reaction, Universe
from tooltoad.orca import orca_calculate

ORCA_CMD = "/Users/julius/Library/orca_6_0_0/orca"
OPEN_MPI_DIR = "/usr/local/openmpi-4.1.1/"
ORCA_DIR = Path(ORCA_CMD).parent
SET_ENV = f'env - PATH="{ORCA_DIR}:{OPEN_MPI_DIR}/bin:$PATH"  XTBEXE="xtb" LD_LIBRARY_PATH="{OPEN_MPI_DIR}/lib:$LD_LIBRARY_PATH" DYLD_LIBRARY_PATH="{OPEN_MPI_DIR}/lib:$DYLD_LIBRARY_PATH"'


n_cores = 8
nsteps = 12


def main(input_smi: str):
    uni = Universe.from_smiles(["O=C=O", "O", input_smi], radius=4)
    uni.find_ncis(n_cores=n_cores)

    conf_ids, interaction_ids = uni.get_unique_interactions()
    atoms = uni.atoms

    ts_guess_coords = []
    for cid, interactions in zip(conf_ids, interaction_ids):
        coords = uni.coords[0][cid]
        if len(interactions) == 0:
            ts_guess_coords.append(None)
            continue
        scan_coords = [
            ScanCoord.from_current_position(atoms, coords, i, nsteps)
            for i in interactions
        ]
        pes = PotentialEnergySurface(atoms, coords, scan_coords=scan_coords)

        pes.xtb(n_cores=n_cores)

        sp = []
        tolerance = 1e-4
        while len(sp) == 0:
            sp = pes.find_stationary_points(
                prune=True, point_type="saddle", tolerance=tolerance
            )
            tolerance += 1e-5
            if tolerance > 1:
                break
        if len(sp) > 0:
            ts_guess_coords.append(pes.traj_tensor[sp[0]["idx"]])
        else:
            ts_guess_coords.append(None)

    clean_ts_coords = [c for c in ts_guess_coords if c is not None]
    clean_interaction_ids = [
        i for i, c in zip(interaction_ids, ts_guess_coords) if c is not None
    ]

    def locate_ts(
        atoms, coords, interaction_indices, orca_options={"XTB2": None}, **kwargs
    ):
        internal_coord_setup = "\n".join(
            [
                f"modify_internal {{ B {i[0]} {i[1]} A }} end"
                for i in interaction_indices
            ]
        )
        TS = {"OPTTS": None, "freq": None}
        TS.update(orca_options)
        ts_results = orca_calculate(
            atoms,
            coords,
            options=TS,
            xtra_inp_str=f"""%geom
        {internal_coord_setup}
        Recalc_Hess 1
        MaxIter 250
        end
        """,
            **kwargs,
        )
        return ts_results

    def run_irc(atoms, coords, orca_options={"XTB2": None}, **kwargs):
        IRC = {"IRC": None}
        IRC.update(orca_options)
        irc_results = orca_calculate(
            atoms,
            coords,
            options=IRC,
            xtra_inp_str="""%irc
        MaxIter    100
        InitHess calc_anfreq
        end
        """,
            **kwargs,
        )
        return irc_results

    ts_results = Parallel(n_jobs=8)(
        delayed(locate_ts)(atoms, c, i, orca_cmd=ORCA_CMD, set_env=SET_ENV)
        for c, i in zip(clean_ts_coords, clean_interaction_ids)
    )

    irc_results = Parallel(n_jobs=n_cores)(
        delayed(run_irc)(atoms, r["opt_coords"], orca_cmd=ORCA_CMD, set_env=SET_ENV)
        for r in ts_results
    )

    reactions = [
        Reaction(
            {"gfn2": irc["irc"]["backward"]},
            {"gfn2": irc["irc"]["forward"]},
            {"gfn2": ts},
        )
        for irc, ts in zip(irc_results, ts_results)
    ]

    for i, r in enumerate(reactions):
        r.save(f"r-{input_smi}-{i:03d}.json")


if __name__ == "__main__":
    import sys

    main(sys.argv[1])
