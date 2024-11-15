import itertools
import json

from tooltoad.chemutils import ac2xyz
from tooltoad.ndscan import PotentialEnergySurface, ScanCoord
from tooltoad.network import Universe
from tooltoad.network.ts_utils import locate_ts, run_irc

starting_species = ["O=C=O", "NCC", "O"]
n_steps = 9
n_cores = 24
for pair in list(itertools.combinations(starting_species, 2)) + list(
    itertools.combinations(starting_species, 3)
):
    print(f"Running {pair}")
    soup = Universe.from_smiles(pair, radius=3.5)
    soup.find_ncis(n_cores=n_cores)
    for conf_id in range(soup.n_conformers):
        interactions = soup.get_interactions(conf_id=conf_id)
        if interactions.shape[0] > 0:
            atoms = soup.atoms
        coords = soup.coords[0][conf_id]
        scan_coords = [
            ScanCoord.from_current_position(atoms, coords, atom_ids=ids, nsteps=n_steps)
            for ids in interactions
        ]
        pes = PotentialEnergySurface(
            atoms, coords, charge=soup.charge, scan_coords=scan_coords
        )
        pes.xtb(
            xtb_options={"gfn": 2, "alpb": "water"},
            n_cores=n_cores,
            max_cycle=50,
            force_constant=1.0,
        )
        pes.refine(orca_options={"r2scan-3c": None, "SMD": "water"}, n_cores=n_cores)
        point_type = "maxima" if interactions.shape[0] == 1 else "saddle"
        stationary_points = pes.find_stationary_points(refined=True)
        if len(stationary_points) > 0:
            for sp in stationary_points:
                ts_guess_coords = pes.traj_tensor(sp["idx"])
                with open(f"{pair}_{point_type}_{conf_id}.xyz", "w") as f:
                    f.write(ac2xyz(atoms, ts_guess_coords))

                ts_results = locate_ts(
                    atoms,
                    ts_guess_coords,
                    interaction_indices=interactions,
                    orca_options={"r2scan-3c": None, "SMD": "water"},
                    n_cores=n_cores,
                )
                with open(f"{pair}_{point_type}_{conf_id}_ts.log", "w") as f:
                    json.dump(ts_results, f)

                if ts_results["normal_termination"]:
                    irc_results = run_irc(
                        atoms,
                        ts_results["opt_coords"],
                        interaction_indices=interactions,
                        orca_options={"r2scan-3c": None, "SMD": "water"},
                        n_cores=n_cores,
                    )
                    with open(f"{pair}_{point_type}_{conf_id}_irc.log", "w") as f:
                        json.dump(irc_results, f)
