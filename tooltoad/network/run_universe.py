import itertools
import json
import logging
import os
from pathlib import Path

import submitit

from tooltoad.chemutils import EnsembleCluster, ac2xyz
from tooltoad.ndscan import PotentialEnergySurface, ScanCoord
from tooltoad.network import Universe
from tooltoad.network.ts_utils import (
    check_ts,
    get_ts_active_bonds,
    locate_ts,
    preoptimize,
)
from tooltoad.orca import orca_calculate

LOGGING_SETTINGS = {
    "level": logging.INFO,
    "handler": logging.StreamHandler(),
}

orca_logger = logging.getLogger("orca")
chemutils_logger = logging.getLogger("chemutils")
universe_logger = logging.getLogger("universe")
ndscan_logger = logging.getLogger("ndscan")

for logger in [orca_logger, chemutils_logger, universe_logger, ndscan_logger]:
    logger.setLevel(LOGGING_SETTINGS["level"])
    logger.addHandler(LOGGING_SETTINGS["handler"])


def save_results(results: dict, file_path: str | Path) -> str:
    with open(file_path, "w") as f:
        json.dump(results, f, indent=2)
    return file_path


L1 = {"gfn2": None, "alpb": "water", "opt": None}
L2 = {
    "r2scan-3c": None,
    "SMD": "water",
}

NCI_N_CORES = 8
TS_N_CORES = 8
MEMORY = 24
SLURM_PARTITION = "kemi1"


def wrap_find_nci(pair, n_cores=6):
    name = "|".join(pair)
    scratch = os.getenv("SCRATCH", ".")
    print(f"Running NCI search for {name} in {scratch}")
    soup = Universe.from_smiles(pair, radius=3)
    soup.find_ncis(n_cores=n_cores)
    soup.save(f"soup-{name}.json")
    interactions = [
        soup.get_interactions(cutoff=0.75, conf_id=idx)
        for idx in range(soup.n_conformers)
    ]
    print(f"Found interactions: {interactions}")
    return soup.atoms, soup.coords[0], soup.charge, interactions, name


def wrap_ts_localization(
    atoms,
    coords,
    charge,
    interactions,
    name,
    n_cores=12,
):
    scratch = os.getenv("SCRATCH", ".")
    nsteps = max([9, 18 - (len(interactions) - 1) * 6])
    scan_coords = [
        ScanCoord.from_current_position(atoms, coords, atom_ids=ids, nsteps=nsteps)
        for ids in interactions
    ]
    print(f"Scanning: {scan_coords} with xtb in {scratch}")
    pes = PotentialEnergySurface(atoms, coords, charge=charge, scan_coords=scan_coords)
    pes.xtb(
        xtb_options=L1,
        n_cores=n_cores,
        max_cycle=50,
        force_constant=1.0,
        scr=scratch,
    )
    print(f"Refining PES in {scratch}")
    pes.refine(orca_options=L2, n_cores=n_cores, scr=scratch)
    point_type = "maxima" if len(interactions) == 1 else "saddle"
    print(f"Looking for {point_type}")
    stationary_points = pes.find_stationary_points(
        refined=True, point_type=point_type, min_samples=len(interactions)
    )
    print(f"Found stationary points: {stationary_points}")
    if len(stationary_points) > 0:
        for sp in stationary_points:
            ts_guess_coords = pes.traj_tensor[sp["idx"]]
            with open(f"{name}-tsguess.xyz", "w") as f:
                f.write(ac2xyz(atoms, ts_guess_coords))
    print(f"Running pre-optimization for {name} in {scratch}")
    preopt_results = preoptimize(
        atoms,
        ts_guess_coords,
        charge,
        interactions,
        orca_options=L2,
        scr=scratch,
        n_cores=n_cores,
        memory=MEMORY,
    )
    with open(f"{name}-preopt.xyz", "w") as f:
        f.write(ac2xyz(atoms, preopt_results["opt_coords"]))

    print(f"Running TS search for {name} in {scratch}")
    ts_results = locate_ts(
        atoms,
        preopt_results["opt_coords"],
        interactions,
        orca_options=L2,
        n_cores=n_cores,
        memory=MEMORY,
        scr=scratch,
    )
    with open(f"{name}-ts.json", "w") as f:
        json.dump(ts_results, f)
    # check if TS search was successful
    if not ts_results["normal_termination"]:
        print(f"TS search failed, see {name}-ts.json for details")
        return ts_results
    if not check_ts(
        ts_results["opt_coords"], ts_results["vibs"][0]["mode"], interactions
    )[0]:
        print(f"TS search didn't yield the expected TS, see {name}-ts.json for details")
        return ts_results
    ts_conf_search_results = wrap_ts_conf_search(
        atoms=ts_results["atoms"],
        coords=ts_results["opt_coords"],
        charge=charge,
        multiplicity=1,
        name=name,
        # multiplicity=ts_results["multiplicity"],
        ts_mode=ts_results["vibs"][0]["mode"],
        n_cores=n_cores,
        memory=MEMORY,
        scr=scratch,
    )
    with open(f"{name}-ts_conf.json", "w") as f:
        # list of dicts to json
        tmp = {
            f"conf_{i}": ts_conf_search_results[i]
            for i in range(len(ts_conf_search_results))
        }
        json.dump(tmp, f)

    return ts_conf_search_results


def wrap_ts_conf_search(
    atoms: list[str],
    coords: list[list[float]],
    charge: int,
    multiplicity: int,
    ts_mode: list[list[float]],
    name: str,
    distance_cutoff: float = 2.75,
    n_cores: int = 8,
    memory: int = 24,
    goat_options: dict = {"xtb2": None, "alpb": "water", "goat": None},
    orca_options: dict = {"r2scan-3c": None, "smd": "water"},
    scr: str = ".",
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
        _type_: _description_
    """

    if isinstance(scr, str):
        scr = Path(scr)
        if not scr.exists():
            scr.mkdir(parents=True)
    active_pairs, _ = get_ts_active_bonds(
        coords, ts_mode, distance_cutoff=distance_cutoff
    )

    def format_orca_constraints(bond_pair_ids: list[tuple[int]]):
        cstr = "%GEOM\n  Constraints\n"
        for pair in bond_pair_ids:
            cstr += f"    {{B {int(pair[0])} {int(pair[1])} C}}\n"
        cstr += "  end\nend"

        return cstr

    print("Running GOAT calculation")
    goat = orca_calculate(
        atoms,
        coords,
        charge=charge,
        multiplicity=multiplicity,
        xtra_inp_str=format_orca_constraints(active_pairs),
        n_cores=n_cores,
        memory=memory,
        options=goat_options,
    )
    print("Clustering GOAT results")
    ec = EnsembleCluster.from_goat(goat)
    clustered_coords = ec()
    res = []
    for cluster_idx, coords in enumerate(clustered_coords):
        # TODO: skip the cluster that contains the original TS
        print(f"Preoptimize for cluster {cluster_idx}")
        preopt = preoptimize(
            atoms,
            coords,
            charge=charge,
            interactions=active_pairs,
            orca_options=orca_options,
            n_cores=n_cores,
            memory=memory,
        )
        save_results(preopt, scr / f"{name}-preopt-cluster-{cluster_idx}.json")
        print(f"TS search for cluster {cluster_idx}")
        result = locate_ts(
            atoms,
            preopt["opt_coords"],
            interaction_indices=active_pairs,
            orca_options=orca_options,
            n_cores=n_cores,
            memory=memory,
        )
        res.append(result)
        save_results(result, scr / f"{name}-ts-cluster-{cluster_idx}.json")
    print("Done with TS conf search")
    return res


if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder="job_logs")
    executor.update_parameters(
        name="crest",
        cpus_per_task=NCI_N_CORES,
        slurm_mem_per_cpu=f"{MEMORY}GB",
        timeout_min=240,
        slurm_partition=SLURM_PARTITION,
    )
    starting_species = ["O=C=O", "NCC", "O"]
    pairs = list(itertools.combinations(starting_species, 2)) + list(
        itertools.combinations(starting_species, 3)
    )
    jobs_nci = [executor.submit(wrap_find_nci, pair, NCI_N_CORES) for pair in pairs]

    executor.update_parameters(
        name="ts_search",
        cpus_per_task=TS_N_CORES,
        slurm_mem_per_cpu=f"{MEMORY}GB",
        timeout_min=360,
    )

    # Track Task B jobs and their results
    job_pes = []
    # Wait for Task A to complete and submit Task B as soon as any Task A finishes
    while jobs_nci:
        # Check for completed Task A jobs
        for job in jobs_nci:
            if job.done():  # If the job is done
                (
                    atoms,
                    nci_coords,
                    charge,
                    interactions,
                    name,
                ) = job.result()  # Get the result of the completed job
                print(f"NCI task for {name} finished")
                for i, (coords, interaction_ids) in enumerate(
                    zip(nci_coords, interactions)
                ):
                    if len(interaction_ids) == 0:
                        print(f"Skipping TS job {name}|CONF{i}")
                        continue
                    job_p = executor.submit(
                        wrap_ts_localization,
                        atoms,
                        coords,
                        charge,
                        interaction_ids,
                        f"{name}-c{i}",
                        TS_N_CORES,
                    )
                    print(f"Submitting TS job {name}|CONF{i}")
                    job_pes.append(job_p)

                # Remove completed Task A job from the list
                jobs_nci.remove(job)
                break  # Break to avoid re-checking the same completed job

    # Wait for all Task B jobs to complete
    for job_p in job_pes:
        print(job_p.result())  # This will block until Task B jobs finish
