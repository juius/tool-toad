import itertools
import json
import os

import submitit

from tooltoad.chemutils import ac2xyz
from tooltoad.ndscan import PotentialEnergySurface, ScanCoord
from tooltoad.network import Universe
from tooltoad.network.ts_utils import locate_ts, preoptimize

L1 = {"gfn2": None, "alpb": "water", "opt": None}
L2 = {
    "r2scan-3c": None,
    "SMD": "water",
}

NCI_N_CORES = 24
TS_N_CORES = 24
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


def wrap_ts_localization(atoms, coords, charge, interactions, name, n_cores=12):
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
    pes.refine(orca_options=L2, n_cores=n_cores)
    point_type = "maxima" if len(interactions) == 1 else "saddle"
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
    preopt_results = preoptimize(atoms, ts_guess_coords, charge, interactions)
    with open(f"{name}-preopt.xyz", "w") as f:
        f.write(ac2xyz(atoms, preopt_results["opt_coords"]))

    print(f"Running TS search for {name} in {scratch}")
    ts_results = locate_ts(
        atoms,
        preopt_results["opt_coords"],
        interactions,
        orca_options=L2,
        n_cores=n_cores,
        scr=scratch,
    )
    with open(f"{name}-ts.json", "w") as f:
        json.dump(ts_results, f)
    return ts_results


if __name__ == "__main__":
    executor = submitit.AutoExecutor(folder="job_logs")
    executor.update_parameters(
        name="crest",
        cpus_per_task=NCI_N_CORES,
        slurm_mem_per_cpu=f"{NCI_N_CORES*2}GB",
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
        slurm_mem_per_cpu=f"{NCI_N_CORES*2}GB",
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
