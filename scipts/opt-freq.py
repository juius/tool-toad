import json
import os
from pathlib import Path

import click
import numpy as np
import submitit
from rdkit import Chem

from tooltoad.orca import orca_calculate

LEVELS = {
    "quick": {"opt": "XTB2", "sp": "R2SCAN-3C"},
    "normal": {"opt": "R2SCAN-3C", "sp": "wB97X-3C"},
}


# SETUP

executor = submitit.AutoExecutor(folder="dft")
executor.update_parameters(
    timeout_min=6000, slurm_partition="kemi1", slurm_array_parallelism=250
)

data_dir = Path(os.getenv("DATA_DIR", "/groups/kemi/julius/data"))
data_dir.mkdir(exist_ok=True)

N_CORES = 16
MEMORY_PER_CORE = 2  # GB


def optimize(
    name, atoms, coords, charge, multiplicity, level, solvent=None, n_cores=1, memory=4
):
    scratch = os.getenv("SCRATCH", ".")
    options = {"opt": None, LEVELS[level]["opt"]: None}
    if solvent:
        options["smd"] = solvent
    results = orca_calculate(
        atoms=atoms,
        coords=coords,
        charge=charge,
        multiplicity=multiplicity,
        options=options,
        scr=scratch,
        n_cores=n_cores,
        memory=memory,
        log_file=(data_dir / (str(name) + f"-opt-{level}.log")).absolute(),
    )
    if results["normal_termination"]:
        # write the json file to the data directory
        json_file = data_dir / (str(name) + f"-opt-{level}.json")
        with open(json_file, "w") as f:
            json.dump(results["json"], f, indent=4)
    return results


def frequencies(
    name, atoms, coords, charge, multiplicity, level, solvent=None, n_cores=1, memory=4
):
    scratch = os.getenv("SCRATCH", ".")
    options = {"freq": None, LEVELS[level]["opt"]: None}
    if solvent:
        options["smd"] = solvent
    results = orca_calculate(
        atoms=atoms,
        coords=coords,
        charge=charge,
        multiplicity=multiplicity,
        options=options,
        scr=scratch,
        n_cores=n_cores,
        memory=memory,
        log_file=(data_dir / (str(name) + f"-freq-{level}.log")).absolute(),
    )
    if results["normal_termination"]:
        # write the json file to the data directory
        json_file = data_dir / (str(name) + f"-freq-{level}.json")
        with open(json_file, "w") as f:
            json.dump(results["json"], f, indent=4)
    return results


def singlepoint(
    name, atoms, coords, charge, multiplicity, level, solvent=None, n_cores=1, memory=4
):
    scratch = os.getenv("SCRATCH", ".")
    options = {LEVELS[level]["sp"]: None}
    if solvent:
        options["smd"] = solvent
    results = orca_calculate(
        atoms=atoms,
        coords=coords,
        charge=charge,
        multiplicity=multiplicity,
        options=options,
        scr=scratch,
        n_cores=n_cores,
        memory=memory,
        log_file=(data_dir / (str(name) + f"-sp-{level}.log")).absolute(),
    )
    if results["normal_termination"]:
        # write the json file to the data directory
        json_file = data_dir / (str(name) + f"-sp-{level}.json")
        with open(json_file, "w") as f:
            json.dump(results["json"], f, indent=4)
    return results


@click.command()
@click.argument("sdf", type=click.Path(exists=True, dir_okay=False))
@click.argument("name", type=str)
@click.option(
    "--level",
    type=click.Choice(["quick", "normal"], case_sensitive=False),
    default="normal",
    help="Level of theory to use for the calculation",
)
@click.option(
    "--multiplicity", type=int, default=1, help="Multiplicity of the molecule"
)
@click.option(
    "--solvent", type=str, default=None, help="Solvent to use for the calculation"
)
@click.option(
    "--resource-multiplier",
    type=int,
    default=1,
    help="Resource multiplier for the calculation",
)
def optfreq(
    sdf, name, level="normal", multiplicity=1, solvent=None, resource_multiplier=1
):
    """Main function to process the SDF file and perform optimization and
    frequency calculations."""
    click.echo(f"Processing {sdf} at level {level}")
    mol = Chem.MolFromMolFile(sdf, removeHs=False)
    atoms = [a.GetSymbol() for a in mol.GetAtoms()]
    coords = mol.GetConformer().GetPositions()
    charge = Chem.GetFormalCharge(mol)
    click.echo(f"Name: {name}")
    click.echo(f"Solvent: {solvent}")

    n_cores = int(N_CORES * resource_multiplier)
    memory = int(N_CORES * MEMORY_PER_CORE * resource_multiplier)
    memory_freq = memory * 2  # Frequencies require more memory

    click.echo(
        f"Running geometry optimization using {n_cores} cores and {memory} GB memory."
    )
    executor.update_parameters(
        slurm_cpus_per_task=n_cores, slurm_mem_gb=memory, job_name=f"{name}-opt"
    )
    job = executor.submit(
        optimize,
        name,
        atoms,
        coords,
        charge,
        multiplicity,
        level,
        solvent,
        n_cores,
        memory,
    )
    opt = job.result()
    if not opt["normal_termination"]:
        click.echo(
            f"Optimization did not terminate normally, see log file {(data_dir / (str(name) + f'-opt-{level}.log')).absolute()}. Exiting."
        )
        return
    mol.SetProp("l1", LEVELS[level]["opt"])
    mol.GetConformer().SetPositions(np.array(opt["opt_coords"]))
    mol.SetDoubleProp("l1_electronic-energy", opt["electronic_energy"])

    click.echo(
        f"Running frequency calculation using {n_cores} cores and {memory_freq} GB memory."
    )
    executor.update_parameters(
        slurm_cpus_per_task=n_cores, slurm_mem_gb=memory_freq, job_name=f"{name}-freq"
    )
    job = executor.submit(
        frequencies,
        name,
        atoms,
        coords,
        charge,
        multiplicity,
        level,
        solvent,
        n_cores,
        memory,
    )
    freq = job.result()
    if not freq["normal_termination"]:
        click.echo(
            f"Frequency calculation did not terminate normally, see log file {(data_dir / (str(name) + f'-freq-{level}.log')).absolute()}. Exiting."
        )
        return
    mol.SetDoubleProp("l1_gibbs-energy", freq["gibbs_energy"])
    mol.SetDoubleProp(
        "l1_gibbs-correction", freq["gibbs_energy"] - freq["electronic_energy"]
    )
    executor.update_parameters(
        slurm_cpus_per_task=n_cores, slurm_mem_gb=memory, job_name=f"{name}-sp"
    )
    click.echo(
        f"Running single point calculation using {n_cores} cores and {memory} GB memory."
    )
    job = executor.submit(
        singlepoint,
        name,
        atoms,
        coords,
        charge,
        multiplicity,
        level,
        solvent,
        n_cores,
        memory,
    )
    sp = job.result()
    if not sp["normal_termination"]:
        click.echo(
            f"Single point calculation did not terminate normally, see log file {(data_dir / (str(name) + f'-sp-{level}.log')).absolute()}. Exiting."
        )
        return

    mol.SetProp("l2", LEVELS[level]["sp"])
    mol.SetDoubleProp("l2_electronic-energy", sp["electronic_energy"])
    mol.SetDoubleProp(
        "l2l1_gibbs-energy",
        sp["electronic_energy"] + freq["gibbs_energy"] - freq["electronic_energy"],
    )

    with Chem.SDWriter((data_dir / (str(name) + f"-{level}.sdf")).absolute()) as writer:
        writer.write(mol)

    return mol


if __name__ == "__main__":
    optfreq()
