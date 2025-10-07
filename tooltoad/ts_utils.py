import logging

import numpy as np
from rdkit.Chem import rdDetermineBonds
from scipy.signal import find_peaks

from tooltoad.chemutils import ac2mol, ac2xyz
from tooltoad.gsm import (
    create_inpfileq,
    create_isomers,
    create_ograd,
    create_scratch_dir,
)
from tooltoad.ndscan import PotentialEnergySurface, ScanCoord
from tooltoad.utils import WorkingDir, stream

_logger = logging.getLogger(__name__)


def sort_start_end(
    reactant, product, energy_key: str = "electronic_energy", reverse=True
):
    pair = [reactant, product]
    if reactant.GetNumBonds() != product.GetNumBonds():
        pair.sort(key=lambda x: x.GetNumBonds(), reverse=reverse)
    else:
        try:
            pair.sort(key=lambda x: x.GetDoubleProp(energy_key), reverse=reverse)
        except KeyError:
            print(f"Key '{energy_key}' not found in molecule properties.")
    return pair


def get_ssm_ts_guess(
    atoms,
    coords,
    bond_changes,
    charge=0,
    multiplicity=1,
    orca_options={"XTB2": None, "alpb": "water"},
    scr=".",
    calc_dir=None,
    orca_cmd="orca",
    gsm_executable="gsm",
    execute=True,
    nnodes: int = 20,
):
    work_dir = WorkingDir(root=scr, name=calc_dir)

    create_scratch_dir(None, parent_path=str(work_dir))
    # write xyz file in scratch dir
    with open(work_dir / "scratch/initial0000.xyz", "w") as f:
        f.write(ac2xyz(atoms, coords))

    create_inpfileq(
        run_name="SSM",
        sm_type="SSM",
        nnodes=nnodes,
        parent_path=str(work_dir),
    )
    create_isomers(bond_changes, parent_path=str(work_dir))

    create_ograd(
        orca_option_string=" ".join(
            f"{key}({value})" if value else f"{key}"
            for key, value in orca_options.items()
        ),
        orca_path=orca_cmd,
        charge=charge,
        multiplicity=multiplicity,
        parent_path=str(work_dir),
    )
    if execute:
        generator = stream(gsm_executable, str(work_dir))
        for line in generator:
            _logger.debug(line.strip("\n"))
        if (work_dir / "scratch/tsq0000.xyz").exists():
            _logger.info("TSQ file created successfully.")
            tsq = Chem.MolFromXYZFile(str(work_dir / "scratch/tsq0000.xyz"))
            rdDetermineBonds.DetermineConnectivity(tsq)
            return tsq

    else:
        return work_dir


def get_scan_ts_guess(
    atoms,
    coords,
    bond_changes,
    charge=0,
    multiplicity=1,
    xtb_options={"alpb": "water"},
    orca_options=None,
    n_points=50,
    max_cycle=25,
    n_cores: int = 1,
    scr=".",
):
    assert len(bond_changes) == 1, "Expected exactly one bond change"
    # either xtb or orca options must be provided
    assert (
        xtb_options or orca_options
    ), "Either xtb_options or orca_options must be provided"
    scs = [
        ScanCoord.from_current_position(
            atoms,
            coords,
            bond_changes[0][1],
            n_points,
            bool(bond_changes[0][0]),
        )
    ]
    pes = PotentialEnergySurface(atoms, coords, charge, scan_coords=scs)
    if xtb_options:
        pes.xtb(n_cores=n_cores, xtb_options=xtb_options, max_cycle=max_cycle)
    elif orca_options:
        pes.orca(n_cores=n_cores, orca_options=orca_options, max_cycle=max_cycle)
    else:
        raise ValueError("Either xtb_options or orca_options must be provided")
    peaks = []
    prominence = 0.1
    max_tries = 100
    while len(peaks) == 0 and max_tries > 0:
        peaks, _ = find_peaks(pes.pes_tensor, prominence=prominence)
        prominence /= 2
        max_tries -= 1
    if max_tries == 0:
        raise ValueError("No peaks found in the potential energy surface.")
    local_maximum = peaks[np.argmax(pes.pes_tensor[peaks])]
    ts_guess_coords = pes.traj_tensor[local_maximum]
    ts_guess = ac2mol(atoms, ts_guess_coords)
    rdDetermineBonds.DetermineConnectivity(ts_guess)
    return ts_guess


if __name__ == "__main__":
    import json
    import os
    from pathlib import Path
    from sys import argv

    from rdkit import Chem
    from rdkit.Chem import rdmolops

    from tooltoad.chemutils import get_bond_change

    reactant_file = argv[1]
    product_file = argv[2]
    reverse = "--reverse" in argv
    calc_dir = Path(os.getcwd())

    with open(reactant_file, "r") as f:
        reactant_data = json.load(f)
    with open(product_file, "r") as f:
        product_data = json.load(f)
    reactant = Chem.MolFromMolBlock(reactant_data["data"]["gfn2-xtb"], removeHs=False)
    product = Chem.MolFromMolBlock(product_data["data"]["gfn2-xtb"], removeHs=False)
    charge = rdmolops.GetFormalCharge(product)
    start, end = sort_start_end(reactant, product, reverse=reverse)
    bond_changes = get_bond_change(start, end)
    print(f"Reaction between {start} and {end}")
    print(f"Bond Changes: {bond_changes}")
    atoms = [a.GetSymbol() for a in start.GetAtoms()]
    coords = start.GetConformer().GetPositions()
    success = False
    assert (
        len(bond_changes) > 0
    ), "No bond changes detected between reactant and product."

    get_ssm_ts_guess(
        atoms,
        coords,
        bond_changes,
        charge=charge,
        multiplicity=1,
        orca_options={"r2SCAN-3c": None, "SMD": "water"},
        scr=".",
        calc_dir=calc_dir,
        gsm_executable="~/opt/GSM/gsm.orca",
        execute=False,
    )
