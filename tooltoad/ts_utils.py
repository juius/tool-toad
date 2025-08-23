import logging

import numpy as np
from rdkit import Chem
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
):
    work_dir = WorkingDir(root=scr, name=calc_dir)

    create_scratch_dir(None, parent_path=str(work_dir))
    # write xyz file in scratch dir
    with open(work_dir / "scratch/initial0000.xyz", "w") as f:
        f.write(ac2xyz(atoms, coords))

    create_inpfileq(
        run_name="SSM",
        sm_type="SSM",
        nnodes=20,
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
    n_points=50,
    max_cycle=25,
    n_cores: int = 1,
    scr=".",
):
    assert len(bond_changes) == 1, "Expected exactly one bond change"
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
    pes.xtb(n_cores=n_cores, xtb_options=xtb_options, max_cycle=max_cycle)
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
