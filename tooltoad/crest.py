import logging
import os
from datetime import datetime

import numpy as np
from joblib import Parallel, delayed
from rdkit import Chem
from tqdm import tqdm

from tooltoad.chemutils import ac2mol, ac2xyz, hartree2kcalmol, xyz2ac
from tooltoad.orca import orca_calculate
from tooltoad.utils import WorkingDir, stream, tqdm_joblib

_logger = logging.getLogger(__name__)


def run_crest(
    atoms: list[str],
    coords: list[list[float]],
    charge: int = 0,
    multiplicity: int = 1,
    n_cores: int = 1,
    calc_dir: None | str = None,
    scr: str = ".",
    keep_files: bool = False,
    bond_constraints: None | list[tuple[int]] = None,
    **crest_kwargs,
):
    crest_kwargs.setdefault("noreftopo", None)
    wd = WorkingDir(root=scr, name=calc_dir)
    # check for fragments
    rdkit_mol = ac2mol(atoms, coords)
    if len(Chem.GetMolFrags(rdkit_mol)) > 1 and "nci" not in [
        s.lower() for s in crest_kwargs.keys()
    ]:
        _logger.warning(
            "Multiple fragments detected. Recommended to run CREST in NCI mode (`nci=True`)."
        )
    with open(wd / "input.xyz", "w") as f:
        f.write(ac2xyz(atoms, coords))

    # crest CREST command
    cmd = f"crest input.xyz --chrg {charge} --uhf {multiplicity-1} --T {n_cores} "
    for key, value in crest_kwargs.items():
        if value is None or value is True:
            cmd += f"--{key} "
        else:
            cmd += f"--{key} {str(value)} "
    # setup constraints
    if bond_constraints:
        _logger.info("Setting up constraints..")
        constaint_str = format_constaints(bond_pair_ids=bond_constraints)
        _logger.debug(constaint_str)
        with open(wd / "detailed.inp", "w") as f:
            f.write(constaint_str)
        cmd += "--cinp detailed.inp "

    cmd += " | tee crest.log"
    _logger.info(f"Running CREST with command: {cmd}")
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["BLIS_NUM_THREADS"] = "1"
    generator = stream(cmd, cwd=str(wd))
    lines = []
    normal_termination = False
    for line in generator:
        _logger.debug(line.rstrip("\n"))
        lines.append(line)
    try:
        with open(wd / "crest_conformers.xyz", "r") as f:
            out_lines = f.readlines()
    except Exception as e:
        _logger.error(f"CREST did not terminate normally.\n{e}")
        _logger.info("".join(lines))
        return None
    for line in lines:
        if "CREST terminated normally" in line:
            normal_termination = True
    if not normal_termination:
        _logger.warning("CREST did not terminate normally.")
    n_atoms = int(out_lines[0].strip())
    xyzs = [
        out_lines[i : i + n_atoms + 2] for i in range(0, len(out_lines), n_atoms + 2)
    ]
    coords = [xyz2ac("".join(xyz))[1] for xyz in xyzs]
    energies = [float(line.strip()) for line in out_lines[1 :: n_atoms + 2]]
    rel_energies = [hartree2kcalmol(e - min(energies)) for e in energies]
    if not keep_files and not calc_dir:
        wd.cleanup()

    results = [
        {"atoms": atoms, "coords": c, "xtb_energy": e}
        for c, e in zip(coords, rel_energies)
    ]
    time = datetime.now()
    results["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    results["timestamp"] = time.timestamp()
    return results


def refine_with_orca(
    crest_out,
    charge=0,
    multiplicity=1,
    options={},
    target="electronic_energy",
    n_cores=1,
    **orca_kwargs,
):
    atoms = crest_out[0]["atoms"]
    all_coords = [r["coords"] for r in crest_out]
    with tqdm_joblib(tqdm(desc="Orca Calculations", total=len(all_coords))):
        results = Parallel(n_jobs=n_cores, prefer="threads")(
            delayed(orca_calculate)(
                atoms=atoms,
                coords=coords,
                charge=charge,
                multiplicity=multiplicity,
                options=options,
                **orca_kwargs,
            )
            for coords in all_coords
        )
    new_energies = np.array([r.get(target, np.nan) for r in results])
    new_energies -= np.nanmin(new_energies)
    for r, e in zip(crest_out, new_energies):
        r["orca_energy"] = hartree2kcalmol(float(e))
    if "opt" in options:
        new_coords = [
            r["opt_coords"] if r["normal_termination"] else None for r in results
        ]
        for r, c in zip(crest_out, new_coords):
            r["xtb_coords"] = r["coords"]
            r["coords"] = c
    # sort by orca energy
    crest_out.sort(key=lambda x: x["orca_energy"])
    return crest_out


def format_constaints(
    atom_ids: list[int] | None = None, bond_pair_ids: list[tuple[int]] | None = None
):
    cstr = "$constrain\n  force constant=0.25 \n"
    if atom_ids:
        raise NotImplementedError
    for pair in bond_pair_ids:
        cstr += f"  distance: {int(pair[0])} {int(pair[1])} auto \n"
    cstr += "$end"
    return cstr


# unified constraints interface in chemutils
