import logging
import math
import tempfile
from pathlib import Path

from tooltoad.utils import stream

_logger = logging.getLogger("orca")
_logger.setLevel(logging.INFO)

ORCA_CMD = "/software/kemi/Orca/orca_5_0_1_linux_x86-64_openmpi411/orca"
SET_ENV = "export PATH=/software/kemi/Orca/orca_5_0_1_linux_x86-64_openmpi411:/software/kemi/openmpi/openmpi-4.1.1/bin:$PATH; export LD_LIBRARY_PATH=/software/kemi/openmpi/openmpi-4.1.1/lib:$LD_LIBRARY_PATH"


def get_header(options, memory, n_cores):
    """Write Orca header."""

    header = "# Automatically generated ORCA input" + 2 * "\n"

    header += "# Number of cores\n"
    header += f"%pal nprocs {n_cores} end\n"
    header += "# RAM per core\n"
    header += f"%maxcore {1024 * memory}" + 2 * "\n"

    for key, value in options.items():
        if (value is None) or (not value):
            header += f"! {key} \n"
        else:
            header += f"! {key}({value}) \n"

    return header


def write_orca_input(
    atoms,
    coords,
    options,
    charge=0,
    multiplicity=1,
    memory=4,
    n_cores=1,
):
    header = get_header(options, memory, n_cores)
    inputstr = header + 2 * "\n"

    # charge, spin, and coordinate section
    inputstr += f"*xyz {charge} {multiplicity} \n"
    for atom_str, coord in zip(atoms, coords):
        inputstr += (
            f"{atom_str}".ljust(5)
            + " ".join(["{:.8f}".format(x).rjust(15) for x in coord])
            + "\n"
        )
    inputstr += "*\n"
    inputstr += "\n"  # magic line

    return inputstr


def normal_termination(lines):
    for line in reversed(lines):
        if line.strip() == "****ORCA TERMINATED NORMALLY****":
            return True
    return False


def read_final_sp_energy(lines):
    for line in reversed(lines):
        if "FINAL SINGLE POINT ENERGY" in line:
            return float(line.split()[-1])
    return None


def read_opt_structure(lines):
    rev_start_idx = 2  # Just a dummy value
    for i, line in enumerate(reversed(lines)):
        if "CARTESIAN COORDINATES (ANGSTROEM)" in line:
            rev_start_idx = i - 1
            break

    atoms = []
    coords = []
    for line in lines[-rev_start_idx:]:
        line = line.strip()
        if len(line) > 0:
            atom, x, y, z = line.split()
            atoms.append(atom)
            coords.append([float(x), float(y), float(z)])
        else:
            break
    return atoms, coords


def get_orca_results(lines, properties=["electronic_energy"]):

    assert isinstance(lines, list), "Input lines must be a list of strings"

    results = {}
    reader = {
        "electronic_energy": read_final_sp_energy,
        "opt_structure": read_opt_structure,
    }

    if not normal_termination(lines):
        raise ValueError("ORCA did not terminate normally")

    for property in properties:
        results[property] = reader[property](lines)

    return results


def orca_calculate(atoms, coords, options, scr, n_cores=1):

    tempdir = tempfile.TemporaryDirectory(dir=scr, prefix="ORCA_")
    tmp_scr = Path(tempdir.name)

    with open(tmp_scr / "input.inp", "w") as f:
        f.write(
            write_orca_input(atoms, coords, options, memory=n_cores, n_cores=n_cores)
        )

    cmd = f"{SET_ENV}; {ORCA_CMD} input.inp | tee output.out"
    _logger.debug(f"Running Orca as: {cmd}")
    out = list(stream(cmd, cwd=tmp_scr))

    if normal_termination(out):
        _logger.info("Orca calculation terminated normally.")
        results = get_orca_results(
            out, properties=["electronic_energy", "opt_structure"]
        )
        energy = results["electronic_energy"]
        atoms, coords = results["opt_structure"]
    else:
        _logger.warning("Orca calculation did not terminate normally.")
        _logger.info("".join(out))
        energy = math.nan

    return atoms, coords, energy
