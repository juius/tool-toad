import logging
import os
import tempfile
from pathlib import Path
from typing import List

from tooltoad.utils import check_executable, stream

_logger = logging.getLogger("orca")

# see https://www.orcasoftware.de/tutorials_orca/first_steps/parallel.html
ORCA_CMD = "/groups/kemi/julius/opt/orca_5_0_4_linux_x86-64_shared_openmpi411/orca"
SET_ENV = 'env - PATH="/groups/kemi/julius/opt/orca_5_0_4_linux_x86-64_shared_openmpi411:/software/kemi/openmpi/openmpi-4.1.1/bin:$PATH" LD_LIBRARY_PATH="/groups/kemi/julius/opt/orca_5_0_4_linux_x86-64_shared_openmpi411:/software/kemi/openmpi/openmpi-4.1.1/lib:$LD_LIBRARY_PATH"'


def orca_calculate(
    atoms: List[str],
    coords: List[list],
    charge: int = 0,
    multiplicity: int = 1,
    options: dict = {},
    xtra_inp_str: str = None,
    scr: str = ".",
    n_cores: int = 1,
    memory: int = 8,
    output_dir=None,
    orca_cmd: str = ORCA_CMD,
    set_env: str = SET_ENV,
) -> tuple:
    """Runs ORCA calculation.

    Args:
        atoms (List[str]): List of atom symbols.
        coords (List[list]): 3xN list of atom coordinates.
        options (dict): ORCA calculation options.
        scr (str, optional): Path to scratch directory. Defaults to '.'.
        n_cores (int, optional): Number of cores used in calculation. Defaults to 1.

    Returns:
        tuple: (atoms, coords, energy)
    """
    check_executable(orca_cmd)
    if output_dir:
        dir_name = str(Path(scr) / output_dir)
        os.makedirs(dir_name)
    else:
        tempdir = tempfile.TemporaryDirectory(dir=scr, prefix="ORCA_")
        dir_name = tempdir.name
    tmp_scr = Path(dir_name)

    with open(tmp_scr / "input.inp", "w") as f:
        f.write(
            write_orca_input(
                atoms,
                coords,
                charge,
                multiplicity,
                options,
                xtra_inp_str=xtra_inp_str,
                memory=memory,
                n_cores=n_cores,
            )
        )

    # cmd = f'{set_env}; {orca_cmd} input.inp "--bind-to-core" | tee orca.out' # "--oversubscribe" "--use-hwthread-cpus"
    cmd = f'{set_env} /bin/bash -c "{orca_cmd} input.inp "--use-hwthread-cpus" | tee orca.out"'
    _logger.debug(f"Running Orca as: {cmd}")

    # Run Orca, capture an log output
    generator = stream(cmd, cwd=tmp_scr)
    lines = []
    for line in generator:
        lines.append(line)
        _logger.debug(line.rstrip("\n"))

    if normal_termination(lines):
        _logger.debug("Orca calculation terminated normally.")
        properties = ["electronic_energy", "mulliken_charges", "loewdin_charges"]
        if "hirshfeld" in [k.lower() for k in options.keys()]:
            properties.append("hirshfeld_charges")
        if "opt" in [k.lower() for k in options.keys()]:
            properties.append("opt_structure")
        results = get_orca_results(lines, properties=properties)
    else:
        _logger.warning("Orca calculation did not terminate normally.")
        _logger.info("".join(lines))
        results = {}
    if output_dir:
        results["calc_dir"] = dir_name
    return results


def get_header(options: dict, xtra_inp_str: str, memory: int, n_cores: int) -> str:
    """Write Orca header."""

    header = "# Automatically generated ORCA input" + 2 * "\n"

    header += "# Number of cores\n"
    header += f"%pal nprocs {n_cores} end\n"
    header += "# RAM per core\n"
    header += f"%maxcore {int(1024 * memory/n_cores)}" + 2 * "\n"

    for key, value in options.items():
        if (value is None) or (not value):
            header += f"! {key} \n"
        else:
            header += f"! {key}({value}) \n"

    if xtra_inp_str:
        header += "\n" + xtra_inp_str + "\n"

    return header


def write_orca_input(
    atoms: List[str],
    coords: List[list],
    charge: int = 0,
    multiplicity: int = 1,
    options: dict = {},
    xtra_inp_str: str = "",
    memory: int = 4,
    n_cores: int = 1,
) -> str:
    """Write Orca input file."""

    header = get_header(options, xtra_inp_str, memory, n_cores)
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


def normal_termination(lines: List[str]) -> bool:
    """Check if ORCA terminated normally."""
    for line in reversed(lines):
        if line.strip() == "****ORCA TERMINATED NORMALLY****":
            return True
    return False


def read_final_sp_energy(lines: List[str]) -> float:
    """Read final single point energy from ORCA output."""
    for line in reversed(lines):
        if "FINAL SINGLE POINT ENERGY" in line:
            return float(line.split()[-1])
    return None


def read_opt_structure(lines: List[str]) -> tuple:
    """Read optimized structure from ORCA output.

    Args:
        lines (List[str]): ORCA output as a list of strings.

    Returns:
        tuple: (atoms, coords)
    """
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


def read_mulliken_charges(lines: List[str]) -> list:
    """Read Mulliken charges from ORCA output.

    Args:
        lines (List[str]): ORCA output as a list of strings.

    Returns:
        list: Mulliken charges.
    """
    rev_start_idx = 2  # Just a dummy value
    for i, line in enumerate(reversed(lines)):
        if "MULLIKEN ATOMIC CHARGES" in line:
            rev_start_idx = i - 1
            break
    mulliken_charges = []
    for line in lines[-rev_start_idx:]:
        line = line.strip()
        if "Sum of atomic charges" in line:
            break
        charge = float(line.split()[-1])
        mulliken_charges.append(charge)
    return mulliken_charges


def read_loewdin_charges(lines: List[str]) -> list:
    """Read Loewdin charges from ORCA output.

    Args:
        lines (List[str]): ORCA output as a list of strings.

    Returns:
        list: Loewdin charges.
    """
    rev_start_idx = 2  # Just a dummy value
    for i, line in enumerate(reversed(lines)):
        if "LOEWDIN ATOMIC CHARGES" in line:
            rev_start_idx = i - 1
            break
    loewdin_charges = []
    for line in lines[-rev_start_idx:]:
        line = line.strip()
        if len(line) == 0:
            break
        charge = float(line.split()[-1])
        loewdin_charges.append(charge)
    return loewdin_charges


def read_hirshfeld_charges(lines: List[str]) -> list:
    """Read Hirshfeld charges from ORCA output.

    Args:
        lines (List[str]): ORCA output as a list of strings.

    Returns:
        list: Hirshfeld charges.
    """
    rev_start_idx = 2  # Just a dummy value
    for i, line in enumerate(reversed(lines)):
        if "HIRSHFELD ANALYSIS" in line:
            rev_start_idx = i - 6
            break
    hirshfeld_charges = []
    for line in lines[-rev_start_idx:]:
        line = line.strip()
        if len(line) == 0:
            break
        charge = float(line.split()[-2])
        hirshfeld_charges.append(charge)
    return hirshfeld_charges


def get_orca_results(
    lines: List[str],
    properties: List[str] = [
        "electronic_energy",
        "mulliken_charges",
        "loewdin_charges",
    ],
) -> dict:
    """Read results from ORCA output.

    Args:
        lines (List[str]): ORCA output as a list of strings.
        properties (List[str], optional): Properties. Defaults to ["electronic_energy"].


    Returns:
        dict: Results dictionary.
    """
    assert isinstance(lines, list), "Input lines must be a list of strings"

    reader = {
        "electronic_energy": read_final_sp_energy,  # always read this
        "opt_structure": read_opt_structure,  # optional
        "mulliken_charges": read_mulliken_charges,  # always read this
        "loewdin_charges": read_loewdin_charges,  # always read this
        "hirshfeld_charges": read_hirshfeld_charges,  # optional
    }

    if not normal_termination(lines):
        raise ValueError("ORCA did not terminate normally")

    results = {}
    for property in properties:
        results[property] = reader[property](lines)

    if "opt_structure" in properties:
        results["atoms"], results["opt_coords"] = results["opt_structure"]

    return results
