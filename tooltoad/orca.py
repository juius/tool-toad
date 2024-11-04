import logging
from typing import List

import numpy as np

from tooltoad.utils import WorkingDir, check_executable, stream

_logger = logging.getLogger("orca")

# see https://www.orcasoftware.de/tutorials_orca/first_steps/parallel.html
ORCA_CMD = "/groups/kemi/julius/opt/orca_5_0_4_linux_x86-64_shared_openmpi411/orca"
SET_ENV = 'export PATH="/groups/kemi/julius/opt/orca_5_0_4_linux_x86-64_shared_openmpi411:/software/kemi/openmpi/openmpi-4.1.1/bin:$PATH" LD_LIBRARY_PATH="/groups/kemi/julius/opt/orca_5_0_4_linux_x86-64_shared_openmpi411:/software/kemi/openmpi/openmpi-4.1.1/lib:$LD_LIBRARY_PATH"'


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
    calc_dir: str = None,
    orca_cmd: str = ORCA_CMD,
    set_env: str = SET_ENV,
) -> dict:
    """Runs ORCA calculation.

    Args:
        atoms (List[str]): List of atom symbols.
        coords (List[list]): 3xN list of atom coordinates.
        charge (int): Formal charge of molecule.
        multiplicity (int): Spin multiplicity of molecule.
        options (dict): ORCA calculation options.
        xtra_inp_str (str): Additional input string to append after # header.
        scr (str, optional): Path to scratch directory. Defaults to '.'.
        n_cores (int, optional): Number of cores used in calculation. Defaults to 1.
        memory (int, optional): Available memory in GB. Defaults to 8.
        calc_dir (str, optional): Name of calculation directory, will be removed after calculation is None. Defaults to None.
        orca_cmd (str): Path to ORCA executable.
        set_env (str): Command to set environmental variables.


    Returns:
         dict: {'atoms': ..., 'coords': ..., ...}
    """
    check_executable(orca_cmd)
    work_dir = WorkingDir(root=scr, name=calc_dir)

    with open(work_dir / "input.inp", "w") as f:
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
    cmd = f'/bin/bash -c "{set_env} {orca_cmd} input.inp "--use-hwthread-cpus" | tee orca.out"'
    _logger.debug(f"Running Orca as: {cmd}")

    # Run Orca, capture an log output
    generator = stream(cmd, cwd=str(work_dir))
    lines = []
    for line in generator:
        lines.append(line)
        _logger.debug(line.rstrip("\n"))

    if normal_termination(lines):
        clean_option_keys = [k.lower() for k in options.keys()]
        _logger.debug("Orca calculation terminated normally.")
        properties = ["electronic_energy"]
        if "cosmors" in clean_option_keys:
            # charges not printed when COSMO-RS is used
            properties.pop(2)
            properties.pop(1)
            properties.append("cosmors_dgsolv")
        if "hirshfeld" in clean_option_keys:
            properties.append("hirshfeld_charges")
        if any(p in clean_option_keys for p in ("opt", "optts")):
            properties.append("opt_structure")
        if any(p in clean_option_keys for p in ("freq", "numfreq")):
            properties.append("vibs")
            properties.append("gibbs_energy")
            properties.append("detailed_contributions")
        results = get_orca_results(lines, properties=properties)
    else:
        _logger.warning("Orca calculation did not terminate normally.")
        _logger.info("".join(lines))
        results = {"normal_termination": False}
    if calc_dir:
        results["calc_dir"] = str(work_dir)
    else:
        work_dir.cleanup()
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


def read_vibrations(lines: List[str]) -> list:
    for i, line in enumerate(lines):
        if "Number of atoms" in line:
            n_atoms = int(line.split()[-1])
        if "VIBRATIONAL FREQUENCIES" in line:
            freq_start_idx = i
        if "NORMAL MODES" in line:
            mode_start_idx = i

    n_vibs = 3 * n_atoms
    frequencies = [
        float(line.split()[1])
        for line in lines[freq_start_idx + 5 : freq_start_idx + n_vibs + 5]
    ]
    modes = []
    for i in range(int(np.ceil(n_vibs / 6))):
        tmp_modes = []
        for line in lines[
            mode_start_idx
            + 8
            + i * (n_vibs + 1) : mode_start_idx
            + 8
            + (i + 1) * n_vibs
            + i
        ]:
            _, *tmp_components = [float(c) for c in line.split()]
            if len(tmp_components) < 6:
                tmp_components += [0.0] * (6 - len(tmp_components))
            tmp_modes.append(tmp_components)
        tmp_modes = np.array(tmp_modes)
        modes.extend(tmp_modes.T.reshape(6, -1, 3).tolist())
    vibrations = [
        {"frequency": f, "mode": list(m)}
        for f, m in zip(frequencies, modes)
        if f != 0.0
    ]
    return vibrations


def read_gibbs_energy(lines: List[str]) -> float:
    for line in reversed(lines):
        if "Final Gibbs free energy" in line:
            return float(line.split()[-2])


def get_detailed_contributions(lines: List[str]) -> dict:
    for i, l in enumerate(lines):
        if "Zero point energy" in l:
            zero_point_energy = float(l.split()[-4])
        elif "Thermal vibrational correction" in l:
            thermal_vibrational_correction = float(l.split()[-4])
        elif "Thermal rotational correction" in l:
            thermal_rotational_correction = float(l.split()[-4])
        elif "Thermal translational correction" in l:
            thermal_translational_correction = float(l.split()[-4])
        elif "Thermal Enthalpy correction" in l:
            thermal_enthalpy_correction = float(l.split()[-4])
        elif "Electronic entropy" in l:
            electronic_entropy = float(l.split()[-4])
        elif "Vibrational entropy               ..." in l:
            vibrational_entropy = float(l.split()[-4])
        elif "Rotational entropy                ..." in l:
            rotational_entropy = float(l.split()[-4])
        elif "Translational entropy             ..." in l:
            translational_entropy = float(l.split()[-4])
        elif "G-E(el)" in l:
            gibbs_correction = float(l.split()[-4])
        elif "rotational entropy values for sn=" in l:
            sn_idx = i
            sn_nums = int(l.split(",")[-1].split()[0].rstrip(":"))
            if lines[sn_idx + 1] == "\n":
                # in orca6 the sn_idx line is followed by an empty line
                sn_idx += 1

    sn_rot_entropy = {}
    if "sn_idx" in locals():
        for i, l in enumerate(lines[sn_idx + 2 : sn_idx + 2 + sn_nums]):
            sn_rot_entropy[i] = float(l.split()[-4])

    # format into dict
    detailed_contributions = {
        "zero_point_energy": zero_point_energy,
        "thermal_vibrational_correction": thermal_vibrational_correction,
        "thermal_rotational_correction": thermal_rotational_correction,
        "thermal_translational_correction": thermal_translational_correction,
        "thermal_enthalpy_correction": thermal_enthalpy_correction,
        "electronic_entropy": electronic_entropy,
        "vibrational_entropy": vibrational_entropy,
        "rotational_entropy": rotational_entropy,
        "translation_entropy": translational_entropy,
        "gibbs_correction": gibbs_correction,
        "sn_rot_entropy": sn_rot_entropy,
    }

    return detailed_contributions


def read_cosmors(lines: List[str]) -> float:
    for line in reversed(lines):
        if "Free energy of solvation (dGsolv)" in line:
            return float(line.split(":")[1].split()[0])


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
        "vibs": read_vibrations,  # optional
        "gibbs_energy": read_gibbs_energy,  # optional
        "mulliken_charges": read_mulliken_charges,  # always read this
        "loewdin_charges": read_loewdin_charges,  # always read this
        "hirshfeld_charges": read_hirshfeld_charges,  # optional
        "detailed_contributions": get_detailed_contributions,  # optional
        "cosmors_dgsolv": read_cosmors,  # optional
    }

    if not normal_termination(lines):
        raise ValueError("ORCA did not terminate normally")

    results = {"normal_termination": True}
    for property in properties:
        try:
            results[property] = reader[property](lines)
        except Exception as e:
            _logger.error(f"Failed to read property {property}: {e}")
            results[property] = None

    if "opt_structure" in properties:
        results["atoms"], results["opt_coords"] = results["opt_structure"]

    return results
