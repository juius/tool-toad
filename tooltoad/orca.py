import copy
import json
import logging
import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List

import numpy as np
from dotenv import load_dotenv

from tooltoad.chemutils import hartree2kcalmol, read_multi_xyz, xyz2ac
from tooltoad.utils import WorkingDir, check_executable, stream

_logger = logging.getLogger("orca")

# see https://www.orcasoftware.de/tutorials_orca/first_steps/parallel.html
PARENT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(PARENT_DIR / ".env")

ORCA_CMD = os.getenv("ORCA_EXE", "orca")
OPEN_MPI_DIR = os.getenv("OPEN_MPI_DIR", "openmpi").rstrip("/")
ORCA_DIR = Path(ORCA_CMD).parent
XTB_EXE = os.getenv("XTB_EXE", "xtb")
SET_ENV = f'env - XTBEXE={XTB_EXE} PATH="{ORCA_DIR}:{OPEN_MPI_DIR}/bin:$PATH" LD_LIBRARY_PATH="{OPEN_MPI_DIR}/lib:$LD_LIBRARY_PATH" DYLD_LIBRARY_PATH="{OPEN_MPI_DIR}/lib:$DYLD_LIBRARY_PATH"'


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
    force: bool = False,
    log_file: str | None = None,
    read_files: list | None = None,
    save_files: list | None = None,
    save_dir: str | None = None,
    data2file: None | dict = None,
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
    if save_files:
        assert save_dir, "save_dir must be provided if save_files are given"
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    check_executable(orca_cmd)
    work_dir = WorkingDir(root=scr, name=calc_dir)

    if data2file:
        for filename, data in data2file.items():
            with open(work_dir / filename, "w") as f:
                f.write(data)

    # append json output
    xtra_inp_str += """%Method
  WriteJSONPropertyfile True
end"""

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

    if not log_file:
        log_file = "orca.out"
    # cmd = f'{set_env}; {orca_cmd} input.inp "--bind-to-core" | tee orca.out' # "--oversubscribe" "--use-hwthread-cpus"
    cmd = f'/bin/bash -c "{set_env} {orca_cmd} input.inp "--use-hwthread-cpus" | tee {log_file}"'
    _logger.debug(f"Running Orca as: {cmd}")

    # Run Orca, capture an log output
    generator = stream(cmd, cwd=str(work_dir))
    lines = []
    for line in generator:
        lines.append(line)
        _logger.debug(line.rstrip("\n"))

    # try copying savefiles
    if save_files:
        for f in save_files:
            try:
                shutil.copy(work_dir / f, save_dir / f)
            except Exception:
                _logger.error(f"Failed to copy {f} to {save_dir}")

    # read results
    if normal_termination(lines) or force:
        clean_option_keys = [k.lower() for k in options.keys()]
        _logger.debug("Orca calculation terminated normally.")
        properties = ["electronic_energy", "timings"]
        additional_properties = []
        if "cosmors" in clean_option_keys:
            properties.append("cosmors_dgsolv")
        if "hirshfeld" in clean_option_keys:
            properties.append("hirshfeld_charges")
        if any(
            p in clean_option_keys for p in ("opt", "optts", "tightopt", "verytightopt")
        ):
            properties.append("opt_structure")
            additional_properties.append("traj")
        if any(p in clean_option_keys for p in ("freq", "numfreq")):
            properties.append("vibs")
            properties.append("gibbs_energy")
            properties.append("enthalpy")
            if not any(p in clean_option_keys for p in ("xtb1", "xtb2")):
                properties.append("detailed_contributions")
        # ---------- results from additional files --------------
        if "irc" in clean_option_keys:
            additional_properties.append("irc")
        if "goat" in clean_option_keys:
            additional_properties.append("goat")
        if "goat-react" in clean_option_keys:
            additional_properties.append("goat")
            additional_properties.append("goat-react")
        # ---------- check for options in xtra_inp_str ----------
        if xtra_inp_str:
            if "scan" in xtra_inp_str.lower():
                additional_properties.append("scan")
        results = get_orca_results(copy.deepcopy(lines), properties=properties)
        additional_results = get_additional_results(
            copy.deepcopy(lines), work_dir, additional_properties
        )
        results.update(additional_results)
        #  read the json
        try:
            with open(work_dir / "input.property.json", "r") as f:
                results["json"] = json.load(f)
        except FileNotFoundError:
            _logger.warning("File 'input.property.json' not found")
    else:
        _logger.warning("Orca calculation did not terminate normally.")
        _logger.info("\n".join(lines))
        results = {"normal_termination": False, "log": "\n".join(lines)}

    if read_files:
        for f in read_files:
            with open(work_dir / f, "r") as _f:
                try:
                    results[f] = _f.read()
                except FileNotFoundError:
                    _logger.warning(f"File {f} not found")
    if calc_dir:
        results["calc_dir"] = str(work_dir)
    else:
        work_dir.cleanup()

    time = datetime.now()
    results["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    results["timestamp"] = time.timestamp()
    results["atoms"] = atoms
    results["coords"] = coords
    results["charge"] = charge
    results["multiplicity"] = multiplicity
    results["options"] = options
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


def read_timings(lines: List[str]) -> dict:
    pattern = r"(\d+) days (\d+) hours (\d+) minutes (\d+) seconds (\d+) msec"
    for line in reversed(lines):
        if "TOTAL RUN TIME" in line:
            match = re.search(pattern, line.strip())
            if match:
                runtime_data = {
                    "days": int(match.group(1)),
                    "hours": int(match.group(2)),
                    "minutes": int(match.group(3)),
                    "seconds": int(match.group(4)),
                    "milliseconds": int(match.group(5)),
                }
                break
    return runtime_data


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


def get_n_atoms(lines: List[str]) -> int:
    for i, line in enumerate(lines):
        if "> *xyz" in line:
            start_idx = i
        elif "> *" in line:
            end_idx = i
            break
    return end_idx - start_idx - 1


def read_vibrations(lines: List[str]) -> list:
    n_atoms = get_n_atoms(lines)
    for i, line in enumerate(lines):
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


def read_enthalpy(lines: List[str]) -> float:
    for line in reversed(lines):
        if "Total Enthalpy" in line:
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
    #     elif "rotational entropy values for sn=" in l:
    #         sn_idx = i
    #         sn_nums = int(l.split(",")[-1].split()[0].rstrip(":"))
    #         if lines[sn_idx + 1] == "\n":
    #             # in orca6 the sn_idx line is followed by an empty line
    #             sn_idx += 1

    # sn_rot_entropy = {}
    # if "sn_idx" in locals():
    #     for i, l in enumerate(lines[sn_idx + 2 : sn_idx + 2 + sn_nums]):
    #         sn_rot_entropy[i] = float(l.split()[-4])

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
        # "sn_rot_entropy": sn_rot_entropy,
    }

    return detailed_contributions


def read_cosmors(lines: List[str]) -> float:
    for line in reversed(lines):
        if "Free energy of solvation (dGsolv)" in line:
            return float(line.split(":")[1].split()[0])


def read_irc(lines: List[str], work_dir: WorkingDir) -> dict:
    # check convergence for each direction
    irc_results = {"forward_converged": False, "backward_converged": False}
    for line in lines:
        if "WARD IRC" in line:
            direction = line.strip(" *").split()[0].lower()
        if "THE IRC HAS CONVERGED" in line:
            irc_results[direction + "_converged"] = True

    for direction in ["forward", "backward"]:
        with open(work_dir / f"input_IRC_{direction[0].upper()}.xyz", "r") as f:
            xyz = f.read()
        atoms, coords = xyz2ac(xyz)
        energy = float(xyz.splitlines()[1].strip().split()[-1])
        irc_results[direction] = {
            "atoms": atoms,
            "opt_coords": coords,
            "electronic_energy": energy,
        }
    traj_atoms, traj_coords, traj_energies = read_multi_xyz(
        work_dir / "input_IRC_Full_trj.xyz",
        extract_property_function=lambda x: float(x.split()[-1]),
    )
    irc_results["traj"] = {
        "atoms": traj_atoms,
        "coords": traj_coords,
        "energies": traj_energies,
    }
    return irc_results


def read_traj(lines: List[str], work_dir: WorkingDir) -> dict:
    atoms, coords, energies = read_multi_xyz(
        work_dir / "input_trj.xyz", lambda line: float(line.split()[-1])
    )
    return {"atoms": atoms, "coords": coords, "energies": energies}


def read_scan(lines: List[str], work_dir: WorkingDir) -> dict:
    scan_results = {}
    scan_results["overview"] = np.loadtxt(work_dir / "input.relaxscanact.dat").tolist()
    with open(work_dir / "input.allxyz", "r") as f:
        lines = f.readlines()
    nAtoms = int(lines[0])
    nFrames = int(len(lines) / (nAtoms + 2))
    pes = []
    traj = []
    for n in range(nFrames):
        pes.append(float(lines[n * (nAtoms + 3) + 1].split("E ")[-1].strip()))
        traj.append(
            xyz2ac("".join(lines[n * (nAtoms + 3) : n * (nAtoms + 3) + (nAtoms + 2)]))[
                1
            ]
        )
    scan_results["pes"] = pes
    scan_results["traj"] = traj
    return scan_results


def read_goat(lines: List[str], work_dir: WorkingDir) -> dict:
    atoms, coords, energies = read_multi_xyz(
        work_dir / "input.finalensemble.xyz", lambda line: float(line.split()[0])
    )
    # sort the ensemble by energy
    coords, energies = zip(*sorted(zip(coords, energies), key=lambda x: x[1]))
    # add relative energies in kcal/mol
    relative_energies = [0.0] + [hartree2kcalmol(e - energies[0]) for e in energies[1:]]
    ensemble = {
        "atoms": atoms[0],
        "coords": coords,
        "energies": energies,
        "relative_energies": relative_energies,
    }
    with open(work_dir / "input.globalminimum.xyz", "r") as f:
        text = f.read()
    a, c = xyz2ac(text)

    return {
        "atoms": a,
        "coords": c,
        "energy": float(text.splitlines()[1].strip()),
        "ensemble": ensemble,
    }


def read_goat_react(lines: List[str], work_dir: WorkingDir) -> dict:
    atoms, coords, energies = read_multi_xyz(
        work_dir / "input.products.unique.xyz", lambda line: float(line.split()[1])
    )
    # sort the ensemble by energy
    coords, energies = zip(*sorted(zip(coords, energies), key=lambda x: x[1]))
    # add relative energies in kcal/mol
    relative_energies = [0.0] + [hartree2kcalmol(e - energies[0]) for e in energies[1:]]
    unique_products = {
        "atoms": atoms,
        "coords": coords,
        "energies": energies,
        "relative_energies": relative_energies,
    }

    return {"unique_products": unique_products}


def read_hessian_str(lines: List[str], work_dir: WorkingDir) -> dict:
    with open(work_dir / "input.hess", "r") as f:
        hessian_str = f.read()
    return {"hessian_str": hessian_str}


def get_additional_results(
    lines: List[str], work_dir: WorkingDir, properties: list
) -> dict:
    reader = {
        "irc": read_irc,
        "scan": read_scan,
        "goat": read_goat,
        "goat-react": read_goat_react,
        "hessian_str": read_hessian_str,
        "traj": read_traj,
    }
    additional_results = {"normal_termination": True}
    for property in properties:
        try:
            additional_results[property] = reader[property](lines, work_dir)
        except Exception as e:
            _logger.error(f"Failed to read property {property}: {e}")
            additional_results[property] = None
    return additional_results


def get_orca_results(
    lines: List[str],
    properties: List[str] = [
        "electronic_energy",
        "mulliken_charges",
        "loewdin_charges",
        "timings",
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
        "enthalpy": read_enthalpy,  # optional
        "mulliken_charges": read_mulliken_charges,  # always read this
        "loewdin_charges": read_loewdin_charges,  # always read this
        "hirshfeld_charges": read_hirshfeld_charges,  # optional
        "detailed_contributions": get_detailed_contributions,  # optional
        "cosmors_dgsolv": read_cosmors,  # optional
        "timings": read_timings,  # always read this
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
