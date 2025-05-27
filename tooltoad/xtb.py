import json
import logging
import os
import re
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path

import numpy as np

from tooltoad.chemutils import read_multi_xyz, xyz2ac
from tooltoad.utils import (
    STANDARD_PROPERTIES,
    WorkingDir,
    check_executable,
    stream,
)

_logger = logging.getLogger(__name__)


def xtb_calculate(
    atoms: list[str],
    coords: list[list],
    charge: int = 0,
    multiplicity: int = 1,
    options: dict = {},
    scr: str = ".",
    n_cores: int = 1,
    detailed_input: None | dict = None,
    detailed_input_str: None | str = None,
    calc_dir: None | str = None,
    xtb_cmd: str = "xtb",
    force: bool = False,
    data2file: None | dict = None,
) -> dict:
    _logger.info("still")
    """Run xTB calculation.

    Args:
        atoms (list[str]): list of atom symbols.
        coords (list[list]): 3xN list of atom coordinates.
        charge (int): Formal charge of molecule.
        multiplicity (int): Spin multiplicity of molecule.
        options (dict): xTB calculation options.
        scr (str, optional): Path to scratch directory. Defaults to '.'.
        n_cores (int, optional): Number of cores used in calculation. Defaults to 1.
        detailed_input (dict, optional): Detailed input for xTB calculation. Defaults to None.
        calc_dir (str, optional): Name of calculation directory, will be removed after calculation is None. Defaults to None.
        xtb_cmd (str): Path to xTB executable.

    Returns:
        dict: {'atoms': ..., 'coords': ..., ...}
    """
    check_executable(xtb_cmd)
    set_threads(n_cores)

    # create TMP directory
    work_dir = WorkingDir(root=scr, name=calc_dir)
    xyz_file = write_xyz(atoms, coords, work_dir)

    if data2file:
        for filename, data in data2file.items():
            with open(work_dir / filename, "w") as f:
                f.write(data)

    # clean xtb method option
    for k, value in options.items():
        if "gfn" in k.lower():
            if value is not None and value is not True:
                options[k + str(value)] = None
                del options[k]
                break

    # options to xTB command
    cmd = f"{xtb_cmd} --chrg {charge} --uhf {multiplicity-1} --norestart --verbose --parallel {n_cores} "
    for key, value in options.items():
        if value is None or value is True:
            cmd += f"--{key} "
        else:
            cmd += f"--{key} {str(value)} "
    if detailed_input is not None:
        fpath = write_detailed_input(detailed_input, work_dir)
        cmd += f"--input {fpath.name} "
    if detailed_input_str is not None:
        fpath = work_dir / "details.inp"
        with open(fpath, "w") as inp:
            inp.write(detailed_input_str)
        cmd += f"--input {fpath.name} "
    _logger.debug(f"Running xTB with command: {cmd}")
    lines = run_xtb((cmd, xyz_file))
    results = normal_termination(lines)
    if not results["normal_termination"] and not force:
        _logger.warning("xTB did not terminate normally")
        _logger.info("".join(lines))
        results["log"] = "".join(lines)
        if calc_dir:
            results["calc_dir"] = str(work_dir)
        else:
            work_dir.cleanup()
        return results

    # read results
    results = read_xtb_results(lines)
    if "hess" in options:
        results.update(read_thermodynamics(lines))
    if "grad" in options:
        with open(work_dir / "mol.engrad", "r") as f:
            grad_lines = f.readlines()
        results["grad"] = read_gradients(grad_lines)
    if "wbo" in options:
        results["wbo"] = read_wbo(work_dir / "wbo", len(atoms))
    if "pop" in options:
        results["mulliken"] = read_mulliken(work_dir / "charges")
    results["atoms"] = atoms
    results["coords"] = coords
    results["charge"] = charge
    results["multiplicity"] = multiplicity
    results["options"] = options
    if "opt" in options:
        results["opt_coords"] = read_opt_structure(lines)[-1]
    if "ohess" in options:
        results.update(read_thermodynamics(lines))
        results["opt_coords"] = read_opt_structure(lines)[-1]
    if any(s in options for s in ["md", "metadyn"]):
        results["traj"] = read_meta_md(work_dir / "xtb.trj")
        results.update(md_normal_termination(lines))
        results.update(read_mdrestart(work_dir / "mdrestart"))
    if detailed_input or detailed_input_str:
        if any(
            "scan" in x for x in (detailed_input, detailed_input_str) if x is not None
        ):
            results["scan"] = read_scan(work_dir / "xtbscan.log")
        if "wall" in detailed_input_str and "sphere" in detailed_input_str:
            results["cavity_radius"] = read_cavity_radius(lines)
    if "json" in options:
        with open(work_dir / "xtbout.json", "r") as f:
            json_data = json.load(f)
        results["json"] = json_data
    if calc_dir:
        results["calc_dir"] = str(work_dir)
    else:
        work_dir.cleanup()

    time = datetime.now()
    results["time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    results["timestamp"] = time.timestamp()

    return results


def set_threads(n_cores: int) -> None:
    """Set threads and procs environment variables."""
    _ = list(stream("ulimit -s unlimited"))
    os.environ["OMP_STACKSIZE"] = "4G"
    os.environ["OMP_NUM_THREADS"] = f"{n_cores},1"
    os.environ["MKL_NUM_THREADS"] = str(n_cores)
    os.environ["OMP_MAX_ACTIVE_LEVELS"] = "1"

    _logger.debug("Set OMP_STACKSIZE to 4G")
    _logger.debug(f"Set OMP_NUM_THREADS to {n_cores},1")
    _logger.debug(f"Set MKL_NUM_THREADS to {n_cores}")
    _logger.debug("Set OMP_MAX_ACTIVE_LEVELS to 1")


def write_xyz(atoms: list[str], coords: list[list], scr: Path) -> Path:
    """Write xyz coordinate file."""
    natoms = len(atoms)
    xyz = f"{natoms} \n \n"
    for atomtype, coord in zip(atoms, coords):
        xyz += f"{atomtype}  {' '.join(list(map(str, coord)))} \n"
    with open(scr / "mol.xyz", "w") as inp:
        inp.write(xyz)
    _logger.debug(f"Written xyz-file to {scr / 'mol.xyz'}")
    return scr / "mol.xyz"


def write_detailed_input(details_dict: dict, scr: Path) -> Path:
    detailed_input_str = ""
    for key, value in details_dict.items():
        detailed_input_str += f"${key}\n"
        for subkey, subvalue in value.items():
            if subkey.lower() == "constraints":
                for constraint in subvalue:
                    detailed_input_str += f"{constraint.xtb}\n"
            else:
                detailed_input_str += f"{subkey}: {subvalue}\n"
    detailed_input_str += "$end\n"
    fpath = scr / "details.inp"
    _logger.debug(f"Writing detailed input to {fpath}")
    _logger.debug(detailed_input_str)
    with open(fpath, "w") as inp:
        inp.write(detailed_input_str)

    return fpath


def run_xtb(args: tuple[str]) -> list[str]:
    """Run xTB command for xyz-file in parent directory, logs and returns
    output."""
    cmd, xyz_file = args
    generator = stream(f"{cmd}-- {xyz_file.name} | tee xtb.out", cwd=xyz_file.parent)
    lines = []
    for line in generator:
        lines.append(line)
        _logger.debug(line.rstrip("\n"))
    return lines


def normal_termination(lines: list[str]) -> dict:
    messages = {"normal_termination": True}
    warnings = []
    errors = []

    i = 0  # Index to track the current position in the list
    while i < len(lines):
        line = lines[i]
        if "[WARNING]" in line:
            while "###" not in line:
                warnings.append(line)
                i += 1
                if i >= len(lines):
                    break
                line = lines[i]
        elif "[ERROR]" in line:
            while "###" not in line:
                errors.append(line)
                i += 1
                if i >= len(lines):
                    break
                line = lines[i]
        i += 1  # Move to the next line after processing

    if warnings:
        messages["warnings"] = warnings
    if errors:
        messages["errors"] = errors
        messages["normal_termination"] = False
    return messages


def read_opt_structure(lines: list[str]) -> tuple[list[str], list[list[float]]]:
    """Read optimized structure from xTB output."""

    def _parse_coordline(line: str) -> tuple[str, list[list[float]]]:
        """Parse coordinate line from xyz-file."""
        line = line.split()
        atom = line[0]
        coord = [float(x) for x in line[-3:]]
        return atom, coord

    for i, l in reversed(list(enumerate(lines))):
        if "final structure" in l:
            break
    n_atoms = int(lines[i + 2].rstrip())
    start = i + 4
    end = start + n_atoms
    atoms = []
    coords = []
    for line in lines[start:end]:
        atom, coord = _parse_coordline(line)
        atoms.append(atom)
        coords.append(coord)

    return atoms, coords


def read_thermodynamics(lines: list[str]) -> dict:
    """Read thermodynamics output of frequency calculation."""
    thermo_idx = np.nan
    thermo_properties = {}
    for i, line in enumerate(lines):
        line = line.strip()
        if "THERMODYNAMIC" in line:
            thermo_idx = i
        if i > (thermo_idx + 2):
            if 20 * ":" in line:
                thermo_idx = np.nan
            elif 20 * "." in line:
                continue
            else:
                tmp = line.strip(":").strip().strip("->").split()[:-1]
                thermo_properties[" ".join(tmp[:-1])] = float(tmp[-1])
    return thermo_properties


def read_gradients(lines: list[str]) -> np.ndarray:
    """Read gradients from engrad file."""
    gradients = []
    gradient_idx = np.nan
    for i, line in enumerate(lines):
        line = line.strip()
        if "gradient" in line:
            gradient_idx = i
        if i > (gradient_idx + 1):
            if line.startswith("#"):
                gradient_idx = np.nan
                break
            gradients.append(float(line))
    gradients = np.asarray(gradients)
    return gradients.reshape(-1, 3)


def read_wbo(wbo_file, n_atoms) -> np.ndarray:
    """Read Wiberg bond order from wbo file."""
    data = np.loadtxt(wbo_file)
    wbo = np.zeros((n_atoms, n_atoms))
    for i, j, o in data:
        i = i.astype(int) - 1
        j = j.astype(int) - 1
        wbo[i, j] = wbo[j, i] = o
    return wbo


def read_mulliken(charges_file) -> np.ndarray:
    """Read Mulliken charges from charges file."""
    return np.loadtxt(charges_file)


def read_scan(scan_file) -> dict:
    """Read scan results from xTB output."""
    with open(scan_file, "r") as f:
        lines = f.readlines()
    nAtoms = int(lines[0])
    nFrames = int(len(lines) / (nAtoms + 2))
    pes = []
    traj = []
    for n in range(nFrames):
        pes.append(float(lines[n * (nAtoms + 2) + 1].split(":")[1].strip("xtb")))
        traj.append(
            xyz2ac("".join(lines[n * (nAtoms + 2) : n * (nAtoms + 2) + (nAtoms + 2)]))[
                1
            ]
        )
    return {"pes": pes, "traj": traj}


def read_meta_md(traj_file) -> dict:
    _, coords, energies = read_multi_xyz(
        traj_file, lambda x: float(x.strip().split()[1])
    )
    return {"coords": coords, "energies": energies}


def read_mdrestart(mdrestart_file):
    with open(mdrestart_file, "r") as f:
        mdrestart = f.read()
    return {"mdrestart": mdrestart}


def md_normal_termination(lines: list[str]) -> bool:
    """Check if MD terminated normally."""
    checks = {"normal_termination_md": False, "md_stable": True}
    for line in reversed(lines):
        if line.strip().startswith("normal exit of md()"):
            checks["normal_termination_md"] = True
        elif line.strip().startswith("MD is unstable, emergency exit"):
            checks["md_stable"] = False
    return checks


def read_cavity_radius(lines: list[str]) -> float:
    """Read cavity radius from xTB output."""
    for line in reversed(lines):
        if "wallpotenial with radius" in line:
            return float(line.strip().split()[-2])
    return None


def read_xtb_results(lines: list[str]) -> dict:
    """Read basic results from xTB log."""

    def parse_time(line):
        pattern = (
            r"\* wall-time:\s+(\d+)\s+d,\s+(\d+)\s+h,\s+(\d+)\s+min,\s+([\d\.]+)\s+sec"
        )
        match = re.search(pattern, line)

        if match:
            return {
                "days": int(match.group(1)),
                "hours": int(match.group(2)),
                "minutes": int(match.group(3)),
                "seconds": float(match.group(4)),
            }
        else:
            return None

    (
        property_start_idx,
        dipole_idx,
        quadrupole_idx,
        runtime_idx,
        polarizability_idx,
        wall_time,
    ) = (np.nan, np.nan, np.nan, np.nan, np.nan, None)
    gfn_offset = 0
    properties = {}
    for i, line in enumerate(lines):
        line = line.strip()
        if "xtb version" in line:
            xtb_version = line.split()[3]
        elif "program call" in line:
            programm_call = line.split(":")[-1]
        elif "SUMMARY" in line:
            property_start_idx = i
        elif "molecular dipole" in line:
            dipole_idx = i
            # hack for gfn-ff calc
            if "gfnff" in programm_call:
                gfn_offset = 1
        elif "molecular quadrupole" in line:
            quadrupole_idx = i
        elif "total:" in line:
            runtime_idx = i
        elif "Mol. α(0) /au" in line:
            polarizability_idx = i

        # read property table
        if i > (property_start_idx + 1):
            if 20 * ":" in line:
                property_start_idx = np.nan
            elif 20 * "." in line:
                continue
            else:
                tmp = line.strip(":").strip().strip("->").split()[:-1]
                properties[" ".join(tmp[:-1])] = float(tmp[-1])

        # read polarizability
        if i == polarizability_idx:
            polarizability = float(line.split()[-1])

        # read dipole moment
        if i > (dipole_idx + 2 - gfn_offset):
            dip_x, dip_y, dip_z, dip_norm = [
                float(x) for x in line.split()[1 + gfn_offset :]
            ]  # norm is in Debye
            dipole_vec = np.array(
                [dip_x, dip_y, dip_z]
            )  # in a.u. (*2.5412 ot convert to Debye)
            dipole_idx = np.nan

        # read quadrupole moment
        if i > (quadrupole_idx + 3):
            quad_vec = np.array([float(x) for x in line.split()[1:]])  # in a.u.
            quad_vec[2], quad_vec[3] = quad_vec[3], quad_vec[2]
            quadrupole_mat = np.zeros((3, 3))
            indices = np.triu_indices(3)
            quadrupole_mat[indices] = quad_vec
            quadrupole_mat[indices[::-1]] = quad_vec
            quadrupole_idx = np.nan

        # read runtimes
        if i == (runtime_idx + 1):
            wall_time = parse_time(line)

    results = {
        "normal_termination": True,
        "programm_call": programm_call,
        "programm_version": xtb_version,
        "timings": wall_time,
    }

    if not np.isnan(polarizability_idx):
        results["polarizability"] = polarizability
    if not np.isnan(dipole_idx):
        results["dipole_vec"] = dipole_vec
        results["dipole_norm"] = dip_norm
    if not np.isnan(quadrupole_idx):
        results["quadrupole_mat"] = quadrupole_mat

    results.update(properties)
    # add standardized property names
    for key, value in STANDARD_PROPERTIES["xtb"].items():
        if key in results:
            results[value] = results[key]
    return results


# --------------------- Detailed Input Options -------------------------


class BaseOptions:
    def __str__(self):
        assert "Options" in self.__class__.__name__
        name = self.__class__.__name__.replace("Options", "").lower()
        lines = []
        for field in fields(self):
            value = getattr(self, field.name)
            if value is None:
                continue
            if isinstance(value, bool):  # Convert boolean to lowercase string
                value = str(value).lower()
            elif isinstance(value, float):  # Format floats with consistent precision
                value = f"{value:.4f}"
            elif isinstance(value, list):
                value = ",".join(map(str, value))
            delimiter = ":" if "," in str(value) else "="
            lines.append(f"   {field.name}{delimiter}{value}")
        return f"${name}\n" + "\n".join(lines) + "\n$end"


@dataclass
class MDOptions(BaseOptions):
    temp: float = 300
    time: float = 10.0  # ps
    dump: float = 10.0  # every x step, dumptrj
    sdump: None | float = 250  # every x step, dumpcoord
    step: float = 0.4  # fs
    velo: bool = False
    shake: int = 0
    hmass: int = 2
    sccacc: float = 2.0
    nvt: bool = True
    restart: bool = False


@dataclass
class MetaDynOptions(BaseOptions):
    save: int = 250  # maximum number of structures to consider for bias potential
    kpush: float = 0.075
    alp: float = 0.3
    coord: None | str = None
    atoms: None | list[int] = None
    # undocumented options
    # https://github.com/grimme-lab/xtb/blob/main/src/set_module.f90#L2541
    static: None | bool = False
    ramp: None | float = 0.03
    bias_input: None | str = None


@dataclass
class WallOptions(BaseOptions):
    potential: str = "logfermi"
    sphere: str = "auto, all"
    autoscale: None | float = None
    beta: None | float = 10.0
    temp: None | float = 6000.0
    ellipsoid: None | str = None
    # TODO: add more options


@dataclass
class SCCOptions(BaseOptions):
    temp: None | float = 6000.0


if __name__ == "__main__":
    atoms = ["C", "C", "C", "N", "C", "C", "N", "H", "H", "H", "H", "H", "H", "H"]
    coords = [
        [-0.150572945378, 1.06902383757551, 0.13369717980808],
        [-1.53340374624082, 0.97192911929508, 0.16014351260219],
        [-2.11046032356157, -0.28452881345742, 0.02365204850958],
        [-1.40326713049944, -1.39105857658305, -0.1287228595083],
        [-0.08701882046636, -1.30624454138783, -0.15461643853251],
        [0.58404976357399, -0.09554934782587, -0.02805090001267],
        [2.04601334595326, -0.0521981720584, -0.06393160516877],
        [0.3301095007743, 2.02983678857648, 0.23606713936603],
        [-2.14949770349705, 1.84686355411139, 0.28440826742643],
        [-3.18339157653865, -0.41415380049138, 0.03929295998759],
        [0.43604958186272, -2.24346585185004, -0.28257385567194],
        [2.37469662935334, 0.91726637325056, 0.04696293388512],
        [2.40041378960073, -0.42116004539483, -0.95829570995538],
        [2.44627963506354, -0.62656052376019, 0.69196732726456],
    ]

    options = {"opt": True, "alpb": "methanol", "wbo": True, "pop": True}
    results = xtb_calculate(atoms=atoms, coords=coords, charge=1, options=options)

    for key, value in results.items():
        print(key)
        print(value)
        print(80 * "*")
