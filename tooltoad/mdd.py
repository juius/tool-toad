import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdmolops
from watchdog.events import FileSystemEventHandler
from watchdog.observers import Observer

from tooltoad.chemutils import ac2mol, read_multi_xyz
from tooltoad.utils import WorkingDir, check_executable
from tooltoad.xtb import set_threads, write_xyz, xtb_calculate

_logger = logging.getLogger(__name__)


def scoords2coords(scoords_file):
    with open(scoords_file, "r") as f:
        lines = f.readlines()
    atoms = []
    coords = []
    for line in lines:
        if line.strip().startswith("$"):
            continue
        *cs, atom = line.split()
        atoms.append(atom)
        coords.append(cs)
    coords = np.array(coords, dtype=float) * 0.529177
    return atoms, coords


def is_small_ring_product(origin, product, max_size=4):
    ac1 = rdmolops.GetAdjacencyMatrix(origin)
    ac2 = rdmolops.GetAdjacencyMatrix(product)
    diff = ac2 - ac1

    if diff.sum() == 2 and np.abs(diff).sum() == 2:
        # only one bond has formed
        _logger.debug("Only one bond has formed, checking for small ring product")
        Chem.SanitizeMol(origin, sanitizeOps=Chem.SanitizeFlags.SANITIZE_SYMMRINGS)
        rinfo = origin.GetRingInfo()

        # are two atoms in the same ring
        atom_rings = rinfo.AtomRings()
        idx1, idx2 = [int(i) for i in np.where(diff == 1)[0]]
        in_same_ring = any(idx1 in ring and idx2 in ring for ring in atom_rings)
        _logger.debug(f"Bond between atom {idx1} and {idx2} is between ring atoms")
        if in_same_ring:
            # now get new bond in product
            new_bond = product.GetBondBetweenAtoms(idx1, idx2)

            is_small_ring = any(
                [new_bond.IsInRingSize(size) for size in range(3, max_size + 1)]
            )
            if is_small_ring:
                _logger.debug(
                    f"Detected small ring product between atoms {idx1} and {idx2} in a ring of size <= {max_size}"
                )
                return True
    return False


def cleanup_processes(observer, executor, xtb_process):
    """Clean up all processes and resources."""
    try:
        # Stop the observer first
        observer.stop()
        observer.join()
        _logger.debug("Observer stopped")

        # Shutdown the thread pool
        executor.shutdown(wait=False)
        _logger.debug("Thread pool shutdown initiated")

        # Terminate the xtb process and its children
        try:
            # First try to terminate the process group
            # try:
            #     1 == 2
            #     # pgid = os.getpgid(xtb_process.pid)
            #     # os.killpg(pgid, signal.SIGTERM)
            # except (ProcessLookupError, OSError):
            # If process group kill fails, try direct process termination
            xtb_process.terminate()

            # Wait for process to terminate
            try:
                xtb_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                _logger.warning(
                    "xTB process did not terminate gracefully, forcing kill"
                )
                # Try to kill the process group again
                try:
                    pgid = os.getpgid(xtb_process.pid)
                    os.killpg(pgid, signal.SIGKILL)
                except (ProcessLookupError, OSError):
                    # If all else fails, use kill
                    xtb_process.kill()
                xtb_process.wait(timeout=2)
        except Exception as e:
            _logger.error(f"Error terminating xTB process: {str(e)}")
        _logger.debug("xTB process terminated")

        # Clean up any remaining processes
        try:
            # Try to clean up the process group
            pass
            # try:
            #     os.killpg(os.getpgid(0), signal.SIGTERM)
            #     time.sleep(1)
            #     os.killpg(os.getpgid(0), signal.SIGKILL)
            # except (ProcessLookupError, OSError):
            #     pass
        except Exception as e:
            _logger.error(f"Error cleaning up processes: {str(e)}")
        _logger.debug("All processes cleaned up")
    except Exception as e:
        _logger.error(f"Error during cleanup: {str(e)}")


def track_tajectory_v2(
    xtb_process,
    traj_file: str,
    init_smiles: str,
    init_mol: Chem.Mol,
    max_products: None | int = None,
    charge: int = 0,
    multiplicity: int = 1,
    opt_options: dict = {},
    frame_interval: int = 10,
    num_workers: int = 1,
    time_interval: int = 2,
    scr: str = ".",
    allow_small_ring_products: bool = False,
):
    work_dir = Path(traj_file).parent
    _logger.info(f"Starting trajectory tracking in directory: {work_dir}")
    _logger.info(f"Initial SMILES: {init_smiles}")

    opt_options["opt"] = "crude"
    _logger.debug(f"opt_options: {opt_options}")

    # Create a shared result variable
    result = None
    result_lock = threading.Lock()
    stop_event = threading.Event()

    class ScoordHandler(FileSystemEventHandler):
        def __init__(self, allow_small_ring_products=allow_small_ring_products):
            self.processed_files = set()
            self.allow_small_ring_products = allow_small_ring_products

        def on_created(self, event):
            nonlocal result
            if not event.is_directory and Path(event.src_path).name.startswith(
                "scoord."
            ):
                filepath = Path(event.src_path)
                time.sleep(2)  # Give some time for the file to be fully written
                if filepath not in self.processed_files:
                    self.processed_files.add(filepath)
                    try:
                        atoms, coords = scoords2coords(filepath)
                        _logger.debug(
                            f"Running optimization for structure from {filepath}"
                        )
                        _logger.debug(f"content of scoord file: {filepath}")
                        _logger.debug(f"atoms: {atoms}")
                        _logger.debug(f"coords: {coords}")
                        # Check if the number of atoms is consistent
                        with open(filepath, "r") as f:
                            lines = f.readlines()
                        _logger.debug(f"Number of lines in scoord file: {len(lines)}")
                        _logger.debug("".join(lines))
                        crude_opt = xtb_calculate(
                            atoms,
                            coords,
                            charge,
                            multiplicity,
                            options=opt_options,
                            scr=scr,
                            # calc_dir=f"opt-{filepath.stem}",
                        )

                        if crude_opt["normal_termination"]:
                            mol = ac2mol(
                                crude_opt["atoms"],
                                crude_opt["opt_coords"],
                                use_xtb=True,
                            )
                            smiles = Chem.MolToSmiles(mol)
                            _logger.debug(f"Initial SMILES: {init_smiles}")
                            _logger.debug(f"Current SMILES: {smiles}")

                            if smiles != init_smiles:
                                if not self.allow_small_ring_products:
                                    Chem.SanitizeMol(
                                        mol,
                                        sanitizeOps=Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
                                    )
                                    if is_small_ring_product(init_mol, mol, max_size=4):
                                        _logger.debug(
                                            "Small ring product detected, skipping..."
                                        )
                                        return
                                _logger.info(
                                    "New SMILES found! Final optimization and terminating process..."
                                )
                                opt_options["opt"] = None
                                opt = xtb_calculate(
                                    atoms,
                                    crude_opt["opt_coords"],
                                    charge,
                                    multiplicity,
                                    options=opt_options,
                                    scr=scr,
                                )
                                opt["smiles"] = smiles
                                opt["frame"] = Path(filepath).suffix.lstrip(".")
                                with result_lock:
                                    result = opt
                                stop_event.set()
                                # Terminate the xtb process
                                try:
                                    pgid = os.getpgid(xtb_process.pid)
                                    os.killpg(pgid, signal.SIGTERM)
                                    time.sleep(1)
                                    if xtb_process.poll() is None:
                                        os.killpg(pgid, signal.SIGKILL)
                                    xtb_process.terminate()
                                    xtb_process.wait(timeout=5)
                                except ProcessLookupError:
                                    pass
                                except subprocess.TimeoutExpired:
                                    _logger.warning(
                                        "xTB process did not terminate gracefully, forcing kill"
                                    )
                                    xtb_process.kill()
                                _logger.debug("xTB process terminated")
                    except Exception as e:
                        _logger.error(f"Error processing {filepath}: {e}")

    # Set up observer
    event_handler = ScoordHandler(allow_small_ring_products=allow_small_ring_products)
    observer = Observer()
    observer.schedule(event_handler, str(work_dir), recursive=False)
    observer.start()
    _logger.debug("File system observer started")

    try:
        # Main monitoring loop
        while not stop_event.is_set():
            if xtb_process.poll() is not None:
                _logger.debug("MD run finished")
                break
            time.sleep(0.1)
    except KeyboardInterrupt:
        _logger.info("Received keyboard interrupt, terminating process...")
    except Exception as e:
        _logger.error(f"Unexpected error: {str(e)}")
    finally:
        # Cleanup
        observer.stop()
        observer.join()
        _logger.debug("Observer stopped")

        try:
            xtb_process.terminate()
            xtb_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _logger.warning("xTB process did not terminate gracefully, forcing kill")
            xtb_process.kill()
        _logger.debug("xTB process terminated")

    # Return the result after cleanup
    with result_lock:
        if result is not None:
            _logger.debug("Returning result with new product")
            return result
        _logger.info("No new product found")
        return None


def md_step(
    atoms: list[str],
    coords: list[list],
    charge: int = 0,
    multiplicity: int = 1,
    options: dict = {},
    scr: str = ".",
    n_md_cores: int = 1,
    n_opt_cores: int = 1,
    max_products: int = 1,
    frame_interval: int = 10,
    detailed_input_str: None | str = None,
    calc_dir: None | str = None,
    xtb_cmd: str = "xtb",
    data2file: None | dict = None,
    save_traj: bool = False,
    allow_small_ring_products: bool = False,
):
    options = options.copy()
    options["md"] = None
    check_executable(xtb_cmd)
    set_threads(n_md_cores)
    env = os.environ.copy()
    init_mol = ac2mol(atoms, coords, use_xtb=True)
    init_smiles = Chem.MolToSmiles(init_mol)
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
    cmd = f"{xtb_cmd} --chrg {charge} --uhf {multiplicity-1} --norestart --verbose --parallel {n_md_cores} "
    for key, value in options.items():
        if value is None or value is True:
            cmd += f"--{key} "
        else:
            cmd += f"--{key} {str(value)} "
    if detailed_input_str is not None:
        fpath = work_dir / "details.inp"
        with open(fpath, "w") as inp:
            inp.write(detailed_input_str)
        cmd += f"--input {fpath.name} "

    process = subprocess.Popen(
        f"{cmd}-- {xyz_file.name} | tee xtb.out",
        cwd=xyz_file.parent,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        preexec_fn=os.setsid,
        env=env,
    )
    opt_options = options.copy()
    opt_options.pop("md")

    result = track_tajectory_v2(
        process,
        traj_file=str((xyz_file.parent / "xtb.trj").absolute()),
        init_smiles=init_smiles,
        init_mol=init_mol,
        charge=charge,
        multiplicity=multiplicity,
        max_products=max_products,
        frame_interval=frame_interval,
        num_workers=n_opt_cores,
        opt_options=opt_options,
        scr=scr,
        allow_small_ring_products=allow_small_ring_products,
    )

    if save_traj:
        if result:
            traj = read_multi_xyz(str((xyz_file.parent / "xtb.trj").absolute()))
            result["traj"] = traj

    # cleanup calc dir
    if not calc_dir:
        work_dir.cleanup()
    return result


if __name__ == "__main__":
    atoms = ["C", "C", "O", "C", "O", "H", "H", "H", "H", "H", "H"]
    coords = [
        [0.26568467572667, -0.27831940366627, -0.58143651098761],
        [-0.90676450908047, -0.32898730272464, 0.40529475847352],
        [-1.41568585464497, 0.94892865156221, 0.71497268931071],
        [1.53863004278233, 0.2010662094895, 0.04239084574045],
        [1.7850842514261, 0.21669083040265, 1.22221509606025],
        [0.03092787573139, 0.35107225545984, -1.44096675309975],
        [0.46739234513469, -1.28482111368172, -0.95677100925294],
        [-0.5614602960097, -0.7422579795195, 1.35610875003709],
        [-1.70335137587454, -0.96875298931832, 0.00696728511019],
        [-1.79732444963634, 1.34360582009679, -0.08220691169635],
        [2.29686729444485, 0.54177502189944, -0.68656823969562],
    ]
    inp_str = """$md
   temp=300
   time=10
   dump=100.0000
   step=0.4000
   velo=false
   shake=0
   hmass=2
   sccacc=2.0000
   nvt=true
   restart=false
$end
$metadyn
   save=100
   kpush=0.1000
   alp=0.3000
   static=false
$end
$wall
   potential=logfermi
   sphere:auto, all
   beta=10.0000
   temp=6000.0000
$end
$scc
   temp=12000
$end
$cma
"""

    _logger.addHandler(logging.StreamHandler())
    _logger.setLevel(logging.DEBUG)
    results = md_step(
        atoms,
        coords,
        detailed_input_str=inp_str,
        n_opt_cores=4,
        options={"alpb": "water", "etemp": 6000},
    )

    print(results)

    def wrap(*args, **kwrags):
        print("got here")
        print(args)
        logger = logging.getLogger("mdd")
        logger.setLevel(logging.DEBUG)
        logger.addHandler(logging.StreamHandler())
        logger1 = logging.getLogger("tooltoad.mdd")
        logger1.setLevel(logging.DEBUG)
        logger1.addHandler(logging.StreamHandler())
        return md_step(*args, **kwrags)

    # executor.submit(
    #     wrap,
    #     atoms,
    #     coords,
    #     detailed_input_str=inp_str,
    #     n_opt_cores=4,
    # )
