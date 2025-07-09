import logging
import os
import signal
import subprocess
import threading
import time
from pathlib import Path
from queue import Empty

import numpy as np
from rdkit import Chem
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
        if line.startswith("$"):
            continue
        *cs, atom = line.split()
        atoms.append(atom)
        coords.append(cs)
    coords = np.array(coords, dtype=float) * 0.529177
    return atoms, coords


def process_scoord_files(
    queue, opt_options, charge, multiplicity, init_smiles, result_queue
):
    # Set up logging in the child process
    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    _logger.addHandler(handler)
    _logger.setLevel(logging.INFO)

    _logger.info("Worker process started processing scoord files")
    while True:
        try:
            filepath = queue.get(timeout=1)
            _logger.info(f"Processing scoord file: {filepath}")
            try:
                atoms, coords = scoords2coords(filepath)
                _logger.info(f"Running optimization for structure from {filepath}")
                crude_opt = xtb_calculate(
                    atoms, coords, charge, multiplicity, options=opt_options
                )
                if crude_opt["normal_termination"]:
                    mol = ac2mol(
                        crude_opt["atoms"], crude_opt["opt_coords"], use_xtb=True
                    )
                    smiles = Chem.MolToSmiles(mol)
                    _logger.info(f"Initial SMILES: {init_smiles}")
                    _logger.info(f"Current SMILES: {smiles}")
                    if smiles != init_smiles:
                        _logger.info("New SMILES found! Terminating processes...")
                        crude_opt["smiles"] = smiles
                        crude_opt["frame"] = Path(filepath).suffix
                        result_queue.put(crude_opt)
                        return
                    else:
                        _logger.info("SMILES unchanged, continuing...")

            except Exception as e:
                _logger.error(f"Error processing {filepath}: {str(e)}")
        except Empty:
            continue
        except Exception as e:
            _logger.error(f"Unexpected error in worker process: {str(e)}")
            return None


def process_structure(
    scoord_file, init_smiles, charge, multiplicity, opt_options, xtb_process
):
    """Process a single structure and check if it's a new product."""
    try:
        atoms, coords = scoords2coords(scoord_file)
        _logger.info(f"Running optimization for structure from {scoord_file}")
        opt_options["opt"] = "crude"
        crude_opt = xtb_calculate(
            atoms, coords, charge, multiplicity, options=opt_options
        )

        if not crude_opt["normal_termination"]:
            _logger.info("Abnormal xtb termination")
            return None

        mol = ac2mol(crude_opt["atoms"], crude_opt["opt_coords"], use_xtb=True)
        smiles = Chem.MolToSmiles(mol)
        _logger.info(f"Processed SMILES: {smiles}")

        if smiles != init_smiles:
            _logger.info(
                f"New product found! Initial SMILES: {init_smiles}, New SMILES: {smiles}"
            )
            crude_opt["smiles"] = smiles
            # Terminate the xtb process and its children
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
            _logger.info("xTB process terminated")
            return crude_opt

        _logger.info("SMILES unchanged, continuing...")
        return None

    except Exception as e:
        _logger.error(f"Error processing {scoord_file}: {str(e)}")
        return None


def cleanup_processes(observer, executor, xtb_process):
    """Clean up all processes and resources."""
    try:
        # Stop the observer first
        observer.stop()
        observer.join()
        _logger.info("Observer stopped")

        # Shutdown the thread pool
        executor.shutdown(wait=False)
        _logger.info("Thread pool shutdown initiated")

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
        _logger.info("xTB process terminated")

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
        _logger.info("All processes cleaned up")
    except Exception as e:
        _logger.error(f"Error during cleanup: {str(e)}")


def track_tajectory_v2(
    xtb_process,
    traj_file: str,
    init_smiles: str,
    max_products: None | int = None,
    charge: int = 0,
    multiplicity: int = 1,
    opt_options: dict = {},
    frame_interval: int = 10,
    num_workers: int = 1,
    time_interval: int = 2,
    scr: str = ".",
):
    work_dir = Path(traj_file).parent
    _logger.info(f"Starting trajectory tracking in directory: {work_dir}")
    _logger.info(f"Initial SMILES: {init_smiles}")

    opt_options["opt"] = "crude"
    _logger.info(f"opt_options: {opt_options}")

    # Create a shared result variable
    result = None
    result_lock = threading.Lock()
    stop_event = threading.Event()

    class ScoordHandler(FileSystemEventHandler):
        def __init__(self):
            self.processed_files = set()

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
                        _logger.info(
                            f"Running optimization for structure from {filepath}"
                        )
                        _logger.info(f"content of scoord file: {filepath}")
                        _logger.info(f"atoms: {atoms}")
                        _logger.info(f"coords: {coords}")
                        # Check if the number of atoms is consistent
                        with open(filepath, "r") as f:
                            lines = f.readlines()
                        _logger.info(f"Number of lines in scoord file: {len(lines)}")
                        _logger.info("".join(lines))
                        crude_opt = xtb_calculate(
                            atoms,
                            coords,
                            charge,
                            multiplicity,
                            options=opt_options,
                            scr=scr,
                        )

                        if crude_opt["normal_termination"]:
                            mol = ac2mol(
                                crude_opt["atoms"],
                                crude_opt["opt_coords"],
                                use_xtb=True,
                            )
                            smiles = Chem.MolToSmiles(mol)
                            _logger.info(f"Initial SMILES: {init_smiles}")
                            _logger.info(f"Current SMILES: {smiles}")

                            if smiles != init_smiles:
                                _logger.info("New SMILES found! Terminating process...")
                                crude_opt["smiles"] = smiles
                                crude_opt["frame"] = Path(filepath).suffix.lstrip(".")
                                with result_lock:
                                    result = crude_opt
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
                                _logger.info("xTB process terminated")
                    except Exception as e:
                        _logger.error(f"Error processing {filepath}: {e}")

    # Set up observer
    event_handler = ScoordHandler()
    observer = Observer()
    observer.schedule(event_handler, str(work_dir), recursive=False)
    observer.start()
    _logger.info("File system observer started")

    try:
        # Main monitoring loop
        while not stop_event.is_set():
            if xtb_process.poll() is not None:
                _logger.info("MD run finished")
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
        _logger.info("Observer stopped")

        try:
            xtb_process.terminate()
            xtb_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _logger.warning("xTB process did not terminate gracefully, forcing kill")
            xtb_process.kill()
        _logger.info("xTB process terminated")

    # Return the result after cleanup
    with result_lock:
        if result is not None:
            _logger.info("Returning result with new product")
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
        charge=charge,
        multiplicity=multiplicity,
        max_products=max_products,
        frame_interval=frame_interval,
        num_workers=n_opt_cores,
        opt_options=opt_options,
        scr=scr,
    )

    if save_traj:
        traj = read_multi_xyz(str((xyz_file.parent / "xtb.trj").absolute()))
        result["traj"] = traj

    # cleanup calc dir
    if not calc_dir:
        work_dir.cleanup()
    return result


if __name__ == "__main__":
    atoms = ["N", "C", "C", "H", "H", "H", "H", "H", "H", "H"]
    coords = [
        [-0.2577470988117524, 0.8838829383188083, -0.8275401594278511],
        [-0.2973839315307961, 0.18801894294025928, 0.44958897759676597],
        [0.5551310303425486, -1.0719018812590675, 0.37795118183108517],
        [0.6944457068755073, 1.1620435600146073, -1.043225225779833],
        [-0.8253498651802484, 1.7238103063235921, -0.7935680193355253],
        [-1.3397283093410641, -0.08443617703404037, 0.6389978412023933],
        [0.045652467786225295, 0.8117651474128725, 1.294316580848435],
        [0.46770576806958214, -1.6405193020514737, 1.2996801209055018],
        [1.6014333054162806, -0.8143243072840862, 0.22592242918279043],
        [0.22776030452129656, -1.689056742105217, -0.4539184478476502],
    ]
    inp_str = """$md
   temp=300
   time=10
   dump=10.0000
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
   kpush=0.1500
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

    import submitit

    executor = submitit.AutoExecutor(
        folder=".tmp",
    )
    executor.update_parameters(
        name="md",
        cpus_per_task=2,
        timeout_min=1200,
        slurm_partition="kemi1",
        slurm_mem_per_cpu=4000,
        slurm_array_parallelism=100,
    )

    _logger.addHandler(logging.StreamHandler())
    _logger.setLevel(logging.DEBUG)
    results = md_step(
        atoms, coords, detailed_input_str=inp_str, n_opt_cores=4, calc_dir="mdd"
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
