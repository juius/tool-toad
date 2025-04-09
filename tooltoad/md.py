import json
import logging
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import networkx as nx
import numpy as np

from tooltoad.chemutils import ac2xyz, read_multi_xyz, xyz2ac
from tooltoad.orca import orca_calculate
from tooltoad.utils import WorkingDir, check_executable, stream
from tooltoad.xtb import set_threads, write_xyz, xtb_calculate

_logger = logging.getLogger(__name__)


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
        stderr=subprocess.PIPE,  # Redirect stderr to suppress messages
        preexec_fn=os.setsid,
        env=env,
    )
    opt_options = options.copy()
    opt_options.pop("md")
    products = track_trajectory(
        process,
        traj_file=str((xyz_file.parent / "xtb.trj").absolute()),
        max_products=max_products,
        frame_interval=frame_interval,
        num_workers=n_opt_cores,
        opt_options=opt_options,
        scr=scr,
    )
    if save_traj:
        traj = read_multi_xyz(str((xyz_file.parent / "xtb.trj").absolute()))
        products = (products, traj)

    # cleanup calc dir
    if not calc_dir:
        work_dir.cleanup()
    return products


def node_match(node1, node2):
    # Check if the atom types match in addition to other structural attributes
    return node1.get("atom_type") == node2.get("atom_type")


def isomorphic_to_any(graph, graph_list):
    graph_properties = (
        len(graph.nodes),
        len(graph.edges),
    )  # Basic properties to compare

    for other_graph in graph_list:
        # Check basic properties first
        if (len(other_graph.nodes), len(other_graph.edges)) != graph_properties:
            continue  # Skip if basic properties don't match
        if nx.is_isomorphic(graph, other_graph, node_match=node_match):
            return True
    return False


def process_frames(frame_queue, products):
    """Processes the frames from the queue using multiple workers."""
    product_graphs = []
    while True:
        # Wait for a frame to be available in the queue
        frame_count, frame_data = frame_queue.get()
        # If the frame_data is None, exit the worker thread
        if frame_data is None:
            break

        if frame_data:
            _logger.info(f"Processing frame {frame_count}...")
            graph, crude_results = analyse_snapshot(frame_count, **frame_data)
            if crude_results is None or not crude_results["normal_termination"]:
                continue
            _logger.info(f"Frame {frame_count} processed.")
            if graph:
                if not isomorphic_to_any(graph, product_graphs):
                    product_graphs.append(graph)
                    if frame_count > 0:
                        _logger.info("New product found.")
                        _logger.info("Performing full optimization...")
                        opt_options = frame_data["options"].copy()
                        opt_options["opt"] = None
                        opt_options["freq"] = None
                        opt_options["XTB2"] = None
                        opt_results = orca_calculate(
                            atoms=crude_results["atoms"],
                            coords=crude_results["opt_coords"],
                            charge=crude_results["charge"],
                            multiplicity=crude_results["multiplicity"],
                            options=opt_options,
                            scr=frame_data["scr"],
                            n_cores=1,
                        )
                        if opt_results["normal_termination"]:
                            _logger.info("Full optimization successful.")
                            products.append((frame_count, opt_results))
                        else:
                            _logger.info("Full optimization failed.")


def track_trajectory(
    xtb_process,
    traj_file: str,
    max_products: None | int = None,
    charge: int = 0,
    multiplicity: int = 1,
    opt_options: dict = {},
    frame_interval: int = 10,
    num_workers: int = 1,
    time_interval: int = 2,
    scr: str = ".",
):
    while not os.path.isfile(traj_file) or os.path.getsize(traj_file) == 0:
        _logger.debug(f"Waiting for trajectory file {traj_file} to be created...")
        time.sleep(1)
    frame_count = 0
    last_position = 0
    last_mod_time = os.path.getmtime(traj_file)
    file_closed = False

    # Create a queue to store the frames for processing
    frame_queue = multiprocessing.Queue()
    manager = multiprocessing.Manager()
    products = manager.list()  # Shared list

    # Start worker processes
    workers = []
    for _ in range(num_workers):
        p = multiprocessing.Process(target=process_frames, args=(frame_queue, products))
        p.start()
        workers.append(p)

    _logger.debug(f"Starting {num_workers} worker processes...")

    with open(traj_file, "r") as file:
        # Read the first line to get the number of atoms
        file.seek(0)
        n_atoms = int(file.readline().strip())
        lines_per_frame = n_atoms + 2
        while not file_closed:
            # Move the pointer to the last position after each frame
            file.seek(last_position)

            # Read a large block of lines in one go
            lines = []
            for _ in range(lines_per_frame):
                line = file.readline()
                if line:
                    lines.append(line.strip())
                else:
                    break

            # If there are lines, put them into the queue and process
            if lines:
                last_position = (
                    file.tell()
                )  # Update the position after reading the frame
                if frame_count % frame_interval == 0:
                    # Only add every frame_interval-th frame
                    atoms, coords = xyz2ac("\n".join(lines))
                    frame_data = {
                        "atoms": atoms,
                        "coords": coords,
                        "charge": charge,
                        "multiplicity": multiplicity,
                        "options": opt_options,
                        "scr": scr,
                    }
                    _logger.info(f"Adding frame {frame_count} to the queue.")
                    frame_queue.put((frame_count, frame_data))
                frame_count += 1

            # If no new data, wait briefly
            if not lines:
                time.sleep(time_interval)
            if (os.path.getmtime(traj_file) == last_mod_time) and (
                file.tell() == os.path.getsize(traj_file)
            ):
                time.sleep(10)
                if (os.path.getmtime(traj_file) == last_mod_time) and (
                    file.tell() == os.path.getsize(traj_file)
                ):
                    file_closed = True
            else:
                last_mod_time = os.path.getmtime(traj_file)
            if max_products and len(products) >= max_products:
                try:
                    os.killpg(os.getpgid(xtb_process.pid), signal.SIGTERM)
                except ProcessLookupError:
                    pass
                xtb_process.wait()
                _logger.info("Maximum number of products reached, terminating MD...")

                time.sleep(0.5)
                break
        # Collect all the products from the result_queue
        _logger.info("All frames read, waiting for workers to finish...")
        # Signal all workers to stop
        for _ in range(num_workers):
            frame_queue.put((None, None))

        # Wait for all worker processes to finish
        for p in workers:
            p.join()

        # Return all found products after file is closed and jobs finished
        _logger.info(f"Found {len(products)} unique products.")

        return list(products)


def gfnff_connectivity(atoms, coords, charge, multiplicity, scr):
    # Determine connectivity based on GFNFF-xTB implementation
    calc_dir = tempfile.TemporaryDirectory(dir=scr)
    tmp_file = Path(calc_dir.name) / "input.xyz"
    with open(tmp_file, "w") as f:
        f.write(ac2xyz(atoms, coords))
    CMD = f"xtb --gfnff {str(tmp_file.name)} --chrg {charge} --uhf {multiplicity-1} --norestart --wrtopo blist"
    _ = list(stream(CMD, cwd=calc_dir.name))
    with open(Path(calc_dir.name) / "gfnff_lists.json", "r") as f:
        data_dict = json.load(f)
    calc_dir.cleanup()
    blist = data_dict["blist"]
    adj = np.zeros((len(atoms), len(atoms)), dtype=int)
    for i, j, _ in blist:
        adj[i - 1, j - 1] = 1
        adj[j - 1, i - 1] = 1
    return adj


def analyse_snapshot(frame_count, atoms, coords, charge, multiplicity, options, scr):
    """Processes the given frame of lines."""
    try:
        if frame_count > 0:
            # crude xTB optimization
            crude_options = {}
            crude_options.update(options)
            crude_options["opt"] = "crude"
            crude_opt = xtb_calculate(
                atoms, coords, charge, multiplicity, options=crude_options, scr=scr
            )
            # TODO: can't i get the connectivity from the gfn2 calc?
            if not crude_opt["normal_termination"]:
                _logger.debug("abnormal termination of xTB")
                return None, None
        else:
            # no optimization on first frame
            crude_opt = {
                "atoms": atoms,
                "opt_coords": coords,
                "charge": charge,
                "multiplicity": multiplicity,
                "normal_termination": True,
            }
        adj = gfnff_connectivity(
            atoms, crude_opt["opt_coords"], charge, multiplicity, scr=scr
        )
        graph = nx.from_numpy_array(adj)
        for i, atom_type in enumerate(atoms):
            graph.nodes[i]["atom_type"] = atom_type
        return graph, crude_opt
    except Exception as e:
        _logger.info(f"Error in frame {frame_count}: {e}")
        return None, None


def graceful_shutdown(executor, frame_queue):
    """Gracefully shuts down the workers and other threads on interruption
    (Ctrl+C)."""
    print("\nGracefully shutting down...")
    # Signal all workers to stop by placing None into the queue
    for _ in range(executor._max_workers):
        frame_queue.put(None)

    # Wait for all workers to finish
    executor.shutdown(wait=True)
    sys.exit(0)


if __name__ == "__main__":
    _logger.setLevel(logging.DEBUG)
    _logger.addHandler(logging.StreamHandler())
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
    products = md_step(atoms, coords, detailed_input_str=inp_str, n_opt_cores=4)
