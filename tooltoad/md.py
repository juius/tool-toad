import json
import logging
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import networkx as nx
import numpy as np

from tooltoad.chemutils import ac2xyz, xyz2ac
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
):
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


def track_trajectory(
    xtb_process,
    traj_file: str,
    max_products: None | int = None,
    charge: int = 0,
    multiplicity: int = 1,
    opt_options: dict = {},
    frame_interval: int = 10,
    num_workers: int = 1,
    time_interval: int = 1,
    scr: str = ".",
):
    while not os.path.isfile(traj_file):
        _logger.debug(f"Waiting for trajectory file {traj_file} to be created...")
        time.sleep(1)
    frame_count = 0
    last_position = 0
    last_mod_time = os.path.getmtime(traj_file)
    file_closed = False

    # Create a queue to store the frames for processing
    frame_queue = queue.Queue()

    products = []

    # Create a thread pool to process frames asynchronously with multiple workers
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        _logger.debug(f"Starting {num_workers} worker threads...")
        for idx in range(num_workers):
            executor.submit(process_frames, frame_queue, products)

        # Register a signal handler for graceful shutdown
        signal.signal(
            signal.SIGINT,
            lambda signum, frame: graceful_shutdown(executor, frame_queue),
        )

        with open(traj_file, "r") as file:
            # Read the first line to get the number of atoms
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
                    time.sleep(5)
                    if (os.path.getmtime(traj_file) == last_mod_time) and (
                        file.tell() == os.path.getsize(traj_file)
                    ):
                        file_closed = True
                else:
                    last_mod_time = os.path.getmtime(traj_file)
                if max_products and len(products) >= max_products:
                    os.killpg(os.getpgid(xtb_process.pid), signal.SIGTERM)
                    xtb_process.wait()
                    _logger.info(
                        "Maximum number of products reached, terminating MD..."
                    )

                    time.sleep(0.5)
                    break
        # cleanup calc dir
        work_dir = Path(traj_file).parent
        shutil.rmtree(work_dir)
        # Collect all the products from the result_queue
        _logger.info("All frames read, waiting for workers to finish...")
        # Signal all workers to stop
        for _ in range(num_workers):
            frame_queue.put(None)  # Send stop signal to each worker

        # Wait for all workers to finish
        executor.shutdown(wait=True)

        # Return all found products after file is closed and jobs finished
        print(f"Found {len(products)} unique products.")
        return products


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
            _logger.info(f"Frame {frame_count} processed.")
            if graph:
                if not isomorphic_to_any(graph, product_graphs):
                    product_graphs.append(graph)
                    if frame_count > 0:
                        _logger.info("New product found.")
                        products.append((frame_count, crude_results))


def gfnff_connectivity(atoms, coords, charge, multiplicity):
    # Determine connectivity based on GFNFF-xTB implementation
    calc_dir = tempfile.TemporaryDirectory()
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


def analyse_snapshot(frame_count, atoms, coords, charge, multiplicity, options):
    """Processes the given frame of lines."""
    try:
        if frame_count > 0:
            # crude xTB optimization
            crude_options = {}
            crude_options.update(options)
            crude_options["opt"] = "crude"
            crude_opt = xtb_calculate(
                atoms,
                coords,
                charge,
                multiplicity,
                options=crude_options,
            )
            # TODO: can't i get the connectivity from the gfn2 calc?
            if not crude_opt["normal_termination"]:
                return None, None
        else:
            # no optimization on first frame
            crude_opt = {"opt_coords": coords}
        adj = gfnff_connectivity(atoms, crude_opt["opt_coords"], charge, multiplicity)
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
    _logger.setLevel(logging.INFO)
    _logger.addHandler(logging.StreamHandler())
    # Replace with the path to your trajectory file
    products = track_trajectory(
        "/Users/julius/opt/d2/notebooks/imiprimine-0/xtb.trj",
        frame_interval=25,
        num_workers=4,
    )
