import contextlib
import os
import random
import shutil
import signal
import string
import subprocess
import threading
import warnings
from pathlib import Path
from queue import Empty, Queue
from typing import Generator

import joblib

alphabet = string.ascii_lowercase + string.digits

STANDARD_PROPERTIES = {"xtb": {"total energy": "electronic_energy"}, "orca": {}}


def stream(
    cmd: str, cwd: None | Path = None, shell: bool = True
) -> Generator[str, None, None]:
    """Execute a command and stream stdout and stderr concurrently."""

    def enqueue_output(pipe, queue):
        """Read lines from the pipe and put them into a queue."""
        try:
            for line in iter(pipe.readline, ""):
                queue.put(line)
        finally:
            pipe.close()

    # Start the subprocess
    popen = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        shell=shell,
        cwd=cwd,
        preexec_fn=os.setsid,  # Start a new process group for better control
    )

    # Queues for stdout and stderr
    stdout_queue = Queue()
    stderr_queue = Queue()

    # Threads for reading stdout and stderr
    stdout_thread = threading.Thread(
        target=enqueue_output, args=(popen.stdout, stdout_queue)
    )
    stderr_thread = threading.Thread(
        target=enqueue_output, args=(popen.stderr, stderr_queue)
    )
    stdout_thread.start()
    stderr_thread.start()

    try:
        while True:
            stdout_done = False
            stderr_done = False

            # Check stdout
            while not stdout_done:
                try:
                    yield stdout_queue.get_nowait()
                except Empty:
                    stdout_done = popen.stdout.closed and stdout_queue.empty()

            # Check stderr
            while not stderr_done:
                try:
                    yield stderr_queue.get_nowait()
                except Empty:
                    stderr_done = popen.stderr.closed and stderr_queue.empty()

            # Break if the process has finished and both queues are empty
            if popen.poll() is not None and stdout_done and stderr_done:
                break

    except KeyboardInterrupt:
        print("\nCtrl+C pressed. Terminating the process...")
        os.killpg(os.getpgid(popen.pid), signal.SIGTERM)
        popen.wait()
        print("Process terminated.")
    finally:
        # Ensure threads are joined and subprocess cleanup
        stdout_thread.join()
        stderr_thread.join()
        popen.wait()


def check_executable(executable: str):
    """Check if executable is in PATH."""
    results = stream(f"which {executable}")
    for result in results:
        break
    if result.startswith("which: no"):
        warnings.warn(f"Executable {executable} not found in PATH")


class WorkingDir:
    def __init__(self, root: str = ".", name: str = None) -> None:
        self.root = Path(root)
        self.name = name if name else self._random_str()
        self.dir = self.root / self.name
        self.create()

    def __str__(self) -> str:
        return str(self.dir.resolve())

    def __repr__(self) -> str:
        return self.__str__()

    def __truediv__(self, name: str) -> str:
        return self.dir / name

    def _random_str(self) -> str:
        name = "_" + "".join(random.choices(alphabet, k=6))
        while (self.root / name).exists():
            name = "".join(random.choices(alphabet, k=6))
        return name

    def create(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)

    def cleanup(self) -> None:
        try:
            # print("removing ", self.dir.absolute())
            shutil.rmtree(self.dir.absolute())
        except FileNotFoundError:
            pass


class WorkingFile:
    def __init__(self, root: str = ".", filename: str = None, mode="w") -> None:
        self.root = Path(root)
        self.filename = filename if filename else self._random_str()
        self.mode = mode
        self.path = self.root / self.filename

    def _random_str(self) -> str:
        name = "".join(random.choices(alphabet, k=6)) + ".ttxt"
        while (self.root / name).exists():
            name = "".join(random.choices(alphabet, k=6)) + ".ttxt"
        return name

    def __str__(self) -> str:
        return str(self.path.resolve())

    def __repr__(self) -> str:
        return self.__str__()

    def create(self) -> None:
        with open(str(self), self.mode) as _:
            pass

    def cleanup(self) -> None:
        try:
            shutil.rmtree(self.path)
        except FileNotFoundError:
            pass

    @property
    def stem(self):
        return str(self.path.stem)


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given
    as argument."""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
