import contextlib
import os
import random
import shutil
import signal
import string
import subprocess
import warnings
from pathlib import Path
from typing import Generator

import joblib

alphabet = string.ascii_lowercase + string.digits

STANDARD_PROPERTIES = {"xtb": {"total energy": "electronic_energy"}, "orca": {}}


def stream(
    cmd: str, cwd: None | Path = None, shell: bool = True
) -> Generator[str, None, None]:
    """Execute a command and stream stdout and stderr concurrently."""
    with subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,  # Use text mode for string-based reading
        shell=shell,
        cwd=cwd,
        preexec_fn=os.setsid,  # Start a new process group for better control
        bufsize=1,  # Line-buffered output for immediate feedback
    ) as process:
        try:
            for line in iter(process.stdout.readline, ""):
                yield line
            for line in iter(process.stderr.readline, ""):
                yield line
        except KeyboardInterrupt:
            print("\nCtrl+C pressed. Terminating the process...")
            os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            process.wait()
            print("Process terminated.")
        finally:
            process.stdout.close()
            process.stderr.close()
            process.wait()


def check_executable(executable: str):
    """Check if executable is in PATH."""
    results = stream(f"which {executable}")
    result = next(results)
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
