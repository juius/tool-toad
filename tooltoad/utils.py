import subprocess
import warnings


def stream(cmd, cwd=None, shell=True):
    """Execute command in directory, and stream stdout."""
    popen = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        shell=shell,
        cwd=cwd,
    )
    for stdout_line in iter(popen.stdout.readline, ""):
        yield stdout_line

    # Yield errors
    stderr = popen.stderr.read()
    popen.stdout.close()
    yield stderr

    return


def check_executable(executable):
    """Check if executable is in PATH."""
    results = stream(f"which {executable}")
    result = list(results)[0]
    if result.startswith("which: no"):
        warnings.warn(f"Executable {executable} not found in PATH")
