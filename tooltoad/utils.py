import subprocess


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
