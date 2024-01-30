## Setup

### Running ORCA calculation in parallel

When using `orca_calculate` with `n_cores` > 1 the full path to the ORCA executable needs to be provided as `orca_cmd`.
Additionally, the path to the OpenMPI executable must be set in the `PATH` and `LD_LIBRARY_PATH` variables via the `set_env` string, see [here](https://github.com/juius/tool-toad/blob/df787b696adccabc68fda45ac9e6fe0f957b8270/tooltoad/orca.py#L10) for an example.

