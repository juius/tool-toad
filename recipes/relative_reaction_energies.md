# Relative Reaction Energy Calculator

### Setup
Requires
* Python 3.10 or higher.
* xTB >=6.7 installed and on your PATH (e.g. via Conda: `conda install -c conda-forge xtb`)
* ORCA (optional, for r2SCAN-3c single‐point calculations)

If using ORCA, set these environment variables before running:

```bash
export ORCA_EXE=/path/to/orca      # orca executable
export OPENMPI_DIR=/path/to/mpi    # OpenMPI installation directory
```

#### Installing Dependencies

```bash
pip install .
```

### Usage

* **--smi** (string, required)
  SMILES of the reactant.

* **--rxn-smarts** (string, required)
  Reaction SMARTS with the reaction center atom labeled `:1` (e.g. `"[#7:1]>>[#7+:1]-[O-]"`).

* **--solvent** (string, optional)
  Solvent name for implicit solvation (passed to xTB ALPB and ORCA SMD).

* **--only-xtb** (flag, optional)
  Skip ORCA; use xTB only.

* **--n-cores** (int, default 4)
  Number of parallel cores to use.



#### N-Oxidation


```bash
python r2e.py --smi "O=C(C1=NN=C(Cl)C=C1)OC(C)(C)C" --only-xtb --rxn-smarts "[#7:1]>>[#7+:1]-[O-]" --solvent "dcm"
```

The relative reaction energies for each reaction center (the atom with atomMapNum 1 in the reaction SMARTS) are returned:

```
Running relative reaction energy calculations for O=C(C1=NN=C(Cl)C=C1)OC(C)(C)C with reaction [#7:1]>>[#7+:1]-[O-] in dcm.

Relative reaction energies (reaction site: energy):
3: 0.00 kcal/mol
4: 2.36 kcal/mol
```

#### Epoxidation

```bash
python r2e.py --smi "CC12CCC3C(CCC4=CC(=O)C=CC43O)C1CCC2O" --only-xtb --rxn-smarts '[C:1]([*:10])([*:11])=,:[C:2]([*:12])([*:13])>>[*;!a:1]1([*:10])([*:11])[*;!a:2]([*:12])([*:13])[#8]1' --solvent "water"
```

```
Running relative reaction energy calculations for CC12CCC3C(CCC4=CC(=O)C=CC43O)C1CCC2O with reaction [C:1]([*:10])([*:11])=,:[C:2]([*:12])([*:13])>>[*;!a:1]1([*:10])([*:11])[*;!a:2]([*:12])([*:13])[#8]1 in water.

Relative reaction energies (reaction site: energy):
8: 9.02 kcal/mol
12: 0.00 kcal/mol
```
