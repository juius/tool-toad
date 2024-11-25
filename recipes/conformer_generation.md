# Functions for generating a conformer ensemble using CREST and reranking the ensemble based on ORCA calculations

> [!NOTE]
> CREST needs to be installed and findable in PATH as `crest` (f.x.: `conda install conda-forge::crest`)

> [!IMPORTANT]
> CREGEN doesn't filter conformers properly on Apple silicon machines as only one conformer is retained.

All functions take a list of atom symbols and coordinates as input:
```python
atoms = ['C', 'C', 'C', 'O', 'O', 'C', 'C', 'O', 'O', 'C', 'C', 'C', 'H', 'H', 'H', 'H', 'H', 'H']

coords = [
    [ 1.19138679, -1.40549202,  0.18324722],
    [ 0.69723073, -0.13106723,  0.04156491],
    [ 1.64007473,  0.95001069,  0.17488094],
    [ 2.86701061,  0.78882044,  0.40954563],
    [ 1.63835925,  2.31448511,  0.12122411],
    [-0.63637865,  0.05717517, -0.21399364],
    [-1.18597588,  1.39223224, -0.36832485],
    [-2.38708634,  1.68583597, -0.60337718],
    [-0.90954568,  2.72768103, -0.36888791],
    [-1.55863875, -0.95355345, -0.34622660],
    [-1.04272317, -2.23278840, -0.20030867],
    [ 0.29342538, -2.44413461,  0.05662380],
    [ 2.22682370, -1.63822406,  0.38504608],
    [ 1.70031900,  2.89189042,  0.94598439],
    [-0.88978837,  3.34347960,  0.45475900],
    [-2.60482277, -0.78823689, -0.54739929],
    [-1.70608146, -3.09695465, -0.29007135],
    [ 0.66641090, -3.46115935,  0.16571343],
]
```


## 1. Optimization of input structure using GFN2-xTB and ALPB solvent model for water

```python
from tooltoad.xtb import xtb_calculate

preopt_results = xtb_calculate(
    atoms=atoms, coords=coords, options={"opt": "verytight", "alpb": "water"}, n_cores=4
)
```

`preopt_results` is a dictionary with the xtb calculation results, we are interested in the optimized coordinates (`opt_coords`)

## 2. Cresting conformer ensemble using CREST

Keywords for CREST are given as keyword arguments to `run_crest`, valid arguments can be found [here](https://crest-lab.github.io/crest-docs/page/documentation/keywords.html).

```python
from tooltoad.crest import run_crest

pre_opt_coords = preopt_results["opt_coords"]

crest_out = run_crest(
    atoms=atoms,
    coords=pre_opt_coords,
    n_cores=4,
    alpb="water",
)
```

`crest_out` is a list of dictionaries with keys `atoms`, `coords` and `xtb_energy`, each dictionary corresonds to one conformer. The conformers are sorted in ascending order of `xtb_energy` which is the electronic energy of the conformer relative to the lowest energy conformer.

## 3. Refine CREST ensemble

The ranking of the conformers can be refined by ORCA singlepoint calculations. Options for ORCA are parsed as a dictionary.

```python
from tooltoad.crest import refine_with_orca

refine_with_orca(crest_out, options={"r2scan-3c": None, "SMD": "water"}, n_cores=4)
```

The dictionaries in `crest_out` now contain also the key `orca_energy` which is the electronic energy calculated at the used level of theory relative to the lowest energy conformer.
The list of dictionaries is sorted so the first item corresponds to the conformer with the lowest energy as calculated by ORCA.

The lowest energy conformer is obtained as:

```python
atoms, coords = crest_out[0]["atoms"], crest_out[0]["coords"]
```
