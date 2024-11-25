# Recepies for common tasks in computational chemistry

* [Performing QM-calculations using xTB and ORCA](qm_calculation.md)
* [Generating low energy conformers](conformer_generation.md)
* [Scanning a potential energy surface along multiple dimensions](pes_scan.md)

---

All function take the molecular structure as list of atom symbols and cartesian coordinates as input:

```python
atoms:list[str] = ["O", "H", "H"]

coords:list[list[float]] = [
    [-0.00170769,  0.38259740, -0.        ],
    [-0.79591444, -0.19194612, -0.        ],
    [ 0.79762213, -0.19065129,  0.        ],
]
```

These can be obtained via RDKit:

```python
from rdkit import Chem
from rdkit.Chem import rdDistGeom

mol = Chem.MolFromSmiles("Cc1ccc(cc1)S(=O)(=O)O")
mol3d = Chem.AddHs(mol)
rdDistGeom.EmbedMolecule(mol3d)

atoms = [a.GetSymbol() for a in mol.GetAtoms()]
coords = mol.GetConformer().GetPositions()
```

or from a xyz-file:

```python
from tooltoad.chemutils import xyz2ac

with open("path/to/file.xyz", "r") as f:
    xyz_str = f.read()

atoms, coords = xyz2ac(xyz_str)
```

The charge and multiplicity of the molecule are parsed as integers to `charge` and `multiplicity` in all functions that perform any QM-calculations, such as `xtb.xtb_calculate`, `orca.orca_calculate`, `crest.run_crest`, `crest.refine_with_orca`, `ndscan.PotentialEnergySurface.xtb`, etc...
