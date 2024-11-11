import json
from dataclasses import dataclass

import numpy as np
from rdkit import Chem

from tooltoad.chemutils import ac2mol, hartree2kcalmol


@dataclass
class Reaction:
    reactant_results: dict
    product_results: dict
    ts_results: None | dict = None

    def get_species_results(self, species: str):
        if species.lower() in ["reactant", "r"]:
            return self.reactant_results
        elif species.lower() in ["product", "p"]:
            return self.product_results
        elif species.lower() in ["ts", "t"]:
            return self.ts_results
        else:
            raise ValueError("Species must be either 'reactant', 'product', or 'ts'")

    def __repr__(self):
        return f"Reaction with reactant{', transition state ' if self.ts_results else ' '}and product"

    def get_smi(self, species: str, level: str):
        results = self.get_species_results(species)
        return Chem.MolToSmiles(
            ac2mol(
                results[level]["atoms"], results[level]["opt_coords"], sanitize=False
            ),
        )

    def get_barrier(
        self, level: str, forward: bool = True, energy_type="electronic_energy"
    ):
        middle = self.get_species_results("ts")[level][energy_type]
        side_species = "product" if forward else "reactant"
        side = self.get_species_results(side_species)[level][energy_type]
        return hartree2kcalmol(middle - side)

    def save(self, file_path: str):
        with open(file_path, "w") as f:
            json.dump(
                {
                    k: (v.tolist() if isinstance(v, np.ndarray) else v)
                    for k, v in self.__dict__.items()
                },
                f,
            )

    @classmethod
    def load(cls, file_path: str):
        with open(file_path, "r") as f:
            data = json.load(f)
        return cls(**data)
