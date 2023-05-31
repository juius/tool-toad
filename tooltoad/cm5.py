import math
import os
from typing import List

path = os.path.dirname(__file__)

# read parameters
COV_RADII = {}
with open(path + "/data/radii.txt") as f:
    lines = f.readlines()
for line in lines:
    if line.startswith("#"):
        continue
    name, symbol, rvdw, rcov = line.strip().split()
    COV_RADII[symbol] = float(rcov)

DZ = {}
with open(path + "/data/cm5_params.txt") as f:
    lines = f.readlines()
for line in lines:
    if line.startswith("#"):
        continue
    symbol, dz = line.strip().split()
    DZ[symbol] = float(dz)

DZZ = {
    "H-C": 0.0502,
    "H-N": 0.1747,
    "H-O": 0.1671,
    "C-N": 0.0556,
    "C-O": 0.0234,
    "N-O": -0.0346,
}
tmp_dict = {}
for k, v in DZZ.items():
    tmp_dict["-".join(k.split("-")[::-1])] = -v
DZZ.update(tmp_dict)

ALPHA = 2.474


def calc_cm5(
    atoms: List[str], coords: List[List[int]], hirshfeld_charges: List[float]
) -> List[float]:
    """Calculate CM5 charges.

    Args:
        atoms (List[str]): Atomic symbols
        coords (List[List[int]]): Atomic coordinates
        hirshfeld_charges (List[float]): Hirshfeld charges

    Returns:
        List[float]: CM5 charges
    """
    cm5_charges = []
    for i, (hc, s1, coord1) in enumerate(zip(hirshfeld_charges, atoms, coords)):
        cm5 = hc
        for j, (s2, coord2) in enumerate(zip(atoms, coords)):
            if i != j:
                rkk = sum([(c1 - c2) ** 2 for c1, c2 in zip(coord1, coord2)]) ** 0.5
                r1 = COV_RADII[s1]
                r2 = COV_RADII[s2]
                bkk = math.exp(-ALPHA * (rkk - r1 - r2))
                if s1 == s2:
                    tkk = 0.0
                elif f"{s1}-{s2}" in DZZ:
                    tkk = DZZ[f"{s1}-{s2}"]
                elif f"{s2}-{s1}" in DZZ:
                    tkk = DZZ[f"{s2}-{s1}"]
                else:
                    tkk = DZ[s1] - DZ[s2]
                cm5 += tkk * bkk
        cm5_charges.append(cm5)
    return cm5_charges
