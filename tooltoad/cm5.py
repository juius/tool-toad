import math
from typing import List

# fmt: off
# flake8: noqa
COV_RADII = {
    "Al": 1.24, "Sb": 1.40, "Ar": 1.01, "As": 1.20, "Ba": 2.06, "Be": 0.99, "Bi": 1.50,
    "B":  0.84, "Br": 1.17, "Cd": 1.40, "Ca": 1.74, "C":  0.75, "Cs": 2.38, "Cl": 1.00,
    "Cr": 1.30, "Co": 1.18, "Cu": 1.22, "F":  0.60, "Ga": 1.23, "Ge": 1.20, "Au": 1.30,
    "Hf": 1.64, "He": 0.37, "H":  0.32, "In": 1.42, "I":  1.36, "Ir": 1.32, "Fe": 1.24,
    "Kr": 1.16, "Pb": 1.45, "Li": 1.30, "Mg": 1.40, "Mn": 1.29, "Hg": 1.32, "Mo": 1.46,
    "Ne": 0.62, "Ni": 1.17, "Nb": 1.56, "N":  0.71, "Os": 1.36, "O":  0.64, "Pd": 1.30,
    "P":  1.09, "Pt": 1.30, "Po": 1.42, "K":  2.00, "Rn": 1.46, "Re": 1.41, "Rh": 1.34,
    "Rb": 2.15, "Ru": 1.36, "Sc": 1.59, "Se": 1.18, "Si": 1.14, "Ag": 1.36, "Na": 1.60,
    "Sr": 1.90, "S":  1.04, "Te": 1.37, "Tl": 1.44, "Sn": 1.40, "Ti": 1.48, "W":  1.50,
    "V":  1.44, "Xe": 1.36, "Y":  1.76, "Zn": 1.20, "Zr": 1.64,
}

DZ = {
   "H":   0.0056, "He": -0.1543, "Li":  0.0000, "Be":  0.0333, "B":  -0.1030,
   "C":  -0.0446, "N":  -0.1072, "O":  -0.0802, "F":  -0.0629, "Ne": -0.1088,
   "Na":  0.0184, "Mg":  0.0000, "Al": -0.0726, "Si": -0.0790, "P":  -0.0756,
   "S":  -0.0565, "Cl": -0.0444, "Ar": -0.0767, "K":   0.0130, "Ca":  0.0000,
   "Sc":  0.0000, "Ti":  0.0000, "V":   0.0000, "Cr":  0.0000, "Mn":  0.0000,
   "Fe":  0.0000, "Co":  0.0000, "Ni":  0.0000, "Cu":  0.0000, "Zn":  0.0000,
   "Ga": -0.0512, "Ge": -0.0557, "As": -0.0533, "Se": -0.0399, "Br": -0.0313,
   "Kr": -0.0541, "Rb":  0.0092, "Sr":  0.0000, "Y":   0.0000, "Zr":  0.0000,
   "Nb":  0.0000, "Mo":  0.0000, "Ru":  0.0000, "Rh":  0.0000, "Pd":  0.0000,
   "Ag":  0.0000, "Cd":  0.0000, "In": -0.0361, "Sn": -0.0393, "Sb": -0.0376,
   "Te": -0.0281, "I":  -0.0220, "Xe": -0.0381, "Cs":  0.0065, "Ba":  0.0000,
   "Hf":  0.0000, "W":   0.0000, "Re":  0.0000, "Os":  0.0000, "Ir":  0.0000,
   "Pt":  0.0000, "Au":  0.0000, "Hg":  0.0000, "Tl": -0.0255, "Pb": -0.0277,
   "Bi": -0.0265, "Po": -0.0198, "Rn": -0.0269,
}

DZZ = {
    "H-C":  0.0502, "H-N":  0.1747, "H-O":  0.1671, "C-N":  0.0556, "C-O":  0.0234,
    "N-O": -0.0346, "C-H": -0.0502, "N-H": -0.1747, "O-H": -0.1671, "N-C": -0.0556,
    "O-C": -0.0234, "O-N":  0.0346,
}

ALPHA = 2.474
# fmt: on


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
