import copy
import os
from io import StringIO

import numpy as np
import py3Dmol
from IPython.display import display
from xyz2mol import xyz2mol
from ppqm import linesio

from rdkit import Chem
from rdkit.Geometry import Point3D

from .gaussian import get_gaussian_geometry

import matplotlib.pyplot as plt
import matplotlib.patches as mpatch
from matplotlib.patches import FancyBboxPatch


def interpolate(mol1, mol2, steps=5):
    c2 = mol2.GetConformer()
    c1 = mol1.GetConformer()
    p2 = c2.GetPositions()
    p1 = c1.GetPositions()
    mols = []
    vec = p2 - p1
    norm = np.sum(np.abs(vec))
    steps = np.min([int(np.round(norm * 20)), 10])
    for x in np.linspace(0, 1, steps):
        new_mol = copy.deepcopy(mol1)
        conf = new_mol.GetConformer()
        new_pos = p1 + x * vec
        for i in range(new_mol.GetNumAtoms()):
            x, y, z = new_pos[i]
            conf.SetAtomPosition(i, Point3D(x, y, z))
        mols.append(new_mol)
    return mols


def parse_coordline(line):
    line = line.split()
    atom = line[1]
    coord = list(map(float, line[3:]))

    return atom, coord


def get_element_symbols(atom_numbers):
    pse = Chem.GetPeriodicTable()
    symb = []
    for n in atom_numbers:
        symb.append(pse.GetElementSymbol(int(n)))
    return symb


def dict2mol(dict):
    mol = xyz2mol(dict["atoms"], dict["coords"])[0]
    return mol


def smooth_traj(traj, steps=5):
    new_traj = []
    for i in range(len(traj)):
        if i > 0:
            inter = interpolate(dict2mol(traj[i - 1]), dict2mol(traj[i]), steps=steps)
            if i > 1:
                inter = inter[1:]
            for i in inter:
                new_traj.append(i)
    return new_traj


def traj2model(traj, smooth_steps=5):
    mols = smooth_traj(traj, smooth_steps)
    sio = StringIO()
    w = Chem.SDWriter(sio)
    for m in mols:
        w.write(m)
    w.flush()
    return sio.getvalue()


# fix smooting step size dependent of len of step
def draw3d(
    mols, multipleConfs=False, confId = -1, atomlabel=False, vibration=True, trajectory=False
):
    width = 900
    height = 600
    p = py3Dmol.view(width=width, height=height)
    if type(mols) is not list:
        mols = [mols]
    for mol in mols:
        if multipleConfs:
            for conf in mol.GetConformers():
                mb = Chem.MolToMolBlock(mol, confId=conf.GetId())
                p.addModel(mb, "sdf")
        else:
            if type(mol) is str:
                if os.path.splitext(mol)[-1] == ".xyz":
                    xyz_f = open(mol)
                    line = xyz_f.read()
                    xyz_f.close()
                    if trajectory:
                        p.addModelsAsFrames(line, "xyz")
                        p.animate({"loop": "forward", "reps": 10})
                    else:
                        p.addModel(line, "xyz")
                elif os.path.splitext(mol)[-1] == ".log":
                    if trajectory:
                        with open(mol) as f:
                            lines = f.readlines()

                        starts = linesio.get_indices(lines, "Standard orientation:")
                        stops = linesio.get_indices(
                            lines, "Rotational constants (GHZ):"
                        )

                        if len(starts) == 0:
                            starts = linesio.get_indices(lines, "Input orientation:")
                        if len(stops) == 0:
                            stops = linesio.get_indices(
                                lines, "Distance matrix (angstroms):"
                            )

                        traj = []
                        for sta, sto in zip(starts, stops):
                            block = lines[sta + 5 : sto - 1]
                            atoms = []
                            coords = []
                            for l in block:
                                atom, coord = parse_coordline(l)
                                atoms.append(int(atom))
                                coords.append(coord)
                            traj.append({"atoms": atoms, "coords": coords})
                        mbs = traj2model(traj, smooth_steps=10)
                        p.addModelsAsFrames(mbs, "sdf")
                        p.animate({"loop": "forward", "reps": 5})
                    else:
                        out = get_gaussian_geometry(mol)
                        mol = xyz2mol(out["atoms"], out["coords"])[0]
                        mb = Chem.MolToMolBlock(mol)
                        p.addModel(mb, "sdf")
                        if vibration:
                            lowest_freq = out["normal_coords"][0]
                            propmap = []
                            for j, m in enumerate(lowest_freq):
                                propmap.append(
                                    {
                                        "index": j,
                                        "props": {
                                            "dx": m[0],
                                            "dy": m[1],
                                            "dz": m[2],
                                        },
                                    }
                                )
                            p.mapAtomProperties(propmap)
                            p.vibrate(20, 1, True)
                            p.animate(
                                {"loop": "backAndForth", "interval": 1, "reps": 20}
                            )

            else:  # if rdkit.mol
                mb = Chem.MolToMolBlock(mol, confId=confId)
                p.addModel(mb, "sdf")
    p.setStyle({"sphere": {"radius": 0.4}, "stick": {}})
    if atomlabel:
        p.addPropertyLabels("index")
    p.zoomTo()
    return p

def format_value(value, decimals):
    """
    Checks the type of a variable and formats it accordingly.
    Floats has 'decimals' number of decimals.
    """

    if isinstance(value, (float, np.float)):
        return f"{value:.{decimals}f}"
    elif isinstance(value, (int, np.integer)):
        return f"{value:d}"
    else:
        return f"{value}"


def values_to_string(values, decimals):
    """
    Loops over all elements of 'values' and returns list of strings
    with proper formating according to the function 'format_value'.
    """

    res = []
    for value in values:
        if isinstance(value, list):
            tmp = [format_value(val, decimals) for val in value]
            res.append(f"{tmp[0]} +/- {tmp[1]}")
        else:
            res.append(format_value(value, decimals))
    return res


def len_of_longest_string(s):
    """Returns the length of the longest string in a list of strings"""
    return len(max(s, key=len))

def nice_string_output(d, extra_spacing=0, decimals=3):
    """
    Takes a dictionary d consisting of names and values to be properly formatted.
    Makes sure that the distance between the names and the values in the printed
    output has a minimum distance of 'extra_spacing'. One can change the number
    of decimals using the 'decimals' keyword.
    """

    names = d.keys()
    max_names = len_of_longest_string(names)

    values = values_to_string(d.values(), decimals=decimals)
    max_values = len_of_longest_string(values)

    string = ""
    for name, value in zip(names, values):
        spacing = extra_spacing + max_values + max_names - len(name) - 1
        string += "{name:s} {value:>{spacing}} \n".format(
            name=name, value=value, spacing=spacing
        )
    return string[:-2]

COLORS = {
    "red": {
        0: "#ffebee",
        1: "#ffcdd2",
        2: "#ef9a9a",
        3: "#e57373",
        4: "#ef5350",
        5: "#f44336",
        6: "#e53935",
        7: "#d32f2f",
        8: "#c62828",
        9: "#b71c1c",
    },
    "pink": {
        0: "#fce4ec",
        1: "#f8bbd0",
        2: "#f48fb1",
        3: "#f06292",
        4: "#ec407a",
        5: "#e91e63",
        6: "#d81b60",
        7: "#c2185b",
        8: "#ad1457",
        9: "#880e4f",
    },
    "purple": {
        0: "#f3e5f5",
        1: "#e1bee7",
        2: "#ce93d8",
        3: "#ba68c8",
        4: "#ab47bc",
        5: "#9c27b0",
        6: "#8e24aa",
        7: "#7b1fa2",
        8: "#6a1b9a",
        9: "#4a148c",
    },
    "d.purple": {
        0: "#ede7f6",
        1: "#d1c4e9",
        2: "#b39ddb",
        3: "#9575cd",
        4: "#7e57c2",
        5: "#673ab7",
        6: "#5e35b1",
        7: "#512da8",
        8: "#4527a0",
        9: "#311b92",
    },
    "indigo": {
        0: "#e8eaf6",
        1: "#c5cae9",
        2: "#9fa8da",
        3: "#7986cb",
        4: "#5c6bc0",
        5: "#3f51b5",
        6: "#3949ab",
        7: "#303f9f",
        8: "#283593",
        9: "#1a237e",
    },
    "blue": {
        0: "#e3f2fd",
        1: "#bbdefb",
        2: "#90caf9",
        3: "#64b5f6",
        4: "#42a5f5",
        5: "#2196f3",
        6: "#1e88e5",
        7: "#1976d2",
        8: "#1565c0",
        9: "#0d47a1",
    },
    "l.blue": {
        0: "#e1f5fe",
        1: "#b3e5fc",
        2: "#81d4fa",
        3: "#4fc3f7",
        4: "#29b6f6",
        5: "#03a9f4",
        6: "#039be5",
        7: "#0288d1",
        8: "#0277bd",
        9: "#01579b",
    },
    "cyan": {
        0: "#e0f7fa",
        1: "#b2ebf2",
        2: "#80deea",
        3: "#4dd0e1",
        4: "#26c6da",
        5: "#00bcd4",
        6: "#00acc1",
        7: "#0097a7",
        8: "#00838f",
        9: "#006064",
    },
    "teal": {
        0: "#e0f2f1",
        1: "#b2dfdb",
        2: "#80cbc4",
        3: "#4db6ac",
        4: "#26a69a",
        5: "#009688",
        6: "#00897b",
        7: "#00796b",
        8: "#00695c",
        9: "#004d40",
    },
    "green": {
        0: "#e8f5e9",
        1: "#c8e6c9",
        2: "#a5d6a7",
        3: "#81c784",
        4: "#66bb6a",
        5: "#4caf50",
        6: "#43a047",
        7: "#388e3c",
        8: "#2e7d32",
        9: "#1b5e20",
    },
    "l.green": {
        0: "#f1f8e9",
        1: "#dcedc8",
        2: "#c5e1a5",
        3: "#aed581",
        4: "#9ccc65",
        5: "#8bc34a",
        6: "#7cb342",
        7: "#689f38",
        8: "#558b2f",
        9: "#33691e",
    },
    "lime": {
        0: "#f9fbe7",
        1: "#f0f4c3",
        2: "#e6ee9c",
        3: "#dce775",
        4: "#d4e157",
        5: "#cddc39",
        6: "#c0ca33",
        7: "#afb42b",
        8: "#9e9d24",
        9: "#827717",
    },
    "yellow": {
        0: "#fffde7",
        1: "#fff9c4",
        2: "#fff59d",
        3: "#fff176",
        4: "#ffee58",
        5: "#ffeb3b",
        6: "#fdd835",
        7: "#fbc02d",
        8: "#f9a825",
        9: "#f57f17",
    },
    "amber": {
        0: "#fff8e1",
        1: "#ffecb3",
        2: "#ffe082",
        3: "#ffd54f",
        4: "#ffca28",
        5: "#ffc107",
        6: "#ffb300",
        7: "#ffa000",
        8: "#ff8f00",
        9: "#ff6f00",
    },
    "orange": {
        0: "#fff3e0",
        1: "#ffe0b2",
        2: "#ffcc80",
        3: "#ffb74d",
        4: "#ffa726",
        5: "#ff9800",
        6: "#fb8c00",
        7: "#f57c00",
        8: "#ef6c00",
        9: "#e65100",
    },
    "d.orange": {
        0: "#fbe9e7",
        1: "#ffccbc",
        2: "#ffab91",
        3: "#ff8a65",
        4: "#ff7043",
        5: "#ff5722",
        6: "#f4511e",
        7: "#e64a19",
        8: "#d84315",
        9: "#bf360c",
    },
    "brown": {
        0: "#efebe9",
        1: "#d7ccc8",
        2: "#bcaaa4",
        3: "#a1887f",
        4: "#8d6e63",
        5: "#795548",
        6: "#6d4c41",
        7: "#5d4037",
        8: "#4e342e",
        9: "#3e2723",
    },
    "grey": {
        0: "#fafafa",
        1: "#f5f5f5",
        2: "#eeeeee",
        3: "#e0e0e0",
        4: "#bdbdbd",
        5: "#9e9e9e",
        6: "#757575",
        7: "#616161",
        8: "#424242",
        9: "#212121",
    },
    "blue grey": {
        0: "#eceff1",
        1: "#cfd8dc",
        2: "#b0bec5",
        3: "#90a4ae",
        4: "#78909c",
        5: "#607d8b",
        6: "#546e7a",
        7: "#455a64",
        8: "#37474f",
        9: "#263238",
    },
}

def show_colors():
    
    nx, dx = 10, 5
    ny, dy = 19, 5
    
    fig = plt.figure(figsize=(8, 10), dpi=100)
    ax = plt.subplot(
        1,
        1,
        1,
        frameon=False,
        xlim=(-1, nx * dx),
        ylim=(-1, (ny - 1) * dy + 2),
        xticks=[],
        yticks=[],
    )


    for palette, y in zip(COLORS.keys(), range(0, ny * dy, dy)):
        for level, x in zip(range(10), range(0, 10 * dx, dx)):
            fancy = FancyBboxPatch(
                (x, y),
                dx - 1,
                0.3 * dy,
                facecolor=COLORS[palette][level],
                edgecolor="black",
                linewidth=0,
            )
            ax.add_patch(fancy)

            name = "%s %s" % (palette.upper(), level)
            ax.text(
                x,
                y - 0.15 * dy,
                name,
                size=8,
    
                weight="regular",
                ha="left",
                va="top",
            )

            value = COLORS[palette][level].upper()
            ax.text(
                x,
                y - 0.375 * dy,
                value,
                size=7.5,
                color="0.5",   
                weight="regular",
                ha="left",
                va="top",
            )


    plt.tight_layout()