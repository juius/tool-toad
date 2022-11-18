from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import py3Dmol
from matplotlib import patches
from rdkit import Chem
from rdkit.Chem import Draw


def draw3d(
    mols, overlay=False, confId=-1, atomlabel=False, vibration=True, trajectory=False
):
    width = 900
    height = 600
    p = py3Dmol.view(width=width, height=height)
    if not isinstance(mols, list):
        mols = [mols]
    for mol in mols:
        if isinstance(mol, (str, Path)):
            p = Path(mol)
            if p.suffix == ".xyz":
                xyz_f = open(mol)
                line = xyz_f.read()
                xyz_f.close()
                p.addModel(line, "xyz")
            else:
                raise NotImplementedError("Only xyz file is supported")
        elif isinstance(mol, Chem.rdchem.Mol):  # if rdkit.mol
            if overlay:
                for conf in mol.GetConformers():
                    mb = Chem.MolToMolBlock(mol, confId=conf.GetId())
                    p.addModel(mb, "sdf")
            else:
                mb = Chem.MolToMolBlock(mol, confId=confId)
                p.addModel(mb, "sdf")
    p.setStyle({"sphere": {"radius": 0.4}, "stick": {}})
    if atomlabel:
        p.addPropertyLabels("index")
    else:
        p.setClickable(
            {},
            True,
            """function(atom,viewer,event,container) {
                   if(!atom.label) {
                    atom.label = viewer.addLabel(atom.index,{position: atom, backgroundColor: 'white', fontColor:'black'});
                   }}""",
        )
    p.zoomTo()
    p.show()
    return p


def drawMolFrame(mol, frameColor="crimson", linewidth=10, size=(300, 250), ax=None):
    if not ax:
        fig, ax = plt.subplots()
    im = Draw.MolToImage(mol, size=size)
    ax.imshow(im)
    frame = patches.Rectangle(
        (0, 0), *im.size, linewidth=linewidth, edgecolor=frameColor, facecolor="none"
    )
    ax.add_patch(frame)
    ax.axis("off")
    return ax.figure


def plot_parity(ax, tick_base=10, **kwargs):
    ax.set_aspect("equal")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    ax_min = min(xlim[0], ylim[0])
    ax_max = max(xlim[1], ylim[1])

    ax.plot(
        [ax_min, ax_max],
        [ax_min, ax_max],
        c="grey",
        linestyle="dashed",
        zorder=0,
        **kwargs
    )
    ax.set_xlim(ax_min, ax_max)
    ax.set_ylim(ax_min, ax_max)

    loc = plticker.MultipleLocator(base=tick_base)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)


def plot_residual_histogram(
    ax,
    x,
    y,
    loc=[0.58, 0.13, 0.4, 0.4],
    bins=15,
    xlabel="Residual (kcal/mol)",
    **kwargs
):
    insert = ax.inset_axes(loc)
    diff = y - x
    insert.hist(diff, bins=bins, **kwargs)
    insert.set_xlim(-np.max(abs(diff)), np.max(abs(diff)))
    insert.set_xlabel(xlabel)
    insert.set_ylabel("Count")
