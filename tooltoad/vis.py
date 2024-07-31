import os
from io import BytesIO
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import py3Dmol
from matplotlib import patches
from PIL import Image
from rdkit import Chem
from rdkit.Chem import Draw, rdDepictor
from rdkit.Geometry import Point2D

from tooltoad.chemutils import ac2xyz

# load matplotlib style
plt.style.use(os.path.dirname(__file__) + "/data/paper.mplstyle")
PAPER = Path("/groups/kemi/julius/opt/tm-catalyst-paper/figures")


def oneColumnFig(square: bool = False):
    """Create a figure that is one column wide.

    Args:
        square (bool, optional): Square figure. Defaults to False.

    Returns:
        (fig, ax): Figure and axes.
    """
    if square:
        size = (6, 6)
    else:
        size = (6, 4.187)
    fig, ax = plt.subplots(figsize=size)
    return fig, ax


def twoColumnFig(**kwargs):
    """Create a figure that is two column wide.

    Args:
        square (bool, optional): Square figure. Defaults to False.

    Returns:
        (fig, ax): Figure and axes.
    """
    size = (12, 4.829)
    fig, axs = plt.subplots(figsize=size, **kwargs)
    return fig, axs


def draw2d(
    mol: Chem.Mol,
    legend: str = None,
    atomLabels: dict = None,
    atomHighlights: dict = None,
    size=(800, 600),
    blackwhite=True,
):
    """Create 2D depiction of molecule for publication.

    Args:
        mol (Chem.Mol): Molecule to render
        legend (str, optional): Legend string. Defaults to None.
        atomLabels (dict, optional): Dictionary of atomindices and atomlabels, f.x.:
                                     {17: 'H<sub>1</sub>', 18: 'H<sub>2</sub>'}.
                                     Defaults to None.
        atomHighlights (dict, optional): List of atoms to highlight,, f.x.:
                                         [(9, False, (0.137, 0.561, 0.984)),
                                         (15, True, (0, 0.553, 0))]
                                         First item is the atomindex, second is whether
                                         or not the highlight should be filled, and third
                                         is the color.
                                         Defaults to None.
        size (tuple, optional): Size of the drawing canvas. Defaults to (800, 600).
        blackwhite (bool, optional): Black and white color palet. Defaults to True.

    Returns:
        PIL.PNG: Image of the molecule.
    """
    d2d = Draw.MolDraw2DCairo(*size)
    rdDepictor.Compute2DCoords(mol)
    rdDepictor.NormalizeDepiction(mol)
    rdDepictor.StraightenDepiction(mol)
    dopts = d2d.drawOptions()
    dopts.legendFraction = 0.15
    dopts.legendFontSize = 45
    dopts.baseFontSize = 0.8
    dopts.additionalAtomLabelPadding = 0.1
    dopts.bondLineWidth = 1
    dopts.scaleBondWidth = False
    if blackwhite:
        dopts.useBWAtomPalette()
    if atomLabels:
        for key, value in atomLabels.items():
            dopts.atomLabels[key] = value

    if legend:
        d2d.DrawMolecule(mol, legend=legend)
    else:
        d2d.DrawMolecule(mol)

    alpha = 0.4
    positions = []
    radii = []
    colors = []
    filled_bools = []
    if atomHighlights:
        for h in atomHighlights:
            filled = False
            color = (0.137, 0.561, 0.984)
            if isinstance(h, int):
                atomIdx = h
            elif len(h) == 2:
                atomIdx, filled = h
            elif len(h) == 3:
                atomIdx, filled, color = h
            else:
                raise ValueError("Invalid atom highlight {}".format(h))
            point = mol.GetConformer().GetAtomPosition(int(atomIdx))
            positions.append(Point2D(point.x, point.y))
            radii.append(0.35)
            colors.append(color)
            filled_bools.append(bool(filled))

        # draw filled circles first
        for pos, radius, color, filled in zip(positions, radii, colors, filled_bools):
            if filled:
                color = (color[0], color[1], color[2], alpha)
                d2d.SetColour(color)
                d2d.SetFillPolys(True)
                d2d.SetLineWidth(0)
                d2d.DrawArc(pos, radius, 0.0, 360.0)

        # # now draw molecule again
        d2d.SetLineWidth(3)
        if legend:
            d2d.DrawMolecule(mol, legend=legend)
        else:
            d2d.DrawMolecule(mol)

        # now draw ring highlights
        for pos, radius, color, filled in zip(positions, radii, colors, filled_bools):
            d2d.SetColour(color)
            d2d.SetFillPolys(False)
            # d2d.SetLineWidth(2.5)
            d2d.SetLineWidth(5)
            d2d.DrawArc(pos, radius, 0.0, 360.0)

        # and draw molecule again for whatever reason
        d2d.SetLineWidth(1)
        if legend:
            d2d.DrawMolecule(mol, legend=legend)
        else:
            d2d.DrawMolecule(mol)

        # now draw ring highlights again
        for pos, radius, color, filled in zip(positions, radii, colors, filled_bools):
            if not filled:
                d2d.SetColour(color)
                d2d.SetFillPolys(False)
                # d2d.SetLineWidth(2.5)
                d2d.SetLineWidth(5)
                d2d.DrawArc(pos, radius, 0.0, 360.0)
    # finish drawing
    d2d.FinishDrawing()
    d2d.GetDrawingText()
    bio = BytesIO(d2d.GetDrawingText())
    img = Image.open(bio)
    return img


def molGrid(images: List, buffer: int = 5, out_file: str = None):
    """Creates a grid of images.

    Args:
        images (List): List of lists of images.
        buffer (int, optional): Buffer between images. Defaults to 5.
        out_file (str, optional): Filename to save image to. Defaults to None.
    """
    max_width = max([max([img.width for img in imgs]) for imgs in images])
    max_height = max([max([img.height for img in imgs]) for imgs in images])
    max_num_rows = max([len(imgs) for imgs in images])
    fig_width = max_width * max_num_rows + buffer * (max_num_rows - 1)
    fig_height = max_height * len(images) + buffer * (len(images) - 1)
    res = Image.new("RGBA", (fig_width, fig_height))

    y = 0
    for imgs in images:
        x = 0
        for img in imgs:
            res.paste(img, (x, y))
            x += img.width + buffer
        y += img.height + buffer
    if out_file:
        res.save(out_file)
    else:
        return res


def draw3d(
    mols: list,
    transparent: bool = True,
    overlay: bool = False,
    confId: int = -1,
    atomlabel: bool = False,
    kekulize: bool = True,
    width: float = 600,
    height: float = 400,
):
    """Draw 3D structures in Jupyter notebook using py3Dmol.

    Args:
        mols (list): List of RDKit molecules.
        overlay (bool, optional): Overlay molecules. Defaults to False.
        confId (int, optional): Conformer ID. Defaults to -1.
        atomlabel (bool, optional): Show all atomlabels. Defaults to False.

    Returns:
        Py3Dmol.view: 3D view object.
    """
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
                    mb = Chem.MolToMolBlock(mol, confId=conf.GetId(), kekulize=kekulize)
                    p.addModel(mb, "sdf")
            else:
                mb = Chem.MolToMolBlock(mol, confId=confId, kekulize=kekulize)
                p.addModel(mb, "sdf")
    p.setStyle({"sphere": {"radius": 0.4}, "stick": {}})
    p.setBackgroundColor("0xeeeeee", int(~transparent))
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
    return p


def show_traj(input: str | dict, width: float = 600, height: float = 400):
    """Show xyz trajectory.

    Args:
        input (str | dict): Trajectory, either as a string or a dict["traj"].
        width (float, optional): Width of py3dmol. Defaults to 600.
        height (float, optional): Height of py3dmol. Defaults to 400.
    """
    if isinstance(input, str):
        traj = input
    elif isinstance(input, dict):
        traj = input.get("traj")
        if not traj:
            raise ValueError
    else:
        raise ValueError
    p = py3Dmol.view(width=width, height=height)
    p.addModelsAsFrames(traj, "xyz")
    p.animate({"loop": "forward", "reps": 3})
    p.setStyle({"sphere": {"radius": 0.4}, "stick": {}})
    return p


def show_vibs(
    results: dict,
    vId: int = 0,
    width: float = 600,
    height: float = 400,
    numFrames: int = 20,
    amplitude: float = 1.0,
):
    """Show normal mode vibration."""
    input = results
    vib = input.get("vibs")[vId]
    mode = vib["mode"]
    frequency = vib["frequency"]
    atoms = results["atoms"]
    opt_coords = results["opt_coords"]
    xyz = ac2xyz(atoms, opt_coords)

    p = py3Dmol.view(width=width, height=height)
    p.addModel(xyz, "xyz")
    propmap = []
    for j, m in enumerate(mode):
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
    p.vibrate(numFrames, amplitude, True)
    p.animate({"loop": "backAndForth", "interval": 1, "reps": 20})
    p.setStyle({"sphere": {"radius": 0.4}, "stick": {}})
    print(f"Normal mode {vId} with frequency {frequency} cm^-1")
    return p


dopts = Chem.Draw.rdMolDraw2D.MolDrawOptions()
dopts.prepareMolsForDrawing = True
dopts.centreMoleculesBeforeDrawing = True
dopts.drawMolsSameScale = True
dopts.legendFontSize = 18
dopts.minFontSize = 30
dopts.padding = 0.05
dopts.atomLabelFontSize = 40


def drawMolInsert(
    ax_below: plt.axes,
    mol: Chem.Mol,
    pos: tuple,
    xSize: float = 0.5,
    aspect: float = 0.33,
    zorder: int = 5,
) -> plt.axes:
    """Draw molecule in a subplot.

    Args:
        ax_below (plt.axes): Axes to draw molecule on top of.
        mol (Chem.Mol): RDKit molecule to draw.
        pos (tuple): (x0, y0) position of insert.
        xSize (float, optional): Size of x dimension of insert. Defaults to 0.5.
        aspect (float, optional): Aspect ratio of insert. Defaults to 0.33.

    Returns:
        plt.axes: Axes of insert.
    """
    ax = ax_below.inset_axes([*pos, xSize, xSize * aspect])
    resolution = 1000
    im = Draw.MolToImage(
        mol,
        size=(int(resolution * xSize), int(resolution * xSize * aspect)),
        options=dopts,
    )
    ax.imshow(im, origin="upper", zorder=zorder)
    ax.axis("off")
    return ax


def addFrame(
    ax_around: plt.axes,
    ax_below: plt.axes,
    linewidth: int = 6,
    edgecolor: str = "crimson",
    nShadows: int = 25,
    shadowLinewidth: float = 0.05,
    molZorder: int = 4,
) -> None:
    """Draw Frame around axes.

    Args:
        ax_around (plt.axes): Axes to draw frame around.
        ax_below (plt.axes): Axes to draw frame on.
        linewidth (int, optional): Linewidth of frame. Defaults to 6.
        edgecolor (str, optional): Color of frame. Defaults to "crimson".
        nShadows (int, optional): Resolution of shadow. Defaults to 25.
        shadowLinewidth (float, optional): Extend of shadow. Defaults to 0.05.
        molZorder (int, optional): ZOrder of Mol. Defaults to 4.
    """
    images = ax_around.get_images()
    assert len(images) == 1, f"Found {len(images)} images in {ax_around}, expected 1"
    img = images[0]
    frame = patches.FancyBboxPatch(
        (0, 0),
        *reversed(img.get_size()),
        boxstyle="round",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor="none",
        transform=ax_around.transData,
        zorder=molZorder,
    )
    ax_below.add_patch(frame)
    if nShadows:
        for i in range(nShadows):
            shadow = patches.FancyBboxPatch(
                (0, 0),
                *reversed(img.get_size()),
                boxstyle="round",
                linewidth=shadowLinewidth * i**2 + 0.2,
                edgecolor="black",
                facecolor="none",
                alpha=0.7 / nShadows,
                transform=ax_around.transData,
                zorder=frame.get_zorder() - 1,
            )
            ax_below.add_patch(shadow)


def plot_parity(ax: plt.axes, tick_base: int = None, **kwargs) -> None:
    """Make square plot with parity line.

    Args:
        ax (plt.axes): Axes to plot on.
        tick_base (int, optional): Tick base. Defaults to 10.
    """
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
        **kwargs,
    )
    ax.set_xlim(ax_min, ax_max)
    ax.set_ylim(ax_min, ax_max)

    if tick_base is None:
        ax_len = ax_max - ax_min
        tick_base = round(ax_len / 4)

    loc = plticker.MultipleLocator(base=tick_base)
    ax.xaxis.set_major_locator(loc)
    ax.yaxis.set_major_locator(loc)


def plot_residual_histogram(
    ax: plt.axes,
    x: np.ndarray,
    y: np.ndarray,
    loc: list = [0.58, 0.13, 0.4, 0.4],
    bins: int = 15,
    xlabel: str = "Residual (kcal/mol)",
    **kwargs,
) -> plt.axes:
    """Plot Histogram insert of residuals.

    Args:
        ax (plt.axes): Axes to plot on.
        x (np.ndarray): x data
        y (np.ndarray): y data
        loc (list, optional): Location of insert. Defaults to [0.58, 0.13, 0.4, 0.4].
        bins (int, optional): Number of bins in histogram. Defaults to 15.
        xlabel (str, optional): Label on x axis. Defaults to "Residual (kcal/mol)".

    Returns:
        plt.axes: _description_
    """
    insert = ax.inset_axes(loc)
    diff = y - x
    insert.hist(diff, bins=bins, **kwargs)
    insert.set_xlim(-np.max(abs(diff)), np.max(abs(diff)))
    insert.set_xlabel(xlabel)
    insert.set_ylabel("Count")
    return insert


def align_axes(axes, align_values):
    # keep format of first axes
    nTicks = len(axes[0].get_yticks())

    idx1 = (np.abs(axes[0].get_yticks() - align_values[0][0])).argmin()
    shiftAx1 = axes[0].get_yticks()[idx1] - align_values[0][0]
    ticksAx1 = axes[0].get_yticks() - shiftAx1
    dy1 = np.mean(
        [
            axes[0].get_yticks()[i] - axes[0].get_yticks()[i - 1]
            for i in range(1, len(axes[0].get_yticks()))
        ]
    )
    ylim1 = (ticksAx1[1] - dy1 / 2, ticksAx1[-2] + dy1 / 2)
    axes[0].set_yticks(ticksAx1)

    for i, ax in enumerate(axes[1:]):
        tmp = np.linspace(align_values[1][i + 1], align_values[0][i + 1], nTicks - 2)
        dy2 = np.mean([tmp[i] - tmp[i - 1] for i in range(1, len(tmp))])
        ticksAx2 = np.linspace(tmp[0] - dy2, tmp[-1] + dy2, nTicks)
        ylim2 = (ticksAx2[1] - dy2 / 2, ticksAx2[-2] + dy2 / 2)
        ax.set_yticks(ticksAx2)
        ax.set_ylim(ylim2)

    axes[0].set_ylim(ylim1)
