import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as plticker
import numpy as np
import py3Dmol
from matplotlib import patches
from rdkit import Chem
from rdkit.Chem import Draw

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


def draw3d(
    mols: list,
    transparent: bool = True,
    overlay: bool = False,
    confId: int = -1,
    atomlabel: bool = False,
    kekulize=True,
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
    p.show()


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


def plot_parity(ax: plt.axes, tick_base: int = 10, **kwargs) -> None:
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
