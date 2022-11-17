from pathlib import Path

import matplotlib.pyplot as plt
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
