from pathlib import Path

import py3Dmol
from rdkit import Chem


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
