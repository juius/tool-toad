import copy

import numpy as np
import xyz2mol

from rdkit import Chem


def translate_atoms(mol, direction, atom_ids, Nsteps=1, trans_range=(-0.5, 0.5)):
    mol = copy.deepcopy(mol)
    org_c = mol.GetConformer()
    if type(direction) == tuple:
        vec = org_c.GetAtomPosition(atom_ids[1]) - org_c.GetAtomPosition(atom_ids[0])
    else:
        raise Exception

    steps = np.linspace(*trans_range, Nsteps)
    for step in steps:
        cid = mol.AddConformer(org_c, assignId=True)  # add new conformer
        c = mol.GetConformer()
        for id in atom_ids:  # set coordinates
            c.SetAtomPosition(id, org_c.GetAtomPosition(id) - vec * step)
    return mol


def mol_from_xyz(xyz_file, charge=0):
    atoms, _, xyz_coordinates = xyz2mol.read_xyz_file(xyz_file)
    mols = xyz2mol.xyz2mol(atoms, xyz_coordinates, charge=charge)
    if len(mols) > 1:
        raise Warning("More than one possible Molecules!")
    return mols[0]

def sdf2mol(sdf_file):
    mol = Chem.SDMolSupplier(sdf_file,removeHs = False, sanitize=True)[0]
    return mol
