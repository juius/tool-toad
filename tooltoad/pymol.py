import subprocess
import tempfile

from rdkit import Chem


def open_pymol(mol, dashed_bonds=None):
    """Open RDKit molecule in PyMOL GUI with publication-quality styling.

    Args:
        mol: RDKit molecule object
        dashed_bonds: List of tuples [(atom1_idx, atom2_idx), ...] for dashed bonds.
                      Optionally include color: [(atom1, atom2, "red"), ...]
                      Atom indices are 0-based (RDKit convention).
    """
    # Export molecule to temp SDF file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sdf", delete=False) as f:
        f.write(Chem.MolToMolBlock(mol))
        sdf_path = f.name

    # Build dashed bond commands (convert 0-based to 1-based for PyMOL)
    dash_commands = ""
    if dashed_bonds:
        for i, bond in enumerate(dashed_bonds):
            atom1, atom2 = bond[0] + 1, bond[1] + 1  # Convert to 1-based
            color = bond[2] if len(bond) > 2 else "gray50"
            dash_commands += f"""
distance dash_{i}, mol and id {atom1}, mol and id {atom2}
hide labels, dash_{i}
set dash_gap, 0.15, dash_{i}
set dash_length, 0.15, dash_{i}
set dash_radius, 0.03, dash_{i}
color {color}, dash_{i}
"""

    # Create PyMOL script
    pml_script = f"""
load {sdf_path}, mol

# Workspace settings
bg_color white
set ray_opaque_background, off
set orthoscopic, 0
set ray_trace_mode, 1
set ray_texture, 2
set antialias, 3
set ambient, 0.5
set spec_count, 5
set shininess, 50
set specular, 1
set reflect, 0.1
space cmyk

# Ball-and-stick representation
show sticks, mol
show spheres, mol
set stick_radius, 0.07, mol
set sphere_scale, 0.18, mol
set sphere_scale, 0.13, mol and elem H
set stick_color, gray20, mol
hide nonbonded, mol
hide lines, mol
set valence, 1, mol
set valence_size, 0.2, mol
hide labels

# Element coloring
color gray85, elem C and mol
color gray98, elem H and mol
color slate, elem N and mol
color red, elem O and mol
color yellow, elem S and mol
color green, elem Cl and mol
color orange, elem P and mol

center mol
zoom mol, buffer=2

{dash_commands}
"""

    # Write script and open PyMOL
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pml", delete=False) as f:
        f.write(pml_script)
        script_path = f.name

    subprocess.Popen(["pymol", script_path])
    print("PyMOL opened!")
    print("To save: ray 2400,1800; png /path/to/output.png, dpi=300")
