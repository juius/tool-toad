import subprocess
import tempfile
from pathlib import Path

from rdkit import Chem


def render_molecule(
    mol, output=None, dashed_bonds=None, width=2400, height=1800, dpi=300
):
    """Render molecule to PNG file without opening GUI.

    Args:
        mol: RDKit molecule object
        output: Output file path. If None, saves to temp file and returns path.
        dashed_bonds: List of tuples [(atom1_idx, atom2_idx), ...] for dashed bonds.
                      Optionally include color: [(atom1, atom2, "red"), ...]
        width: Image width in pixels (default 2400)
        height: Image height in pixels (default 1800)
        dpi: Image DPI (default 300)

    Returns:
        Path to the rendered PNG file.
    """
    if output is None:
        output = tempfile.NamedTemporaryFile(suffix=".png", delete=False).name
    output = str(Path(output).resolve())

    # Export molecule to temp SDF file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".sdf", delete=False) as f:
        f.write(Chem.MolToMolBlock(mol))
        sdf_path = f.name

    # Build dashed bond commands
    dash_commands = _build_dash_commands(dashed_bonds)

    # Create PyMOL script for headless rendering
    pml_script = f"""
load {sdf_path}, mol

{_get_style_commands()}

{dash_commands}

# Orient molecule optimally and render
orient mol
zoom mol, buffer=2
ray {width}, {height}
png {output}, dpi={dpi}
quit
"""

    # Write script and run PyMOL in headless mode
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pml", delete=False) as f:
        f.write(pml_script)
        script_path = f.name

    subprocess.run(["pymol", "-cq", script_path], check=True)
    print(f"Rendered: {output}")
    return output


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

    # Build dashed bond commands
    dash_commands = _build_dash_commands(dashed_bonds)

    # Create PyMOL script
    pml_script = f"""
load {sdf_path}, mol

{_get_style_commands()}

# Set selection mode to atoms (not molecules)
set mouse_selection_mode, 0
set seq_view, 0

{dash_commands}

# Custom shortcut: select two atoms, then press F1 to add a dashed bond
python
dash_count = [0]
def add_dash():
    # Get atoms in selection
    atoms = cmd.get_model("sele").atom
    if len(atoms) != 2:
        print(f"Error: Select exactly 2 atoms (currently {{len(atoms)}} selected)")
        return

    # Create selections for each atom
    name = f"dash_{{dash_count[0]}}"
    idx1, idx2 = atoms[0].id, atoms[1].id
    cmd.distance(name, f"mol and id {{idx1}}", f"mol and id {{idx2}}")
    cmd.hide("labels", name)
    cmd.set("dash_gap", 0.15, name)
    cmd.set("dash_length", 0.15, name)
    cmd.set("dash_radius", 0.03, name)
    cmd.color("gray50", name)
    cmd.deselect()
    dash_count[0] += 1
    print(f"Added {{name}} between atoms {{idx1}} and {{idx2}}")

cmd.extend("add_dash", add_dash)
cmd.set_key("F1", add_dash)
python end
"""

    # Write script and open PyMOL
    with tempfile.NamedTemporaryFile(mode="w", suffix=".pml", delete=False) as f:
        f.write(pml_script)
        script_path = f.name

    subprocess.Popen(["pymol", script_path])
    print("PyMOL opened!")
    print("To save: ray 2400,1800; png /path/to/output.png, dpi=300")


def _build_dash_commands(dashed_bonds):
    """Build PyMOL commands for dashed bonds."""
    if not dashed_bonds:
        return ""

    commands = ""
    for i, bond in enumerate(dashed_bonds):
        atom1, atom2 = bond[0] + 1, bond[1] + 1  # Convert to 1-based
        color = bond[2] if len(bond) > 2 else "gray50"
        commands += f"""
distance dash_{i}, mol and id {atom1}, mol and id {atom2}
hide labels, dash_{i}
set dash_gap, 0.15, dash_{i}
set dash_length, 0.15, dash_{i}
set dash_radius, 0.03, dash_{i}
color {color}, dash_{i}
"""
    return commands


def _get_style_commands():
    """Get PyMOL commands for publication-quality styling."""
    return """
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
"""
