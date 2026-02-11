import subprocess
import tempfile
from pathlib import Path

import numpy as np
from rdkit import Chem


def render_normal_mode(
    mol,
    normal_mode,
    output=None,
    amplitude=0.3,
    n_frames=20,
    fps=30,
    gui=False,
    width=1200,
    height=900,
    ray=True,
    dashed_bonds=None,
):
    """Visualize a vibrational normal mode as an oscillating animation.

    Args:
        mol: RDKit molecule with 3D coordinates
        normal_modes: Normal mode displacements, shape (NModes, Natoms, 3)
        mode_index: Which mode to visualize (0-indexed)
        output: Output GIF path. If None and gui=False, saves to temp file.
        amplitude: Maximum displacement amplitude in Angstroms (default 0.3)
        n_frames: Frames per half-oscillation (total frames = 2*n_frames)
        fps: Animation frames per second (default 30)
        gui: If True, open PyMOL GUI for interactive viewing
        width: Image width in pixels (for GIF export)
        height: Image height in pixels (for GIF export)
        ray: If True, use ray-tracing for high-quality frames (slower)
        dashed_bonds: List of tuples [(atom1_idx, atom2_idx), ...] for dashed bonds.
                      Optionally include color: [(atom1, atom2, "red"), ...]

    Returns:
        Path to output GIF (if gui=False), None otherwise.
    """
    mode = np.asarray(normal_mode)
    assert mode.shape == (mol.GetNumAtoms(), 3), "Normal mode shape must be (Natoms, 3)"

    # Normalize mode vector and scale by amplitude
    norm = np.linalg.norm(mode)
    if norm > 0:
        mode = mode / norm * amplitude

    # Get equilibrium coordinates
    conf = mol.GetConformer()
    n_atoms = mol.GetNumAtoms()
    coords = np.array(
        [
            [
                conf.GetAtomPosition(i).x,
                conf.GetAtomPosition(i).y,
                conf.GetAtomPosition(i).z,
            ]
            for i in range(n_atoms)
        ]
    )

    # Generate oscillating frames (full cycle: 0 → +amp → 0 → -amp → 0)
    frames = []
    total_frames = 2 * n_frames
    for i in range(total_frames):
        phase = 2 * np.pi * i / total_frames
        displaced = coords + mode * np.sin(phase)
        frames.append(displaced)

    # Write multi-state XYZ file
    xyz_content = _write_multistate_xyz(mol, frames)
    with tempfile.NamedTemporaryFile(mode="w", suffix=".xyz", delete=False) as f:
        f.write(xyz_content)
        xyz_path = f.name

    # Build dashed bond commands
    dash_commands = _build_dash_commands(dashed_bonds)

    if gui:
        # Open in PyMOL GUI with animation playing
        pml_script = f"""
load {xyz_path}, mol

{_get_style_commands()}

{dash_commands}

# Animation setup
mset 1 -{total_frames}
set movie_fps, {fps}
set movie_loop, 1
mplay
"""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".pml", delete=False) as f:
            f.write(pml_script)
            script_path = f.name

        subprocess.Popen(["pymol", script_path])
        print("PyMOL opened with normal mode animation!")
        print("Controls: mplay/mstop to play/stop, set movie_fps, N to change speed")
        return None

    else:
        # Headless render to GIF
        if output is None:
            output = tempfile.NamedTemporaryFile(suffix=".gif", delete=False).name
        output = str(Path(output).resolve())

        # Create temp directory for PNG frames
        with tempfile.TemporaryDirectory() as tmpdir:
            pml_script = f"""
load {xyz_path}, mol

{_get_style_commands()}

{dash_commands}

orient mol
zoom mol, buffer=2

# Render each frame
python
import os
for state in range(1, {total_frames + 1}):
    cmd.frame(state)
    {"cmd.ray({}, {})".format(width, height) if ray else ""}
    cmd.png(os.path.join("{tmpdir}", f"frame{{state:04d}}.png"), width={width}, height={height}, dpi=150)
python end
quit
"""
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".pml", delete=False
            ) as f:
                f.write(pml_script)
                script_path = f.name

            subprocess.run(["pymol", "-cq", script_path], check=True)

            # Combine PNGs to GIF using imageio
            import imageio.v3 as iio

            frame_files = sorted(Path(tmpdir).glob("frame*.png"))
            images = [iio.imread(f) for f in frame_files]
            duration = 1000 / fps  # ms per frame
            iio.imwrite(output, images, duration=duration, loop=0)

        print(f"Rendered: {output}")
        return output


def _write_multistate_xyz(mol, frames):
    """Write multi-state XYZ file from molecule and coordinate frames."""
    n_atoms = mol.GetNumAtoms()
    symbols = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(n_atoms)]

    xyz_content = ""
    for frame_idx, coords in enumerate(frames):
        xyz_content += f"{n_atoms}\nFrame {frame_idx}\n"
        for sym, (x, y, z) in zip(symbols, coords):
            xyz_content += f"{sym} {x:.6f} {y:.6f} {z:.6f}\n"

    return xyz_content


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
