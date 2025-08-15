import os
from pathlib import Path
from typing import List

import typer
from typing_extensions import Annotated


def create_inpfileq(
    run_name="GSM_run", sm_type="SSM", nnodes=20, bond_delta=True, parent_path="."
):
    """Create inpfileq file."""
    content = f"""# FSM/GSM/SSM inpfileq

------------- QCHEM Scratch Info ------------------------
$QCSCRATCH/    # path for scratch dir. end with "/"
{run_name}     # name of run
---------------------------------------------------------

------------ String Info --------------------------------
SM_TYPE                 {sm_type}      # SSM, FSM or GSM
RESTART                 0      # read restart.xyz
MAX_OPT_ITERS           80     # maximum iterations
STEP_OPT_ITERS          30     # for FSM/SSM
CONV_TOL                0.0005 # perp grad
ADD_NODE_TOL            0.1    # for GSM
SCALING                 1.0    # for opt steps
SSM_DQMAX               0.8    # add step size
GROWTH_DIRECTION        0      # normal/react/prod: 0/1/2
INT_THRESH              2.0    # intermediate detection
MIN_SPACING             5.0    # node spacing SSM
BOND_FRAGMENTS          1      # make IC's for fragments
INITIAL_OPT             0      # opt steps first node
FINAL_OPT               150    # opt steps last SSM node
PRODUCT_LIMIT           100.0  # kcal/mol
TS_FINAL_TYPE           {int(bond_delta)}      # any/delta bond: 0/1
NNODES                  {nnodes}      # including endpoints
---------------------------------------------------------
"""
    output_path = Path(parent_path) / "inpfileq"
    with open(output_path, "w") as f:
        f.write(content)
    print(f"Created: {output_path}")


def create_isomers(bond_changes: List[tuple[str, int, int]], parent_path="."):
    """Create ISOMERS0000 file.

    !!! GSM atom index starts at 1.
    """
    change_types = {
        "ADD": "ADD",
        "BREAK": "BREAK",
    }
    content = "NEW\n"
    for action, atom1, atom2 in bond_changes:
        content += f"{change_types[action]} {atom1+1} {atom2+1}\n"
    content += "\n"
    output_path = Path(parent_path) / "scratch/ISOMERS0000"
    with open(output_path, "w") as f:
        f.write(content)
    print(f"Created: {output_path}")


def create_ograd(
    orca_option_string="XTB2",
    orca_path="/groups/kemi/julius/orca_6_1_0/orca",
    charge=0,
    multiplicity=1,
    parent_path=".",
):
    """Create ograd executable script (content string)."""
    content = """#!/usr/bin/env bash
tag="$1"
ncpu="$2"

WORKDIR="scratch"
molfile="$WORKDIR/structure${{tag}}"
infile="$WORKDIR/orcain${{tag}}.in"
outfile="$WORKDIR/orcain${{tag}}.out"
export OMP_NUM_THREADS="$ncpu"

# ORCA binary (assumes it's on PATH; otherwise set ORCA_BIN=/path/to/orca)
ORCA_BIN={orca_path}

# Detect standard XYZ: first line is an integer atom count
first_line="$(head -n1 "$molfile" | tr -d '\\r')"
is_xyz=0
if [[ "$first_line" =~ ^[0-9]+$ ]]; then
  is_xyz=1
fi

# header
cat > "$infile" <<EOF
! {orca_option_string} ENGRAD
! nomoprint
%pal
  nprocs $ncpu
end
%scf
  MaxIter 350
end

* xyz {charge} {multiplicity}
EOF

# coordinates, then closing *
if [ "$is_xyz" -eq 1 ]; then
  # Skip atom count + comment line
  tail -n +3 "$molfile" >> "$infile"
else
  # Assume already ORCA-style atom lines
  cat "$molfile" >> "$infile"
fi
echo "*" >> "$infile"

"$ORCA_BIN" "$infile" > "$outfile"

# Extract the value from "FINAL SINGLE POINT ENERGY"
energy=$(grep "FINAL SINGLE POINT ENERGY" "$outfile" | awk '{{print $5}}')
eV=$(awk -v eh="$energy" 'BEGIN {{ printf("%.5f", eh * 27.211386245988) }}')
found=0
awk -v newE="$energy" -v newEV="$eV" '
/Total Energy       :/ {{
    printf("Total Energy       :       % .14f Eh           % .5f eV\\n", newE, newEV)
    found=1
    next
}}
/Total Energy calculation/ {{ next }} # remove line
{{ print }}
END {{
    if (found == 0) {{
        printf("Total Energy       :       % .14f Eh           % .5f eV\\n", newE, newEV)
    }}
}}
' "$outfile" > "${{outfile}}.tmp" && mv "${{outfile}}.tmp" "$outfile"
echo "Done with ORCA and fixed output. Output at: $outfile"
""".format(
        orca_option_string=orca_option_string,
        orca_path=orca_path,
        charge=charge,
        multiplicity=multiplicity,
    )
    output_path = Path(parent_path) / "ograd"
    with open(output_path, "w") as f:
        f.write(content)
    os.chmod(output_path, 0o755)
    print(f"Created: {output_path}")


def create_scratch_dir(xyz_file=None, parent_path="."):
    """Create scratch directory."""
    scratch_path = Path(parent_path) / "scratch"
    scratch_path.mkdir(exist_ok=True)
    if xyz_file and Path(xyz_file).exists():
        import shutil

        shutil.copy2(xyz_file, scratch_path / "initial0000.xyz")
        print(f"Copied {xyz_file} to {scratch_path / 'initial0000.xyz'}")
    print(f"Created: {scratch_path}/")


def parse_bond_changes(bond_args: List[str]) -> List[tuple[str, int, int]]:
    """Parse bond change arguments from command line.

    Args:
        bond_args: List of strings in format "ADD:1:2" or "BREAK:3:4"

    Returns:
        List of tuples (action, atom1, atom2)
    """
    bond_changes = []
    for bond_arg in bond_args:
        try:
            parts = bond_arg.split(":")
            if len(parts) != 3:
                raise ValueError(
                    f"Invalid bond format: {bond_arg}. Use ACTION:ATOM1:ATOM2"
                )

            action = parts[0].upper()
            if action not in ["ADD", "BREAK"]:
                raise ValueError(f"Invalid action: {action}. Use ADD or BREAK")

            atom1 = int(parts[1])
            atom2 = int(parts[2])
            bond_changes.append((action, atom1, atom2))

        except (ValueError, IndexError) as e:
            typer.echo(f"Error parsing bond change '{bond_arg}': {e}", err=True)
            raise typer.Exit(1)

    return bond_changes


# Create Typer app
app = typer.Typer(
    help="Generate GSM (Growing String Method) input files",
    epilog="Examples:\n"
    "  gsm ssm input.xyz ADD:6:4 BREAK:5:1 -o output_dir\n"
    "  gsm ssm molecule.xyz ADD:1:2 ADD:3:4 -n MySSM\n"
    "  gsm gsm input.xyz ADD:1:2  # (not implemented yet)",
)


@app.command()
def ssm(
    xyz_file: Annotated[Path, typer.Argument(help="Initial XYZ file path")],
    bond_changes: Annotated[
        List[str],
        typer.Argument(
            help="Bond changes in format ACTION:ATOM1:ATOM2 (e.g., ADD:6:4 BREAK:5:1)"
        ),
    ],
    output: Annotated[
        Path, typer.Option("-o", "--output", help="Output directory for SSM files")
    ] = Path("."),
    name: Annotated[str, typer.Option("-n", "--name", help="Run name")] = "SSM_run",
    nodes: Annotated[
        int, typer.Option("--nodes", help="Number of nodes including endpoints")
    ] = 20,
    method: Annotated[
        str, typer.Option("--method", help="ORCA method string")
    ] = "XTB2",
    orca_path: Annotated[
        str, typer.Option("--orca-path", help="Path to ORCA executable")
    ] = "/groups/kemi/julius/orca_6_1_0/orca",
    charge: Annotated[int, typer.Option("--charge", help="Molecular charge")] = 0,
    multiplicity: Annotated[
        int, typer.Option("--multiplicity", help="Spin multiplicity")
    ] = 1,
):
    """Generate SSM (Single-Ended String Method) input files."""
    if not xyz_file.exists():
        typer.echo(f"Error: XYZ file '{xyz_file}' does not exist", err=True)
        raise typer.Exit(1)

    if not bond_changes:
        typer.echo(
            "Error: At least one bond change must be specified",
            err=True,
        )
        raise typer.Exit(1)

    parsed_bond_changes = parse_bond_changes(bond_changes)

    # Create output directory
    output.mkdir(parents=True, exist_ok=True)

    typer.echo(f"=== Creating SSM files in {output} ===")
    typer.echo(f"Run name: {name}")
    typer.echo(f"XYZ file: {xyz_file}")
    typer.echo(f"Bond changes: {parsed_bond_changes}")

    create_scratch_dir(str(xyz_file), parent_path=str(output))

    create_inpfileq(
        run_name=name,
        sm_type="SSM",
        nnodes=nodes,
        bond_delta=True,
        parent_path=str(output),
    )

    create_isomers(parsed_bond_changes, parent_path=str(output))

    create_ograd(
        orca_option_string=method,
        orca_path=orca_path,
        charge=charge,
        multiplicity=multiplicity,
        parent_path=str(output),
    )

    typer.echo("\n=== Done! ===")
    typer.echo("Files created:")
    typer.echo(f"  - {output / 'inpfileq'}")
    typer.echo(f"  - {output / 'scratch/ISOMERS0000'}")
    typer.echo(f"  - {output / 'ograd'}")
    typer.echo(f"  - {output / 'scratch'}/")
    typer.echo(f"  - {output / 'scratch' / 'initial0000.xyz'}")
    typer.echo("\nReady to run SSM calculations!")


@app.command()
def gsm():
    raise NotImplementedError(
        "GSM (Growing String Method) is not yet implemented. "
        "Please use the SSM command for Single-Ended String Method."
    )


if __name__ == "__main__":
    app()
