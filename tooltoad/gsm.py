import argparse
import os
import sys
from pathlib import Path


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


def create_isomers(bond_changes: list[tuple[int, tuple[int, int]]], parent_path="."):
    """Create ISOMERS0000 file.

    !!! GSM atom index starts at 1.
    """
    change_types = {
        -1: "ADD",
        1: "BREAK",
    }  # reverse here bc we want to to the opposite of what happened in the adj diff
    content = "NEW\n"
    for change_type, (atom1, atom2) in bond_changes:
        content += f"{change_types[change_type]} {atom1+1} {atom2+1}\n"
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


def create_slurm(
    job_name="orcagsm",
    ntasks=8,
    walltime="3-00:00:00",
    partition="kemi1",
    gsm_exe="/groups/kemi/julius/gsmtest/gsm.orca",
    parent_path=".",
):
    """Create SLURM submission script."""
    content = f"""#!/bin/bash
#SBATCH --array=1
#SBATCH --job-name={job_name}
#SBATCH --time={walltime}
#SBATCH --nodes=1
#SBATCH --ntasks={ntasks}
#SBATCH --mem-per-cpu=4G
#SBATCH -o orca.output
#SBATCH -e orca.error
#SBATCH -p {partition}

item=$SLURM_ARRAY_TASK_ID
ID=`printf "%0*d\\n" 4 ${{item}}`

export PATH=/groups/kemi/julius/orca_6_1_0:/software/kemi/openmpi/openmpi-4.1.1/bin:$PATH
export LD_LIBRARY_PATH=/software/kemi/openmpi/openmpi-4.1.1/lib:$LD_LIBRARY_PATH
export OMP_NUM_THREADS=1

{gsm_exe} ${{item}} {ntasks} > scratch/paragsm$ID
"""
    output_path = Path(parent_path) / "submit.sh"
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

        shutil.copy2(xyz_file, scratch_path / "input.xyz")
        print(f"Copied {xyz_file} to {scratch_path / 'input.xyz'}")
    print(f"Created: {scratch_path}/")


def parse_bond_changes(bond_args):
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
            print(f"Error parsing bond change '{bond_arg}': {e}")
            sys.exit(1)

    return bond_changes


def main():
    """CLI main function."""
    parser = argparse.ArgumentParser(
        description="Generate GSM (Growing String Method) input files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s -o output_dir -b ADD:6:4 -b BREAK:5:1
  %(prog)s -o test_run -n MyGSM -b ADD:1:2 -b ADD:3:4 -x input.xyz
  %(prog)s -o . -b ADD:6:4  # Single bond addition
        """,
    )

    # Required arguments
    parser.add_argument(
        "-o", "--output", required=True, help="Output directory for GSM files"
    )

    parser.add_argument(
        "-b",
        "--bonds",
        action="append",
        required=True,
        help="Bond changes in format ACTION:ATOM1:ATOM2 (e.g., ADD:6:4, BREAK:5:1). Can be used multiple times.",
    )

    # Optional arguments
    parser.add_argument(
        "-n", "--name", default="GSM_run", help="Run name (default: GSM_run)"
    )

    parser.add_argument("-x", "--xyz", help="Initial XYZ file path")

    parser.add_argument(
        "-t",
        "--type",
        choices=["SSM", "FSM", "GSM"],
        default="SSM",
        help="String method type (default: SSM)",
    )

    parser.add_argument(
        "--nodes",
        type=int,
        default=20,
        help="Number of nodes including endpoints (default: 20)",
    )

    parser.add_argument(
        "--ntasks", type=int, default=8, help="Number of SLURM tasks (default: 8)"
    )

    parser.add_argument(
        "--walltime", default="3-00:00:00", help="SLURM walltime (default: 3-00:00:00)"
    )

    parser.add_argument(
        "--partition", default="kemi1", help="SLURM partition (default: kemi1)"
    )

    parser.add_argument(
        "--method", default="XTB2", help="ORCA method (default: DFT B3LYP)"
    )

    parser.add_argument(
        "--charge", type=int, default=0, help="Molecular charge (default: 0)"
    )

    parser.add_argument(
        "--multiplicity", type=int, default=1, help="Spin multiplicity (default: 1)"
    )

    args = parser.parse_args()

    # Parse bond changes
    bond_changes = parse_bond_changes(args.bonds)

    # Create output directory
    output_path = Path(args.output)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"=== Creating GSM files in {output_path} ===")
    print(f"Run name: {args.name}")
    print(f"Bond changes: {bond_changes}")

    # Create all files
    create_inpfileq(
        run_name=args.name,
        sm_type=args.type,
        nnodes=args.nodes,
        parent_path=args.output,
    )

    create_isomers(bond_changes, parent_path=args.output)

    create_ograd(
        method=args.method,
        charge=args.charge,
        multiplicity=args.multiplicity,
        parent_path=args.output,
    )

    create_slurm(
        job_name=f"orcagsm_{args.name}",
        ntasks=args.ntasks,
        walltime=args.walltime,
        partition=args.partition,
        parent_path=args.output,
    )

    create_scratch_dir(args.xyz, parent_path=args.output)

    print("\n=== Done! ===")
    print("Files created:")
    print(f"  - {output_path / 'inpfileq'}")
    print(f"  - {output_path / 'ISOMERS0001'}")
    print(f"  - {output_path / 'ograd'}")
    print(f"  - {output_path / 'submit.sh'}")
    print(f"  - {output_path / 'scratch'}/")
    if args.xyz:
        print(f"  - {output_path / 'scratch' / 'input.xyz'}")

    print(f"\nTo submit: cd {output_path} && sbatch submit.sh")


if __name__ == "__main__":
    main()
