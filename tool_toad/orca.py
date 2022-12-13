from rdkit import Chem


def get_header(options, memory, n_cores):
    """Write Orca header."""

    header = "# Automatically generated ORCA input" + 2 * "\n"

    header += "# Number of cores\n"
    header += f"%pal nprocs {n_cores} end\n"
    header += "# RAM per core\n"
    header += f"%maxcore {1024 * memory}" + 2 * "\n"

    for key, value in options.items():
        if (value is None) or (not value):
            header += f"! {key} \n"
        else:
            header += f"! {key}({value}) \n"

    return header


def write_orca_input(
    mol,
    options,
    spin=0,
    confId=0,
    memory=4,
    n_cores=1,
):
    header = get_header(options, memory, n_cores)
    atom_strs = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coordinates = mol.GetConformer(confId).GetPositions()
    inputstr = header + 2 * "\n"

    # charge, spin, and coordinate section
    inputstr += f"*xyz {Chem.GetFormalCharge(mol)} {spin} \n"
    for atom_str, coord in zip(atom_strs, coordinates):
        inputstr += (
            f"{atom_str}".ljust(5)
            + " ".join(["{:.8f}".format(x).rjust(15) for x in coord])
            + "\n"
        )
    inputstr += "*\n"
    inputstr += "\n"  # magic line

    return inputstr
