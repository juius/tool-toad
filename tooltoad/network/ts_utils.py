from tooltoad.orca import orca_calculate


def preoptimize(
    atoms, coords, charge, interactions, orca_options={"XTB2": None}, **kwargs
):
    constraint_str = "%geom\n  Constraints\n"
    for i in interactions:
        constraint_str += f"    {{ B {' '.join([str(idx) for idx in i])} C }} \n"
    constraint_str += "  end\nend"

    print(constraint_str)

    PREOPT = {"OPT": None}
    PREOPT.update(orca_options)
    preopt_results = orca_calculate(
        atoms,
        coords,
        charge=charge,
        options=PREOPT,
        xtra_inp_str=constraint_str,
        **kwargs,
    )
    return preopt_results


def locate_ts(
    atoms, coords, interaction_indices, orca_options={"XTB2": None}, **kwargs
):
    internal_coord_setup = "\n".join(
        [f"modify_internal {{ B {i[0]} {i[1]} A }} end" for i in interaction_indices]
    )
    active_atoms = " ".join(
        [item for sublist in interaction_indices for item in sublist]
    )
    TS = {"OPTTS": None, "freq": None}
    TS.update(orca_options)
    ts_results = orca_calculate(
        atoms,
        coords,
        options=TS,
        xtra_inp_str=f"""%geom
    {internal_coord_setup}
    TS_Active_Atoms {{ {active_atoms} }} end
    TS_Active_Atoms_Factor 1.5
    Recalc_Hess 5
    MaxIter 250
    end
    """,
        **kwargs,
    )
    return ts_results


def run_irc(atoms, coords, interaction_indices, orca_options={"XTB2": None}, **kwargs):
    IRC = {"IRC": None}
    IRC.update(orca_options)
    irc_results = orca_calculate(
        atoms,
        coords,
        options=IRC,
        calc_dir="ts_irc",
        xtra_inp_str="""%irc
    MaxIter    100
    InitHess calc_anfreq
    end
    """,
        **kwargs,
    )
    return irc_results
