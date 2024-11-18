from tooltoad.orca import orca_calculate


def locate_ts(
    atoms, coords, interaction_indices, orca_options={"XTB2": None}, **kwargs
):
    internal_coord_setup = "\n".join(
        [f"modify_internal {{ B {i[0]} {i[1]} A }} end" for i in interaction_indices]
    )
    TS = {"OPTTS": None, "freq": None}
    TS.update(orca_options)
    ts_results = orca_calculate(
        atoms,
        coords,
        options=TS,
        xtra_inp_str=f"""%geom
    {internal_coord_setup}
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
