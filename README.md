# ToolToad
## Functions and utilities for computational chemistry

System dependent variables like paths to executables are set in [config.toml](config.toml).

## Overview of functionalities:

* QM python wrappers for:
    * Orca (`orca_calculate`)
    * xTB (`xtb_calculate`)
* Calculation of properties:
    * Gibbs free energy with qRRHO approximation and adjusted standart state (`Thermochemistry`)
    * CM5 charges (`calc_cm5`)
    * SA score (`calculateScore` and `sa_target_score_clipped`)
* Visualization and plotting settings
