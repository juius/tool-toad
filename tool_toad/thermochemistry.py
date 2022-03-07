import numpy as np

# CONVERSION FACTORS

JOULES2CAL = 0.2390057361
HARTREE2JOULES = 4.3597482e-18
HARTEE2CALMOL = 627509.5
AMU2KG = 1.6605402e-27
ATM2PASCAL = 101325

# CONSTANTS

GAS_CONSTANT = 8.31446261815324  # J/(K x mol)
BOLTZMANN_CONSTANT = 1.380649e-23  # J/K
PLANCK_CONSTANT = 6.626070150e-34  # J/s
SPEED_OF_LIGHT = 299792458  # m/s
AVOGARDOS_CONSTANT = 6.02214076e23  # 1/mol


def get_indices(lines, pattern, stop_pattern=None):
    indices = []
    for i, l in enumerate(lines):
        if pattern in l:
            indices.append(i)
        if stop_pattern and stop_pattern in l:
            break
    return indices


def get_frequencies(lines):
    indices = get_indices(lines, pattern="Frequencies --")
    frequencies = []
    for idx in indices:
        line = lines[idx]
        frequencies.extend([float(f) for f in line.split()[-3:]])
    return frequencies


def get_electronic_energy(lines):
    indices = get_indices(lines, pattern="SCF Done:")
    line = lines[indices[-1]]
    energy = float(line.split()[4])
    return energy


def get_enthalpy_correction(lines):
    indices = get_indices(lines, pattern="Thermal correction to Enthalpy=")
    line = lines[indices[-1]]
    correction = float(line.split()[-1])
    return correction


def get_translational_entropy(lines):
    indices = get_indices(lines, pattern="Translational")
    line = lines[indices[0]]
    St = float(line.split()[-1])
    return St


def get_rotational_entropy(lines):
    indices = get_indices(lines, pattern="Rotational")
    line = lines[indices[-2]]
    Sr = float(line.split()[-1])
    return Sr


def get_temperature_and_pressure(lines):
    indices = get_indices(lines, "Temperature")
    line = lines[indices[0]]
    T, p = line.split()[1::3]
    return float(T), float(p)


def get_molar_mass(lines):
    indices = get_indices(lines, pattern="Molecular mass:")
    line = lines[indices[-1]]
    M = float(line.split()[2])
    return M


def calc_translational_entropy(molar_mass, temperature=298.15, M=None, p=None):
    """Calculates translational component of Entropy

    S_t = R (ln((2pi m kT/h^2)^(3/2) * V) + 5/2)
    with the mass of the molecule m in kg and the volume V in m^3
    following 'Cramer, C. J. Essentials of Computational Chemistry: Theories
    and Models, 2nd ed' p:362, eq:10.18

    Parameters:
    molar_mass : float
        Molar mass of Molecule in atomic units
    temperature : float
        Temperature in Kelvin
    M : float (optional)
        Concentration of Standard State in mol/L
    p : float (optional)
        Pressure of Standard State in atm

    Returns:
        Translational Contribution to Entropy in Cal/Mol-Kelvin
    """

    if p and M:
        raise Warning("Choose Concentration OR Pressure for Standard State")
    elif M:
        V = 1 / (M * 1000 * AVOGARDOS_CONSTANT)  # m^3
    elif p:
        V = GAS_CONSTANT * temperature / (p * ATM2PASCAL * AVOGARDOS_CONSTANT)  # m^3
    else:
        raise ValueError("No Standard State specified")

    return (
        GAS_CONSTANT
        * (
            np.log(
                (
                    (2 * np.pi * molar_mass * AMU2KG * BOLTZMANN_CONSTANT * temperature)
                    / PLANCK_CONSTANT ** 2
                )
                ** (3 / 2)
                * V
            )
            + 2.5
        )
    ) * JOULES2CAL


def calc_vibrational_entropy(
    frequencies, f_cutoff=None, temperature=298.15, verbose=True
):
    """Calculates vibrational component of Entropy

    S_v = R \sum( hv / (kT (exp(hv/kT) - 1)) - ln(1 - exp(-hv/kT)))
    'Cramer, C. J. Essentials of Computational Chemistry: Theories
    and Models, 2nd ed' p:365, eq:10.30

    with the possibility to shift all frequencies below a cutoff value
    to the cutoff value following Truhlar and Cramer
    (doi.org/10.1021/jp205508z, p:14559, bottom right)

    Parameters:
    frequencies : list
        Frequencies in cm^-1
    f_cutoff : float
        Frequency Cutoff in cm^-1
    temperature : float
        Temperature in Kelvin

    Returns:
        Vibrational Contribution to Entropy in Cal/Mol-Kelvin
    """

    frequencies = np.asarray(frequencies)
    imag_freq = frequencies <= 0
    if np.sum(imag_freq > 0):
        if verbose:
            print(
                f"1 imaginary frequencies ignored: {float(frequencies[imag_freq])}"
            )
        frequencies[imag_freq] = np.nan

    energy_factor = PLANCK_CONSTANT * SPEED_OF_LIGHT * 100
    thermal_energy = BOLTZMANN_CONSTANT * temperature

    clip_freq = np.clip(frequencies, f_cutoff, np.nan)
    energies = clip_freq * energy_factor / thermal_energy

    return (
        np.nansum(
            GAS_CONSTANT * energies / (np.exp(energies) - 1)
            - GAS_CONSTANT * np.log(1 - np.exp(-energies))
        )
        * JOULES2CAL
    )
    


def recalc_gibbs(
    log_file, f_cutoff=None, standard_state_M=None, standard_state_p=None, verbose=True
):
    """Calculates the Gibbs Free Energy in Hartree from a Gaussian LOG-file with the
    option to adjust the Standard State and treat low frequencies as proposed by
    Truhlar and Cramer (doi.org/10.1021/jp205508z)

    Parameters:
    log_file : str
        Path to LOG-file
    f_cutoff : float
        Frequency Cutoff in cm^-1
    standard_state_M : float (optional)
        Concentration of Standard State in mol/L
    standard_state_p : float (optional)
        Pressure of Standard State in atm

    Returns:
        Gibbs Free Energy in Hartree/Particle
    """

    with open(log_file, "r") as f:
        lines = f.readlines()

    # check if terminated successfully
    if not "Normal termination of Gaussian" in next(
        s for s in reversed(lines) if s != "\n"
    ):
        print(f"Abnormal Termination: of {log_file}")
        return np.nan

    T, p0 = get_temperature_and_pressure(lines)  # Kelvin
    electronic_energy = get_electronic_energy(lines)  # Hartree/Particle
    enthalpy_correction = get_enthalpy_correction(lines)

    Sr = get_rotational_entropy(lines)
    m = get_molar_mass(lines)
    St = calc_translational_entropy(m, T, standard_state_M, standard_state_p)

    if verbose:
        stxt = (
            f"{standard_state_M} M" if standard_state_M else f"{standard_state_p} atm"
        )
        print(f"Calculating the Gibbs Free Energy at {T} K")
        print(f"with a Standard State of {stxt}")
        if f_cutoff:
            print(f"and with a frequency cutoff of {f_cutoff} cm^-1")

    frequencies = get_frequencies(lines)
    Sv = calc_vibrational_entropy(
        frequencies=frequencies, f_cutoff=f_cutoff, temperature=T, verbose=False
    )

    S = St + Sr + Sv  # Cal/Mol-Kelvin
    entropy_correction = S * T / HARTEE2CALMOL

    gibbs_free_energy = electronic_energy + enthalpy_correction - entropy_correction
    
    return gibbs_free_energy  # Hartree/Particle
