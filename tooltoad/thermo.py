import math

import numpy as np
import pymsym
from rdkit import Chem

# Physical constants and conversions
NA = 6.02214076e23  # 1/mol
kB = 1.380649e-23  # J/K
h = 6.62607015e-34  # J*s
R = 8.314462618  # J/(mol*K)
c_cm = 2.99792458e10  # cm/s
P_ATM = 101325.0  # Pa
HARTREE_J_PER_MOL = 2625499.638  # J/mol per Eh
AMU_TO_KG = 1.66053906660e-27
AMU_A2_TO_KG_M2 = 1.66053906660e-47  # 1 amu*Å^2 in kg*m^2


def jmol_to_eh(x_jmol: float) -> float:
    return x_jmol / HARTREE_J_PER_MOL


class Thermochemistry:
    """Minimal thermochemistry driver with qRRHO vibrational entropy
    (Grimme/ORCA style), rigid-rotor (nonlinear) and ideal-gas translations.

    Parameters
    ----------
    atoms : list[str] | list[int]
        Atomic symbols or atomic numbers.
    coords : array-like (N,3)
        Cartesian coordinates in Å.
    vibs : iterable of float
        Harmonic vibrational frequencies in cm^-1 (imaginaries allowed; will be floored to 0).

    Notes
    -----
    - Assumes a nonlinear rotor for rotation (common for most molecules).
    - Rotational constants are returned in cm^-1 (A ≥ B ≥ C).
    - Symmetry number is obtained via `pymsym`.
    """

    # ---- Config for qRRHO ----
    QRRHO_REF_CM = 100.0
    QRRHO_ALPHA = 4.0
    CutOffFreq = 1  # cm^-1

    def __init__(self, atoms, coords, vibs):
        self.atoms = list(atoms)
        self.coords = np.asarray(coords, dtype=float)
        self.vibs = np.asarray(vibs, dtype=float)
        self._pt = Chem.GetPeriodicTable()

    @property
    def atom_numbers(self):
        return [self._pt.GetAtomicNumber(str(a)) for a in self.atoms]

    # ---- Cached properties ----
    @property
    def mass(self) -> float:
        if not hasattr(self, "_mass"):
            self._mass = sum(
                self._pt.GetAtomicWeight(aNum) for aNum in self.atom_numbers
            )
        return self._mass

    @property
    def rotational_constants(self):
        if not hasattr(self, "_rotconst"):
            self._rotconst = self._calc_rotational_constants_cm()
        return self._rotconst

    @property
    def symmetry_number(self) -> int:
        if not hasattr(self, "_sym"):
            self._sym = max(
                1, int(pymsym.get_symmetry_number(self.atom_numbers, self.coords))
            )
        return self._sym

    @property
    def is_linear(self) -> bool:
        """Check if molecule is linear by examining moments of inertia."""
        if not hasattr(self, "_is_linear"):
            self._is_linear = bool(self._principal_moments_amuA2()[0] < 1e-6)
        return self._is_linear

    # ---- Public API ----
    def get_contributions(
        self, T: float, p_atm: float | None = None, c_M: float | None = None
    ):
        """Return a dict of energy/entropy contributions (Eh and kcal/mol)."""
        assert bool(p_atm) != bool(
            c_M
        ), "Either pressure or concentration must be provided."
        vibs = self.vibs[self.vibs > 0]
        vibs = [float(f) for f in vibs if f > self.CutOffFreq]
        if p_atm:
            p_Pa = p_atm * P_ATM
        else:
            p_Pa = None
        # Moments of inertia for rotation/qRRHO mapping
        I_amuA2 = self._principal_moments_amuA2()
        Ikgm2 = tuple(m * AMU_A2_TO_KG_M2 for m in I_amuA2)
        I_iso = sum(Ikgm2) / 3.0

        # Energies (Eh)
        zpe = self._zpe_from_freqs_Eh(vibs)
        vibEnergy = self._thermal_vib_energy_Eh(vibs, T)
        rotEnergy = jmol_to_eh(R * T * (1 if self.is_linear else 1.5))
        transEnergy = jmol_to_eh(1.5 * R * T)

        # Entropies (J/mol/K) -> T*S (Eh)
        S_vib = self._qrrho_vib_entropy_JmolK(
            vibs, T, I_iso, self.QRRHO_REF_CM, self.QRRHO_ALPHA
        )
        S_rot = self.rot_entropy_JmolK(T, Ikgm2, self.symmetry_number, self.is_linear)
        S_trans = self.trans_entropy(T, self.mass, p_Pa, c_M)
        TS_vib = jmol_to_eh(T * S_vib)
        TS_rot = jmol_to_eh(T * S_rot)
        TS_trans = jmol_to_eh(T * S_trans)

        TS_total = TS_vib + TS_rot + TS_trans
        gibbs_correction_eh = (
            zpe + vibEnergy + rotEnergy + transEnergy + jmol_to_eh(R * T) - TS_total
        )

        return {
            "zpe": zpe,
            "vibEnergy": vibEnergy,
            "rotEnergy": rotEnergy,
            "transEnergy": transEnergy,
            "TS_vib_Eh": TS_vib,
            "TS_rot_Eh": TS_rot,
            "TS_trans_Eh": TS_trans,
            "TS_total_Eh": TS_total,
            "gibbs_correction_Eh": gibbs_correction_eh,
            "gibbs_correction_kcal": gibbs_correction_eh * 627.509474,
        }

    # ---- Internals ----

    def _principal_moments_amuA2(self):
        """Principal moments of inertia in amu·Å^2 (ascending)."""
        masses_amu = np.array(
            [self._pt.GetAtomicWeight(aNum) for aNum in self.atom_numbers]
        )
        r = np.asarray(self.coords) - np.average(
            self.coords, axis=0, weights=masses_amu
        )
        x, y, z = r.T

        inertia = np.array(
            [
                [
                    np.sum(masses_amu * (y**2 + z**2)),
                    -np.sum(masses_amu * x * y),
                    -np.sum(masses_amu * x * z),
                ],
                [
                    -np.sum(masses_amu * x * y),
                    np.sum(masses_amu * (x**2 + z**2)),
                    -np.sum(masses_amu * y * z),
                ],
                [
                    -np.sum(masses_amu * x * z),
                    -np.sum(masses_amu * y * z),
                    np.sum(masses_amu * (x**2 + y**2)),
                ],
            ]
        )
        return np.maximum(np.linalg.eigvalsh(inertia), 1e-12)

    def _calc_rotational_constants_cm(self):
        """Return rotational constants in cm^-1, handling linear molecules."""
        inertia = self._principal_moments_amuA2()

        if self.is_linear:
            I_perp = max(inertia[1], inertia[2])
            if I_perp < 1e-12:
                return (0.0, 0.0)
            B = 16.857631 / I_perp
            return (B, B)
        else:
            B = 16.857631 / inertia
            return tuple(np.sort(B)[::-1])

    # ----- Vibrational pieces -----
    @staticmethod
    def _zpe_from_freqs_Eh(freqs_cm):
        """Zero-point energy in Eh: sum 1/2 hv over modes (ignoring imaginary parts)."""
        zpe_jmol = sum(0.5 * h * c_cm * max(nu, 0.0) * NA for nu in freqs_cm)
        return jmol_to_eh(zpe_jmol)

    @staticmethod
    def _thermal_vib_energy_Eh(freqs_cm, T: float):
        """Finite-T HO vibrational energy in Eh."""
        Evib_jmol = 0.0
        for nu in freqs_cm:
            nu = max(nu, 1e-12)
            x = h * c_cm * nu / (kB * T)
            Evib_jmol += h * c_cm * nu * NA / math.expm1(x)
        return jmol_to_eh(Evib_jmol)

    @staticmethod
    def _qrrho_vib_entropy_JmolK(
        freqs_cm, T: float, I_iso: float, ref_cm: float, alpha: float
    ) -> float:
        """QRRHO vibrational entropy (J/mol/K): w*HO + (1-w)*free-rotor per
        mode.

        I_iso in kg*m^2 (isotropic average molecular inertia).
        """
        Svib = 0.0
        for nu in freqs_cm:
            nu = max(nu, 1e-12)

            # HO entropy
            x = h * c_cm * nu / (kB * T)
            S_ho = R * (x / math.expm1(x) - math.log1p(-math.exp(-x)))

            # Free-rotor mapping
            nu_hz = c_cm * nu
            mu_K = h / (8.0 * math.pi**2 * nu_hz)
            mu_eff = mu_K * I_iso / (mu_K + I_iso)
            S_fr = R * (
                0.5 + 0.5 * math.log(8.0 * math.pi**3 * mu_eff * kB * T / h**2)
            )

            # Damping weight
            w = 1.0 / (1.0 + (ref_cm / nu) ** alpha)
            Svib += w * S_ho + (1.0 - w) * S_fr
        return Svib

    @staticmethod
    def rot_entropy_JmolK(T: float, Ikgm2_tuple, sigma: int, is_linear: bool) -> float:
        """Unified rotational entropy (J/mol/K) for both linear and nonlinear
        rotors."""
        I_A, I_B, I_C = Ikgm2_tuple
        sigma = max(sigma, 1)

        if is_linear:
            I_perp = max(I_B, I_C)
            if I_perp < 1e-47:
                return 0.0
            q_rot = 8.0 * math.pi**2 * I_perp * kB * T / (sigma * h**2)
            return R * (math.log(q_rot) + 1.0)
        else:
            pref = math.sqrt(math.pi) / sigma
            term = ((8.0 * math.pi**2 * kB * T) / h**2) ** 1.5 * (
                I_A * I_B * I_C
            ) ** 0.5
            return R * (math.log(pref * term) + 1.5)

    @staticmethod
    def trans_entropy(T: float, mass_amu: float, p_Pa: float, c_M: float) -> float:
        """Translational entropy (J/mol/K) at pressure p:

        S = R [ ln( (2π m kT / h^2)^(3/2) * (kT/p) ) + 5/2 ], with m =
        molecular mass (kg).
        """
        assert p_Pa or c_M, "Either pressure or concentration must be provided."
        if p_Pa:
            V = kB * T / p_Pa  # m^3
        elif c_M:
            V = 1 / (NA * c_M * 1000)  # m^3
        m = mass_amu * AMU_TO_KG
        q = ((2.0 * math.pi * m * kB * T) / h**2) ** 1.5 * V
        return R * (math.log(q) + 2.5)
