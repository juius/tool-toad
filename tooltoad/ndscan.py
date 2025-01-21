import itertools
import logging
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from joblib import Parallel, delayed
from matplotlib import cm
from scipy.ndimage import gaussian_filter, maximum_filter, minimum_filter
from sklearn.cluster import DBSCAN
from tqdm import tqdm

from tooltoad.chemutils import COVALENT_RADII, VDW_RADII, hartree2kcalmol
from tooltoad.orca import orca_calculate
from tooltoad.utils import tqdm_joblib
from tooltoad.xtb import xtb_calculate

_logger = logging.getLogger(__name__)


@dataclass
class ScanCoord:
    atom_ids: list[int]
    start: float
    end: float
    nsteps: int

    # TODO: have a metric on how many points i need per angstrom

    def __post_init__(self):
        if len(self.atom_ids) == 2:
            self.xtb_ctype = "distance"
            self.orca_ctype = "B"
        elif len(self.atom_ids) == 3:
            self.xtb_ctype = "angle"
            self.orca_ctype = "A"
        elif len(self.atom_ids) == 4:
            self.xtb_ctype = "dihedral"
            self.orca_ctype = "D"
        else:
            raise ValueError("Invalid number of atom_ids")

    @property
    def range(self):
        return np.linspace(self.start, self.end, self.nsteps)

    @property
    def xtb_atom_ids_str(self):
        return ", ".join([str(idx + 1) for idx in self.atom_ids])

    @property
    def orca_atom_ids_str(self):
        return " ".join([str(idx) for idx in self.atom_ids])

    @classmethod
    def from_current_position(
        cls,
        atoms: list[str],
        coords: list[list[float]] | np.ndarray,
        atom_ids: list[int],
        nsteps: int,
        bond_breaking: bool = False,
    ):
        """Create a ScanCoord instance from the current position of atoms.

        Parameters:
        - atoms: List of atom symbols.
        - coords: Numpy array of atomic coordinates.
        - atom_ids: List of atom IDs involved in the scan.
        - nsteps: Number of steps in the scan.
        - bond_breaking: Boolean indicating if bond breaking is considered.

        Returns:
        - ScanCoord instance.
        """
        coords = np.asarray(coords)
        if len(atom_ids) == 2:
            start = np.linalg.norm(coords[atom_ids[0]] - coords[atom_ids[1]])
            data_dict = VDW_RADII if bond_breaking else COVALENT_RADII
            end = sum(data_dict[atoms[i]] for i in atom_ids)
        else:
            raise NotImplementedError("Only distance scan is supported")
        return cls(atom_ids=atom_ids, start=start, end=end, nsteps=nsteps)


class PotentialEnergySurface:
    # TODO: get partial scan info
    def __init__(
        self,
        atoms: list[str],
        coords: list[list[float]] | np.ndarray,
        charge: int = 0,
        multiplicity: int = 1,
        scan_coords: None | list[dict] | list[ScanCoord] = None,
    ):
        """Initialize the PotentialEnergySurface instance.

        Parameters:
        - atoms: List of atom symbols.
        - coords: Numpy array of atomic coordinates.
        - charge: Molecular charge.
        - multiplicity: Spin multiplicity.
        - scan_coords: List of scan coordinates.
        """
        self.coords = np.asarray(coords)
        self.atoms = atoms
        self.charge = charge
        self.multiplicity = multiplicity
        self.scan_coords = [
            ScanCoord(**d) if isinstance(d, dict) else d for d in scan_coords
        ]

    def __repr__(self):
        return f"{self.__class__.__name__} with {len(self.atoms)} atoms and {len(self.scan_coords)} scan coordinates"

    @property
    def ndims(self):
        return len(self.scan_coords)

    @staticmethod
    def _construct_xtb_scan(
        scan_coords=list[dict] | list[ScanCoord],
        force_constant: float = 0.5,
        max_cycle: int = 10,
    ):
        """Construct xTB input strings for a multi-dimensional scan.

        Args:
            scan_coords (list[], optional): List of scan coordinates. Defaults to list[dict] | list[ScanCoord].
            force_constant (float, optional): Force constant for the scan.. Defaults to 0.5.
            max_cycle (int, optional): Maximum number of optimization cycles.. Defaults to 10.

        Returns:
            scan_value_tensor (np.ndarray): Array of scan values.
            input_strings (list[str]): List of input strings for xTB.
        """

        scan_coord = scan_coords[0]
        if len(scan_coords) > 1:
            dims = np.meshgrid(*[s.range for s in scan_coords[1:]], indexing="ij")
            constrain_points = np.vstack([d.reshape(-1) for d in dims]).T
        else:
            constrain_points = np.array([[]])
        input_strings = []
        for constains in constrain_points:
            input_str = f"$constrain\n force constant={force_constant}\n"
            input_str += (
                f" {scan_coord.xtb_ctype}: {scan_coord.xtb_atom_ids_str}, auto\n"
            )
            for i, c in enumerate(constains):
                sc = scan_coords[i + 1]
                input_str += f" {sc.xtb_ctype}: {sc.xtb_atom_ids_str}, {c}\n"
            input_str += "$scan\n"
            input_str += (
                f" 1: {scan_coord.start}, {scan_coord.end}, {scan_coord.nsteps}\n"
            )
            input_str += f"$opt\n maxcycle={max_cycle}\n"
            input_str += "$end\n"
            input_strings.append(input_str)
        scan_value_tensor = np.stack(
            np.meshgrid(*[s.range for s in scan_coords], indexing="ij"), axis=-1
        )
        _logger.info(
            f"Constructed {len(scan_coords)}D scan with {scan_value_tensor[..., 0].size} points via {len(input_strings)} 1D scans"
        )
        return scan_value_tensor, input_strings

    def xtb(
        self,
        n_cores: int = 1,
        force_constant: float = 1.0,
        max_cycle: int = 10,
        xtb_options: dict = {},
        scr: str = ".",
    ):
        """Evaluate the PES using xTB.

        Args:
            n_cores (int, optional): Number of CPU cores to use. Defaults to 1.
            force_constant (float, optional): Force constant for the scan. Defaults to 1.0.
            max_cycle (int, optional): Maximum number of optimization cycles. Defaults to 10.
            xtb_options (dict, optional): Dictionary of xTB options. Defaults to {}.
        """
        xtb_options.setdefault("opt", None)
        self.scan_value_tensor, detailed_strings = self._construct_xtb_scan(
            self.scan_coords, force_constant=force_constant, max_cycle=max_cycle
        )
        with tqdm_joblib(tqdm(desc="1-dimensional scans", total=len(detailed_strings))):
            results = Parallel(n_jobs=n_cores, prefer="threads")(
                delayed(xtb_calculate)(
                    atoms=self.atoms,
                    coords=self.coords,
                    charge=self.charge,
                    multiplicity=self.multiplicity,
                    options=xtb_options,
                    detailed_input_str=s,
                    scr=scr,
                )
                for s in detailed_strings
            )

        # construct results tensors
        pes_values = np.asarray(
            [
                (
                    r["scan"]["pes"]
                    if r["normal_termination"]
                    else np.ones(self.scan_value_tensor.shape[0]) * np.nan
                )
                for r in results
            ]
        )
        coord_shape = self.scan_value_tensor[..., 0].shape
        pes_tensor = pes_values.reshape(*(coord_shape[1:] + (coord_shape[0],)))
        self.pes_tensor = np.moveaxis(pes_tensor, -1, 0)

        traj_values = np.asarray(
            [
                (
                    r["scan"]["traj"]
                    if r["normal_termination"]
                    else np.ones((self.scan_value_tensor.shape[0],) + self.coords.shape)
                    * np.nan
                )
                for r in results
            ]
        )

        traj_tensor_shape = (*coord_shape[1:], *traj_values.shape[-3:])
        traj_tensor = traj_values.reshape(traj_tensor_shape)
        self.traj_tensor = np.moveaxis(traj_tensor, -3, 0)
        self.check_scan_quality()

    @staticmethod
    def _construct_orca_scan(
        scan_coords=list[dict] | list[ScanCoord],
        max_cycle: int = 10,
    ):
        """Construct Orca input strings for a multi-dimensional scan.

        Args:
            scan_coords (list[], optional): List of scan coordinates. Defaults to list[dict] | list[ScanCoord].
            force_constant (float, optional): Force constant for the scan.. Defaults to 0.5.
            max_cycle (int, optional): Maximum number of optimization cycles.. Defaults to 10.

        Returns:
            scan_value_tensor (np.ndarray): Array of scan values.
            input_strings (list[str]): List of input strings for xTB.
        """

        scan_coord = scan_coords[0]
        if len(scan_coords) > 1:
            dims = np.meshgrid(*[s.range for s in scan_coords[1:]], indexing="ij")
            constrain_points = np.vstack([d.reshape(-1) for d in dims]).T
        else:
            constrain_points = np.array([[]])
        scan_coord = scan_coords[0]
        if len(scan_coords) > 1:
            dims = np.meshgrid(*[s.range for s in scan_coords[1:]], indexing="ij")
            constrain_points = np.vstack([d.reshape(-1) for d in dims]).T
        else:
            constrain_points = np.array([[]])
        input_strings = []
        for constains in constrain_points:
            input_str = f"%geom\n  MaxIter {max_cycle}\n  Convergence loose \n  Scan\n"
            input_str += f"    {scan_coord.orca_ctype} {scan_coord.orca_atom_ids_str} = {scan_coord.start:.06f}, {scan_coord.end:.06f}, {scan_coord.nsteps}\n"
            input_str += "  end\n"
            for i, c in enumerate(constains):
                if i == 0:
                    input_str += "  Constraints\n"
                sc = scan_coords[i + 1]
                input_str += (
                    f"    {{{sc.orca_ctype} {sc.orca_atom_ids_str} {c:.06f} C}}\n"
                )
                if i == len(constains) - 1:
                    input_str += "  end\n"
            input_str += "end"
            input_strings.append(input_str)
        scan_value_tensor = np.stack(
            np.meshgrid(*[s.range for s in scan_coords], indexing="ij"), axis=-1
        )
        _logger.info(
            f"Constructed {len(scan_coords)}D scan with {scan_value_tensor[..., 0].size} points via {len(input_strings)} 1D scans"
        )
        return scan_value_tensor, input_strings

    def orca(
        self,
        n_cores: int = 1,
        max_cycle: int = 10,
        orca_options: dict = {},
        scr: str = ".",
    ):
        orca_options.setdefault("opt", None)
        self.scan_value_tensor, detailed_strings = self._construct_orca_scan(
            self.scan_coords, max_cycle=max_cycle
        )
        n_scans = len(detailed_strings)
        n_parallel = min(n_cores, n_scans)
        n_per_scan = n_cores // n_parallel
        with tqdm_joblib(tqdm(desc="1-dimensional scans", total=len(detailed_strings))):
            results = Parallel(n_jobs=n_parallel, prefer="threads")(
                delayed(orca_calculate)(
                    atoms=self.atoms,
                    coords=self.coords,
                    charge=self.charge,
                    multiplicity=self.multiplicity,
                    options=orca_options,
                    xtra_inp_str=s,
                    force=True,
                    scr=scr,
                    n_cores=n_per_scan,
                )
                for i, s in enumerate(detailed_strings)
            )
        # construct results tensors
        pes_values = np.asarray(
            [
                (
                    r["scan"]["pes"]
                    if r["normal_termination"]
                    else np.ones(self.scan_value_tensor.shape[0]) * np.nan
                )
                for r in results
            ]
        )
        coord_shape = self.scan_value_tensor[..., 0].shape
        pes_tensor = pes_values.reshape(*(coord_shape[1:] + (coord_shape[0],)))
        self.pes_tensor = np.moveaxis(pes_tensor, -1, 0)

        traj_values = np.asarray(
            [
                (
                    r["scan"]["traj"]
                    if r["normal_termination"]
                    else np.ones((self.scan_value_tensor.shape[0],) + self.coords.shape)
                    * np.nan
                )
                for r in results
            ]
        )

        traj_tensor_shape = (*coord_shape[1:], *traj_values.shape[-3:])
        traj_tensor = traj_values.reshape(traj_tensor_shape)
        self.traj_tensor = np.moveaxis(traj_tensor, -3, 0)
        self.check_scan_quality()

    def refine(self, n_cores: int = 1, orca_options: dict = {}, scr: str = "."):
        assert "opt" not in [k.lower() for k in orca_options.keys()]

        with tqdm_joblib(
            tqdm(desc="Single Point Calculations", total=self.pes_tensor.size)
        ):
            results = Parallel(n_jobs=n_cores, prefer="threads")(
                delayed(orca_calculate)(
                    atoms=self.atoms,
                    coords=c,
                    charge=self.charge,
                    multiplicity=self.multiplicity,
                    options=orca_options,
                    force=True,
                    scr=scr,
                )
                for c in self.traj_tensor.reshape(-1, *self.traj_tensor.shape[-2:])
            )
        self.refined_pes_tensor = np.array(
            [
                r["electronic_energy"] if r["normal_termination"] else np.nan
                for r in results
            ]
        ).reshape(self.pes_tensor.shape)

    @property
    def dimension_arrays(self):
        ranges = []

        for n in range(self.ndims):
            indices = [n] * (self.ndims + 1)
            indices[n] = slice(None)
            ranges.append(self.scan_value_tensor[tuple(indices)])
        return ranges

    def plot_2d(
        self,
        coord_slice: list[int] = [slice(None), slice(None)],
        ax: None | plt.Axes = None,
        cbar: bool = True,
        refined: bool = False,
        **plt_kwargs,
    ):
        """Plot a 2D slice of the potential energy surface.

        Args:
            coord_slice (list[int], optional): List of slices for the coordinates. Defaults to [slice(None), slice(None)].
            ax (plt.ax, optional): Matplotlib axis to plot on. Defaults to None.
            cbar (bool, optional): Boolean indicating if a colorbar should be added. Defaults to True.

        Returns:
            fig: Matplotlib figure.
            ax: Matplotlib axis.
            cbar: Matplotlib colorbar.
        """
        assert (
            len(coord_slice) == self.scan_value_tensor.shape[-1]
        ), f"Length of scan_coord_ids must match number of dimensions of PES ({self.scan_value_tensor.shape[-1]} dimensions), got {coord_slice}"
        slice_ids = [i for i, elem in enumerate(coord_slice) if isinstance(elem, slice)]
        assert (
            len(slice_ids) == 2
        ), f"Only 2D slices are supported, got {len(slice_ids)} slices"
        plt_kwargs.setdefault("levels", 20)
        plt_kwargs.setdefault("cmap", cm.PuBu)

        ranges = [self.dimension_arrays[i] for i in slice_ids]
        xx, yy = np.meshgrid(*ranges, indexing="ij")
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.get_figure()
        pes_tensor = self.refined_pes_tensor if refined else self.pes_tensor
        pes_slice = pes_tensor[tuple(coord_slice)].copy()
        pes_slice -= pes_slice.min()
        pes_slice = hartree2kcalmol(pes_slice)
        cs = ax.contourf(xx, yy, pes_slice, **plt_kwargs)
        if cbar:
            cbar = fig.colorbar(cs, pad=0.1)
            cbar.set_label("Relative Energy [kcal/mol]", rotation=270, labelpad=15)
            cbar.ax.yaxis.set_ticks_position("left")
            cbar.ax.yaxis.set_label_position("right")
        else:
            cbar = None
        for set_label, sc in zip(
            [ax.set_xlabel, ax.set_ylabel], [self.scan_coords[i] for i in slice_ids]
        ):
            set_label(
                f"{sc.xtb_ctype.capitalize()} {'-'.join([str(i) for i in sc.atom_ids])} {'[Å]' if sc.xtb_ctype == 'distance' else '[°]'}"
            )

        return fig, ax, cbar

    def plot_point(
        self, point: np.ndarray, mark_point: bool = True, refined: bool = False
    ):
        """Plot all slices through a specific point of the potential energy
        surface.

        Args:
            point (np.ndarray): Coordinates of the point to plot.
            mark_point (bool, optional): Boolean indicating if the point should be marked.. Defaults to True.

        Returns:
            fig: Matplotlib figure.
            ax: Matplotlib axis.
        """
        ncols = min([3, self.ndims])
        nrows = (self.ndims + ncols - 1) // ncols
        fig, axs = plt.subplots(
            ncols=ncols, nrows=nrows, figsize=(5 * ncols, 5 * nrows)
        )
        faxs = axs.flatten()
        coord_slices = []
        mark_slices = []
        for slice_ids in itertools.combinations(np.arange(len(point)), 2):
            tmp = np.array(point, dtype=object)
            tmp[list(slice_ids)] = slice(None)
            coord_slices.append(tuple(tmp))
            mark_slices.append(self.scan_value_tensor[point][np.array(slice_ids)])

        for ax, coord_slice, mark_slide in zip(faxs, coord_slices, mark_slices):
            self.plot_2d(coord_slice, ax=ax, cbar=False, refined=refined)
            if mark_point:
                ax.axvline(mark_slide[0], color="red", linestyle="dotted", marker="")
                ax.axhline(mark_slide[1], color="red", linestyle="dotted", marker="")

        return fig, ax

    def find_stationary_points(
        self,
        point_type: str = "saddle",
        tolerance: float = 1e-2,
        curvature_threshold: float = 1e-3,
        prune: bool = True,
        eps: float = 1.0,
        min_samples: int = 1,
        refined: bool = False,
    ):
        """Locates stationary points (minima, maxima, saddle points) on the
        PES.

        Parameters:
        - point_type (str): Type of stationary point to find ('minima', 'maxima', 'saddle').
        - tolerance (float): Convergence tolerance for identifying stationary points.

        Returns:
        - List of stationary points with coordinates and PES values.
        """
        # Apply Gaussian filter to smooth the PES tensor for stable derivative estimation
        pes_tensor = self.refined_pes_tensor if refined else self.pes_tensor
        smoothed_pes = gaussian_filter(
            pes_tensor, sigma=0.5
        )  # TODO: have this depend on scan step size
        gradient = np.gradient(smoothed_pes)
        smoothed_pes = gaussian_filter(
            pes_tensor, sigma=0.5
        )  # TODO: have this depend on scan step size
        gradient = np.gradient(smoothed_pes)

        if smoothed_pes.ndim == 1:
            gradient_magnitude = np.abs(gradient)
        else:
            gradient_magnitude = np.linalg.norm(gradient, axis=0)

        tolerance = 1e-2
        max_iterations = 100
        max_true_fraction = 0.25
        current_tolerance = tolerance
        adjustment_factor = 0.9

        for _ in range(max_iterations):
            stationary_mask = gradient_magnitude < current_tolerance
            true_fraction = np.sum(stationary_mask) / stationary_mask.size
            if true_fraction > max_true_fraction:
                current_tolerance *= adjustment_factor  # Decrease tolerance
            elif true_fraction < max_true_fraction * 0.5:
                current_tolerance /= adjustment_factor  # Increase tolerance
            else:
                break

        if point_type == "minima":
            footprint = np.ones((3,) * smoothed_pes.ndim, dtype=bool)
            local_min = minimum_filter(
                smoothed_pes, footprint=footprint, mode="nearest"
            )
            local_minima_mask = smoothed_pes == local_min
            minima = local_minima_mask & stationary_mask
            stationary_points = np.argwhere(minima)
        elif point_type == "maxima":
            footprint = np.ones((3,) * smoothed_pes.ndim, dtype=bool)
            local_min = maximum_filter(
                smoothed_pes, footprint=footprint, mode="nearest"
            )
            local_maxima_mask = smoothed_pes == local_min
            maxima = local_maxima_mask & stationary_mask
            stationary_points = np.argwhere(maxima)
        elif point_type == "saddle":
            stationary_points = self._detect_saddle_points(
                smoothed_pes, stationary_mask, curvature_threshold
            )
        else:
            raise ValueError(
                "Invalid point_type. Choose 'minima', 'maxima', or 'saddle'."
            )
        # sort stationary points by magnitude of gradient
        stationary_points = sorted(
            stationary_points, key=lambda idx: gradient_magnitude[tuple(idx)]
        )
        stationary_points_info = [
            {
                "idx": tuple(idx),
                "energy": pes_tensor[tuple(idx)],
                "grad_norm": gradient_magnitude[tuple(idx)],
            }
            for idx in stationary_points
        ]
        if prune & (len(stationary_points_info) > 0):
            if len(stationary_points_info[0]["idx"]) == 0:
                return []
            clustered_stationary_points = self._cluster_stationary_points(
                stationary_points_info, eps, min_samples
            )
        else:
            clustered_stationary_points = stationary_points_info
        return clustered_stationary_points

    @staticmethod
    def _detect_saddle_points(
        smoothed_pes: np.ndarray,
        stationary_mask: np.ndarray,
        curvature_threshold: float = 1e-3,
    ):
        """Detects saddle points by calculating the Hessian and analyzing the
        eigenvalues. A saddle point has a mix of positive and negative
        eigenvalues with significant curvature in the Hessian.

        Parameters:
        - smoothed_pes (np.ndarray): The PES tensor (after Gaussian smoothing).
        - stationary_mask (np.ndarray): Boolean array indicating potential stationary points.
        - curvature_threshold (float): Minimum magnitude of eigenvalues to consider significant.

        Returns:
        - List of indices of detected saddle points.
        """
        saddle_points = []
        hessian = PotentialEnergySurface._compute_hessian(smoothed_pes)

        for idx in np.argwhere(stationary_mask):
            hess_matrix = hessian[tuple(idx)]
            # Eigenvalue decomposition to check for mixed curvatures
            eigvals = np.linalg.eigvals(hess_matrix)

            # Check for mixed-sign eigenvalues with significant curvature
            if np.any(eigvals > curvature_threshold) and np.any(
                eigvals < -curvature_threshold
            ):
                saddle_points.append(tuple(idx))

        return saddle_points

    @staticmethod
    def _compute_hessian(smoothed_pes: np.ndarray):
        """Computes the Hessian matrix (second derivatives) of the PES using
        finite differences.

        Parameters:
        - smoothed_pes (np.ndarray): The PES tensor (after Gaussian smoothing).

        Returns:
        - Hessian (np.ndarray): The second derivatives of the PES.
        """
        hessian = np.zeros(smoothed_pes.shape + (smoothed_pes.ndim, smoothed_pes.ndim))

        for i in range(smoothed_pes.ndim):
            hessian[..., i, i] = np.gradient(np.gradient(smoothed_pes, axis=i), axis=i)
            for j in range(i + 1, smoothed_pes.ndim):
                hessian[..., i, j] = np.gradient(
                    np.gradient(smoothed_pes, axis=i), axis=j
                )
                hessian[..., j, i] = hessian[..., i, j]

        return hessian

    @staticmethod
    def _cluster_stationary_points(
        data: np.ndarray, eps: float = 1.0, min_samples: int = 3
    ):
        points = np.array([entry["idx"] for entry in data])
        grad_mags = np.array([entry["grad_norm"] for entry in data])

        dbscan = DBSCAN(eps=eps, min_samples=min_samples)
        labels = dbscan.fit_predict(points)

        # Find the point with the lowest grad_mag in each cluster
        clustered_points = []
        for cluster_label in set(labels):
            if cluster_label == -1:  # Skip noise points (-1)
                continue
            cluster_indices = np.where(labels == cluster_label)[0]
            best_point_idx = cluster_indices[np.argmin(grad_mags[cluster_indices])]
            clustered_points.append(data[best_point_idx])
        return clustered_points

    def check_scan_quality(
        self, mean_threshold: float = 1e-2, max_threshold: float = 1e-1
    ):
        distance_tensors = []
        for sc in self.scan_coords:
            if sc.xtb_ctype == "distance":
                ids = sc.atom_ids
                distance_tensors.append(
                    np.linalg.norm(
                        self.traj_tensor[..., ids[0], :]
                        - self.traj_tensor[..., ids[1], :],
                        axis=-1,
                    )
                )
            else:
                _logger.warning(
                    "Quality of scan can only be checked for distance scans"
                )
                return None
        value_tensor = np.stack(distance_tensors, axis=-1)
        diff = np.abs(value_tensor - self.scan_value_tensor)
        mean_diff = diff.mean()
        max_diff = diff.max()
        if mean_diff > mean_threshold or max_diff > max_threshold:
            _logger.warning(
                f"Scan quality is poor:\n\tDmean diff = {mean_diff:.06f}, max diff = {max_diff:.06f}"
            )
            _logger.warning(
                "It is recommended to increase the number of optimization steps (`max_cycle`) or increase the force constant (`force_constant`)"
            )
