"""Helpers for building Fock matrices from density matrices with PySCF.

This module focuses on relativistic references that are frequently used in
real-time TDDFT experiments: the full two-component X2C1e Hamiltonian and the
four-component Dirac-Kohn-Sham (DKS) Hamiltonian.  PySCF already exposes the
density-to-Fock mapping for both cases through :meth:`get_fock`, but wrapping
that logic here keeps the propagation code independent from the PySCF object
model.

While inspecting ``pyscf.x2c.x2c.SCF`` and ``pyscf.dft.dks.DKS`` we observe
that, once the Fock matrix is assembled, PySCF solves the generalized
eigenvalue problem with ``scipy.linalg.eigh(f, s)`` (see
``pyscf.scf.hf.eig``).  This corresponds to a Löwdin orthonormalization of the
AO basis using :math:`S^{\pm 1/2}`.  Two convenience helpers are therefore
provided to transform density and Fock matrices to the orthonormal
representation when needed (e.g. for external propagators that expect an
orthonormal basis).
"""

from __future__ import annotations

import base64
import json
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Sequence, Tuple, Union

import numpy as np
from numpy.typing import ArrayLike

from pyscf.dft import dks as dks_mod
from pyscf.x2c import x2c as x2c_mod


Array = np.ndarray
_C_AU = 137.035999084  # speed of light in atomic units
_HARTREE_TO_EV = 27.211386245988
_CART_AXES = ("x", "y", "z")
_AXIS_TO_INDEX = {axis: idx for idx, axis in enumerate(_CART_AXES)}


def dm_to_fock_x2c1e(
    mf: x2c_mod.SCF,
    dm: ArrayLike,
    *,
    hermi: int = 1,
    diis: Optional[object] = None,
    cycle: int = -1,
    **kwargs,
) -> Array:
    """Return the X2C Fock matrix for a given density matrix.

    Parameters
    ----------
    mf
        A fully configured :class:`pyscf.x2c.x2c.SCF` (or one of its subclasses).
        ``mf.with_x2c.approx`` should contain ``'1E'`` to ensure the full
        one-electron picture-change Hamiltonian is used.
    dm
        Density matrix in the spinor AO basis (2-component representation).
    hermi
        Hermiticity flag forwarded to :meth:`pyscf.scf.hf.get_fock`.
    diis, cycle
        Optional arguments forwarded to :meth:`get_fock` so the caller can
        control DIIS behaviour during out-of-core propagation.  With the
        defaults (:data:`diis=None`, ``cycle=-1``) PySCF performs a single
        evaluation without entering the SCF macro-iteration loop.
    kwargs
        Any additional keyword arguments are passed straight through to
        ``mf.get_fock``.

    Returns
    -------
    numpy.ndarray
        The Fock matrix in the AO spinor basis.
    """
    if not hasattr(mf, "with_x2c"):
        raise TypeError("Expected an X2C mean-field object that exposes 'with_x2c'.")
    approx = getattr(mf.with_x2c, "approx", "")
    if "1E" not in str(approx).upper():
        raise ValueError(
            "The supplied X2C object is not configured for the full 1e approximation."
        )
    dm_arr = np.asarray(dm)
    fock = mf.get_fock(dm=dm_arr, hermi=hermi, diis=diis, cycle=cycle, **kwargs)
    return np.asarray(fock)


def dm_to_fock_dks(
    mf: dks_mod.DKS,
    dm: ArrayLike,
    *,
    hermi: int = 1,
    diis: Optional[object] = None,
    cycle: int = -1,
    **kwargs,
) -> Array:
    """Return the DKS Fock matrix for a given density matrix.

    Parameters
    ----------
    mf
        A :class:`pyscf.dft.dks.DKS` (or subclass such as :class:`RDKS`),
        pre-built for the molecule of interest.
    dm
        Four-component density matrix expressed in the AO spinor basis.
    hermi
        Hermiticity flag forwarded to :meth:`pyscf.scf.hf.get_fock`.
    diis, cycle
        Optional DIIS controls forwarded to :meth:`get_fock`.  Leaving the
        defaults unchanged triggers just one evaluation of the Coulomb and XC
        potentials without running an SCF cycle.
    kwargs
        Extra keyword arguments forwarded to ``mf.get_fock``.

    Returns
    -------
    numpy.ndarray
        The Fock matrix in the AO spinor basis.
    """
    if not isinstance(mf, dks_mod.KohnShamDFT):
        raise TypeError("Expected a Dirac-Kohn-Sham mean-field object.")
    dm_arr = np.asarray(dm)
    # Some PySCF releases (e.g., 2.3.x) do not accept hermi/diis/cycle in DKS.get_fock.
    # Keep the parameters for API symmetry but avoid passing unsupported keywords.
    fock = mf.get_fock(dm=dm_arr, **kwargs)
    return np.asarray(fock)


def density_to_orthonormal(
    mf: Any,
    dm: ArrayLike,
    *,
    lindep_tol: float = 1e-12,
) -> Array:
    """Transform a density matrix to the Löwdin orthonormal AO basis.

    PySCF transforms orbitals via :math:`C' = S^{1/2} C` before diagonalising
    the Fock matrix (see ``pyscf.scf.hf.eig``).  Applying the same change of
    basis to the density matrix yields:

    .. math::
        P' = S^{1/2} P S^{1/2}

    Parameters
    ----------
    mf
        Mean-field object providing :meth:`get_ovlp` (e.g., X2C or DKS).
    dm
        Density matrix in the original AO basis.
    lindep_tol
        Eigenvalues of :math:`S` below this threshold are treated as linearly
        dependent and discarded.

    Returns
    -------
    numpy.ndarray
        Density matrix in the orthonormal basis.
    """
    if not hasattr(mf, "get_ovlp"):
        raise TypeError("Mean-field object does not provide get_ovlp.")
    s_half, _ = _lowdin_factors(mf.get_ovlp(), lindep_tol=lindep_tol)
    dm_arr = np.asarray(dm)
    return s_half @ dm_arr @ s_half


def fock_to_orthonormal(
    mf: Any,
    fock: ArrayLike,
    *,
    lindep_tol: float = 1e-12,
) -> Array:
    """Transform a Fock (or any operator) matrix to the orthonormal AO basis.

    The Löwdin transformation that converts the generalized eigenvalue problem
    :math:`F C = S C \\varepsilon` into a standard eigenvalue problem reads
    :math:`F' = S^{-1/2} F S^{-1/2}`.

    Parameters
    ----------
    mf
        Mean-field object providing :meth:`get_ovlp` (e.g., X2C or DKS).
    fock
        Operator matrix in the original AO basis.
    lindep_tol
        Eigenvalues of :math:`S` below this threshold are treated as linearly
        dependent and discarded.

    Returns
    -------
    numpy.ndarray
        Operator matrix in the orthonormal basis.
    """
    if not hasattr(mf, "get_ovlp"):
        raise TypeError("Mean-field object does not provide get_ovlp.")
    _, s_inv_half = _lowdin_factors(mf.get_ovlp(), lindep_tol=lindep_tol)
    fock_arr = np.asarray(fock)
    return s_inv_half @ fock_arr @ s_inv_half


def _lowdin_factors(
    overlap: ArrayLike,
    *,
    lindep_tol: float = 1e-12,
) -> Tuple[Array, Array]:
    """Compute :math:`S^{1/2}` and :math:`S^{-1/2}` via eigen-decomposition."""
    s = np.asarray(overlap)
    herm_s = 0.5 * (s + s.conj().T)
    eigvals, eigvecs = np.linalg.eigh(herm_s)
    keep = eigvals > lindep_tol
    if not np.all(keep):
        dropped = (~keep).sum()
        if dropped == eigvals.size:
            raise np.linalg.LinAlgError("Overlap matrix is singular within tolerance.")
        eigvals = eigvals[keep]
        eigvecs = eigvecs[:, keep]
    sqrt_e = np.sqrt(eigvals)
    inv_sqrt_e = 1.0 / sqrt_e
    s_half = (eigvecs * sqrt_e) @ eigvecs.conj().T
    s_inv_half = (eigvecs * inv_sqrt_e) @ eigvecs.conj().T
    return s_half, s_inv_half


def _hermitize(matrix: ArrayLike) -> Array:
    """Return the Hermitian part of ``matrix``."""
    arr = np.asarray(matrix)
    return 0.5 * (arr + arr.conj().T)


def _unitary_from_hamiltonian(hamiltonian: ArrayLike, dt: float) -> Array:
    """Build the midpoint propagator ``exp[-i H dt]`` via eigen-decomposition."""
    herm_h = _hermitize(hamiltonian)
    eigvals, eigvecs = np.linalg.eigh(herm_h)
    phases = np.exp(-1j * dt * eigvals)
    return (eigvecs * phases) @ eigvecs.conj().T


def _relative_norm(delta: ArrayLike, reference: ArrayLike) -> float:
    """Return ``||delta|| / max(||reference||, 1)`` to avoid division by zero."""
    delta_norm = np.linalg.norm(delta)
    ref_norm = np.linalg.norm(reference)
    if ref_norm < 1e-18:
        ref_norm = 1.0
    return float(delta_norm / ref_norm)


@dataclass
class TimeStepResult:
    """Container returned at every real-time TDDFT step."""

    step: int
    time: float
    density_ao: Array
    density_orth: Array
    fock_ao: Array
    hamiltonian_orth: Array
    midpoint_hamiltonian: Array
    midpoint_iterations: int
    midpoint_residual: float
    converged: bool


@dataclass
class DeltaKickResponse:
    """Hold the full response extracted from a delta-kick propagation."""

    directions: Tuple[str, ...]
    field_strength: float
    damping: float
    times: Array
    omega: Array
    dipole_moments: Array
    induced_dipoles: Array
    static_dipole: Array
    polarizability: Array
    strength_function: Array
    strength_integral: float
    plot_path: Optional[Path] = None

    def alpha_trace(self) -> Array:
        """Return :math:`\\alpha_{xx} + \\alpha_{yy} + \\alpha_{zz}`."""
        trace = np.zeros_like(self.omega, dtype=np.complex128)
        for dir_idx, axis_label in enumerate(self.directions):
            axis_idx = _AXIS_TO_INDEX.get(axis_label.lower())
            if axis_idx is None:
                continue
            if axis_idx >= self.polarizability.shape[0]:
                continue
            if dir_idx >= self.polarizability.shape[1]:
                continue
            trace += self.polarizability[axis_idx, dir_idx]
        return trace


def _build_spinor_dipole_operator(
    mf: Any,
    *,
    engine: str,
    origin: Sequence[float],
    picture_change: bool,
) -> Array:
    """Return the electric dipole operators in the spinor AO basis."""
    if not hasattr(mf, "mol"):
        raise TypeError("Mean-field object does not expose `mol`.")
    mol = mf.mol
    if len(origin) != 3:
        raise ValueError("`origin` must contain three cartesian components.")
    origin_vec = tuple(float(x) for x in origin)
    if engine == "x2c1e":
        with mol.with_common_orig(origin_vec):
            if picture_change:
                if not hasattr(mf, "with_x2c"):
                    raise ValueError("X2C mean-field object is required.")
                ao_dip = mf.with_x2c.picture_change(
                    ("int1e_r_spinor", "int1e_sprsp_spinor")
                )
            else:
                ao_dip = mol.intor_symmetric("int1e_r_spinor", comp=3)
        ao_dip = np.asarray(ao_dip, dtype=np.complex128)
        if ao_dip.shape[0] != 3:
            raise ValueError("Dipole integrals are expected to have three components.")
        return -ao_dip

    # DKS (four-component) case: build a block matrix matching the 4c density.
    with mol.with_common_orig(origin_vec):
        ll_dip = mol.intor_symmetric("int1e_r_spinor", comp=3)
        ss_dip = mol.intor_symmetric("int1e_sprsp_spinor", comp=3)
    ll_dip = np.asarray(ll_dip, dtype=np.complex128)
    ss_dip = np.asarray(ss_dip, dtype=np.complex128)
    if ll_dip.shape[0] != 3 or ss_dip.shape[0] != 3:
        raise ValueError("Dipole integrals are expected to have three components.")
    n2c = mol.nao_2c()
    n4c = 2 * n2c
    dipole = np.zeros((3, n4c, n4c), dtype=np.complex128)
    dipole[:, :n2c, :n2c] = -ll_dip
    dipole[:, n2c:, n2c:] = -ss_dip * (0.5 / _C_AU) ** 2
    return dipole


def _apply_delta_kick(dm: ArrayLike, operator: ArrayLike, strength: float) -> Array:
    """Apply :math:`e^{i P \\cdot E}` to ``dm`` for a delta-like external field."""
    unitary = _unitary_from_hamiltonian(operator, strength)
    dm_arr = np.asarray(dm)
    return _hermitize(unitary @ dm_arr @ unitary.conj().T)


def _dipole_expectation(dipole_ops: ArrayLike, dm: ArrayLike) -> Array:
    """Return the expectation value of the dipole operator for ``dm``."""
    ops = np.asarray(dipole_ops)
    dm_arr = np.asarray(dm)
    values = np.einsum("aij,ji->a", ops, dm_arr)
    return np.asarray(values, dtype=np.complex128)


class DensityMatrixStore:
    """Append-only helper that keeps density matrices on disk.

    Each entry is serialized as a JSON record containing the step index, the
    propagation time, the array metadata, and a base64 payload for the raw
    bytes.  This keeps the file human-inspectable (it's plain text) while
    avoiding the prohibitive size explosion that would accompany a fully
    expanded textual representation of large complex matrices.
    """

    def __init__(self, path: Union[str, Path]):
        self.path = Path(path)
        if self.path.parent and not self.path.parent.exists():
            self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, step: int, time: float, density: ArrayLike) -> None:
        """Append a density matrix snapshot."""
        dm_arr = np.ascontiguousarray(np.asarray(density))
        record = {
            "step": int(step),
            "time": float(time),
            "shape": list(dm_arr.shape),
            "dtype": dm_arr.dtype.str,
            "payload": base64.b64encode(dm_arr.tobytes()).decode("ascii"),
        }
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record))
            handle.write("\n")

    def read(self, step: int) -> Array:
        """Load the density matrix recorded for ``step``."""
        if not self.path.exists():
            raise FileNotFoundError(f"Density store '{self.path}' does not exist.")
        with self.path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("step") != step:
                    continue
                payload = base64.b64decode(record["payload"])
                arr = np.frombuffer(payload, dtype=np.dtype(record["dtype"]))
                return arr.reshape(tuple(record["shape"]))
        raise KeyError(f"Step {step} not found in '{self.path}'.")


class RealTimeTDDFT:
    """Propagate the density matrix with the Liouville-von Neumann equation.

    Parameters
    ----------
    mf
        A pre-built PySCF mean-field object.  Only X2C1e (two-component) and
        DKS (four-component) references are currently supported.
    engine
        Force the propagator to use ``'x2c1e'`` or ``'dks'``.  When omitted the
        engine is derived automatically from ``mf``.
    lindep_tol
        Linear-dependency threshold used while building the Löwdin factors.
    dm_store_path
        Optional path to a text file where every density snapshot is persisted.
        This is useful for long time traces that would otherwise exhaust
        memory; snapshots can be loaded later with :meth:`DensityMatrixStore.read`.
    """

    def __init__(
        self,
        mf: Any,
        *,
        engine: Optional[str] = None,
        lindep_tol: float = 1e-12,
        dm_store_path: Optional[Union[str, Path]] = None,
    ) -> None:
        self.mf = mf
        self.engine = self._resolve_engine(mf, engine)
        self._dm_to_fock: Callable[[Any, Array], Array]
        if self.engine == "x2c1e":
            self._dm_to_fock = dm_to_fock_x2c1e
        else:
            self._dm_to_fock = dm_to_fock_dks
        self._s_half, self._s_inv_half = _lowdin_factors(
            self.mf.get_ovlp(), lindep_tol=lindep_tol
        )
        self._dm_store = DensityMatrixStore(dm_store_path) if dm_store_path else None

    @property
    def dm_store(self) -> Optional[DensityMatrixStore]:
        """Return the active density-matrix store, if any."""
        return self._dm_store

    def propagate(
        self,
        *,
        dt: float,
        n_steps: int,
        dm0: Optional[ArrayLike] = None,
        t0: float = 0.0,
        max_midpoint_cycles: int = 30,
        midpoint_tol: float = 1e-10,
        callback: Optional[Callable[[TimeStepResult], None]] = None,
        copy_outputs: bool = True,
        store_initial: bool = True,
    ) -> Iterator[TimeStepResult]:
        """Yield density snapshots at ``t0 + k * dt`` for ``k=1..n_steps``.

        Parameters
        ----------
        dt, n_steps
            Time step (in atomic units) and number of propagation steps.
        dm0
            Optional initial density matrix in the AO basis.  When omitted the
            converged SCF density from ``mf`` is used.
        max_midpoint_cycles, midpoint_tol
            Control the midpoint fixed-point iteration that refines
            :math:`H(t_{i-1} + \\Delta t / 2)`.
        callback
            Callable invoked with every :class:`TimeStepResult`.
        copy_outputs
            When :data:`False`, the arrays exposed by :class:`TimeStepResult`
            alias the propagator's internal state and must be treated as
            read-only (mutating them would corrupt the next step).
        store_initial
            If :data:`True`, the initial density is written to the store before
            propagation starts.
        """
        if dt <= 0.0:
            raise ValueError("Time step `dt` must be positive.")
        if n_steps < 1:
            raise ValueError("`n_steps` must be at least 1.")
        if max_midpoint_cycles < 1:
            raise ValueError("`max_midpoint_cycles` must be at least 1.")
        if midpoint_tol <= 0.0:
            raise ValueError("`midpoint_tol` must be positive.")

        dm_prev_ao = np.asarray(dm0 if dm0 is not None else self.mf.make_rdm1())
        dm_prev_ao = _hermitize(dm_prev_ao)
        dm_prev_orth = self._density_to_orth(dm_prev_ao)
        fock_prev_ao = _hermitize(self._dm_to_fock(self.mf, dm_prev_ao))
        h_prev_orth = _hermitize(self._fock_to_orth(fock_prev_ao))

        time = float(t0)
        prev_midpoint: Optional[Array] = None

        if self._dm_store and store_initial:
            self._dm_store.append(0, time, dm_prev_ao)

        for step in range(1, n_steps + 1):
            time += dt
            (
                dm_new_ao,
                dm_new_orth,
                fock_new_ao,
                h_new_orth,
                midpoint,
                cycles,
                residual,
                converged,
            ) = self._propagate_step(
                dt=dt,
                dm_prev_ao=dm_prev_ao,
                dm_prev_orth=dm_prev_orth,
                fock_prev_ao=fock_prev_ao,
                h_prev_orth=h_prev_orth,
                prev_midpoint=prev_midpoint,
                max_cycles=max_midpoint_cycles,
                tol=midpoint_tol,
            )

            if self._dm_store:
                self._dm_store.append(step, time, dm_new_ao)

            if not converged:
                warnings.warn(
                    f"Step {step} failed to converge within "
                    f"{max_midpoint_cycles} midpoint cycles "
                    f"(residual={residual:.3e}).",
                    RuntimeWarning,
                )

            dm_prev_ao = dm_new_ao
            dm_prev_orth = dm_new_orth
            fock_prev_ao = fock_new_ao
            h_prev_orth = h_new_orth
            prev_midpoint = midpoint

            result = TimeStepResult(
                step=step,
                time=time,
                density_ao=dm_new_ao.copy() if copy_outputs else dm_new_ao,
                density_orth=dm_new_orth.copy() if copy_outputs else dm_new_orth,
                fock_ao=fock_new_ao.copy() if copy_outputs else fock_new_ao,
                hamiltonian_orth=h_new_orth.copy() if copy_outputs else h_new_orth,
                midpoint_hamiltonian=midpoint.copy() if copy_outputs else midpoint,
                midpoint_iterations=cycles,
                midpoint_residual=residual,
                converged=converged,
            )

            if callback is not None:
                callback(result)

            yield result

    def _density_to_orth(self, dm: ArrayLike) -> Array:
        return self._s_half @ np.asarray(dm) @ self._s_half

    def _density_to_ao(self, dm_orth: ArrayLike) -> Array:
        return self._s_inv_half @ np.asarray(dm_orth) @ self._s_inv_half

    def _fock_to_orth(self, fock: ArrayLike) -> Array:
        return self._s_inv_half @ np.asarray(fock) @ self._s_inv_half

    def _propagate_step(
        self,
        *,
        dt: float,
        dm_prev_ao: Array,
        dm_prev_orth: Array,
        fock_prev_ao: Array,
        h_prev_orth: Array,
        prev_midpoint: Optional[Array],
        max_cycles: int,
        tol: float,
    ) -> Tuple[Array, Array, Array, Array, Array, int, float, bool]:
        """Single midpoint step following the Liouville-von Neumann equation."""
        midpoint = (
            h_prev_orth if prev_midpoint is None else 2.0 * h_prev_orth - prev_midpoint
        )
        dm_last = None
        residual = np.inf
        converged = False

        for cycle in range(1, max_cycles + 1):
            unitary = _unitary_from_hamiltonian(midpoint, dt)
            dm_new_orth = _hermitize(unitary @ dm_prev_orth @ unitary.conj().T)
            dm_new_ao = _hermitize(self._density_to_ao(dm_new_orth))
            fock_new_ao = _hermitize(self._dm_to_fock(self.mf, dm_new_ao))
            h_new_orth = _hermitize(self._fock_to_orth(fock_new_ao))
            midpoint_updated = 0.5 * (h_prev_orth + h_new_orth)

            dm_diff = (
                _relative_norm(dm_new_orth - dm_last, dm_new_orth)
                if dm_last is not None
                else np.inf
            )
            mid_diff = _relative_norm(midpoint_updated - midpoint, midpoint_updated)
            residual = float(max(mid_diff, 0.0 if np.isinf(dm_diff) else dm_diff))
            dm_last = dm_new_orth
            midpoint = midpoint_updated

            if not np.isinf(dm_diff) and residual < tol:
                converged = True
                break

        return (
            dm_new_ao,
            dm_new_orth,
            fock_new_ao,
            h_new_orth,
            midpoint,
            cycle,
            residual,
            converged,
        )

    @staticmethod
    def _resolve_engine(mf: Any, explicit: Optional[str]) -> str:
        """Inspect ``mf`` (and ``explicit``) to decide which backend to use."""
        if explicit is not None:
            normalized = explicit.strip().lower()
            if normalized not in {"x2c1e", "dks"}:
                raise ValueError("`engine` must be either 'x2c1e' or 'dks'.")
            if normalized == "x2c1e" and not hasattr(mf, "with_x2c"):
                raise ValueError("The supplied mean-field object is not X2C-enabled.")
            if normalized == "dks" and not isinstance(mf, dks_mod.KohnShamDFT):
                raise ValueError("The supplied mean-field object is not a DKS reference.")
            return normalized
        if hasattr(mf, "with_x2c") and getattr(mf.with_x2c, "approx", None):
            approx = str(mf.with_x2c.approx).upper()
            if "1E" in approx:
                return "x2c1e"
        if isinstance(mf, dks_mod.KohnShamDFT):
            return "dks"
        raise ValueError(
            "Could not infer the engine type. Please pass engine='x2c1e' or engine='dks'."
        )


def simulate_delta_kick_response(
    mf: Any,
    *,
    engine: Optional[str] = None,
    directions: Sequence[str] = ("x", "y", "z"),
    dt: float,
    n_steps: int,
    field_strength: float = 5e-4,
    damping: float = 5e-4,
    origin: Sequence[float] = (0.0, 0.0, 0.0),
    picture_change: bool = True,
    max_midpoint_cycles: int = 30,
    midpoint_tol: float = 1e-10,
    dm_store_dir: Optional[Union[str, Path]] = None,
    store_initial: bool = True,
    callback: Optional[Callable[[str, TimeStepResult], None]] = None,
    plot_path: Optional[Union[str, Path]] = None,
) -> DeltaKickResponse:
    """Propagate the density matrix under a delta-like external field.

    Parameters
    ----------
    mf
        Configured PySCF mean-field object (X2C1e or DKS).
    directions
        Iterable containing any combination of ``'x'``, ``'y'``, ``'z'`` that
        specifies the orientation of the pulsed field.
    field_strength
        Strength :math:`\\kappa` of the delta kick (atomic units).
    damping
        Exponential damping factor :math:`\\gamma` applied before the Fourier
        transform to avoid delta-like peaks.
    origin
        Origin for evaluating the dipole operators.
    picture_change
        When :data:`True`, the X2C dipole operator is picture-change corrected.
    dm_store_dir
        Optional directory where per-direction density snapshots are written.
    plot_path
        Optional path where the strength function :math:`S(\\omega)` is stored.

    Returns
    -------
    DeltaKickResponse
        Container with the time- and frequency-domain observables.
    """

    if not directions:
        raise ValueError("At least one Cartesian direction must be provided.")
    normalized_dirs = tuple(axis.lower() for axis in directions)
    for axis in normalized_dirs:
        if axis not in _AXIS_TO_INDEX:
            raise ValueError(f"Invalid direction '{axis}'. Use x, y, or z.")

    engine_name = RealTimeTDDFT._resolve_engine(mf, engine)
    dipole_ops = _build_spinor_dipole_operator(
        mf, engine=engine_name, origin=origin, picture_change=picture_change
    )
    dm_reference = _hermitize(np.asarray(mf.make_rdm1()))
    static_dipole = _dipole_expectation(dipole_ops, dm_reference).real

    n_dirs = len(normalized_dirs)
    n_times = n_steps + 1
    dipoles = np.zeros((n_dirs, 3, n_times), dtype=np.complex128)
    axis_indices = [_AXIS_TO_INDEX[axis] for axis in normalized_dirs]
    reference_times: Optional[Array] = None

    dm_store_dir_path: Optional[Path] = None
    if dm_store_dir is not None:
        dm_store_dir_path = Path(dm_store_dir)
        dm_store_dir_path.mkdir(parents=True, exist_ok=True)

    for dir_idx, axis_idx in enumerate(axis_indices):
        store_path = None
        if dm_store_dir_path is not None:
            store_path = dm_store_dir_path / f"density_{normalized_dirs[dir_idx]}.txt"
        propagator = RealTimeTDDFT(mf, engine=engine_name, dm_store_path=store_path)
        dm_kicked = _apply_delta_kick(
            dm_reference, dipole_ops[axis_idx], field_strength
        )
        dipoles[dir_idx, :, 0] = _dipole_expectation(dipole_ops, dm_kicked)
        local_times = np.zeros(n_times, dtype=float)
        local_times[0] = 0.0

        def _capture(result: TimeStepResult, axis_label=normalized_dirs[dir_idx], idx=dir_idx):
            local_times[result.step] = result.time
            dipoles[idx, :, result.step] = _dipole_expectation(
                dipole_ops, result.density_ao
            )
            if callback is not None:
                callback(axis_label, result)

        for _ in propagator.propagate(
            dt=dt,
            n_steps=n_steps,
            dm0=dm_kicked,
            t0=0.0,
            max_midpoint_cycles=max_midpoint_cycles,
            midpoint_tol=midpoint_tol,
            callback=_capture,
            store_initial=store_initial,
        ):
            pass

        if reference_times is None:
            reference_times = local_times
        elif not np.allclose(reference_times, local_times, atol=1e-12):
            raise RuntimeError("Propagation timelines differ between directions.")

    assert reference_times is not None
    time_axis = np.asarray(reference_times, dtype=float)
    damping_window = np.exp(-damping * time_axis)
    freqs = np.fft.rfftfreq(time_axis.size, d=dt)
    omega = 2.0 * np.pi * freqs
    n_freq = omega.size
    polarizability = np.zeros((3, n_dirs, n_freq), dtype=np.complex128)
    induced = np.zeros_like(dipoles.real)

    for dir_idx, axis_idx in enumerate(axis_indices):
        mu_ind = dipoles[dir_idx].real - static_dipole[:, None]
        induced[dir_idx] = mu_ind
        damped = mu_ind * damping_window
        mu_freq = dt * np.fft.rfft(damped, axis=-1)
        polarizability[:, dir_idx, :] = mu_freq / field_strength

    alpha_trace = np.zeros(n_freq, dtype=np.complex128)
    for dir_idx, axis_idx in enumerate(axis_indices):
        alpha_trace += polarizability[axis_idx, dir_idx, :]

    strength = (4.0 * np.pi * omega / (3.0 * _C_AU)) * np.imag(alpha_trace)
    strength_real = np.real_if_close(strength, tol=1e-12)
    if np.iscomplexobj(strength_real):
        raise RuntimeError("Strength function has a non-negligible imaginary part.")
    strength = np.asarray(strength_real, dtype=float)
    strength_integral = float(np.trapz(strength, omega))

    plot_path_obj: Optional[Path] = None
    if plot_path is not None:
        plot_path_obj = _plot_strength_function(omega, strength, plot_path)

    return DeltaKickResponse(
        directions=normalized_dirs,
        field_strength=field_strength,
        damping=damping,
        times=time_axis,
        omega=omega,
        dipole_moments=dipoles,
        induced_dipoles=induced,
        static_dipole=static_dipole,
        polarizability=polarizability,
        strength_function=strength,
        strength_integral=strength_integral,
        plot_path=plot_path_obj,
    )


def _plot_strength_function(
    omega: ArrayLike, strength: ArrayLike, path: Union[str, Path]
) -> Path:
    """Save a plot of ``S(omega)`` vs energy (eV) and return the path."""
    omega_arr = np.asarray(omega, dtype=float)
    energy_ev = omega_arr * _HARTREE_TO_EV  # omega (a.u.) equals energy in Ha
    strength_arr = np.asarray(strength, dtype=float)
    target = Path(path)
    if target.parent and not target.parent.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib

        if matplotlib.get_backend().lower() == "agg":
            pass
        else:
            matplotlib.use("Agg", force=True)
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(
            "Matplotlib is required to plot the strength function."
        ) from exc

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(energy_ev, strength_arr, lw=1.5)
    ax.set_xlabel("Energy (eV)")
    ax.set_ylabel(r"$S(\omega)$")
    ax.set_title("Dipole Strength Function")
    ax.set_xlim(0.0, 20.0)
    ax.grid(True, which="both", ls="--", lw=0.5, alpha=0.5)
    fig.tight_layout()
    fig.savefig(target, dpi=300)
    plt.close(fig)
    return target


__all__ = [
    "dm_to_fock_x2c1e",
    "dm_to_fock_dks",
    "density_to_orthonormal",
    "fock_to_orthonormal",
    "TimeStepResult",
    "DeltaKickResponse",
    "DensityMatrixStore",
    "RealTimeTDDFT",
    "simulate_delta_kick_response",
]
