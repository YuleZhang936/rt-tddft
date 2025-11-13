"""CLI driver for delta-kick real-time TDDFT simulations.

The script is intentionally lightweight so it can be launched on an HPC node
without editing source files.  It prepares a relativistic PySCF reference,
applies a delta-like external field, propagates the density matrix in real
time, and finally reports the strength function integral together with a plot
of :math:`S(\omega)`.

Usage example (Rubidium, X2C1e reference)::

    python -m rttddft.run_delta_kick --engine x2c1e --n-steps 2000 --dt 2e-3 \
        --plot-path rb_strength.png --dm-store-dir ./rb_densities
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Sequence, Tuple

ROOT_PYSCF = Path(__file__).resolve().parents[2] / "pyscf-master" / "pyscf-master"
if ROOT_PYSCF.is_dir() and str(ROOT_PYSCF) not in sys.path:
    sys.path.insert(0, str(ROOT_PYSCF))

from pyscf import dft, gto  # type: ignore  # pylint: disable=import-error

from .evolve import DeltaKickResponse, simulate_delta_kick_response


def _parse_directions(raw: str) -> Tuple[str, ...]:
    if not raw:
        raise argparse.ArgumentTypeError("Direction string must not be empty.")
    cleaned = tuple(c.lower() for c in raw if not c.isspace())
    allowed = {"x", "y", "z"}
    invalid = [c for c in cleaned if c not in allowed]
    if invalid:
        raise argparse.ArgumentTypeError(
            f"Invalid Cartesian directions: {', '.join(invalid)}."
        )
    return cleaned


def build_mf(args: argparse.Namespace):
    mol = gto.Mole()
    mol.atom = args.atom
    mol.basis = args.basis
    mol.spin = args.spin
    mol.charge = args.charge
    mol.verbose = args.verbose
    mol.build()

    if args.engine == "x2c1e":
        mf = dft.GKS(mol, xc=args.xc).x2c1e()
    else:
        mf = dft.DKS(mol, xc=args.xc)
    mf.conv_tol = args.conv_tol
    energy = mf.kernel()
    print(f"SCF energy ({args.engine}) = {energy:.12f} Ha")
    return mf


def run(args: argparse.Namespace) -> DeltaKickResponse:
    mf = build_mf(args)
    response = simulate_delta_kick_response(
        mf,
        engine=args.engine,
        directions=args.directions,
        dt=args.dt,
        n_steps=args.n_steps,
        field_strength=args.field_strength,
        damping=args.damping,
        origin=tuple(args.origin),
        picture_change=not args.no_picture_change,
        max_midpoint_cycles=args.max_midpoint_cycles,
        midpoint_tol=args.midpoint_tol,
        dm_store_dir=args.dm_store_dir,
        store_initial=True,
        callback=None,
        plot_path=args.plot_path,
    )
    print(f"Strength integral int S(omega) d omega = {response.strength_integral:.8e}")
    if response.plot_path:
        print(f"Strength plot written to: {response.plot_path}")
    if args.dm_store_dir:
        print(f"Density snapshots stored under: {args.dm_store_dir}")
    return response


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", choices=("x2c1e", "dks"), default="x2c1e")
    parser.add_argument("--atom", default="Rb 0 0 0", help="Molecule specification")
    parser.add_argument("--basis", default="dyall_v2z")
    parser.add_argument("--xc", default="LDA,VWN")
    parser.add_argument("--spin", type=int, default=1)
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--dt", type=float, default=5e-4)
    parser.add_argument("--n-steps", type=int, default=2000)
    parser.add_argument("--field-strength", type=float, default=5e-4)
    parser.add_argument("--damping", type=float, default=5e-4)
    parser.add_argument("--directions", type=_parse_directions, default=("x", "y", "z"))
    parser.add_argument("--origin", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    parser.add_argument("--max-midpoint-cycles", type=int, default=30)
    parser.add_argument("--midpoint-tol", type=float, default=1e-10)
    parser.add_argument("--dm-store-dir", type=str, default=None)
    parser.add_argument("--plot-path", type=str, default="strength_function.png")
    parser.add_argument("--conv-tol", type=float, default=1e-10)
    parser.add_argument("--no-picture-change", action="store_true")
    parser.add_argument("--verbose", type=int, default=0)
    args = parser.parse_args(argv)
    args.directions = (
        args.directions
        if isinstance(args.directions, tuple)
        else tuple(args.directions)
    )
    run(args)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
