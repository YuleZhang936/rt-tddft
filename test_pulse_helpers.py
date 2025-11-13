"""Unit tests that cover the pulsed-field helpers without invoking PySCF."""

import sys
import types
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

if "pyscf" not in sys.modules:
    pyscf_stub = types.ModuleType("pyscf")
    sys.modules["pyscf"] = pyscf_stub
    pyscf_stub.dft = types.ModuleType("pyscf.dft")
    sys.modules["pyscf.dft"] = pyscf_stub.dft
    pyscf_stub.dft.dks = types.ModuleType("pyscf.dft.dks")
    sys.modules["pyscf.dft.dks"] = pyscf_stub.dft.dks

    class _DummyKS:
        def __init__(self, *args, **kwargs):
            pass

    pyscf_stub.dft.dks.DKS = _DummyKS
    pyscf_stub.dft.dks.KohnShamDFT = _DummyKS

    pyscf_stub.x2c = types.ModuleType("pyscf.x2c")
    sys.modules["pyscf.x2c"] = pyscf_stub.x2c
    pyscf_stub.x2c.x2c = types.ModuleType("pyscf.x2c.x2c")
    sys.modules["pyscf.x2c.x2c"] = pyscf_stub.x2c.x2c

    class _DummyX2C:
        def __init__(self, *args, **kwargs):
            self.with_x2c = types.SimpleNamespace(approx="1E")

    pyscf_stub.x2c.x2c.SCF = _DummyX2C

from rttddft.evolve import (
    DeltaKickResponse,
    _apply_delta_kick,
    _dipole_expectation,
)


def test_delta_kick_preserves_trace():
    dm = np.array(
        [
            [0.4, 0.05, 0.02j, 0.0],
            [0.05, 0.3, -0.01j, 0.03],
            [-0.02j, 0.01j, 0.2, 0.04],
            [0.0, 0.03, 0.04, 0.1],
        ],
        dtype=np.complex128,
    )
    dm = 0.5 * (dm + dm.conj().T)
    dm /= np.trace(dm).real
    operator = np.diag([0.1, -0.2, 0.3, -0.4])
    kicked = _apply_delta_kick(dm, operator, strength=1e-3)
    assert np.allclose(kicked, kicked.conj().T)
    assert np.isclose(np.trace(kicked), np.trace(dm))


def test_dipole_expectation_matches_manual_result():
    dm = np.array([[1.0, 0.2j], [-0.2j, 0.5]])
    ops = np.stack(
        [
            np.array([[0.0, 1.0], [1.0, 0.0]]),
            np.array([[1.0, 0.0], [0.0, -1.0]]),
            np.eye(2),
        ]
    )
    expected = np.array(
        [
            np.trace(ops[0] @ dm),
            np.trace(ops[1] @ dm),
            np.trace(ops[2] @ dm),
        ]
    )
    values = _dipole_expectation(ops, dm)
    assert np.allclose(values, expected)


def test_delta_kick_response_alpha_trace():
    omega = np.linspace(0.0, 1.0, 5)
    polarizability = np.zeros((3, 2, omega.size), dtype=np.complex128)
    polarizability[0, 0, :] = 1.0 + 0.5j
    polarizability[1, 1, :] = 0.5 + 0.25j
    response = DeltaKickResponse(
        directions=("x", "y"),
        field_strength=1.0,
        damping=1.0,
        times=np.zeros(omega.size),
        omega=omega,
        dipole_moments=np.zeros((2, 3, omega.size), dtype=np.complex128),
        induced_dipoles=np.zeros((2, 3, omega.size), dtype=np.complex128),
        static_dipole=np.zeros(3),
        polarizability=polarizability,
        strength_function=np.zeros_like(omega),
        strength_integral=0.0,
    )
    trace = response.alpha_trace()
    assert np.allclose(trace, polarizability[0, 0, :] + polarizability[1, 1, :])
