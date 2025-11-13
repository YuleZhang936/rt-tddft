import shutil
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

_PYSCF_ROOT = _PROJECT_ROOT / "pyscf-master" / "pyscf-master"
if _PYSCF_ROOT.is_dir():
    sys.path.insert(0, str(_PYSCF_ROOT))

import numpy as np
from pyscf import dft, gto

from rttddft.evolve import simulate_delta_kick_response


def _build_cu(spin: int = 1):
    """Return a Copper atom suited for relativistic runs."""
    mol = gto.Mole()
    mol.atom = "Cu 0 0 0"
    mol.basis = "dyall_v2z"
    mol.spin = spin
    mol.charge = 0
    mol.verbose = 0
    mol.build()
    return mol


def test_rttddft_x2c1e():
    """Perform a delta-kick RT-TDDFT run for an X2C1e reference."""
    mol = _build_cu()
    mf = dft.GKS(mol, xc="LDA,VWN").x2c1e()
    mf.conv_tol = 1e-10
    mf.kernel()

    dm_store_dir = Path(__file__).with_name("cu_x2c1e_dm_store")
    if dm_store_dir.exists():
        shutil.rmtree(dm_store_dir)
    plot_path = Path(__file__).with_name("cu_x2c1e_strength.png")
    if plot_path.exists():
        plot_path.unlink()

    response = simulate_delta_kick_response(
        mf,
        engine="x2c1e",
        directions=("x", "y", "z"),
        dt=0.2,
        n_steps=30000,
        field_strength=5e-4,
        damping=5e-4,
        dm_store_dir=str(dm_store_dir),
        plot_path=str(plot_path),
        max_midpoint_cycles=30,
        midpoint_tol=1e-9,
    )

    print(f"[X2C1e] S(omega) integral = {response.strength_integral:.8e}")
    if response.plot_path:
        print(f"[X2C1e] Strength plot stored at: {response.plot_path}")
    print(
        "[X2C1e] Max strength = "
        f"{response.strength_function.max():.8e} at omega="
        f"{response.omega[np.argmax(response.strength_function)]:.6f}"
    )

    assert response.plot_path is not None
    assert Path(response.plot_path).exists()
    assert dm_store_dir.exists()
    assert response.dipole_moments.shape[0] == 3  # x, y, z kicks
    assert not np.isnan(response.strength_integral)


if __name__ == "__main__":
    test_rttddft_x2c1e()
