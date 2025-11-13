import os
import sys

# Ensure the local PySCF checkout is importable when running from this repo.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'pyscf-master', 'pyscf-master'))
if os.path.isdir(_ROOT) and _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pyscf import gto, dft, scf


def _build_rb(spin=1):
    """Return a Rubidium molecule suited for relativistic runs."""
    mol = gto.Mole()
    mol.atom = 'Rb 0 0 0'
    mol.basis = 'dyall_v2z'
    mol.spin = spin
    mol.charge = 0
    mol.verbose = 4
    mol.build()
    return mol


def run_x2c_smearing():
    mol = _build_rb()
    mf = dft.GKS(mol, xc='LDA,VWN').x2c1e()
    mf = scf.addons.smearing(mf, sigma=0.02, method='fermi')
    return mf.kernel()


def run_dks_smearing():
    mol = _build_rb()
    mf = dft.DKS(mol, xc='LDA,VWN')
    mf = scf.addons.smearing(mf, sigma=0.02, method='fermi')
    return mf.kernel()


if __name__ == '__main__':
    e_x2c = run_x2c_smearing()
    print(f'X2C1e smeared total energy (Rb): {e_x2c:.12f}')

    e_dks = run_dks_smearing()
    print(f'DKS smeared total energy (Rb): {e_dks:.12f}')
