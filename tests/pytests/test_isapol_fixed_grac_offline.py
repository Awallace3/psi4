"""Correction admission test doubles; no native LibXC/SCF numerical claim.

Run with --noconftest -o addopts='' to avoid the repository's Psi4 conftest.
The test double checks admission logic, not the LibXC/SCF numerical contract.
"""
from dataclasses import FrozenInstanceError, replace
import importlib.util
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import pytest

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'psi4/driver/procrouting/isapol_native_correction.py'
SEAL = ROOT / 'psi4/driver/procrouting/scf_proc/_scf_seal.py'


class Component:
    def __init__(self, name, unpolarized=True):
        self.label, self.scale, self.polarity = name, 1., unpolarized
        self.tweaks = {}
    def name(self): return self.label
    def alpha(self): return self.scale
    def set_alpha(self, value): self.scale = value
    def omega(self): return 0.
    def is_gga(self): return 'GGA' in self.label
    def is_meta(self): return False
    def is_lrc(self): return False
    def is_unpolarized(self): return self.polarity
    def get_mix_data(self): return [(self.label, 0, 1.)]
    def query_libxc(self, key): return {'OMEGA': 0., 'ALPHA': 0., 'BETA': 0.}
    def get_tweak(self): return dict(self.tweaks)
    def density_cutoff(self): return 1e-15


class Functional:
    def __init__(self):
        self.controls = dict(x_alpha=.25, x_beta=0., x_omega=0., c_alpha=0., c_omega=0.,
            c_os_alpha=0., c_ss_alpha=0., vv10_b=0., vv10_c=0., grac_shift=0.,
            grac_alpha=.5, grac_beta=40., ansatz=1, is_libxc_func=True, needs_vv10=False,
            needs_grac=False, is_x_lrc=False, is_c_lrc=False)
        self.x, self.c = None, None
        self.base = Component('XC_HYB_GGA_XC_PBEH')
    def __getattr__(self, key):
        if key in self.controls:
            return lambda: self.controls[key]
        raise AttributeError(key)
    def name(self): return 'PBE0'
    def x_functionals(self): return []
    def c_functionals(self): return [self.base]
    def grac_x_functional(self): return self.x
    def grac_c_functional(self): return self.c


@pytest.fixture
def module(monkeypatch):
    psi = ModuleType('psi4')
    psi.core = SimpleNamespace(LibXCFunctional=Component,
        SuperFunctional=SimpleNamespace(XC_build=lambda *a: Functional()))
    monkeypatch.setitem(sys.modules, 'psi4', psi)
    # The seal helpers live in scf_proc/_scf_seal.py (imported relatively), so the
    # source files are loaded under a private package with the same layout.
    for name in ('_fixed_grac_offline', '_fixed_grac_offline.scf_proc'):
        package = ModuleType(name)
        package.__path__ = []
        monkeypatch.setitem(sys.modules, name, package)
    loaded = None
    for name, path in (('_fixed_grac_offline.scf_proc._scf_seal', SEAL),
                       ('_fixed_grac_offline.isapol_native_correction', SOURCE)):
        spec = importlib.util.spec_from_file_location(name, path)
        loaded = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, spec.name, loaded)
        spec.loader.exec_module(loaded)
    return loaded


def profile():
    f = Functional()
    f.controls.update(needs_grac=True, grac_shift=.06)
    f.x, f.c = Component('XC_GGA_X_LB'), Component('XC_LDA_C_VWN')
    f.x.set_alpha(.75)
    return f, SimpleNamespace(functional=lambda: f)


def test_none_and_fixed_profile_owned(module, monkeypatch):
    f = Functional()
    assert module.validate_correction(SimpleNamespace(functional=lambda: f), require_canonical=True).policy == 'NONE'
    f, w = profile()
    calls = []
    monkeypatch.setattr(module, 'require_scf_seal', lambda wfn: calls.append(wfn))
    p = module.validate_correction(w, scf_correction='FIXED_GRAC', expected_grac_shift=.06)
    assert calls == [w]
    assert p.shift == .06 and 'no GRAC kernel derivative' in p.response_description
    with pytest.raises(FrozenInstanceError):
        p.shift = .1
    old = p.components
    f.x.set_alpha(.9)
    assert p.components == old and p != replace(p, shift=.1)


@pytest.mark.parametrize('shift', [None, 0, True, '0.06', float('nan'), .061])
def test_bad_declaration_never_reaches_seal(module, monkeypatch, shift):
    _, w = profile()
    monkeypatch.setattr(module, 'require_scf_seal', lambda w: pytest.fail('seal entered'))
    with pytest.raises(ValueError):
        module.validate_correction(w, scf_correction='FIXED_GRAC', expected_grac_shift=shift)


@pytest.mark.parametrize('mutation', ['alpha', 'beta', 'nan', 'scale', 'identity', 'tweak',
                                     'polarized', 'underlying', 'flag_only', 'missing_x'])
def test_actual_mismatch_never_reaches_seal(module, monkeypatch, mutation):
    f, w = profile()
    if mutation == 'alpha': f.controls['grac_alpha'] = .6
    if mutation == 'beta': f.controls['grac_beta'] = 41.
    if mutation == 'nan': f.controls['grac_beta'] = float('nan')
    if mutation == 'scale': f.x.scale = 1.
    if mutation == 'identity': f.c.label = 'XC_LDA_C_PW'
    if mutation == 'tweak': f.x.tweaks['beta'] = .4
    if mutation == 'polarized': f.x.polarity = False
    if mutation == 'underlying': f.base.scale = .9
    if mutation == 'flag_only': f.controls['needs_grac'] = False
    if mutation == 'missing_x': f.x = None
    monkeypatch.setattr(module, 'require_scf_seal', lambda w: pytest.fail('seal entered'))
    with pytest.raises(ValueError):
        module.validate_correction(w, scf_correction='FIXED_GRAC', expected_grac_shift=.06)


def test_none_rejects_attachments_even_without_flag(module):
    f, w = profile()
    f.controls.update(needs_grac=False, grac_shift=0.)
    with pytest.raises(ValueError, match='NONE'):
        module.validate_correction(w)
    with pytest.raises(ValueError, match='NONE'):
        module.validate_correction(w, expected_grac_shift=.06)
