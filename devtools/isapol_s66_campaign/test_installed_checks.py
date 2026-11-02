"""Operational harness tests only; no Psi4 import or scheduler contact."""
import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest

spec = importlib.util.spec_from_file_location(
    'installed_checks', Path(__file__).with_name('phoenix_installed_checks.py'))
checks = importlib.util.module_from_spec(spec)
spec.loader.exec_module(checks)


def test_junit_requires_nonempty_successful_test_run(tmp_path):
    report = tmp_path/'junit.xml'
    report.write_text('<testsuites><testsuite tests="4" failures="0" errors="0" skipped="0"/></testsuites>')
    assert checks.junit_summary(report) == dict(tests=4, failures=0, errors=0, skipped=0)
    report.write_text('<testsuites><testsuite tests="4" failures="1" errors="0" skipped="0"/></testsuites>')
    with pytest.raises(ValueError, match='successful'):
        checks.junit_summary(report)
    report.write_text('<testsuites><testsuite tests="0" failures="0" errors="0" skipped="0"/></testsuites>')
    with pytest.raises(ValueError, match='successful'):
        checks.junit_summary(report)


def test_sealed_runtime_and_external_dependency_changes_are_refused(tmp_path):
    environment = tmp_path/'env'
    environment.mkdir()
    module = environment/'module.py'
    module.write_text('original')
    library = tmp_path/'library.so'
    library.write_bytes(b'library')
    manifest = dict(runtime_files={'module.py': checks.digest(module)}, runtime_directories=['.'],
                    external_runtime_files={str(library): checks.digest(library)})
    checks.verify_runtime(environment, manifest)
    module.write_text('changed')
    with pytest.raises(RuntimeError, match='private installed runtime changed'):
        checks.verify_runtime(environment, manifest)
    module.write_text('original')
    library.write_bytes(b'changed')
    with pytest.raises(RuntimeError, match='external runtime changed'):
        checks.verify_runtime(environment, manifest)


def test_undeclared_runtime_symlink_is_refused(tmp_path):
    environment = tmp_path/'env'
    environment.mkdir()
    foreign = tmp_path/'foreign.py'
    foreign.write_text('not in the declared dependencies')
    (environment/'module.py').symlink_to(foreign)
    manifest = dict(runtime_files={'module.py': checks.digest(foreign)}, runtime_directories=['.'],
                    external_runtime_files={})
    with pytest.raises(RuntimeError, match='leaves declared dependencies'):
        checks.verify_runtime(environment, manifest)


def test_added_site_startup_file_is_refused_even_when_original_files_match(tmp_path):
    environment = tmp_path/'env'
    environment.mkdir()
    module = environment/'module.py'
    module.write_text('unchanged')
    manifest = dict(runtime_files={'module.py': checks.digest(module)}, runtime_directories=['.'],
                    external_runtime_files={})
    checks.verify_runtime(environment, manifest)
    (environment/'extra.pth').write_text('import unsealed_code')
    with pytest.raises(RuntimeError, match='inventory'):
        checks.verify_runtime(environment, manifest)


def test_inherited_pytest_selection_and_plugins_are_removed():
    values = checks.test_environment(dict(
        PATH='/declared/bin', PYTEST_ADDOPTS='-k only_one', PYTEST_PLUGINS='unsealed',
        PYTEST_DISABLE_PLUGIN_AUTOLOAD='0', PYTEST_CURRENT_TEST='outside'))
    assert values == dict(PATH='/declared/bin', PYTEST_DISABLE_PLUGIN_AUTOLOAD='1')


def test_installed_child_uses_a_fresh_explicit_bytecode_cache(tmp_path):
    command = checks.installed_python_command('/private/env/bin/python', tmp_path/'fresh-cache', '-m', 'pytest')
    assert command == [
        '/private/env/bin/python', '-I', '-B', '-X',
        'pycache_prefix='+str(tmp_path/'fresh-cache'), '-m', 'pytest']


def test_unsealed_sourceless_sitecustomize_bytecode_is_refused(tmp_path):
    environment = tmp_path/'env'
    environment.mkdir()
    # A standalone .pyc is importable without a .py source even with an
    # alternate cache prefix; do not confuse it with unused __pycache__ data.
    (environment/'sitecustomize.pyc').write_bytes(b'unsealed bytecode fixture')
    manifest = dict(runtime_files={}, runtime_directories=['.'], external_runtime_files={})
    with pytest.raises(RuntimeError, match='standalone bytecode'):
        checks.verify_runtime(environment, manifest)


def test_real_pytest_runs_all_explicit_tests_despite_inherited_selection(tmp_path):
    sample = tmp_path/'test_sample.py'
    sample.write_text('def test_first():\n    assert True\n\ndef test_second():\n    assert True\n')
    report = tmp_path/'junit.xml'
    env = checks.test_environment(dict(
        os.environ, PYTEST_ADDOPTS='-k first', PYTEST_PLUGINS='not_a_real_plugin'))
    command = checks.installed_python_command(
        sys.executable, tmp_path/'fresh-cache', '-m', 'pytest',
        '--noconftest', '-c', '/dev/null', '-p', 'no:cacheprovider',
        '-q', '--junitxml='+str(report), str(sample))
    result = subprocess.run(command, env=env, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, (result.stdout+result.stderr)[-2000:]
    assert checks.junit_summary(report)['tests'] == 2


def test_newer_test_revision_cannot_change_scientific_or_build_code():
    checks.validate_test_revision([
        'tests/pytests/test_isapol_bounded.py',
        'devtools/isapol_s66_campaign/phoenix_installed_checks.py',
        'devtools/isapol_s66_campaign/README.md',
    ])
    for forbidden in ('psi4/driver/procrouting/isapol_bounded.py', 'CMakeLists.txt',
                      'psi4/src/psi4/libisapol/pfit.cc', 'cmake/psi4OptionsTools.cmake'):
        with pytest.raises(ValueError, match='scientific/build'):
            checks.validate_test_revision([forbidden])


def test_installed_suite_requires_the_exact_declared_case_count(tmp_path):
    report = tmp_path/'junit.xml'
    expected = checks.EXPECTED_TESTS
    assert expected == 430
    assert {'test_isapol_pfit_workspace.py', 'test_isapol_pfit_forwarding.py',
            'test_isapol_extended_resources.py', 'test_isapol_bounded_response.py'} <= set(checks.TEST_FILES)
    for count in (expected-1, expected+1):
        report.write_text(
            f'<testsuites><testsuite tests="{count}" failures="0" errors="0" skipped="0"/></testsuites>')
        with pytest.raises(ValueError, match=f'exactly{expected}'):
            checks.junit_summary(report, expected=expected)
    report.write_text(
        f'<testsuites><testsuite tests="{expected}" failures="0" errors="0" skipped="0"/></testsuites>')
    assert checks.junit_summary(report, expected=expected)['tests'] == expected
