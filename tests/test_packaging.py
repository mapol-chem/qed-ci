"""Packaging, installed-package, example, and documentation contracts."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tomllib
import zipfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
METADATA = ROOT / 'pyproject.toml'
WHEEL_GLOB = 'qed_ci-*.whl'


def native_prefix():
    prefix = Path(os.environ.get('QED_BUILD_PREFIX', sys.base_prefix)).resolve()
    if sys.platform == 'darwin':
        required = ['include/cblas.h', 'include/lapacke.h', 'include/omp.h',
                    'lib/libopenblas.dylib', 'lib/libomp.dylib']
        return prefix if all((prefix / item).exists() for item in required) else None
    if sys.platform.startswith('linux'):
        mklroot = os.environ.get('MKLROOT')
        if mklroot and (Path(mklroot) / 'include/mkl.h').is_file() and shutil.which('icx'):
            return prefix
    return None


PREFIX = native_prefix()
NEEDS_NATIVE = pytest.mark.skipif(
    PREFIX is None,
    reason='native build dependencies not found; set QED_BUILD_PREFIX or install '
           'OpenBLAS/LAPACKE/LLVM OpenMP')


def build_environment():
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    for key in list(env):
        if key.startswith(('QED_', 'CPP_CASSCF_')):
            env.pop(key)
    env['QED_BUILD_PREFIX'] = str(PREFIX)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return env


def run_environment():
    env = dict(os.environ)
    env.pop('PYTHONPATH', None)
    for key in list(env):
        if key.startswith(('QED_', 'CPP_CASSCF_')):
            env.pop(key)
    env.update(OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')
    return env


def project_metadata():
    return tomllib.loads(METADATA.read_text(encoding='utf-8'))


def build_wheel(tmp_path):
    output = tmp_path / 'wheelhouse'
    output.mkdir()
    subprocess.run([sys.executable, '-m', 'pip', 'wheel', '--no-build-isolation',
                    '--no-deps', '--no-cache-dir', str(ROOT), '-w', str(output)],
                   cwd=tmp_path, env=build_environment(), check=True,
                   capture_output=True, text=True, timeout=300)
    wheels = list(output.glob(WHEEL_GLOB))
    assert len(wheels) == 1, wheels
    return wheels[0]


def make_venv(tmp_path, name):
    target = tmp_path / name
    subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(target)],
                   check=True, timeout=120)
    return target


def run_in_venv(venv, code, cwd):
    script = (
        'import json, pathlib\n'
        'from qed_ci import CQEDConfig, CISolverOptions, calculate\n'
        'from qed_ci.backend import backend_path\n'
        'import qed_ci\n'
        + code
    )
    process = subprocess.run([str(venv / 'bin' / 'python'), '-B', '-c', script],
                             cwd=cwd, env=run_environment(), check=True,
                             capture_output=True, text=True, timeout=180)
    return process


def test_project_metadata_and_version():
    import qed_ci
    metadata = project_metadata()
    project = metadata['project']
    assert project['name'] == 'qed-ci'
    assert project['version'] == qed_ci.__version__
    assert project['requires-python'] == '>=3.12,<3.14'
    assert project['license'] == 'GPL-3.0-only'
    requirements = project['dependencies']
    for package in ('numpy', 'scipy', 'numba', 'llvmlite', 'psutil'):
        assert any(item.startswith(package) for item in requirements), package
    modules = metadata['tool']['setuptools']['py-modules']
    for module in ('helper_PFCI', 'helper_cqed_rhf', 'nuclear_grad',
                   'orbital_sigma', 'trust_region_solvers', 'mecp_optimizer'):
        assert module in modules
    assert metadata['build-system']['build-backend'] == 'setuptools.build_meta'


def test_documentation_links_and_option_reference():
    process = subprocess.run([sys.executable, str(ROOT / 'tools/check_docs.py')],
                             cwd=ROOT, env=run_environment(), check=True,
                             capture_output=True, text=True, timeout=60)
    assert 'Documentation checks passed' in process.stdout


def test_quickstart_examples_run_from_source(tmp_path):
    if not any(path.is_file() for path in (
            ROOT / 'src/cfunctions.so', ROOT / 'src/qed_ci/_native/cfunctions.so')):
        pytest.skip('no source-checkout native library')
    env = run_environment()
    env['PYTHONPATH'] = str(ROOT / 'src')
    process = subprocess.run([sys.executable, str(ROOT / 'examples/quickstart/methods.py'),
                              'cis', '--quiet'], cwd=tmp_path, env=env, check=True,
                             capture_output=True, text=True, timeout=180)
    assert 'Final energies (Eh):' in process.stdout
    assert 'Convergence status:' in process.stdout


@NEEDS_NATIVE
def test_wheel_is_platform_tagged_and_contains_native_library(tmp_path):
    wheel = build_wheel(tmp_path)
    assert 'py3-none-any' not in wheel.name
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        metadata_name = next(name for name in names
                             if name.endswith('.dist-info/METADATA'))
        metadata = archive.read(metadata_name).decode()
        wheel_metadata = archive.read(
            next(name for name in names if name.endswith('.dist-info/WHEEL'))
        ).decode()
    assert 'qed_ci/_native/cfunctions.so' in names
    assert 'helper_PFCI.py' in names and 'qed_ci/config.py' in names
    assert 'mecp_optimizer.py' in names
    assert not any(name.endswith(('.o', '.npy', '.npz', '.dylib')) for name in names)
    assert not any(name.startswith(
        ('src/', 'docs/', 'examples/', 'tests/', 'validation/', 'build/')
    ) for name in names)
    requires_python = next(line for line in metadata.splitlines()
                           if line.startswith('Requires-Python:'))
    assert '3.12' in requires_python and '3.14' in requires_python
    for requirement in ('numpy>=1.26', 'scipy>=1.11', 'numba>=0.60',
                        'llvmlite>=0.43', 'psutil>=5.9'):
        assert f'Requires-Dist: {requirement}' in metadata, requirement
    assert 'Tag: py3-none-' in wheel_metadata and 'py3-none-any' not in wheel_metadata
    assert 'Root-Is-Purelib: false' in wheel_metadata


@NEEDS_NATIVE
def test_source_distribution_contents_exclude_artifacts(tmp_path):
    output = tmp_path / 'sdist'
    output.mkdir()
    spec = importlib.util.find_spec('build')
    wheel_spec = importlib.util.find_spec('wheel')
    if (spec is not None and spec.origin is not None and
            wheel_spec is not None and wheel_spec.origin is not None):
        command = [sys.executable, '-m', 'build', '--sdist', '--no-isolation',
                   '--outdir', str(output), str(ROOT)]
        cwd = tmp_path
    else:
        # setuptools' legacy sdist command is enough for content inspection when
        # the optional build frontend is absent.
        command = [sys.executable, str(ROOT / 'setup.py'), 'sdist',
                   '--dist-dir', str(output)]
        cwd = ROOT
    subprocess.run(command, cwd=cwd, env=build_environment(), check=True,
                   capture_output=True, text=True, timeout=300)
    archives = list(output.glob('qed_ci-*.tar.gz'))
    assert len(archives) == 1, archives
    with tarfile.open(archives[0]) as archive:
        names = [member.name for member in archive.getmembers()]
    suffix = archives[0].name[:-len('.tar.gz')] + '/'
    for required in ('src/ci_solver.c', 'src/orbital.c', 'src/qed_blas.h',
                     'tools/build_backend.py', 'docs/installation.md',
                     'docs/configuration-and-output.md',
                     'examples/canonical/README.md',
                     'examples/canonical/qed_cis_energy.py',
                     'examples/canonical/qed_fci_energy.py',
                     'examples/canonical/lih_polariton_surface_scan.py',
                     'examples/canonical/qed_casci_energy.py',
                     'examples/canonical/qed_casscf_energy.py',
                     'examples/canonical/sa_qed_casscf_energy.py',
                     'examples/canonical/sa_qed_casscf_energy_gradient.py',
                     'examples/canonical/sa_qed_casscf_coupling_vectors.py',
                     'examples/canonical/lih_polaritonic_mecp.py',
                     'src/mecp_optimizer.py',
                     'examples/quickstart/methods.py', 'tests/test_packaging.py',
                     'tests/test_helperPFCI.py',
                     'tests/data/lstrs_hard_case_lih.npz',
                     'validation/numerical_nac_check.py',
                     'validation/benchmarks/c10h8_casscf.py',
                     'validation/drivers/ethylene_mecp_scan.py',
                     'docs/theory/numerical_nac_method.md'):
        assert suffix + required in names, required
    assert not any(name.endswith(('.so', '.o', '.npy', '.dylib')) for name in names)
    npz_files = [name for name in names if name.endswith('.npz')]
    assert npz_files == [suffix + 'tests/data/lstrs_hard_case_lih.npz']
    assert not any('/build/' in name or 'egg-info' in name for name in names)
    assert not any(name.endswith(('timer.dat', 'output.dat')) for name in names)
    # Check the artifact users receive, not only links in the full checkout.
    unpacked = tmp_path / 'unpacked'
    with tarfile.open(archives[0]) as archive:
        archive.extractall(unpacked, filter='data')
    subprocess.run([sys.executable, str(unpacked / suffix / 'tools/check_docs.py')],
                   cwd=tmp_path, env=run_environment(), check=True,
                   capture_output=True, text=True, timeout=60)


@NEEDS_NATIVE
def test_installed_wheel_works_outside_checkout(tmp_path):
    wheel = build_wheel(tmp_path)
    venv = make_venv(tmp_path, 'wheel-venv')
    subprocess.run([str(venv / 'bin' / 'python'), '-m', 'pip', 'install',
                    '--no-deps', '--force-reinstall', str(wheel)],
                   cwd=tmp_path, env=run_environment(), check=True,
                   capture_output=True, text=True, timeout=180)
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    process = run_in_venv(venv, '''
assert 'site-packages' in qed_ci.__file__, qed_ci.__file__
path = backend_path()
assert 'site-packages' in str(path), path
assert path.is_file()
config = CQEDConfig(method='casci', active_electrons=2, active_orbitals=2,
    n_roots=2, max_photons=0,
    psi4_options={'basis': 'sto-3g', 'e_convergence': 1e-10,
                  'd_convergence': 1e-10},
    ci_solver=CISolverOptions(threshold=1e-9), quiet=True)
job = calculate('Li\\nH 1 1.4\\nsymmetry c1', config)
energies = job.result.energies['final']
assert abs(energies[0] + 7.860729506592616) < 1e-8, energies
assert job.result.provenance['package_version'], '0.1.0'
assert job.result.provenance['backend'] == str(path)
''', run_dir)
    assert process.returncode == 0


@NEEDS_NATIVE
def test_editable_install_uses_source_native_library(tmp_path):
    venv = make_venv(tmp_path, 'editable-venv')
    subprocess.run([str(venv / 'bin' / 'python'), '-m', 'pip', 'install',
                    '--no-build-isolation', '--no-deps', '-e', str(ROOT)],
                   cwd=ROOT, env=build_environment(), check=True,
                   capture_output=True, text=True, timeout=300)
    run_dir = tmp_path / 'run'
    run_dir.mkdir()
    process = run_in_venv(venv, '''
source = pathlib.Path(qed_ci.__file__).resolve()
assert source.is_relative_to(pathlib.Path(%r).resolve()), source
path = backend_path()
assert path.is_file()
assert path.parent.name == '_native'
assert path.is_relative_to(pathlib.Path(%r).resolve()), path
config = CQEDConfig(method='casci', active_electrons=2, active_orbitals=2,
    n_roots=2, max_photons=0,
    psi4_options={'basis': 'sto-3g', 'e_convergence': 1e-10,
                  'd_convergence': 1e-10},
    ci_solver=CISolverOptions(threshold=1e-9), quiet=True)
job = calculate('Li\\nH 1 1.4\\nsymmetry c1', config)
assert len(job.result.energies['final']) == 2
''' % (str(ROOT / 'src'), str(ROOT / 'src')), run_dir)
    assert process.returncode == 0
