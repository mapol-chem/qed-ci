"""Build the ctypes library using the same helper as source-checkout builds."""
import os
from pathlib import Path
import subprocess
import sys

from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext
from setuptools.command.sdist import sdist

try:
    from setuptools.command.bdist_wheel import bdist_wheel
except ImportError:  # setuptools < 70 with the external wheel command
    from wheel.bdist_wheel import bdist_wheel


class BackendBuild(build_ext):
    """Compile the ctypes library with tools/build_backend.py.

    The library has no CPython extension ABI, but it is platform- and runtime-
    library-specific, so the wheel tag is handled explicitly below.
    """

    def get_ext_filename(self, fullname):
        return os.path.join(*fullname.split('.')) + '.so'

    def build_extension(self, ext):
        root = Path(__file__).resolve().parent
        destination = Path(self.get_ext_fullpath(ext.name)).resolve()
        destination.parent.mkdir(parents=True, exist_ok=True)
        # Editable builds copy the built library back into src/qed_ci/_native;
        # setuptools assumes that directory already exists.
        (root / 'src' / 'qed_ci' / '_native').mkdir(parents=True, exist_ok=True)
        command = [sys.executable, str(root / 'tools/build_backend.py'),
                   '--output', str(destination),
                   '--prefix', os.environ.get('QED_BUILD_PREFIX', sys.base_prefix)]
        if os.environ.get('CC'):
            command += ['--compiler', os.environ['CC']]
        subprocess.run(command, check=True)


class PlatformWheel(bdist_wheel):
    def get_tag(self):
        # ctypes has no CPython ABI, but DOES depend on platform/runtime libraries.
        _, _, platform = super().get_tag()
        return 'py3', 'none', platform


class SourceDistribution(sdist):
    """Do not ship the generated local egg-info directory."""

    def make_release_tree(self, base_dir, files):
        files = [name for name in files
                 if not any('egg-info' in part for part in Path(name).parts)]
        super().make_release_tree(base_dir, files)


setup(ext_modules=[Extension('qed_ci._native.cfunctions', sources=[])],
      cmdclass={'build_ext': BackendBuild, 'bdist_wheel': PlatformWheel,
                'sdist': SourceDistribution})
