"""Shim so wheels that bundle the native librt_c get a platform tag.

All real metadata lives in pyproject.toml. setuptools tags a wheel
py3-none-any unless the distribution claims extension modules; a bundled
prebuilt .dylib/.so is invisible to that check, so claim them iff the
library sits next to the package sources (build_wheel.sh puts it there).
A source build without the library stays a pure (any) wheel that falls
back to RELATIVEDB_RT_LIB / the monorepo build tree at runtime.
"""

import os

from setuptools import setup
from setuptools.dist import Distribution

try:
    from setuptools.command.bdist_wheel import bdist_wheel as _bdist_wheel
except ImportError:  # older setuptools
    from wheel.bdist_wheel import bdist_wheel as _bdist_wheel


class BinaryDistribution(Distribution):
    def has_ext_modules(self):
        pkg = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "src", "relativedb")
        try:
            names = os.listdir(pkg)
        except OSError:
            return False
        return any(n.startswith(("librt_c", "rt_c")) for n in names)


class bdist_wheel(_bdist_wheel):
    def get_tag(self):
        python, abi, plat = super().get_tag()
        if not self.distribution.has_ext_modules():
            return python, abi, plat
        # librt_c is loaded via ctypes: platform-specific but independent of
        # the CPython version/ABI. On macOS bdist_wheel also refuses to tag
        # below the interpreter's own build target; the bundled dylib is
        # compiled for MACOSX_DEPLOYMENT_TARGET (build_wheel.sh), so let that
        # drive the platform tag.
        target = os.environ.get("MACOSX_DEPLOYMENT_TARGET")
        if target and plat.startswith("macosx_"):
            major, _, minor = target.partition(".")
            arch = plat.rsplit("_", 1)[-1]
            plat = f"macosx_{major}_{int(minor or 0)}_{arch}"
        return "py3", "none", plat


setup(distclass=BinaryDistribution, cmdclass={"bdist_wheel": bdist_wheel})
