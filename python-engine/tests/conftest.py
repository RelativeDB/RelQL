"""Engine-package test setup: reuse the base package's fixtures.

These tests exercise the native scorer/backends against the same schemas and
wirings the base tests use, so the base conftest is loaded wholesale — by
file path, because pytest already owns the module name ``conftest`` for THIS
file. The monorepo layout puts both source trees on the path (an installed
pair works the same through site-packages).
"""
import importlib.util
import sys
from pathlib import Path

_here = Path(__file__).resolve()
_repo = _here.parents[2]
for p in (str(_repo / "python" / "src"),
          str(_repo / "python-engine" / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

_base_path = _repo / "python" / "tests" / "conftest.py"
_spec = importlib.util.spec_from_file_location("_base_conftest", _base_path)
_base = importlib.util.module_from_spec(_spec)
sys.modules["_base_conftest"] = _base
_spec.loader.exec_module(_base)

# Re-export everything public (fixtures included: pytest discovers fixtures
# per conftest module, so they must exist as attributes of THIS module).
for _name in dir(_base):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_base, _name)
