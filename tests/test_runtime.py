"""Guard the standalone deploy runtime's dependency contract."""

import ast
import importlib.util
import pathlib

RUNTIME = (
  pathlib.Path(__file__).resolve().parents[1]
  / "src"
  / "mjlab_homierl"
  / "runtime.py"
)

# The whole point of runtime.py is to be vendorable into consumer repos
# (BiGym plugin, robot-side deploy) without the training stack.
ALLOWED_TOP_LEVEL = {"__future__", "json", "collections", "numpy"}


def test_runtime_top_level_imports_stay_clean() -> None:
  tree = ast.parse(RUNTIME.read_text())
  roots = set()
  for node in tree.body:
    if isinstance(node, ast.Import):
      roots.update(a.name.split(".")[0] for a in node.names)
    elif isinstance(node, ast.ImportFrom):
      roots.add((node.module or "").split(".")[0])
  assert roots <= ALLOWED_TOP_LEVEL, (
    f"runtime.py grew top-level imports {roots - ALLOWED_TOP_LEVEL}; it must "
    "stay stdlib+numpy so consumers can vendor the single file."
  )


def test_runtime_loads_standalone_by_file_path() -> None:
  spec = importlib.util.spec_from_file_location("homie_runtime_isolated", RUNTIME)
  module = importlib.util.module_from_spec(spec)
  spec.loader.exec_module(module)
  assert len(module.G1_MOTOR_ORDER) == 29
  gravity = module.gravity_orientation([1.0, 0.0, 0.0, 0.0])
  assert gravity.tolist() == [0.0, 0.0, -1.0]
