from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

project = "mjlab-homierl"
copyright = "2026, mjlab-homierl Developers"
author = "mjlab-homierl Developers"

extensions: list[str] = []

source_suffix = {
  ".rst": "restructuredtext",
}

templates_path = ["_templates"]
exclude_patterns = [
  "_build",
  "Thumbs.db",
  ".DS_Store",
  "source/actuators.rst",
  "source/api/**",
  "source/distributed_training.rst",
  "source/faq.rst",
  "source/installation.rst",
  "source/migration_isaac_lab.rst",
  "source/motivation.rst",
  "source/nan_guard.rst",
  "source/observation.rst",
  "source/randomization.rst",
  "source/raycast_sensor.rst",
  "source/sensors.rst",
  "source/walkthrough/debugging_perf.rst",
  "source/walkthrough/how_to_add_g1_task.rst",
  "source/walkthrough/manager_based_env.rst",
  "source/walkthrough/managers_and_terms.rst",
  "source/walkthrough/quickstart.rst",
  "source/walkthrough/rewards_and_terminations.rst",
  "source/walkthrough/tasks_tracking_g1.rst",
  "source/walkthrough/tasks_velocity_g1.rst",
  "source/walkthrough_en/debugging_perf.rst",
  "source/walkthrough_en/how_to_add_g1_task.rst",
  "source/walkthrough_en/manager_based_env.rst",
  "source/walkthrough_en/managers_and_terms.rst",
  "source/walkthrough_en/quickstart.rst",
  "source/walkthrough_en/rewards_and_terminations.rst",
  "source/walkthrough_en/tasks_tracking_g1.rst",
  "source/walkthrough_en/tasks_velocity_g1.rst",
]

language = "en"
root_doc = "index"

html_title = "mjlab-homierl Documentation"
html_theme = "alabaster"
html_static_path = ["source/_static"]
html_css_files = ["css/custom.css"]
