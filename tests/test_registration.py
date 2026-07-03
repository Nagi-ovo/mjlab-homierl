from mjlab.tasks.registry import list_tasks

import mjlab_homierl  # noqa: F401


def test_registered_task_ids() -> None:
  tasks = set(list_tasks())
  assert "Mjlab-Homie-Unitree-G1" in tasks
  assert "Mjlab-Homie-Unitree-H1" in tasks
  assert "Mjlab-Homie-Unitree-H1-with_hands" in tasks
