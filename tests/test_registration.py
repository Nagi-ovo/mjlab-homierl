import mjlab_homierl  # noqa: F401

from mjlab.tasks.registry import list_tasks


def test_registered_task_ids() -> None:
  tasks = set(list_tasks())
  assert "Mjlab-Homie-Unitree-H1" in tasks
  assert "Mjlab-Homie-Unitree-H1-with_hands" in tasks
