"""WandB utilities."""

from __future__ import annotations

from typing import Sequence


def add_wandb_tags(tags: Sequence[str]) -> None:
  """Add tags to the current wandb run."""
  if not tags:
    return

  try:
    import wandb

    if wandb.run is not None:
      existing_tags = list(wandb.run.tags) if wandb.run.tags else []
      new_tags = list(set(existing_tags + list(tags)))
      wandb.run.tags = new_tags
  except ImportError:
    pass
