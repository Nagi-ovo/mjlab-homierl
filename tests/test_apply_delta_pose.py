"""Tests for apply_delta_pose numerical stability."""

import torch

from mjlab.utils.lab_api.math import apply_delta_pose


def test_apply_delta_pose_zero_rotation_is_finite_and_identity():
  source_pos = torch.zeros(2, 3)
  source_rot = torch.tensor([[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]])
  delta_pose = torch.zeros(2, 6)

  target_pos, target_rot = apply_delta_pose(source_pos, source_rot, delta_pose)

  assert torch.isfinite(target_pos).all()
  assert torch.isfinite(target_rot).all()
  assert torch.allclose(target_pos, source_pos)
  assert torch.allclose(target_rot, source_rot)
