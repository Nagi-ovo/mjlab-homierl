"""Tests for h1_constants.py."""

import re

import mujoco
import numpy as np
import pytest

from mjlab.asset_zoo.robots.unitree_h1 import h1_constants
from mjlab.entity import Entity
from mjlab.utils.string import resolve_expr


@pytest.fixture(scope="module")
def h1_entity() -> Entity:
  return Entity(h1_constants.get_h1_robot_cfg())


@pytest.fixture(scope="module")
def h1_model(h1_entity: Entity) -> mujoco.MjModel:
  return h1_entity.spec.compile()


# fmt: off
@pytest.mark.parametrize(
  "actuator_config,stiffness,damping",
  [
    (h1_constants.H1_ACTUATOR_HIP_KNEE, h1_constants.STIFFNESS_HIP_KNEE, h1_constants.DAMPING_HIP_KNEE),
    (h1_constants.H1_ACTUATOR_ANKLE_TORSO, h1_constants.STIFFNESS_ANKLE_TORSO, h1_constants.DAMPING_ANKLE_TORSO),
    (h1_constants.H1_ACTUATOR_ARM, h1_constants.STIFFNESS_ARM, h1_constants.DAMPING_ARM),
  ],
)
# fmt: on
def test_actuator_parameters(h1_model, actuator_config, stiffness, damping):
  for i in range(h1_model.nu):
    actuator = h1_model.actuator(i)
    actuator_name = actuator.name
    matches = any(
      re.match(pattern, actuator_name) for pattern in actuator_config.joint_names_expr
    )
    if matches:
      assert actuator.gainprm[0] == stiffness
      assert actuator.biasprm[1] == -stiffness
      assert actuator.biasprm[2] == -damping
      assert actuator.forcerange[0] == -actuator_config.effort_limit
      assert actuator.forcerange[1] == actuator_config.effort_limit


def test_keyframe_base_position(h1_model) -> None:
  data = mujoco.MjData(h1_model)
  mujoco.mj_resetDataKeyframe(h1_model, data, 0)
  mujoco.mj_forward(h1_model, data)
  np.testing.assert_array_equal(data.qpos[:3], h1_constants.KNEES_BENT_KEYFRAME.pos)
  np.testing.assert_array_equal(data.qpos[3:7], h1_constants.KNEES_BENT_KEYFRAME.rot)


def test_keyframe_joint_positions(h1_entity, h1_model) -> None:
  """Test that keyframe joint positions match the configuration."""
  key = h1_model.key("init_state")
  expected_joint_pos = h1_constants.KNEES_BENT_KEYFRAME.joint_pos
  expected_values = resolve_expr(expected_joint_pos, h1_entity.joint_names, 0.0)
  for joint_name, expected_value in zip(
    h1_entity.joint_names, expected_values, strict=True
  ):
    joint = h1_model.joint(joint_name)
    qpos_idx = joint.qposadr[0]
    actual_value = key.qpos[qpos_idx]
    np.testing.assert_allclose(
      actual_value,
      expected_value,
      rtol=1e-5,
      err_msg=f"Joint {joint_name} position mismatch: "
      f"expected {expected_value}, got {actual_value}",
    )


def test_foot_collision_geoms(h1_model) -> None:
  foot_pattern = r"^(left|right)_foot[1-3]$"
  for i in range(h1_model.ngeom):
    geom = h1_model.geom(i)
    geom_name = geom.name
    # Foot collision geoms should have condim=3, priority=1, and friction=0.6.
    if re.match(foot_pattern, geom_name):
      assert geom.condim == 3
      assert geom.priority == 1
      assert geom.friction[0] == 0.6


def test_non_foot_collision_geoms(h1_model) -> None:
  """Test that body collision geoms exist and have proper configuration.
  
  H1 uses named geoms for body collision detection.
  These geoms inherit their condim from XML defaults.
  """
  named_collision_geoms = ["head", "helmet", "torso", "hip", "left_shoulder", "right_shoulder"]
  
  found_geoms = []
  for i in range(h1_model.ngeom):
    geom = h1_model.geom(i)
    if geom.name in named_collision_geoms:
      found_geoms.append(geom.name)
      # These geoms should have collision detection enabled (condim > 0)
      assert geom.condim[0] > 0, f"Geom {geom.name} should have condim > 0"
  
  # Verify all expected collision geoms were found
  assert set(found_geoms) == set(named_collision_geoms)


def test_collision_geom_count(h1_model) -> None:
  """Test that H1 has the correct number of foot geoms.
  
  H1 uses 3 capsules per foot (6 total) for ground contact.
  """
  all_geoms = [h1_model.geom(i).name for i in range(h1_model.ngeom) if h1_model.geom(i).name]
  
  foot_geoms = [
    name
    for name in all_geoms
    if re.match(r"^(left|right)_foot[1-3]$", name)
  ]
  assert len(foot_geoms) == 6
  
  # Check that body collision geoms exist
  body_collision_geoms = ["head", "helmet", "torso", "hip", "left_shoulder", "right_shoulder"]
  for geom_name in body_collision_geoms:
    assert geom_name in all_geoms, f"Expected collision geom '{geom_name}' not found"


def test_h1_entity_creation(h1_entity) -> None:
  assert h1_entity.num_actuators == 19
  assert h1_entity.num_joints == 19
  assert h1_entity.is_actuated
  assert not h1_entity.is_fixed_base


def test_h1_actuators_configured_correctly(h1_model):
  """Verify that all H1 actuators have correct control and force limiting.

  All 19 H1 actuators should have ctrllimited=False (allowing setpoints beyond
  joint limits) and forcelimited=True (limiting forces to effort limits).
  """
  for i in range(h1_model.nu):
    actuator_name = h1_model.actuator(i).name
    assert h1_model.actuator_ctrllimited[i] == 0, (
      f"Actuator '{actuator_name}' has ctrllimited=True, expected False"
    )
    assert h1_model.actuator_forcelimited[i] == 1, (
      f"Actuator '{actuator_name}' has forcelimited=False, expected True"
    )

