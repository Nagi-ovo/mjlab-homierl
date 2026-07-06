"""Standalone HOMIE/HOMIE+ ONNX runtime: the consumer side of the metadata
contract.

This module is deliberately dependency-light — stdlib + numpy + onnxruntime
only, and no imports from the rest of this package. It is THE canonical
implementation of the exported policy's conventions (observation assembly,
history handling, action-to-target mapping, command semantics), shared by:

- the real-robot deploy script (``scripts/deploy_g1_homie.py``),
- the BiGym controller plugin (vendor this single file into that repo),
- any other consumer that should not depend on mjlab/torch.

Deployment story: copy this one file. Do not reimplement the observation
builder elsewhere — it has been validated bit-for-bit against the mjlab
training plant (see deploy_g1_homie.py --sim), and that guarantee only holds
for this implementation.
"""

from __future__ import annotations

import json
from collections import deque

import numpy as np

# Canonical Unitree G1 29-dof low-level motor order (hg LowCmd/LowState).
G1_MOTOR_ORDER: tuple[str, ...] = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)


def gravity_orientation(quat_wxyz: np.ndarray) -> np.ndarray:
  """Project the world -z gravity direction into the base frame."""
  w, x, y, z = quat_wxyz
  return np.array(
    [
      -2.0 * (x * z - w * y),
      -2.0 * (y * z + w * x),
      -(1.0 - 2.0 * (x * x + y * y)),
    ],
    dtype=np.float32,
  )


class HomieOnnxPolicy:
  """ONNX session + the metadata-driven observation builder.

  Works for both 4-dim-command HOMIE and 5-dim-command HOMIE+ exports; every
  convention is read from the ONNX ``metadata_props``.
  """

  def __init__(self, onnx_path: str):
    import onnxruntime as ort

    self.session = ort.InferenceSession(
      onnx_path, providers=["CPUExecutionProvider"]
    )
    self.meta = dict(self.session.get_modelmeta().custom_metadata_map)
    self.input_name = self.session.get_inputs()[0].name

    m = self.meta
    self.joint_names = m["joint_names"].split(",")
    self.action_joint_names = m["action_joint_names"].split(",")
    self.default_pos = np.array(
      [float(v) for v in m["default_joint_pos"].split(",")], dtype=np.float32
    )
    self.kps = np.array(
      [float(v) for v in m["joint_stiffness"].split(",")], dtype=np.float32
    )
    self.kds = np.array(
      [float(v) for v in m["joint_damping"].split(",")], dtype=np.float32
    )
    self.action_scale = float(m["action_scale"])
    self.history_length = int(m["obs_history_length"])
    self.num_one_step_obs = int(m["num_one_step_obs"])
    layout = json.loads(m["one_step_obs_layout"])
    self.num_commands = int(layout["command"])
    self.has_pitch = self.num_commands >= 5
    self.scale_lin_vel = float(m["obs_scale_lin_vel"])
    self.scale_ang_vel = float(m["obs_scale_ang_vel"])
    self.scale_dof_pos = float(m["obs_scale_dof_pos"])
    self.scale_dof_vel = float(m["obs_scale_dof_vel"])
    twist = json.loads(m["twist_command_ranges"])
    self.vx_range = tuple(twist["lin_vel_x"])
    self.vy_range = tuple(twist["lin_vel_y"])
    self.wz_range = tuple(twist["ang_vel_z"])
    self.height_range = tuple(
      float(v) for v in m["height_command_range"].split(",")
    )
    self.standing_height = float(m["standing_height"])
    self.pitch_range = (0.0, 0.0)
    if self.has_pitch:
      pr = json.loads(m["pitch_command_ranges"])
      lows, highs = zip(pr["walk"], pr["squat"])
      self.pitch_range = (min(lows), max(highs))

    expected = self.num_commands + 6 + 2 * len(self.joint_names) + len(
      self.action_joint_names
    )
    if expected != self.num_one_step_obs:
      raise ValueError(
        f"Metadata inconsistent: layout sums to {expected}, "
        f"num_one_step_obs is {self.num_one_step_obs}."
      )

    self.action_ids = [self.joint_names.index(n) for n in self.action_joint_names]
    self.last_action = np.zeros(len(self.action_joint_names), dtype=np.float32)
    self.history: deque[np.ndarray] = deque(maxlen=self.history_length)

  def reset(self) -> None:
    self.last_action[:] = 0.0
    self.history.clear()

  def one_step_obs(
    self,
    command: np.ndarray,
    ang_vel: np.ndarray,
    quat_wxyz: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
  ) -> np.ndarray:
    """Assemble one observation step. Joint arrays follow ``joint_names``."""
    cmd = command.astype(np.float32).copy()
    cmd[0:2] *= self.scale_lin_vel
    cmd[2] *= self.scale_ang_vel  # height (cmd[3]) and pitch (cmd[4]) unscaled
    return np.concatenate(
      [
        cmd,
        ang_vel.astype(np.float32) * self.scale_ang_vel,
        gravity_orientation(quat_wxyz),
        (joint_pos - self.default_pos) * self.scale_dof_pos,
        joint_vel.astype(np.float32) * self.scale_dof_vel,
        self.last_action,
      ]
    )

  def act(self, one_step: np.ndarray) -> np.ndarray:
    """Push one observation step; return 29-dim joint position targets.

    Non-action joints stay at the default pose; the caller decides what to do
    with them (hold via PD on the real robot, ignore in a plugin that drives
    the upper body separately).
    """
    if not self.history:
      self.history.extend([one_step] * self.history_length)
    else:
      self.history.append(one_step)
    obs = np.concatenate(list(self.history))[None].astype(np.float32)
    action = self.session.run(None, {self.input_name: obs})[0][0]
    self.last_action = action.astype(np.float32)
    targets = self.default_pos.copy()
    targets[self.action_ids] += self.action_scale * action
    return targets


class CommandState:
  """Joystick-driven (vx, vy, wz, height[, pitch]) command with slewed
  height/pitch channels, scaled to the policy's training ranges.

  ``speed_scale`` caps full stick deflection at that fraction of the training
  twist range (1.0 = full range; teleop wants ~0.4-0.6). ``deadzone`` zeroes
  small stick values so drift does not creep the robot.
  """

  def __init__(
    self,
    policy: HomieOnnxPolicy,
    speed_scale: float = 1.0,
    deadzone: float = 0.05,
  ):
    self.p = policy
    self.speed_scale = float(speed_scale)
    self.deadzone = float(deadzone)
    self.height = policy.standing_height
    self.pitch = 0.0

  def _stick(self, v: float) -> float:
    return 0.0 if abs(v) < self.deadzone else float(v)

  def vector(self, lx, ly, rx, height_step=0.0, pitch_step=0.0) -> np.ndarray:
    p = self.p
    lx, ly, rx = self._stick(lx), self._stick(ly), self._stick(rx)
    s = self.speed_scale
    vx = s * ly * (p.vx_range[1] if ly >= 0 else -p.vx_range[0])
    vy = s * -lx * p.vy_range[1]
    wz = s * -rx * p.wz_range[1]
    self.height = float(
      np.clip(self.height + height_step, p.height_range[0], p.height_range[1])
    )
    cmd = [vx, vy, wz, self.height]
    if p.has_pitch:
      self.pitch = float(
        np.clip(self.pitch + pitch_step, p.pitch_range[0], p.pitch_range[1])
      )
      cmd.append(self.pitch)
    return np.array(cmd, dtype=np.float32)
