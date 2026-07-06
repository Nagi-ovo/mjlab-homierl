"""Minimal DDS command/remote helpers for real-robot deployment.

Adapted from unitree_rl_gym deploy/deploy_real/common (BSD-3): wireless-remote
byte parsing and the zero-torque / damping / init LowCmd constructors for the
unitree_hg message family (G1).
"""

from __future__ import annotations

import struct


class KeyMap:
  R1 = 0
  L1 = 1
  start = 2
  select = 3
  R2 = 4
  L2 = 5
  F1 = 6
  F2 = 7
  A = 8
  B = 9
  X = 10
  Y = 11
  up = 12
  right = 13
  down = 14
  left = 15


class RemoteController:
  """Parses the 40-byte wireless_remote buffer from LowState."""

  def __init__(self):
    self.lx = 0.0
    self.ly = 0.0
    self.rx = 0.0
    self.ry = 0.0
    self.button = [0] * 16

  def set(self, data) -> None:
    keys = struct.unpack("H", bytes(data[2:4]))[0]
    for i in range(16):
      self.button[i] = (keys & (1 << i)) >> i
    self.lx = struct.unpack("f", bytes(data[4:8]))[0]
    self.rx = struct.unpack("f", bytes(data[8:12]))[0]
    self.ry = struct.unpack("f", bytes(data[12:16]))[0]
    self.ly = struct.unpack("f", bytes(data[20:24]))[0]


class MotorMode:
  PR = 0  # Series control for pitch/roll ankle joints.
  AB = 1


def init_cmd_hg(cmd, mode_machine: int) -> None:
  cmd.mode_machine = mode_machine
  cmd.mode_pr = MotorMode.PR
  size = len(cmd.motor_cmd)
  for i in range(size):
    cmd.motor_cmd[i].mode = 1
    cmd.motor_cmd[i].q = 0.0
    cmd.motor_cmd[i].qd = 0.0
    cmd.motor_cmd[i].kp = 0.0
    cmd.motor_cmd[i].kd = 0.0
    cmd.motor_cmd[i].tau = 0.0


def create_zero_cmd(cmd) -> None:
  for i in range(len(cmd.motor_cmd)):
    cmd.motor_cmd[i].q = 0.0
    cmd.motor_cmd[i].qd = 0.0
    cmd.motor_cmd[i].kp = 0.0
    cmd.motor_cmd[i].kd = 0.0
    cmd.motor_cmd[i].tau = 0.0


def create_damping_cmd(cmd) -> None:
  for i in range(len(cmd.motor_cmd)):
    cmd.motor_cmd[i].q = 0.0
    cmd.motor_cmd[i].qd = 0.0
    cmd.motor_cmd[i].kp = 0.0
    cmd.motor_cmd[i].kd = 8.0
    cmd.motor_cmd[i].tau = 0.0
