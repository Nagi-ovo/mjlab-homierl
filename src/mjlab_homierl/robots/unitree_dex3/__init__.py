"""Unitree Dex3 hand attachments for the G1.

The MJCF attachments under ``xmls/`` are extracted from the AMO
``g1_29dof_with_dex3.xml`` model (Unitree Dex3 three-finger hands): fingers are
welded at the open pose with their inertials intact, geoms are visual-only
(``contype=conaffinity=0``). Like the H1 Robotiq mounts, the hands are
inertial/disturbance attachments (~0.53 kg per hand: 0.20 kg palm net of the
bare wrist + 0.32 kg fingers), not contact tools.
"""

from pathlib import Path

import mujoco

_XML_DIR = Path(__file__).resolve().parent / "xmls"
DEX3_LEFT_XML = _XML_DIR / "left_dex3.xml"
DEX3_RIGHT_XML = _XML_DIR / "right_dex3.xml"
assert DEX3_LEFT_XML.exists() and DEX3_RIGHT_XML.exists()

# Body name of the attached hand root, per side. mjlab entity name lookups
# strip attach prefixes, so this matches the bare body name.
DEX3_MOUNT_BODY_PATTERN = r"^(left|right)_dex3_mount$"


def attach_dex3_hands(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Attach Dex3 hands to a G1 spec at both wrist_yaw links."""
  for side, xml in (("left", DEX3_LEFT_XML), ("right", DEX3_RIGHT_XML)):
    hand_spec = mujoco.MjSpec.from_file(str(xml))
    wrist = spec.body(f"{side}_wrist_yaw_link")
    frame = wrist.add_frame()
    spec.attach(hand_spec, frame=frame, prefix=f"{side}_dex3/")
  return spec
