"""Inspire RH56 hand attachments for the G1.

The MJCF attachments under ``xmls/`` are extracted from the official Unitree
``g1_29dof_rev_1_0_with_inspire_hand_DFQ.urdf`` (unitree_ros, BSD-3-Clause):
each hand subtree is welded at the open pose and fused into a single body
with a properly composed inertial, geoms are visual-only
(``contype=conaffinity=0``), and the root body carries Unitree's official
wrist mount transform. The vendor URDF inertials sum to only ~0.19 kg per
hand, so masses/inertias are rescaled to the Inspire RH56DFX spec weight of
0.54 kg per hand (en.inspire-robots.com; DFQ meshes, same RH56 hand family).
Like the Dex3 mounts, these are inertial/disturbance attachments, not
contact tools.
"""

from pathlib import Path

import mujoco

_XML_DIR = Path(__file__).resolve().parent / "xmls"
INSPIRE_LEFT_XML = _XML_DIR / "left_inspire.xml"
INSPIRE_RIGHT_XML = _XML_DIR / "right_inspire.xml"
assert INSPIRE_LEFT_XML.exists() and INSPIRE_RIGHT_XML.exists()

# Body name of the attached hand root, per side. mjlab entity name lookups
# strip attach prefixes, so this matches the bare body name.
INSPIRE_MOUNT_BODY_PATTERN = r"^(left|right)_inspire_mount$"


def attach_inspire_hands(spec: mujoco.MjSpec) -> mujoco.MjSpec:
  """Attach Inspire RH56 hands to a G1 spec at both wrist_yaw links."""
  for side, xml in (("left", INSPIRE_LEFT_XML), ("right", INSPIRE_RIGHT_XML)):
    hand_spec = mujoco.MjSpec.from_file(str(xml))
    wrist = spec.body(f"{side}_wrist_yaw_link")
    frame = wrist.add_frame()
    spec.attach(hand_spec, frame=frame, prefix=f"{side}_inspire/")
  return spec
