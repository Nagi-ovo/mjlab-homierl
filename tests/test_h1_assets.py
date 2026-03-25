import mujoco

from mjlab_homierl.env_cfgs import _default_hands_cfg
from mjlab_homierl.robots.unitree_h1 import DEFAULT_2F85_XML, get_h1_robot_cfg


def test_h1_asset_paths_exist() -> None:
  cfg = get_h1_robot_cfg()
  spec = cfg.spec_fn()
  assert spec is not None
  assert DEFAULT_2F85_XML.exists()


def test_homie_with_hands_disables_hand_collisions() -> None:
  cfg = get_h1_robot_cfg(hands=_default_hands_cfg(True))
  spec = cfg.spec_fn()

  assert spec.geom("left_hand_collision").contype == 0
  assert spec.geom("left_hand_collision").conaffinity == 0
  assert spec.geom("right_hand_collision").contype == 0
  assert spec.geom("right_hand_collision").conaffinity == 0

  model = spec.compile()
  for body_id in range(model.nbody):
    body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
    if not body or not body.startswith(("left_gripper/", "right_gripper/")):
      continue
    geom_adr = model.body_geomadr[body_id]
    geom_num = model.body_geomnum[body_id]
    for geom_id in range(geom_adr, geom_adr + geom_num):
      assert model.geom_contype[geom_id] == 0
      assert model.geom_conaffinity[geom_id] == 0
