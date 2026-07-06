"""Foot contact-compliance domain randomization (soft / foam floors).

Real-robot observation (2026-07-06): standing on EVA foam gym mats produces a
mild fore-aft rocking that a rigid-floor-trained policy never saw; a
compliant floor in sim reproduces it. Following arXiv:2504.13619 (HRP-5P on
mattresses/foam/grass), the compliance is randomized on the FOOT geoms: the
feet already carry contact ``priority=1`` over the terrain, so their solref
governs the foot-ground contact — no terrain changes needed.

``geom_solref`` uses MuJoCo's negative direct form: axis 0 = -stiffness
[N/m], axis 1 = -damping [N*s/m]. mjlab's DR engine expands the field
per-world (verified shape (num_envs, ngeom, 2)), so each environment trains
on its own floor softness.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from mjlab.envs.mdp.dr._core import (
  _DEFAULT_ASSET_CFG,
  Ranges,
  _randomize_model_field,
)
from mjlab.managers.event_manager import requires_model_fields
from mjlab.managers.scene_entity_config import SceneEntityCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@requires_model_fields("geom_solref")
def foot_compliance(
  env: "ManagerBasedRlEnv",
  env_ids: torch.Tensor | None,
  ranges: Ranges,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
  distribution: str = "uniform",
  shared_random: bool = False,
) -> None:
  """Randomize foot geom solref (negative stiffness/damping form).

  Pass ``ranges`` as ``{0: (-K_max, -K_min), 1: (-D_max, -D_min)}``; e.g.
  ``{0: (-1e5, -8e3), 1: (-1000, -100)}`` spans near-rigid (~2 mm sink under
  a 35 kg robot) down to soft gym-mat foam (~2 cm sink).
  """
  _randomize_model_field(
    env,
    env_ids,
    "geom_solref",
    entity_type="geom",
    ranges=ranges,
    distribution=distribution,
    operation="abs",
    asset_cfg=asset_cfg,
    shared_random=shared_random,
    default_axes=[0, 1],
    valid_axes=[0, 1],
  )
