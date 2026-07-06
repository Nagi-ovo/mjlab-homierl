"""Foot contact-compliance domain randomization (soft / foam floors).

Real-robot observation (2026-07-06): standing on EVA foam gym mats produces a
mild fore-aft rocking that a rigid-floor-trained policy never saw; a
compliant floor in sim reproduces it. Following arXiv:2504.13619 (HRP-5P on
mattresses/foam/grass), the compliance is randomized on the FOOT geoms: the
feet already carry contact ``priority=1`` over the terrain, so their solref
governs the foot-ground contact — no terrain changes needed.

``geom_solref`` uses MuJoCo's positive form: axis 0 = time constant [s]
(0.02 = rigid default, larger = softer), axis 1 = damping ratio. The positive
form is numerically safe across the whole sampled box; the negative
stiffness/damping form produced NaNs when independent axis draws landed on
high-stiffness/low-damping (a ~224 rad/s underdamped contact at a 5 ms
step). mjlab's DR engine expands the field per-world (verified shape
(num_envs, ngeom, 2)), so each environment trains on its own floor softness.
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
  """Randomize foot geom solref (positive timeconst/dampratio form).

  Pass ``ranges`` as ``{0: (T_min, T_max), 1: (zeta_min, zeta_max)}``; e.g.
  ``{0: (0.02, 0.1), 1: (0.7, 1.5)}`` spans MuJoCo's rigid default to a soft
  gym-mat foam feel. Keep T_min >= 2x the physics step.
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
