"""Recorder terms for HOMIE training."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from mjlab.managers.recorder_manager import RecorderTerm, RecorderTermCfg

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

TERMINAL_CRITIC_OBS_KEY = "terminal_critic_obs"


class TerminalCriticObsRecorder(RecorderTerm):
  """Capture the terminal critic observation before terminated envs reset.

  OpenHomie's runner substitutes the pre-reset privileged observation into
  ``next_critic_obs`` for terminated envs (him_on_policy_runner.py:144), so
  the HIM estimator's targets on done transitions are the terminal state.
  mjlab's ``auto_reset=True`` step never computes that observation, so this
  recorder rebuilds it in ``record_pre_reset`` by calling the critic
  observation function directly (single step, history-free) and stashes it
  in ``env.extras`` for :meth:`HIMPPO.process_env_step` to consume.
  """

  def __init__(self, cfg: RecorderTermCfg, env: "ManagerBasedRlEnv"):
    super().__init__(cfg, env)
    term_cfg = env.observation_manager.get_term_cfg("critic", "him_privileged_obs")
    if int(term_cfg.history_length) > 1:
      raise ValueError(
        "TerminalCriticObsRecorder assumes a single-step critic observation; "
        f"got history_length={term_cfg.history_length}."
      )
    self._func = term_cfg.func
    self._params = dict(term_cfg.params)
    self._clip = term_cfg.clip

  def record_pre_reset(self, env_ids: torch.Tensor) -> None:
    # Sim state is terminal here (derived quantities carry the same
    # one-substep staleness the termination/reward managers accepted).
    obs = self._func(self._env, **self._params)
    if self._clip is not None:
      obs = obs.clip(min=self._clip[0], max=self._clip[1])
    self._env.extras[TERMINAL_CRITIC_OBS_KEY] = (
      env_ids.detach().clone(),
      obs[env_ids].detach().clone(),
    )
