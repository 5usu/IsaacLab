# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Runtime implementations for MicroDuck command generators."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp import UniformVelocityCommand
from isaaclab.managers import CommandTerm

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

    from .commands import (
        CyclicPhaseCommandCfg,
        MicroDuckVelocityCommandCfg,
        RelativeHeadingVelocityCommandCfg,
        SitStandCommandCfg,
        UniformVectorCommandCfg,
    )


class MicroDuckVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command with explicit turn-in-place samples."""

    cfg: MicroDuckVelocityCommandCfg

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        super()._resample_command(env_ids)
        if self.cfg.rel_turn_in_place_envs <= 0.0:
            return

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        turn_ids = env_ids[torch.rand(len(env_ids), device=self.device) < self.cfg.rel_turn_in_place_envs]
        if len(turn_ids) == 0:
            return

        max_rate = max(abs(value) for value in self.cfg.ranges.ang_vel_z)
        signs = torch.where(
            torch.rand(len(turn_ids), device=self.device) < 0.5,
            -torch.ones(len(turn_ids), device=self.device),
            torch.ones(len(turn_ids), device=self.device),
        )
        magnitudes = torch.empty(len(turn_ids), device=self.device).uniform_(0.4 * max_rate, max_rate)
        self.vel_command_b[turn_ids, :2] = 0.0
        self.vel_command_b[turn_ids, 2] = signs * magnitudes
        self.is_standing_env[turn_ids] = False


class RelativeHeadingVelocityCommand(MicroDuckVelocityCommand):
    """Encode relative heading error in the yaw-rate command slot."""

    cfg: RelativeHeadingVelocityCommandCfg

    def __init__(self, cfg: RelativeHeadingVelocityCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._target_heading_w = torch.zeros(self.num_envs, device=self.device)
        self._heading_max = max(abs(value) for value in cfg.ranges.ang_vel_z)

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        super()._resample_command(env_ids)
        self._target_heading_w[env_ids] = torch.empty(len(env_ids), device=self.device).uniform_(-math.pi, math.pi)
        self.vel_command_b[env_ids, 2] = 0.0

    def _update_command(self) -> None:
        quat = self.robot.data.root_link_quat_w.torch
        w, x, y, z = quat.unbind(dim=-1)
        yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
        error = torch.atan2(torch.sin(self._target_heading_w - yaw), torch.cos(self._target_heading_w - yaw))
        self.vel_command_b[:, 2] = error.clamp(-self._heading_max, self._heading_max)

    def _update_metrics(self) -> None:
        return


class CyclicPhaseCommand(CommandTerm):
    """Continuous phase command represented as cosine and sine."""

    cfg: CyclicPhaseCommandCfg

    def __init__(self, cfg: CyclicPhaseCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._phase = torch.zeros(self.num_envs, device=self.device)
        self._command = torch.zeros(self.num_envs, 3, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    @property
    def phase(self) -> torch.Tensor:
        return self._phase

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        if self.cfg.randomize_phase:
            self._phase[env_ids] = torch.rand(len(env_ids), device=self.device)
        else:
            self._phase[env_ids] = 0.0
        self.command_counter[env_ids] = 0
        self._write_command()
        return {}

    def compute(self, dt: float) -> None:
        self._phase.add_(dt / self.cfg.period).remainder_(1.0)
        self._write_command()

    def _write_command(self) -> None:
        angle = 2.0 * torch.pi * self._phase
        self._command[:, 0] = torch.cos(angle)
        self._command[:, 1] = torch.sin(angle)
        self._command[:, 2] = 0.0

    def _update_metrics(self) -> None:
        return

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        return

    def _update_command(self) -> None:
        return


class UniformVectorCommand(CommandTerm):
    """Sample and hold a vector with independent uniform ranges."""

    cfg: UniformVectorCommandCfg

    def __init__(self, cfg: UniformVectorCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._command = torch.zeros(self.num_envs, len(cfg.ranges), device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self) -> None:
        return

    def _update_command(self) -> None:
        return

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        if len(env_ids) == 0:
            return
        for index, value_range in enumerate(self.cfg.ranges):
            self._command[env_ids, index].uniform_(*value_range)
        if self.cfg.zero_command_prob > 0.0:
            zero_ids = torch.as_tensor(env_ids, device=self.device)[
                torch.rand(len(env_ids), device=self.device) < self.cfg.zero_command_prob
            ]
            self._command[zero_ids] = 0.0


class SitStandCommand(UniformVelocityCommand):
    """Binary posture command with a slew-limited reward target."""

    cfg: SitStandCommandCfg

    def __init__(self, cfg: SitStandCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._env = env
        self._alpha = torch.zeros(self.num_envs, device=self.device)

    @property
    def alpha(self) -> torch.Tensor:
        """Target blend where zero is standing and one is sitting."""
        return self._alpha

    def _resample_command(self, env_ids: Sequence[int]) -> None:
        self.vel_command_b[env_ids] = 0.0
        self.vel_command_b[env_ids, 0] = (
            torch.rand(len(env_ids), device=self.device) < self.cfg.sit_probability
        ).float()

    def _update_metrics(self) -> None:
        return

    def _update_command(self) -> None:
        return

    def compute(self, dt: float) -> None:
        super().compute(dt)
        fresh = self._env.episode_length_buf <= 1
        if fresh.any():
            height = self.robot.data.root_link_pos_w.torch[:, 2] - self._env.scene.env_origins[:, 2]
            from_height = ((self.cfg.stand_height - height) / (self.cfg.stand_height - self.cfg.sit_height)).clamp(
                0.0, 1.0
            )
            self._alpha = torch.where(fresh, torch.nan_to_num(from_height), self._alpha)
        step = dt / max(self.cfg.ramp_time, 1.0e-6)
        self._alpha.add_((self.vel_command_b[:, 0] - self._alpha).clamp(-step, step))


# Runtime aliases retained for checkpoint/config compatibility.
VelocityCommandCommandOnly = MicroDuckVelocityCommand
GroundPickPhaseCommand = CyclicPhaseCommand
UniformPoseCommand = UniformVectorCommand
