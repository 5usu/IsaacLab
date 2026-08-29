# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.actuators import ActuatorBase
from isaaclab.utils import DelayBuffer
from isaaclab.utils.types import ArticulationActions

if TYPE_CHECKING:
    from .bam_cfg import BamActuatorCfg


class BamActuator(ActuatorBase):
    """Vectorized XL330 actuator using the fitted BAM M6 dynamics.

    The model applies firmware position control, current limiting, motor
    back-electromotive force, randomized command delay and supply voltage,
    load-dependent voltage sag, and BAM's M6 friction law. Passive articulation
    joints receive zero effort. For backlash models, firmware position feedback
    can include the matching serial gear-play joint.
    """

    cfg: BamActuatorCfg

    _ENCODER_COUNTS_PER_REV = 4096.0
    _FIRMWARE_KP_DIVISOR = 256.0
    _PWM_LIMIT = 885.0
    _MAX_CURRENT = 1.75

    def __init__(self, cfg: BamActuatorCfg, *args, **kwargs):
        if cfg.minimum_delay < 0 or cfg.maximum_delay < cfg.minimum_delay:
            raise ValueError("BAM delay bounds must satisfy 0 <= minimum_delay <= maximum_delay.")
        if cfg.supply_voltage_range[0] <= 0.0 or cfg.supply_voltage_range[1] < cfg.supply_voltage_range[0]:
            raise ValueError("BAM supply voltage bounds must be positive and ordered.")
        super().__init__(cfg, *args, **kwargs)

        active_ids = [index for index, name in enumerate(self.joint_names) if not name.startswith("passive_")]
        if not active_ids:
            raise ValueError("BamActuator requires at least one non-passive joint.")
        self._active_ids = torch.tensor(active_ids, dtype=torch.long, device=self._device)

        name_to_index = {name: index for index, name in enumerate(self.joint_names)}
        backlash_ids: list[int] = []
        backlash_mask: list[float] = []
        for index in active_ids:
            backlash_id = name_to_index.get(f"passive_{self.joint_names[index]}_backlash")
            backlash_ids.append(0 if backlash_id is None else backlash_id)
            backlash_mask.append(0.0 if backlash_id is None else 1.0)
        self._backlash_ids = torch.tensor(backlash_ids, dtype=torch.long, device=self._device)
        self._backlash_mask = torch.tensor(backlash_mask, dtype=torch.float32, device=self._device)

        self._position_delay = DelayBuffer(cfg.maximum_delay, self._num_envs, device=self._device)
        self._supply_voltage = torch.empty(self._num_envs, 1, device=self._device).uniform_(*cfg.supply_voltage_range)
        self._voltage_drop_gain = torch.empty(self._num_envs, 1, device=self._device).uniform_(
            *cfg.voltage_drop_gain_range
        )
        self.kp_scale = torch.ones(self._num_envs, 1, device=self._device)
        self.kd_scale = torch.ones(self._num_envs, 1, device=self._device)
        self.friction_scale = torch.full(
            (self._num_envs, 1), cfg.friction_scale, dtype=torch.float32, device=self._device
        )
        self._default_kp_scale = self.kp_scale.clone()
        self._default_kd_scale = self.kd_scale.clone()
        self._default_friction_scale = self.friction_scale.clone()
        self._previous_motor_effort = torch.zeros(self._num_envs, len(active_ids), device=self._device)
        self._sample_delay(torch.arange(self._num_envs, device=self._device))

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        """Reset command history and resample delay for selected environments."""
        ids = self._resolve_env_ids(env_ids)
        self._sample_delay(ids)
        self._position_delay.reset(ids)
        self._previous_motor_effort[ids] = 0.0
        self.kp_scale[ids] = self._default_kp_scale[ids]
        self.kd_scale[ids] = self._default_kd_scale[ids]
        self.friction_scale[ids] = self._default_friction_scale[ids]

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        """Convert desired joint positions to BAM actuator efforts [N m]."""
        if control_action.joint_positions is None:
            raise ValueError("BamActuator requires joint position targets.")

        position_target = self._position_delay.compute(control_action.joint_positions[:, self._active_ids])
        position = joint_pos[:, self._active_ids]
        if self.cfg.encoder_through_backlash:
            position = position + joint_pos[:, self._backlash_ids] * self._backlash_mask
        velocity = joint_vel[:, self._active_ids]

        supply_voltage = self._supply_voltage - self._voltage_drop_gain * self._previous_motor_effort.abs().sum(
            dim=-1, keepdim=True
        )
        supply_voltage.clamp_(min=self.cfg.minimum_supply_voltage)

        scaled_velocity = velocity * self.kd_scale
        error_gain = self._ENCODER_COUNTS_PER_REV / (2.0 * math.pi * self._FIRMWARE_KP_DIVISOR * self._PWM_LIMIT)
        duty_cycle = (position_target - position) * self.cfg.firmware_kp * self.kp_scale * error_gain

        back_emf = self.cfg.torque_constant * scaled_velocity
        duty_span = self.cfg.motor_resistance * self._MAX_CURRENT / supply_voltage
        duty_center = back_emf / supply_voltage
        duty_cycle = torch.maximum(torch.minimum(duty_cycle, duty_center + duty_span), duty_center - duty_span)
        duty_cycle.clamp_(min=-1.0, max=1.0)

        motor_effort = self.cfg.torque_constant * supply_voltage * duty_cycle / self.cfg.motor_resistance
        motor_effort -= self.cfg.torque_constant**2 * scaled_velocity / self.cfg.motor_resistance
        self.computed_effort.zero_()
        self.computed_effort[:, self._active_ids] = motor_effort

        external_effort = -self._previous_motor_effort
        friction_budget = self._friction_budget(self._previous_motor_effort, external_effort, velocity)
        moving = velocity.abs() >= self.cfg.static_velocity_threshold
        moving_effort = motor_effort - torch.sign(velocity) * friction_budget
        static_effort = torch.sign(motor_effort) * torch.relu(motor_effort.abs() - friction_budget)
        active_effort = torch.where(moving, moving_effort, static_effort)
        active_effort -= self.cfg.friction_viscous * velocity
        active_effort = torch.clip(
            active_effort,
            min=-self.effort_limit[:, self._active_ids],
            max=self.effort_limit[:, self._active_ids],
        )

        self.applied_effort.zero_()
        self.applied_effort[:, self._active_ids] = active_effort
        self._previous_motor_effort.copy_(motor_effort.detach())
        control_action.joint_positions = None
        control_action.joint_velocities = None
        control_action.joint_efforts = self.applied_effort
        return control_action

    def set_gains(
        self,
        env_ids: Sequence[int] | slice,
        kp_scale: torch.Tensor | None = None,
        kd_scale: torch.Tensor | None = None,
    ) -> None:
        """Set firmware and back-electromotive-force gain scales."""
        if kp_scale is not None:
            self.kp_scale[env_ids] = kp_scale
        if kd_scale is not None:
            self.kd_scale[env_ids] = kd_scale

    def set_friction_scale(self, env_ids: Sequence[int] | slice, friction_scale: torch.Tensor) -> None:
        """Set velocity-independent friction multipliers for selected environments."""
        self.friction_scale[env_ids] = friction_scale

    def _friction_budget(
        self, motor_effort: torch.Tensor, external_effort: torch.Tensor, velocity: torch.Tensor
    ) -> torch.Tensor:
        stribeck = torch.exp(-torch.pow(velocity.abs() / self.cfg.stribeck_velocity, self.cfg.stribeck_exponent))
        budget = torch.full_like(motor_effort, self.cfg.friction_base)
        budget += stribeck * self.cfg.friction_stribeck
        budget += torch.abs(
            external_effort * self.cfg.load_friction_external - motor_effort * self.cfg.load_friction_motor
        )
        budget += stribeck * torch.abs(
            external_effort * self.cfg.load_friction_external_stribeck
            - motor_effort * self.cfg.load_friction_motor_stribeck
        )
        drive = (motor_effort.abs() > external_effort.abs()).to(motor_effort.dtype)
        quadratic = drive * self.cfg.load_friction_external_quad * external_effort.square()
        quadratic += (1.0 - drive) * self.cfg.load_friction_motor_quad * motor_effort.square()
        budget += stribeck * quadratic
        return budget * self.friction_scale

    def _sample_delay(self, env_ids: torch.Tensor) -> None:
        delays = torch.randint(
            self.cfg.minimum_delay,
            self.cfg.maximum_delay + 1,
            (len(env_ids),),
            dtype=torch.int,
            device=self._device,
        )
        self._position_delay.set_time_lag(delays, env_ids)

    def _resolve_env_ids(self, env_ids: Sequence[int] | slice | None) -> torch.Tensor:
        if env_ids is None or isinstance(env_ids, slice):
            return torch.arange(self._num_envs, device=self._device)[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self._device)
