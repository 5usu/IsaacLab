# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.actuators import ActuatorBaseCfg
from isaaclab.utils.configclass import configclass

if TYPE_CHECKING:
    from .bam import BamActuator


@configclass
class BamActuatorCfg(ActuatorBaseCfg):
    """Configuration for the MicroDuck XL330 BAM M6 actuator model."""

    class_type: type[BamActuator] | str = "{DIR}.bam:BamActuator"

    joint_names_expr: list[str] = MISSING
    stiffness: float | dict[str, float] = 0.0
    damping: float | dict[str, float] = 0.0
    armature: float | dict[str, float] = 0.0018077432831600838
    effort_limit: float | dict[str, float] = 1.0675
    velocity_limit: float | dict[str, float] = 20.0

    firmware_kp: float = 200.0
    """Firmware proportional gain."""

    supply_voltage_range: tuple[float, float] = (6.5, 8.2)
    """Startup supply-voltage sampling range [V]."""

    voltage_drop_gain_range: tuple[float, float] = (0.0, 0.2)
    """Startup load-dependent voltage-drop gain range [V/(N m)]."""

    minimum_supply_voltage: float = 6.0
    """Minimum effective supply voltage after load-dependent drop [V]."""

    minimum_delay: int = 3
    """Minimum position-command delay in physics steps."""

    maximum_delay: int = 6
    """Maximum position-command delay in physics steps."""

    encoder_through_backlash: bool = False
    """Add each matching passive backlash-joint angle to encoder feedback."""

    static_velocity_threshold: float = 1.0e-3
    """Absolute velocity below which the static-friction branch is used [rad/s]."""

    friction_scale: float = 1.0
    """Nominal multiplier for the velocity-independent BAM friction budget."""

    torque_constant: float = 0.36601349688984386
    """Fitted XL330 torque constant [N m/A]."""

    motor_resistance: float = 2.8113923539223227
    """Fitted XL330 motor resistance [ohm]."""

    friction_base: float = 0.004771183165566
    """Coulomb friction magnitude [N m]."""

    friction_stribeck: float = 0.004676345799486616
    """Stribeck friction magnitude [N m]."""

    load_friction_motor: float = 0.2667860954283698
    load_friction_external: float = 8.515871897059342e-6
    load_friction_motor_stribeck: float = 1.0722918395099123e-5
    load_friction_external_stribeck: float = 0.08077928978935671
    load_friction_motor_quad: float = 0.009972471242139415
    load_friction_external_quad: float = 0.004902565732332559
    stribeck_velocity: float = 2.890372094130307
    """Characteristic Stribeck velocity [rad/s]."""

    stribeck_exponent: float = 8.683259907618984
    friction_viscous: float = 0.005359668274599504
    """Viscous friction coefficient [N m s/rad]."""
