# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Asset configurations for MicroDuck and its XL330 test bench."""

from __future__ import annotations

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.sim.schemas import ArticulationRootBaseCfg, RigidBodyBaseCfg

from ..actuators import BamActuatorCfg

_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "robots"
_MICRODUCK_DIR = _DATA_DIR / "microduck"
_TEST_BENCH_DIR = _DATA_DIR / "xl330_test_bench"
_USD_DIR = _DATA_DIR / "usd"

MICRODUCK_ACTION_JOINT_NAMES = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "head_roll",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]
"""Canonical 14-element policy action and actuated-joint order."""

MICRODUCK_HOME_JOINT_POS = {
    "left_hip_yaw": 0.0,
    "right_hip_yaw": 0.0,
    "left_hip_roll": -0.0873,
    "right_hip_roll": 0.0873,
    "left_hip_pitch": -0.4579,
    "right_hip_pitch": 0.4579,
    "left_knee": -0.0049,
    "right_knee": 0.0049,
    "left_ankle": 0.4530,
    "right_ankle": -0.4530,
    "neck_pitch": 0.3491,
    "head_pitch": 0.3491,
    "head_yaw": 0.0,
    "head_roll": 0.0,
}
"""Standing pose shared by MicroDuck task families [rad]."""


def _spawn_from_source(
    source: Path, *, self_collision: bool, fix_base: bool = False, articulation_enabled: bool = True
) -> sim_utils.SpawnerCfg:
    usd_path = _USD_DIR / source.stem / f"{source.stem}.usda"
    common = {
        "activate_contact_sensors": True,
        "rigid_props": RigidBodyBaseCfg(disable_gravity=False),
        "articulation_props": ArticulationRootBaseCfg(
            articulation_enabled=articulation_enabled, fix_root_link=fix_base
        ),
    }
    if usd_path.exists():
        return sim_utils.UsdFileCfg(usd_path=str(usd_path), **common)
    return sim_utils.MjcfFileCfg(
        asset_path=str(source),
        usd_dir=str(usd_path.parent),
        usd_file_name=usd_path.name,
        self_collision=self_collision,
        import_physics_scene=False,
        run_asset_transformer=True,
        run_multi_physics_conversion=True,
        **common,
    )


def _robot_cfg(source_name: str, *, self_collision: bool, backlash: bool = False) -> ArticulationCfg:
    return ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=_spawn_from_source(_MICRODUCK_DIR / source_name, self_collision=self_collision),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
            joint_pos=MICRODUCK_HOME_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
        soft_joint_pos_limit_factor=0.9,
        actuators={
            "bam": BamActuatorCfg(
                joint_names_expr=MICRODUCK_ACTION_JOINT_NAMES,
                encoder_through_backlash=backlash,
            )
        },
    )


MICRODUCK_WALK_CFG = _robot_cfg("robot_walk.xml", self_collision=False)
"""MicroDuck walking configuration with reduced collision geometry."""

MICRODUCK_ALL_COLLISIONS_CFG = _robot_cfg("robot_allcollisions.xml", self_collision=True)
"""MicroDuck configuration with full body collision geometry."""

MICRODUCK_ROLLERS_CFG = _robot_cfg("robot_allcollisions_rollers.xml", self_collision=True)
"""MicroDuck configuration with four passive roller joints."""

MICRODUCK_WALK_BACKLASH_CFG = _robot_cfg("robot_walk_backlash.xml", self_collision=False, backlash=True)
"""Walking configuration with serial gearbox backlash joints."""

MICRODUCK_ALL_COLLISIONS_BACKLASH_CFG = _robot_cfg(
    "robot_allcollisions_backlash.xml", self_collision=True, backlash=True
)
"""Full collision configuration with serial gearbox backlash joints."""

MICRODUCK_ROLLERS_BACKLASH_CFG = _robot_cfg(
    "robot_allcollisions_rollers_backlash.xml", self_collision=True, backlash=True
)
"""Passive roller configuration with serial gearbox backlash joints."""

MICRODUCK_BALL_CFG = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Ball",
    spawn=_spawn_from_source(_MICRODUCK_DIR / "ball.xml", self_collision=False, articulation_enabled=False),
    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.3, 0.0, 0.035)),
)
"""Seventy-millimeter, 15-gram ball used by the kick task."""

XL330_TEST_BENCH_CFG = ArticulationCfg(
    prim_path="{ENV_REGEX_NS}/XL330TestBench",
    spawn=_spawn_from_source(_TEST_BENCH_DIR / "xl330_test_bench.xml", self_collision=False),
    # Isaac Sim sanitizes the numeric MJCF joint name "1" to this USD identifier.
    init_state=ArticulationCfg.InitialStateCfg(joint_pos={"tn__1_": 0.0}, joint_vel={".*": 0.0}),
    actuators={"bam": BamActuatorCfg(joint_names_expr=["tn__1_"])},
)
"""Fixed-base XL330 pendulum used for simulation-to-real validation."""
