# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Fall recovery, stand-up, and sit/stand task configurations."""

from __future__ import annotations

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.configclass import configclass

from isaaclab_microduck.assets import MICRODUCK_ALL_COLLISIONS_CFG
from isaaclab_microduck.tasks import mdp

from .base_env_cfg import MicroDuckVelocityRoughEnvCfg

LEG_JOINTS = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
SIT_Z = 0.060
STAND_Z = 0.115
SITTING_JOINT_OVERRIDES = {
    1: 0.0,
    2: -0.4079,
    3: 1.35,
    4: 0.0,
    10: 0.0,
    11: 0.4079,
    12: -1.35,
    13: 0.0,
}
TRUNK_CFG = SceneEntityCfg("robot", body_names=("trunk_base",))


def _use_full_collision_robot(cfg) -> None:
    cfg.scene.robot = MICRODUCK_ALL_COLLISIONS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.scene.robot.init_state.pos = (0.0, 0.0, STAND_Z)


def _clear_rewards(cfg) -> None:
    for name in list(vars(cfg.rewards)):
        if not name.startswith("_"):
            setattr(cfg.rewards, name, None)


def _make_flat(cfg) -> None:
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None
    cfg.curriculum.terrain_levels = None


def _make_play(cfg) -> None:
    cfg.scene.num_envs = 32
    cfg.events.push_robot = None


def _configure_velstand(cfg) -> None:
    _use_full_collision_robot(cfg)
    cfg.terminations.body_contact = None
    cfg.rewards.upright_progress = RewTerm(func=mdp.upright_progress, weight=5.0, params={"asset_cfg": TRUNK_CFG})
    cfg.rewards.height_progress = RewTerm(
        func=mdp.height_progress, weight=30.0, params={"asset_cfg": TRUNK_CFG, "ceiling": STAND_Z}
    )
    cfg.rewards.joint_torque_rate = RewTerm(func=mdp.joint_torque_rate_l2, weight=-0.002)
    cfg.rewards.feet_air_time = RewTerm(
        func=mdp.feet_air_time_upright,
        weight=3.0,
        params={
            "command_name": "twist",
            "sensor_names": ("left_foot_contact", "right_foot_contact"),
            "threshold": 0.125,
            "gate_tilt_above_deg": 40.0,
        },
    )
    cfg.rewards.fallen_tax = RewTerm(
        func=mdp.fallen_state_penalty,
        weight=0.5,
        params={
            "asset_cfg": TRUNK_CFG,
            "gate_tilt_above_deg": 40.0,
            "release_tilt_below_deg": 25.0,
            "release_z_above": 0.09,
        },
    )
    cfg.rewards.recovery_success = RewTerm(
        func=mdp.recovery_success,
        weight=5.0,
        params={
            "asset_cfg": TRUNK_CFG,
            "fallen_tilt_deg": 40.0,
            "min_fallen_s": 0.5,
            "up_tilt_deg": 25.0,
            "up_z": 0.09,
        },
    )
    cfg.events.random_prone_init = EventTerm(
        func=mdp.maybe_set_random_prone_orientation,
        mode="reset",
        params={
            "prone_prob": 0.45,
            "face_down_prob": 0.5,
            "prone_z_min": 0.05,
            "prone_z_max": 0.09,
            "crouch_prob": 0.15,
        },
    )


@configclass
class MicroDuckVelStandRoughEnvCfg(MicroDuckVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_velstand(self)


@configclass
class MicroDuckVelStandFlatEnvCfg(MicroDuckVelStandRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_flat(self)


@configclass
class MicroDuckVelStandFlatEnvCfg_PLAY(MicroDuckVelStandFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_play(self)


@configclass
class MicroDuckVelStandRoughEnvCfg_PLAY(MicroDuckVelStandRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_play(self)
        self.scene.terrain.terrain_generator.curriculum = False


def _configure_standup(cfg) -> None:
    _use_full_collision_robot(cfg)
    _clear_rewards(cfg)
    cfg.terminations.body_contact = None
    cfg.commands.twist.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.twist.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.twist.ranges.ang_vel_z = (0.0, 0.0)
    cfg.rewards.pose_stand = RewTerm(
        func=mdp.pose_target_match,
        weight=2.0,
        params={"std": 0.5, "joint_indices": LEG_JOINTS, "target_overrides": None},
    )
    cfg.rewards.pose_l1 = RewTerm(
        func=mdp.pose_l1_penalty,
        weight=1.25,
        params={"joint_indices": LEG_JOINTS, "target_overrides": None},
    )
    cfg.rewards.height = RewTerm(
        func=mdp.height_target_gaussian,
        weight=2.0,
        params={"std": 0.04, "target_height": STAND_Z, "asset_cfg": TRUNK_CFG},
    )
    cfg.rewards.height_l1 = RewTerm(
        func=mdp.height_l1_penalty,
        weight=7.5,
        params={"target_height": STAND_Z, "asset_cfg": TRUNK_CFG},
    )
    cfg.rewards.rise = RewTerm(
        func=mdp.com_upward_velocity,
        weight=0.75,
        params={"asset_cfg": TRUNK_CFG, "max_height": 0.125},
    )
    cfg.rewards.upright = RewTerm(func=mdp.body_upright_linear, weight=1.5, params={"asset_cfg": TRUNK_CFG})
    cfg.rewards.upright_sharp = RewTerm(
        func=mdp.upright_gaussian_at_height,
        weight=1.5,
        params={"std": 0.3, "height_low": SIT_Z, "height_high": STAND_Z, "asset_cfg": TRUNK_CFG},
    )
    cfg.rewards.standing = RewTerm(
        func=mdp.standing_composite_score,
        weight=3.75,
        params={
            "target_height": STAND_Z,
            "height_std": 0.04,
            "upright_std": 0.4,
            "pose_std": 0.4,
            "joint_indices": LEG_JOINTS,
            "target_overrides": None,
            "asset_cfg": TRUNK_CFG,
        },
    )
    cfg.rewards.action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    cfg.events.set_ground_state = EventTerm(
        func=mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.2,
            "face_up_prob": 0.0,
            "sitting_prob": 0.4,
            "standing_prob": 0.4,
            "prone_z_min": 0.05,
            "prone_z_max": 0.09,
            "face_up_roll_max": math.radians(90.0),
            "sitting_joint_overrides": SITTING_JOINT_OVERRIDES,
            "sitting_joint_noise_std": 0.12,
            "sitting_tilt_max": math.radians(10.0),
            "sitting_z_min": 0.05,
            "sitting_z_max": 0.09,
            "standing_z_min": 0.11,
            "standing_z_max": 0.12,
        },
    )


@configclass
class MicroDuckStandUpRoughEnvCfg(MicroDuckVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_standup(self)


@configclass
class MicroDuckStandUpFlatEnvCfg(MicroDuckStandUpRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_flat(self)


@configclass
class MicroDuckStandUpFlatEnvCfg_PLAY(MicroDuckStandUpFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_play(self)


@configclass
class MicroDuckStandUpRoughEnvCfg_PLAY(MicroDuckStandUpRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_play(self)
        self.scene.terrain.terrain_generator.curriculum = False


def _configure_sitstand(cfg) -> None:
    _configure_standup(cfg)
    _clear_rewards(cfg)
    cfg.commands.twist = mdp.SitStandCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 8.0),
        rel_standing_envs=0.0,
        rel_heading_envs=0.0,
        heading_command=False,
        ranges=mdp.SitStandCommandCfg.Ranges(lin_vel_x=(0.0, 0.0), lin_vel_y=(0.0, 0.0), ang_vel_z=(0.0, 0.0)),
        sit_probability=0.5,
        ramp_time=2.0,
        sit_height=SIT_Z,
        stand_height=STAND_Z,
    )
    posture_params = {"command_name": "twist", "sit_z": SIT_Z, "stand_z": STAND_Z}
    pose_params = {
        "command_name": "twist",
        "joint_indices": LEG_JOINTS,
        "sit_overrides": SITTING_JOINT_OVERRIDES,
    }
    cfg.rewards.pose = RewTerm(
        func=mdp.posture_pose_match,
        weight=3.0,
        params={**pose_params, "std": 0.4},
    )
    cfg.rewards.pose_l1 = RewTerm(
        func=mdp.posture_pose_l1,
        weight=6.0,
        params=pose_params,
    )
    cfg.rewards.height = RewTerm(func=mdp.posture_height_gaussian, weight=3.0, params={**posture_params, "std": 0.04})
    cfg.rewards.height_l1 = RewTerm(func=mdp.posture_height_l1, weight=6.0, params=posture_params)
    cfg.rewards.rise = RewTerm(
        func=mdp.posture_rise_bootstrap,
        weight=0.75,
        params={"command_name": "twist", "max_height": 0.125, "max_vz": 0.08},
    )
    cfg.rewards.descent_speed = RewTerm(
        func=mdp.trunk_downward_velocity_penalty, weight=10.0, params={"max_down_vel": 0.05, "asset_cfg": TRUNK_CFG}
    )
    cfg.rewards.upright = RewTerm(func=mdp.body_upright_linear, weight=1.0, params={"asset_cfg": TRUNK_CFG})
    cfg.rewards.stillness = RewTerm(func=mdp.posture_stillness, weight=1.0, params=posture_params)
    cfg.rewards.composite = RewTerm(
        func=mdp.posture_composite,
        weight=4.0,
        params={
            **posture_params,
            "height_std": 0.035,
            "upright_std": 0.4,
            "pose_std": 0.4,
            "joint_indices": LEG_JOINTS,
            "sit_overrides": SITTING_JOINT_OVERRIDES,
        },
    )


@configclass
class MicroDuckSitStandRoughEnvCfg(MicroDuckVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_sitstand(self)


@configclass
class MicroDuckSitStandFlatEnvCfg(MicroDuckSitStandRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_flat(self)


@configclass
class MicroDuckSitStandFlatEnvCfg_PLAY(MicroDuckSitStandFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_play(self)


@configclass
class MicroDuckSitStandRoughEnvCfg_PLAY(MicroDuckSitStandRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_play(self)
        self.scene.terrain.terrain_generator.curriculum = False
