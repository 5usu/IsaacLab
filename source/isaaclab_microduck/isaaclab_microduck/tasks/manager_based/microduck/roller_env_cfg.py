# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Passive-wheel locomotion and roller skill task configurations."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_microduck.assets import MICRODUCK_ROLLERS_CFG
from isaaclab_microduck.tasks import mdp

from .base_env_cfg import MicroDuckVelocityFlatEnvCfg
from .recovery_env_cfg import LEG_JOINTS, _clear_rewards, _make_play
from .terrain import MicroDuckDownhillTerrainCfg

WHEEL_CFG = SceneEntityCfg("robot", joint_names=(r"^passive_.*wheel$",))
TRUNK_CFG = SceneEntityCfg("robot", body_names=("trunk_base",))
ROLLER_FEET_CFG = SceneEntityCfg("robot", body_names=("ankle_l_v1", "ankle_r_v1"), preserve_order=True)

_ROLLER_LEFT_FOOT_PATH = "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/yaw2roll/hip_l/upper_leg_left/leg/ankle_l_v1"
_ROLLER_RIGHT_FOOT_PATH = (
    "{ENV_REGEX_NS}/Robot/Geometry/trunk_base/bearing_roll/hip_l_2/upper_leg_right/leg_2/ankle_r_v1"
)

STAND_POSE = {
    "left_hip_yaw": -0.0476,
    "left_hip_roll": -0.0629,
    "left_hip_pitch": -0.2869,
    "left_knee": 0.9618,
    "left_ankle": 1.1674,
    "neck_pitch": 0.6029,
    "head_pitch": 0.543,
    "head_yaw": -0.069,
    "head_roll": -0.0414,
    "right_hip_yaw": -0.0337,
    "right_hip_roll": -0.0061,
    "right_hip_pitch": 0.1534,
    "right_knee": -0.9725,
    "right_ankle": -1.0646,
}
CROUCH_POSE = {
    "left_hip_yaw": -0.0184,
    "left_hip_roll": 0.0307,
    "left_hip_pitch": 1.4082,
    "left_knee": 1.5248,
    "left_ankle": -0.0675,
    "neck_pitch": 1.0937,
    "head_pitch": 1.2149,
    "head_yaw": -0.0184,
    "head_roll": -0.0368,
    "right_hip_yaw": 0.0184,
    "right_hip_roll": -0.0169,
    "right_hip_pitch": -1.4757,
    "right_knee": -1.5907,
    "right_ankle": 0.0568,
}


def _use_roller_robot(cfg) -> None:
    cfg.scene.robot = MICRODUCK_ROLLERS_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    cfg.scene.robot.init_state.pos = (0.0, 0.0, 0.12)
    cfg.scene.left_foot_contact.prim_path = _ROLLER_LEFT_FOOT_PATH
    cfg.scene.right_foot_contact.prim_path = _ROLLER_RIGHT_FOOT_PATH
    cfg.observations.critic.wheel_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": WHEEL_CFG})


def _configure_roller_velocity(cfg) -> None:
    _use_roller_robot(cfg)
    cfg.rewards.com_height = RewTerm(
        func=mdp.com_height_target,
        weight=2.0,
        params={"target_height_min": 0.0935, "target_height_max": 0.1235},
    )
    cfg.rewards.feet_flat = RewTerm(
        func=mdp.feet_flat_penalty,
        weight=-2.0,
        params={"asset_cfg": ROLLER_FEET_CFG, "sensor_name": "feet_ground_contact"},
    )
    cfg.rewards.wheel_speed = RewTerm(
        func=mdp.wheel_speed_reward,
        weight=10.0,
        params={"command_name": "twist", "vel_scale": 0.3},
    )
    cfg.rewards.braking = RewTerm(func=mdp.braking_reward, weight=1.0, params={"command_name": "twist", "vel_std": 0.3})
    cfg.rewards.skating_air_time = RewTerm(
        func=mdp.skating_air_time_reward,
        weight=1.5,
        params={
            "sensor_name": "feet_ground_contact",
            "command_name": "twist",
            "threshold_min": 0.15,
            "threshold_max": 0.45,
            "vel_gate_ref": 0.2,
        },
    )
    cfg.rewards.glide = RewTerm(
        func=mdp.glide_reward,
        weight=4.0,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist", "vel_ref": 0.2},
    )
    cfg.rewards.single_support = RewTerm(
        func=mdp.single_support_reward,
        weight=3.0,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist", "vel_gate_ref": 0.2},
    )
    cfg.rewards.gait_symmetry = RewTerm(
        func=mdp.gait_symmetry_penalty,
        weight=-1.0,
        params={"sensor_name": "feet_ground_contact"},
    )
    cfg.rewards.forward_lean = RewTerm(
        func=mdp.forward_lean_reward,
        weight=1.5,
        params={"command_name": "twist", "target_pitch": 0.262, "std": 0.1},
    )
    cfg.rewards.heading = RewTerm(func=mdp.heading_hold_reward, weight=1.0, params={"std": 0.4})
    cfg.rewards.neck_rate = RewTerm(func=mdp.neck_action_rate_l2, weight=-0.5)


@configclass
class MicroDuckVelocityRollersFlatEnvCfg(MicroDuckVelocityFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_roller_velocity(self)


@configclass
class MicroDuckVelocityRollersFlatEnvCfg_PLAY(MicroDuckVelocityRollersFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_play(self)


@configclass
class MicroDuckSwizzleFlatEnvCfg(MicroDuckVelocityRollersFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.rewards.skating_air_time = None
        self.rewards.glide = None
        self.rewards.single_support = None
        self.rewards.gait_symmetry = None
        self.rewards.leg_symmetry = RewTerm(func=mdp.leg_symmetry_reward, weight=2.0)
        self.rewards.grounded = RewTerm(
            func=mdp.grounded_reward,
            weight=1.0,
            params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
        )


@configclass
class MicroDuckSwizzleFlatEnvCfg_PLAY(MicroDuckSwizzleFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_play(self)


def _configure_crouch(cfg) -> None:
    _configure_roller_velocity(cfg)
    _clear_rewards(cfg)
    cfg.commands.twist = mdp.CyclicPhaseCommandCfg(period=5.0, randomize_phase=True)
    pose_params = {
        "command_name": "twist",
        "crouch_pose": CROUCH_POSE,
        "stand_pose": STAND_POSE,
        "descent_end": 0.10,
        "hold_end": 0.50,
        "rise_end": 0.60,
    }
    cfg.rewards.pose = RewTerm(func=mdp.crouch_glide_pose_by_phase, weight=6.0, params={**pose_params, "std": 0.4})
    cfg.rewards.pose_l1 = RewTerm(func=mdp.crouch_glide_pose_l1, weight=2.0, params=pose_params)
    cfg.rewards.forward_speed = RewTerm(func=mdp.forward_speed_reward, weight=1.0, params={"vel_ref": 0.2})
    cfg.rewards.forward_lean = RewTerm(
        func=mdp.crouch_forward_lean,
        weight=1.0,
        params={
            "command_name": "twist",
            "target_pitch": 0.08,
            "std": 0.1,
            "descent_end": 0.1,
            "hold_end": 0.5,
            "rise_end": 0.6,
        },
    )
    cfg.rewards.feet_flat = RewTerm(
        func=mdp.feet_flat_penalty,
        weight=-2.0,
        params={"asset_cfg": ROLLER_FEET_CFG, "sensor_name": "feet_ground_contact"},
    )
    cfg.rewards.neck_rate = RewTerm(func=mdp.neck_action_rate_l2, weight=-0.5)
    cfg.rewards.torque = RewTerm(func=mdp.joint_torques_l2, weight=-0.001)
    cfg.events.reset_rolling_entry = EventTerm(
        func=mdp.reset_rolling_entry, mode="reset", params={"speed_range": (0.2, 0.5)}
    )


@configclass
class MicroDuckRollerCrouchFlatEnvCfg(MicroDuckVelocityFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_crouch(self)


@configclass
class MicroDuckRollerCrouchFlatEnvCfg_PLAY(MicroDuckRollerCrouchFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_play(self)


def _configure_slope(cfg) -> None:
    _use_roller_robot(cfg)
    _clear_rewards(cfg)
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=42,
        curriculum=True,
        size=(15.0, 4.0),
        border_width=0.0,
        num_rows=10,
        num_cols=1,
        difficulty_range=(0.0, 1.0),
        sub_terrains={"downhill": MicroDuckDownhillTerrainCfg(proportion=1.0)},
    )
    cfg.commands.twist.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.twist.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.twist.ranges.ang_vel_z = (0.0, 0.0)
    cfg.rewards.upright = RewTerm(
        func=mdp.body_upright_gaussian, weight=3.0, params={"asset_cfg": TRUNK_CFG, "std": 0.2}
    )
    cfg.rewards.alive = RewTerm(func=mdp.is_alive, weight=1.0)
    cfg.rewards.wheel_glide = RewTerm(func=mdp.wheel_glide_reward, weight=2.0, params={"cap_speed": 0.35})
    cfg.rewards.heading = RewTerm(func=mdp.heading_hold_reward, weight=1.5, params={"std": 0.4})
    cfg.rewards.feet_flat = RewTerm(
        func=mdp.feet_flat_penalty,
        weight=-2.0,
        params={"asset_cfg": ROLLER_FEET_CFG, "sensor_name": "feet_ground_contact"},
    )
    cfg.rewards.neck_rate = RewTerm(func=mdp.neck_action_rate_l2, weight=-0.5)
    cfg.rewards.neck_pos = RewTerm(func=mdp.neck_joint_pos_l2, weight=-0.75)
    cfg.rewards.torque = RewTerm(func=mdp.joint_torques_l2, weight=-0.001)
    cfg.terminations.body_contact = None
    cfg.terminations.fell_over = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.0})
    cfg.terminations.fell_into_void = DoneTerm(func=mdp.root_height_below, params={"min_height": -3.41})
    cfg.events.push_robot = None
    cfg.events.reset_rolling_entry = EventTerm(
        func=mdp.reset_rolling_entry, mode="reset", params={"speed_range": (0.25, 0.45)}
    )


@configclass
class MicroDuckRollerSlopeEnvCfg(MicroDuckVelocityFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_slope(self)


@configclass
class MicroDuckRollerSlopeEnvCfg_PLAY(MicroDuckRollerSlopeEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 10
        self.scene.terrain.terrain_generator.curriculum = False


def _configure_roller_standup(cfg) -> None:
    _use_roller_robot(cfg)
    _clear_rewards(cfg)
    cfg.episode_length_s = 6.0
    cfg.terminations.body_contact = None
    cfg.rewards.pose = RewTerm(
        func=mdp.pose_target_match,
        weight=8.0,
        params={"std": 0.5, "joint_indices": LEG_JOINTS, "target_overrides": None},
    )
    cfg.rewards.pose_l1 = RewTerm(
        func=mdp.pose_l1_penalty,
        weight=5.0,
        params={"joint_indices": LEG_JOINTS, "target_overrides": None},
    )
    cfg.rewards.height = RewTerm(
        func=mdp.height_target_gaussian,
        weight=8.0,
        params={"std": 0.04, "target_height": 0.138, "asset_cfg": TRUNK_CFG},
    )
    cfg.rewards.height_l1 = RewTerm(
        func=mdp.height_l1_penalty,
        weight=30.0,
        params={"target_height": 0.138, "asset_cfg": TRUNK_CFG},
    )
    cfg.rewards.rise = RewTerm(
        func=mdp.com_upward_velocity,
        weight=3.0,
        params={"asset_cfg": TRUNK_CFG, "max_height": 0.148},
    )
    cfg.rewards.upright = RewTerm(func=mdp.body_upright_linear, weight=6.0, params={"asset_cfg": TRUNK_CFG})
    cfg.rewards.standing = RewTerm(
        func=mdp.standing_composite_score,
        weight=15.0,
        params={
            "target_height": 0.138,
            "height_std": 0.04,
            "upright_std": 0.4,
            "pose_std": 0.4,
            "joint_indices": LEG_JOINTS,
            "target_overrides": None,
            "asset_cfg": TRUNK_CFG,
        },
    )
    cfg.events.set_ground_state = EventTerm(
        func=mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.4,
            "face_up_prob": 0.4,
            "sitting_prob": 0.0,
            "standing_prob": 0.2,
            "prone_z_min": 0.07,
            "prone_z_max": 0.10,
            "standing_z_min": 0.13,
            "standing_z_max": 0.145,
        },
    )


@configclass
class MicroDuckRollerStandUpFlatEnvCfg(MicroDuckVelocityFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_roller_standup(self)


@configclass
class MicroDuckRollerStandUpFlatEnvCfg_PLAY(MicroDuckRollerStandUpFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_play(self)


def _configure_spin(cfg) -> None:
    _use_roller_robot(cfg)
    _clear_rewards(cfg)
    cfg.commands.twist = mdp.CyclicPhaseCommandCfg(period=mdp.SPIN_PERIOD, randomize_phase=True)
    cfg.rewards.rate = RewTerm(func=mdp.spin_rate_track, weight=6.0, params={"command_name": "twist", "std": 1.5})
    cfg.rewards.rate_l1 = RewTerm(func=mdp.spin_rate_l1, weight=0.5, params={"command_name": "twist"})
    cfg.rewards.position = RewTerm(
        func=mdp.spin_stay_in_place,
        weight=-3.0,
        params={"command_name": "twist", "launch_scale": mdp.SPIN_LAUNCH_DRIFT_SCALE, "accel_end": mdp.SPIN_ACCEL_END},
    )
    cfg.rewards.wheels = RewTerm(
        func=mdp.spin_wheel_differential,
        weight=1.0,
        params={"command_name": "twist", "omega_scale": mdp.SPIN_WHEEL_OMEGA_SCALE},
    )
    cfg.rewards.legs = RewTerm(
        func=mdp.leg_antisymmetry,
        weight=1.0,
        params={"command_name": "twist", "joint_bases": ("hip_pitch", "knee")},
    )
    cfg.rewards.grounded = RewTerm(
        func=mdp.spin_grounded,
        weight=0.5,
        params={"sensor_name": "feet_ground_contact", "command_name": "twist"},
    )
    cfg.rewards.feet_flat = RewTerm(
        func=mdp.feet_flat_penalty,
        weight=-2.0,
        params={"asset_cfg": ROLLER_FEET_CFG, "sensor_name": "feet_ground_contact"},
    )
    cfg.rewards.neck_rate = RewTerm(func=mdp.neck_action_rate_l2, weight=-0.5)
    cfg.rewards.torque = RewTerm(func=mdp.joint_torques_l2, weight=-0.001)


@configclass
class MicroDuckSpinFlatEnvCfg(MicroDuckVelocityFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_spin(self)


@configclass
class MicroDuckSpinFlatEnvCfg_PLAY(MicroDuckSpinFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_play(self)
