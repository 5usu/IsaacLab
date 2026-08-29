# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ground pick, ball kick, acrobatics, and parkour task configurations."""

from __future__ import annotations

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.terrains import TerrainGeneratorCfg
from isaaclab.utils.configclass import configclass

from isaaclab_microduck.assets import MICRODUCK_BALL_CFG
from isaaclab_microduck.tasks import mdp

from .base_env_cfg import MicroDuckVelocityFlatEnvCfg, MicroDuckVelocityRoughEnvCfg
from .recovery_env_cfg import LEG_JOINTS, STAND_Z, _clear_rewards, _make_play, _use_full_collision_robot
from .terrain import (
    COURSE_LENGTH,
    FINISH_X,
    START_X,
    TRAINING_CHECKPOINTS,
    MicroDuckParkourTerrainCfg,
)

NECK_JOINTS = [5, 6, 7, 8]
FEET_CFG = SceneEntityCfg("robot", body_names=("ankle_left", "ankle_right"), preserve_order=True)
MOUTH_CFG = SceneEntityCfg("robot", body_names=("jaw_soft",))
LANDING_GATE_LO = math.radians(260.0)
LANDING_GATE_HI = math.radians(330.0)
TUCK_OVERRIDES = {
    2: -1.15,
    3: 1.25,
    4: 1.05,
    5: -1.0,
    6: 1.0,
    11: 1.15,
    12: -1.25,
    13: -1.05,
}


def _zero_velocity_command(cfg) -> None:
    cfg.commands.twist.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.twist.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.twist.ranges.ang_vel_z = (0.0, 0.0)


def _configure_ground_pick(cfg) -> None:
    _use_full_collision_robot(cfg)
    _clear_rewards(cfg)
    cfg.terminations.body_contact = None
    cfg.commands.twist = mdp.CyclicPhaseCommandCfg(period=4.0, randomize_phase=True)
    cfg.rewards.mouth_height = RewTerm(
        func=mdp.mouth_ground_proximity_phased,
        weight=3.0,
        params={
            "asset_cfg": MOUTH_CFG,
            "std": 0.10,
            "target_height": 0.0,
            "command_name": "twist",
            "descent_end": 0.375,
            "hold_end": 0.425,
            "rise_end": 0.80,
        },
    )
    cfg.rewards.mouth_orientation = RewTerm(
        func=mdp.mouth_perpendicular_phased,
        weight=2.0,
        params={
            "asset_cfg": MOUTH_CFG,
            "command_name": "twist",
            "descent_end": 0.375,
            "hold_end": 0.425,
            "rise_end": 0.80,
        },
    )
    cfg.rewards.return_legs = RewTerm(
        func=mdp.ground_pick_return_pose_phased,
        weight=6.0,
        params={"std": 0.3, "command_name": "twist", "joint_indices": LEG_JOINTS, "hold_end": 0.425, "rise_end": 0.80},
    )
    cfg.rewards.return_neck = RewTerm(
        func=mdp.ground_pick_return_pose_phased,
        weight=6.0,
        params={
            "std": 0.15,
            "command_name": "twist",
            "joint_indices": NECK_JOINTS,
            "hold_end": 0.425,
            "rise_end": 0.80,
        },
    )
    cfg.rewards.return_upright = RewTerm(
        func=mdp.ground_pick_return_upright_phased,
        weight=4.0,
        params={"std": 0.4, "command_name": "twist", "hold_end": 0.425, "rise_end": 0.80},
    )
    cfg.rewards.feet_grounded = RewTerm(
        func=mdp.feet_grounded_reward, weight=3.0, params={"sensor_name": "feet_ground_contact"}
    )
    cfg.rewards.feet_flat = RewTerm(func=mdp.feet_flat_penalty, weight=-2.0, params={"asset_cfg": FEET_CFG})
    cfg.rewards.head_impact = RewTerm(
        func=mdp.body_impact_cost,
        weight=-2.0,
        params={"sensor_name": "head_ground_contact", "threshold": 1.0},
    )
    cfg.rewards.action_rate = RewTerm(func=mdp.action_rate_l2, weight=-2.0)
    cfg.rewards.neck_action_rate = RewTerm(func=mdp.neck_action_rate_l2, weight=-1.0)
    cfg.rewards.joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-0.005)
    cfg.rewards.payload = RewTerm(
        func=mdp.apply_mouth_payload_force,
        weight=0.0,
        params={"asset_cfg": MOUTH_CFG, "command_name": "twist", "hold_end": 0.425},
    )
    cfg.events.sample_mouth_payload = EventTerm(
        func=mdp.sample_mouth_payload, mode="reset", params={"min_kg": 0.01, "max_kg": 0.04}
    )


@configclass
class MicroDuckGroundPickFlatEnvCfg(MicroDuckVelocityFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_ground_pick(self)


@configclass
class MicroDuckGroundPickRoughEnvCfg(MicroDuckVelocityRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_ground_pick(self)


@configclass
class MicroDuckGroundPickFlatEnvCfg_PLAY(MicroDuckGroundPickFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_play(self)


@configclass
class MicroDuckGroundPickRoughEnvCfg_PLAY(MicroDuckGroundPickRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_play(self)
        self.scene.terrain.terrain_generator.curriculum = False


def _configure_ball_kick(cfg) -> None:
    _use_full_collision_robot(cfg)
    _clear_rewards(cfg)
    cfg.episode_length_s = 5.0
    cfg.terminations.body_contact = None
    _zero_velocity_command(cfg)
    cfg.scene.ball = MICRODUCK_BALL_CFG.replace(prim_path="{ENV_REGEX_NS}/Ball")
    cfg.rewards.ball_velocity = RewTerm(
        func=mdp.ball_forward_velocity, weight=12.0, params={"asset_name": "ball", "max_speed": 1.0}
    )
    cfg.rewards.ball_overshoot = RewTerm(
        func=mdp.ball_speed_overshoot_penalty,
        weight=-4.0,
        params={"asset_name": "ball", "target_speed": 1.0},
    )
    cfg.rewards.support_foot = RewTerm(
        func=mdp.single_foot_grounded_reward,
        weight=2.0,
        params={"sensor_name": "left_foot_contact"},
    )
    cfg.rewards.pose_legs = RewTerm(
        func=mdp.pose_target_match,
        weight=2.0,
        params={"std": 0.5, "joint_indices": LEG_JOINTS, "target_overrides": None},
    )
    cfg.rewards.pose_neck = RewTerm(
        func=mdp.pose_target_match,
        weight=1.0,
        params={"std": 0.3, "joint_indices": NECK_JOINTS, "target_overrides": None},
    )
    cfg.observations.critic.ball_position = ObsTerm(func=mdp.ball_pos_in_base, params={"asset_name": "ball"})
    cfg.observations.critic.ball_velocity = ObsTerm(func=mdp.ball_vel_in_base, params={"asset_name": "ball"})
    cfg.events.set_ground_state = EventTerm(
        func=mdp.set_random_ground_state,
        mode="reset",
        params={
            "face_down_prob": 0.0,
            "face_up_prob": 0.0,
            "sitting_prob": 0.0,
            "standing_prob": 1.0,
            "sitting_tilt_max": math.radians(5.0),
            "standing_z_min": 0.11,
            "standing_z_max": 0.12,
        },
    )
    cfg.events.reset_ball = EventTerm(
        func=mdp.reset_ball_in_front_of_foot,
        mode="reset",
        params={"offset": (0.09, -0.042), "noise_xy": 0.015, "ball_radius": 0.035, "asset_name": "ball"},
    )


@configclass
class MicroDuckBallKickFlatEnvCfg(MicroDuckVelocityFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_ball_kick(self)


@configclass
class MicroDuckBallKickFlatEnvCfg_PLAY(MicroDuckBallKickFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_play(self)


def _configure_roulade(cfg) -> None:
    _use_full_collision_robot(cfg)
    _clear_rewards(cfg)
    cfg.episode_length_s = 5.0
    cfg.terminations.body_contact = None
    _zero_velocity_command(cfg)
    cfg.rewards.progress = RewTerm(
        func=mdp.roulade_progress,
        weight=8.0,
        params={"target_angle": 2.0 * math.pi, "max_paid_rate": 5.0},
    )
    cfg.rewards.overspeed = RewTerm(func=mdp.roulade_overspeed_penalty, weight=-0.1, params={"omega_max": 7.0})
    cfg.rewards.head_pivot = RewTerm(
        func=mdp.roulade_head_pivot,
        weight=0.5,
        params={
            "sensor_name": "head_ground_contact",
            "angle_lo": math.radians(30.0),
            "angle_hi": math.radians(240.0),
            "rate_norm": 2.0,
        },
    )
    cfg.rewards.landing = RewTerm(
        func=mdp.roulade_landing_composite,
        weight=4.0,
        params={
            "target_height": STAND_Z,
            "height_std": 0.04,
            "upright_std": 0.4,
            "pose_std": 0.4,
            "joint_indices": LEG_JOINTS,
            "gate_lo": LANDING_GATE_LO,
            "gate_hi": LANDING_GATE_HI,
            "target_overrides": None,
        },
    )
    cfg.rewards.upright = RewTerm(
        func=mdp.roulade_upright_after_roll,
        weight=1.5,
        params={"gate_lo": LANDING_GATE_LO, "gate_hi": LANDING_GATE_HI},
    )
    cfg.rewards.height = RewTerm(
        func=mdp.roulade_height_after_roll,
        weight=1.0,
        params={"target_height": STAND_Z, "std": 0.04, "gate_lo": LANDING_GATE_LO, "gate_hi": LANDING_GATE_HI},
    )
    cfg.rewards.stand_tax = RewTerm(
        func=mdp.roulade_stand_tax,
        weight=5.0,
        params={"target_height": STAND_Z, "gate_lo": LANDING_GATE_LO, "gate_hi": LANDING_GATE_HI},
    )
    cfg.rewards.rise = RewTerm(
        func=mdp.roulade_rise_velocity,
        weight=0.75,
        params={"max_height": 0.125, "gate_lo": math.radians(180.0), "gate_hi": LANDING_GATE_LO},
    )
    cfg.rewards.sagittal = RewTerm(func=mdp.roulade_sagittal_penalty, weight=-0.1)
    cfg.rewards.lateral_velocity = RewTerm(func=mdp.roulade_lateral_velocity_penalty, weight=-0.5)
    cfg.rewards.flatness = RewTerm(func=mdp.roulade_flatness_penalty, weight=-0.5)
    cfg.rewards.action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    cfg.events.set_roulade_state = EventTerm(
        func=mdp.reset_roulade_state,
        mode="reset",
        params={
            "standing_prob": 0.5,
            "midroll_prob": 0.5,
            "standing_z_min": 0.11,
            "standing_z_max": 0.12,
            "midroll_pitch_min": math.radians(50.0),
            "midroll_pitch_max": math.radians(340.0),
            "midroll_z_min": 0.05,
            "midroll_z_max": 0.10,
            "midroll_omega_range": (0.0, 4.0),
            "tuck_overrides": TUCK_OVERRIDES,
            "tuck_factor_range": (0.3, 1.0),
            "joint_noise_std": 0.08,
        },
    )


@configclass
class MicroDuckRouladeFlatEnvCfg(MicroDuckVelocityFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_roulade(self)


@configclass
class MicroDuckRouladeFlatEnvCfg_PLAY(MicroDuckRouladeFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_play(self)


def _configure_backflip(cfg) -> None:
    _configure_roulade(cfg)
    for name in list(vars(cfg.rewards)):
        if name in {"progress", "overspeed", "head_pivot", "landing", "upright", "height", "stand_tax", "rise"}:
            setattr(cfg.rewards, name, None)
    cfg.episode_length_s = 4.0
    cfg.events.set_roulade_state = None
    gate_lo, gate_hi = math.radians(300.0), math.radians(345.0)
    cfg.rewards.takeoff = RewTerm(
        func=mdp.backflip_takeoff_progress, weight=1.5, params={"target_vz": 1.0, "target_omega": 8.0}
    )
    cfg.rewards.air_progress = RewTerm(
        func=mdp.backflip_air_progress,
        weight=10.0,
        params={"target_angle": 2.0 * math.pi, "max_paid_rate": 12.0},
    )
    cfg.rewards.flight_height = RewTerm(
        func=mdp.backflip_flight_height_progress,
        weight=4.0,
        params={"standing_height": STAND_Z, "target_height": 0.20},
    )
    cfg.rewards.landing = RewTerm(
        func=mdp.backflip_landing_composite,
        weight=8.0,
        params={
            "target_height": STAND_Z,
            "height_std": 0.035,
            "upright_std": 0.35,
            "pose_std": 0.4,
            "joint_indices": LEG_JOINTS,
            "gate_lo": gate_lo,
            "gate_hi": gate_hi,
        },
    )
    cfg.rewards.upright = RewTerm(
        func=mdp.backflip_upright_after_flip, weight=3.0, params={"gate_lo": gate_lo, "gate_hi": gate_hi}
    )
    cfg.rewards.height = RewTerm(
        func=mdp.backflip_height_after_flip,
        weight=2.0,
        params={"target_height": STAND_Z, "std": 0.03, "gate_lo": gate_lo, "gate_hi": gate_hi},
    )
    cfg.rewards.stand_tax = RewTerm(
        func=mdp.backflip_stand_tax,
        weight=5.0,
        params={"target_height": STAND_Z, "gate_lo": gate_lo, "gate_hi": gate_hi},
    )
    cfg.rewards.body_contact = RewTerm(func=mdp.backflip_body_contact_cost, weight=-0.5)
    cfg.rewards.wrong_direction = RewTerm(func=mdp.backflip_wrong_direction_cost, weight=-0.1)
    cfg.rewards.sagittal = RewTerm(func=mdp.roulade_sagittal_penalty, weight=-0.05)
    cfg.rewards.lateral_velocity = RewTerm(func=mdp.roulade_lateral_velocity_penalty, weight=-0.25)
    cfg.rewards.flatness = RewTerm(func=mdp.roulade_flatness_penalty, weight=-0.25)
    cfg.events.set_backflip_state = EventTerm(
        func=mdp.reset_backflip_state,
        mode="reset",
        params={"standing_z_min": 0.11, "standing_z_max": STAND_Z},
    )


@configclass
class MicroDuckBackflipFlatEnvCfg(MicroDuckVelocityFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_backflip(self)


@configclass
class MicroDuckBackflipFlatEnvCfg_PLAY(MicroDuckBackflipFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _make_play(self)


def _configure_parkour(cfg) -> None:
    cfg.episode_length_s = 35.0
    cfg.scene.terrain.terrain_type = "generator"
    cfg.scene.terrain.terrain_generator = TerrainGeneratorCfg(
        seed=42,
        curriculum=False,
        size=(COURSE_LENGTH + 0.2, 2.0),
        border_width=0.0,
        num_rows=1,
        num_cols=1,
        color_scheme="none",
        difficulty_range=(1.0, 1.0),
        sub_terrains={"parkour": MicroDuckParkourTerrainCfg(proportion=1.0)},
    )
    cfg.curriculum.terrain_levels = None
    cfg.commands.twist.rel_standing_envs = 0.0
    cfg.commands.twist.rel_turn_in_place_envs = 0.0
    cfg.commands.twist.ranges.lin_vel_x = (0.30, 0.34)
    cfg.commands.twist.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.twist.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.head_pose.ranges = ((0.0, 0.0),) * 4
    cfg.commands.body_pose.ranges = ((0.0, 0.0),) * 6
    cfg.events.push_robot = None
    cfg.events.reset_parkour_state = EventTerm(
        func=mdp.reset_parkour_state,
        mode="reset",
        params={"checkpoints": TRAINING_CHECKPOINTS, "start_x": START_X, "root_height": 0.12},
    )
    cfg.rewards.track_lin_vel.weight = 3.0
    cfg.rewards.track_ang_vel.weight = 1.0
    cfg.rewards.upright.weight = -3.0
    finish_offset = FINISH_X - START_X
    cfg.rewards.parkour_progress = RewTerm(
        func=mdp.parkour_forward_progress, weight=8.0, params={"finish_offset": finish_offset}
    )
    cfg.rewards.parkour_centerline = RewTerm(func=mdp.parkour_centerline_cost, weight=-2.0, params={"free_width": 0.04})
    cfg.rewards.parkour_finish = RewTerm(
        func=mdp.parkour_finish_bonus, weight=20.0, params={"finish_offset": finish_offset}
    )
    cfg.terminations.parkour_finished = DoneTerm(
        func=mdp.parkour_finished, time_out=True, params={"finish_offset": finish_offset}
    )
    cfg.terminations.parkour_left_course = DoneTerm(func=mdp.parkour_left_course)


@configclass
class MicroDuckParkourEnvCfg(MicroDuckVelocityFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_parkour(self)


@configclass
class MicroDuckParkourEnvCfg_PLAY(MicroDuckParkourEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.events.reset_parkour_state.params["checkpoints"] = ((START_X, 0.0, 0.0),)
        _make_play(self)
