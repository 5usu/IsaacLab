# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Shared manager-based environment configuration for MicroDuck."""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
import isaaclab.terrains as terrain_gen
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_microduck.assets import MICRODUCK_ACTION_JOINT_NAMES, MICRODUCK_WALK_CFG
from isaaclab_microduck.tasks import mdp

MICRODUCK_ROUGH_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=8.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.001,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": terrain_gen.MeshPlaneTerrainCfg(proportion=0.25),
        "stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.25,
            step_height_range=(0.0, 0.015),
            step_width=0.15,
            platform_width=2.0,
            border_width=1.0,
            holes=False,
        ),
        "grid": terrain_gen.MeshRandomGridTerrainCfg(
            proportion=0.30,
            grid_width=0.45,
            grid_height_range=(0.0, 0.010),
            platform_width=1.5,
        ),
        "slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.20,
            slope_range=(0.03, 0.10),
            platform_width=2.0,
            border_width=0.25,
        ),
    },
)
"""Terrain distribution scaled for MicroDuck's short legs."""


@configclass
class MicroDuckSceneCfg(InteractiveSceneCfg):
    """Robot, terrain, sensors, and lighting shared by every task."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        terrain_generator=MICRODUCK_ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = MICRODUCK_WALK_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    left_foot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base/yaw2roll/hip_l/upper_leg_left/leg/ankle_left",
        history_length=3,
        track_air_time=True,
    )
    right_foot_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base/bearing_roll/hip_l_2/upper_leg_right/leg_2/ankle_right",
        history_length=3,
        track_air_time=True,
    )
    trunk_ground_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base",
        history_length=3,
        track_air_time=True,
    )
    head_ground_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/Geometry/trunk_base/neck/neck_pitch_1/yaw_roll_motion/jaw_soft",
        history_length=3,
        track_air_time=True,
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.86, 0.90, 1.0)),
    )


@configclass
class CommandsCfg:
    """Unified 13-dimensional command layout."""

    twist = mdp.MicroDuckVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.0, 8.0),
        rel_standing_envs=0.02,
        rel_heading_envs=0.0,
        rel_turn_in_place_envs=0.15,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.MicroDuckVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.4, 0.4),
            lin_vel_y=(-0.3, 0.3),
            ang_vel_z=(-1.0, 1.0),
        ),
    )
    head_pose = mdp.UniformVectorCommandCfg(
        resampling_time_range=(2.0, 5.0),
        ranges=((-0.05, 0.05), (-0.05, 0.05), (-0.07, 0.07), (-0.015, 0.015)),
        zero_command_prob=0.05,
    )
    body_pose = mdp.UniformVectorCommandCfg(
        resampling_time_range=(2.0, 5.0),
        ranges=((-0.005, 0.005),) * 3 + ((-0.05, 0.05),) * 3,
        zero_command_prob=0.05,
    )


@configclass
class ActionsCfg:
    """Canonical 14-joint absolute position action."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=MICRODUCK_ACTION_JOINT_NAMES,
        preserve_order=True,
        scale=1.0,
        use_default_offset=True,
    )


_ACTION_JOINT_CFG = SceneEntityCfg("robot", joint_names=MICRODUCK_ACTION_JOINT_NAMES, preserve_order=True)


@configclass
class ObservationsCfg:
    """Policy and privileged critic observations."""

    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.03, n_max=0.03))
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity_imu_misaligned,
            params={"max_angle_deg": 6.0},
            noise=Unoise(n_min=-0.01, n_max=0.01),
        )
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "twist"})
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": _ACTION_JOINT_CFG},
            noise=Unoise(n_min=-0.001, n_max=0.001),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": _ACTION_JOINT_CFG},
            noise=Unoise(n_min=-0.25, n_max=0.25),
        )
        actions = ObsTerm(func=mdp.last_action)
        head_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "head_pose"})
        body_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "body_pose"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventsCfg:
    """Resets, pushes, and domain randomization."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25), "yaw": (-math.pi, math.pi)},
            "velocity_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "yaw": (-0.1, 0.1)},
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.05, 0.05),
            "velocity_range": (-0.1, 0.1),
            "asset_cfg": _ACTION_JOINT_CFG,
        },
    )
    reset_action_history = EventTerm(func=mdp.reset_action_history, mode="reset")
    randomize_joint_friction = EventTerm(
        func=mdp.randomize_bam_friction,
        mode="reset",
        params={"scale_range": (0.9, 1.1)},
    )
    randomize_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="trunk_base"),
            "mass_distribution_params": (0.95, 1.05),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={
            "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class RewardsCfg:
    """Default walking objective."""

    track_lin_vel = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=2.0,
        params={"command_name": "twist", "std": math.sqrt(0.1)},
    )
    track_ang_vel = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=2.0,
        params={"command_name": "twist", "std": math.sqrt(0.5)},
    )
    upright = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-1.0)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)
    joint_torque = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-4, params={"asset_cfg": _ACTION_JOINT_CFG})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    feet_air_time = RewTerm(
        func=mdp.paired_feet_air_time,
        weight=3.0,
        params={
            "command_name": "twist",
            "threshold": 0.125,
            "sensor_names": ("left_foot_contact", "right_foot_contact"),
        },
    )
    head_pose = RewTerm(
        func=mdp.head_pose_tracking,
        weight=2.0,
        params={"command_name": "head_pose", "std": 0.5},
    )
    body_pose = RewTerm(
        func=mdp.body_pose_tracking_6d,
        weight=0.0,
        params={"command_name": "body_pose", "nominal_height": 0.115},
    )


@configclass
class TerminationsCfg:
    """Default episode termination conditions."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    nan_state = DoneTerm(func=mdp.robot_state_is_nan)
    body_contact = DoneTerm(
        func=mdp.any_sensor_contact,
        params={
            "sensor_names": ("trunk_ground_contact", "head_ground_contact"),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    """Default terrain curriculum."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class MicroDuckVelocityRoughEnvCfg(ManagerBasedRLEnvCfg):
    """Velocity tracking on gentle generated terrain."""

    scene: MicroDuckSceneCfg = MicroDuckSceneCfg(num_envs=4096, env_spacing=2.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=4)

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 20.0
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.12)
        self.scene.left_foot_contact.update_period = self.sim.dt
        self.scene.right_foot_contact.update_period = self.sim.dt
        self.scene.trunk_ground_contact.update_period = self.sim.dt
        self.scene.head_ground_contact.update_period = self.sim.dt
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.curriculum = self.curriculum.terrain_levels is not None


@configclass
class MicroDuckVelocityFlatEnvCfg(MicroDuckVelocityRoughEnvCfg):
    """Velocity tracking on a flat plane."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.curriculum.terrain_levels = None


@configclass
class MicroDuckVelocityFlatEnvCfg_PLAY(MicroDuckVelocityFlatEnvCfg):
    """Small flat velocity environment for interactive playback."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.events.push_robot = None


@configclass
class MicroDuckVelocityRoughEnvCfg_PLAY(MicroDuckVelocityRoughEnvCfg):
    """Small rough velocity environment for interactive playback."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 32
        self.scene.terrain.terrain_generator.num_rows = 5
        self.scene.terrain.terrain_generator.num_cols = 5
        self.scene.terrain.terrain_generator.curriculum = False
        self.events.push_robot = None
