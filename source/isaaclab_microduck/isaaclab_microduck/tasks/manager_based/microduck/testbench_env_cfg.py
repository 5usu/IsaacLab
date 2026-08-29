# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Single-joint XL330 tracking task for actuator sim-to-real validation."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils.configclass import configclass
from isaaclab.utils.noise import UniformNoiseCfg as Unoise

from isaaclab_microduck.assets import XL330_TEST_BENCH_CFG
from isaaclab_microduck.tasks import mdp

if TYPE_CHECKING:
    from isaaclab.assets import Articulation
    from isaaclab.envs import ManagerBasedRLEnv

JOINT_CFG = SceneEntityCfg("robot", joint_names=("tn__1_",))


def target_angle_tracking(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = JOINT_CFG,
):
    """Track a scalar joint-angle command with a Gaussian reward."""
    asset: Articulation = env.scene[asset_cfg.name]
    target = env.command_manager.get_command(command_name)[:, 0]
    joint_pos = asset.data.joint_pos.torch[:, asset_cfg.joint_ids[0]]
    return torch.exp(-torch.square((joint_pos - target) / std))


@configclass
class TestBenchSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    robot: ArticulationCfg = XL330_TEST_BENCH_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.86, 0.90, 1.0)),
    )


@configclass
class TestBenchActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=("tn__1_",),
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class TestBenchCommandsCfg:
    target_angle = mdp.UniformVectorCommandCfg(
        resampling_time_range=(4.0, 4.0),
        ranges=((-math.radians(80.0), math.radians(80.0)),),
    )


@configclass
class TestBenchObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": JOINT_CFG},
            noise=Unoise(n_min=-0.0006, n_max=0.0006),
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": JOINT_CFG},
            noise=Unoise(n_min=-0.24, n_max=0.24),
        )
        actions = ObsTerm(func=mdp.last_action)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "target_angle"})

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        def __post_init__(self):
            super().__post_init__()
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class TestBenchEventsCfg:
    reset_joint = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
            "asset_cfg": JOINT_CFG,
        },
    )


@configclass
class TestBenchRewardsCfg:
    track_target = RewTerm(
        func=target_angle_tracking,
        weight=3.0,
        params={"command_name": "target_angle", "std": math.sqrt(0.15), "asset_cfg": JOINT_CFG},
    )
    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": JOINT_CFG})
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.6)
    joint_torques = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-3, params={"asset_cfg": JOINT_CFG})
    joint_velocity = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-3, params={"asset_cfg": JOINT_CFG})


@configclass
class TestBenchTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class TestBenchEnvCfg(ManagerBasedRLEnvCfg):
    scene: TestBenchSceneCfg = TestBenchSceneCfg(num_envs=64, env_spacing=1.0)
    observations: TestBenchObservationsCfg = TestBenchObservationsCfg()
    actions: TestBenchActionsCfg = TestBenchActionsCfg()
    commands: TestBenchCommandsCfg = TestBenchCommandsCfg()
    events: TestBenchEventsCfg = TestBenchEventsCfg()
    rewards: TestBenchRewardsCfg = TestBenchRewardsCfg()
    terminations: TestBenchTerminationsCfg = TestBenchTerminationsCfg()
    curriculum = None
    sim: SimulationCfg = SimulationCfg(dt=0.005, render_interval=4)

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 8.0


@configclass
class TestBenchEnvCfg_PLAY(TestBenchEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
