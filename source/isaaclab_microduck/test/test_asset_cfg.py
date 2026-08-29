# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
from isaaclab_microduck.assets import (
    MICRODUCK_ACTION_JOINT_NAMES,
    MICRODUCK_ALL_COLLISIONS_BACKLASH_CFG,
    MICRODUCK_ALL_COLLISIONS_CFG,
    MICRODUCK_BALL_CFG,
    MICRODUCK_ROLLERS_BACKLASH_CFG,
    MICRODUCK_ROLLERS_CFG,
    MICRODUCK_WALK_BACKLASH_CFG,
    MICRODUCK_WALK_CFG,
    XL330_TEST_BENCH_CFG,
)

from pxr import Usd, UsdPhysics

from isaaclab.sim import MjcfFileCfg, UsdFileCfg


def test_action_joint_contract_is_stable() -> None:
    assert len(MICRODUCK_ACTION_JOINT_NAMES) == 14
    assert len(set(MICRODUCK_ACTION_JOINT_NAMES)) == 14
    assert not any(name.startswith("passive_") for name in MICRODUCK_ACTION_JOINT_NAMES)

    robot_configs = [
        MICRODUCK_WALK_CFG,
        MICRODUCK_ALL_COLLISIONS_CFG,
        MICRODUCK_ROLLERS_CFG,
        MICRODUCK_WALK_BACKLASH_CFG,
        MICRODUCK_ALL_COLLISIONS_BACKLASH_CFG,
        MICRODUCK_ROLLERS_BACKLASH_CFG,
    ]
    for configuration in robot_configs:
        assert configuration.actuators["bam"].joint_names_expr == MICRODUCK_ACTION_JOINT_NAMES


def test_all_source_assets_exist() -> None:
    configurations = [
        MICRODUCK_WALK_CFG,
        MICRODUCK_ALL_COLLISIONS_CFG,
        MICRODUCK_ROLLERS_CFG,
        MICRODUCK_WALK_BACKLASH_CFG,
        MICRODUCK_ALL_COLLISIONS_BACKLASH_CFG,
        MICRODUCK_ROLLERS_BACKLASH_CFG,
        MICRODUCK_BALL_CFG,
        XL330_TEST_BENCH_CFG,
    ]
    for configuration in configurations:
        assert isinstance(configuration.spawn, MjcfFileCfg | UsdFileCfg)
        path = (
            configuration.spawn.asset_path
            if isinstance(configuration.spawn, MjcfFileCfg)
            else configuration.spawn.usd_path
        )
        assert Path(path).is_file()


def test_backlash_feedback_is_enabled_only_for_backlash_models() -> None:
    assert MICRODUCK_WALK_BACKLASH_CFG.actuators["bam"].encoder_through_backlash
    assert MICRODUCK_ALL_COLLISIONS_BACKLASH_CFG.actuators["bam"].encoder_through_backlash
    assert MICRODUCK_ROLLERS_BACKLASH_CFG.actuators["bam"].encoder_through_backlash
    assert not MICRODUCK_WALK_CFG.actuators["bam"].encoder_through_backlash


def test_backlash_sources_use_distinct_serial_bodies() -> None:
    robot_dir = Path(__file__).parents[1] / "isaaclab_microduck" / "data" / "robots" / "microduck"
    for source_name in [
        "robot_walk_backlash.xml",
        "robot_allcollisions_backlash.xml",
        "robot_allcollisions_rollers_backlash.xml",
    ]:
        source = robot_dir / source_name
        root = ET.parse(source).getroot()
        body_by_joint = {
            joint.get("name"): body.get("name") for body in root.iter("body") for joint in body.findall("joint")
        }
        for name in MICRODUCK_ACTION_JOINT_NAMES:
            assert body_by_joint[name] != body_by_joint[f"passive_{name}_backlash"]

        model = mujoco.MjModel.from_xml_path(str(source))
        assert model.nu == 14
        assert model.njnt >= 29


def test_converted_usd_joint_counts() -> None:
    expected_counts = {
        "robot_walk": 14,
        "robot_allcollisions": 14,
        "robot_allcollisions_rollers": 18,
        "robot_walk_backlash": 28,
        "robot_allcollisions_backlash": 28,
        "robot_allcollisions_rollers_backlash": 32,
        "xl330_test_bench": 2,
        "ball": 0,
    }
    usd_dir = Path(__file__).parents[1] / "isaaclab_microduck" / "data" / "robots" / "usd"
    for model_name, expected_count in expected_counts.items():
        stage = Usd.Stage.Open(str(usd_dir / model_name / f"{model_name}.usda"))
        assert stage is not None
        joints = [prim for prim in stage.Traverse() if prim.IsA(UsdPhysics.Joint)]
        assert len(joints) == expected_count


def test_converted_usd_articulation_api_is_above_root_body() -> None:
    usd_dir = Path(__file__).parents[1] / "isaaclab_microduck" / "data" / "robots" / "usd"
    for model_name in (
        "robot_walk",
        "robot_allcollisions",
        "robot_allcollisions_rollers",
        "robot_walk_backlash",
        "robot_allcollisions_backlash",
        "robot_allcollisions_rollers_backlash",
        "xl330_test_bench",
    ):
        stage = Usd.Stage.Open(str(usd_dir / model_name / f"{model_name}.usda"))
        root = stage.GetDefaultPrim()
        assert root.HasAPI(UsdPhysics.ArticulationRootAPI)
        assert all(not prim.HasAPI(UsdPhysics.ArticulationRootAPI) for prim in stage.Traverse() if prim != root)

    ball_stage = Usd.Stage.Open(str(usd_dir / "ball" / "ball.usda"))
    assert all(not prim.HasAPI(UsdPhysics.ArticulationRootAPI) for prim in ball_stage.Traverse())


def test_xl330_test_bench_is_fixed_to_world() -> None:
    usd_path = (
        Path(__file__).parents[1]
        / "isaaclab_microduck"
        / "data"
        / "robots"
        / "usd"
        / "xl330_test_bench"
        / "xl330_test_bench.usda"
    )
    stage = Usd.Stage.Open(str(usd_path))
    fixed_joint = UsdPhysics.FixedJoint.Get(stage, "/xl330_test_bench/world_fixed_joint")

    assert fixed_joint
    assert fixed_joint.GetBody0Rel().GetTargets() == []
    assert fixed_joint.GetBody1Rel().GetTargets() == [
        stage.GetPrimAtPath("/xl330_test_bench/Geometry/bench_holder").GetPath()
    ]
