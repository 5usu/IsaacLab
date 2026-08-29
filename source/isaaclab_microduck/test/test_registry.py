# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Tests for the public MicroDuck Gym registrations."""

import sys

import gymnasium as gym
import isaaclab_microduck.tasks  # noqa: F401
from isaaclab_microduck import register_tasks

from isaaclab.envs import ManagerBasedRLEnvCfg

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg

from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry


def _registered_task_ids() -> list[str]:
    return sorted(task_id for task_id in gym.registry if "MicroDuck" in task_id or "XL330-TestBench" in task_id)


def test_all_train_and_play_tasks_are_registered() -> None:
    task_ids = _registered_task_ids()
    train_ids = [task_id for task_id in task_ids if "-Play-v0" not in task_id]

    assert len(task_ids) == 74
    assert len(train_ids) == 37
    assert "Isaac-Backflip-Flat-MicroDuck-v0" in train_ids
    assert "Isaac-Parkour-MicroDuck-v0" in train_ids
    assert "Isaac-Position-XL330-TestBench-v0" in train_ids
    for task_id in train_ids:
        assert task_id.removesuffix("-v0") + "-Play-v0" in task_ids


def test_external_callback_preserves_command_line(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["train.py", "--task", "example", "physics=physx"])

    assert register_tasks() == ["--task", "example", "physics=physx"]


def test_all_registered_configs_load() -> None:
    for task_id in _registered_task_ids():
        env_cfg = load_cfg_from_registry(task_id, "env_cfg_entry_point")
        runner_cfg = load_cfg_from_registry(task_id, "rsl_rl_cfg_entry_point")

        assert isinstance(env_cfg, ManagerBasedRLEnvCfg), task_id
        assert isinstance(runner_cfg, RslRlBaseRunnerCfg), task_id
