# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Ensure task configuration remains safe to load before SimulationApp."""

import subprocess
import sys
import textwrap


def test_configs_do_not_import_simulator_runtime() -> None:
    script = textwrap.dedent(
        """
        import builtins
        import gymnasium as gym

        import isaaclab_microduck
        from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry

        forbidden = {"pxr", "omni", "carb", "isaacsim", "scipy"}
        failures = {}
        task_ids = sorted(
            task_id
            for task_id in gym.registry
            if "MicroDuck" in task_id or "XL330-TestBench" in task_id
        )
        for task_id in task_ids:
            imported = set()
            original_import = builtins.__import__

            def import_hook(name, *args, **kwargs):
                top_level = name.split(".")[0]
                if top_level in forbidden:
                    imported.add(top_level)
                return original_import(name, *args, **kwargs)

            builtins.__import__ = import_hook
            try:
                load_cfg_from_registry(task_id, "env_cfg_entry_point")
            finally:
                builtins.__import__ = original_import
            if imported:
                failures[task_id] = sorted(imported)

        if failures:
            raise RuntimeError(f"Forbidden imports during config loading: {failures}")
        """
    )
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)

    assert result.returncode == 0, result.stdout + result.stderr
