# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""MicroDuck-scale parkour and downhill roller terrains."""

from __future__ import annotations

import math

import numpy as np

from isaaclab.terrains import SubTerrainBaseCfg
from isaaclab.utils.configclass import configclass

COURSE_LENGTH = 9.0
COURSE_WIDTH = 1.2
COURSE_CENTER_Y = 1.0
START_X = 0.35
FINISH_X = 8.35
BRIDGE_HEIGHT = 0.035
TRAINING_CHECKPOINTS = (
    (START_X, 0.0, 0.0),
    (0.90, 0.0, 0.0),
    (2.15, 0.0, 0.0),
    (3.45, 0.0, 0.0),
    (4.30, 0.0, BRIDGE_HEIGHT * 0.875),
    (5.30, 0.0, BRIDGE_HEIGHT),
    (6.30, 0.0, 0.0),
    (7.55, 0.0, 0.0),
)


def _box(x0: float, x1: float, width: float, top: float, center_y: float, thickness: float = 0.12):
    import trimesh

    center = ((x0 + x1) / 2.0, center_y, top - thickness / 2.0)
    transform = trimesh.transformations.translation_matrix(center)
    return trimesh.creation.box((x1 - x0, width, thickness), transform=transform)


def _ramp(x0: float, x1: float, z0: float, z1: float, width: float, center_y: float, thickness: float = 0.12):
    import trimesh

    angle = -math.atan2(z1 - z0, x1 - x0)
    length = math.hypot(x1 - x0, z1 - z0)
    center = (
        (x0 + x1) / 2.0 - thickness / 2.0 * math.sin(angle),
        center_y,
        (z0 + z1) / 2.0 - thickness / 2.0 * math.cos(angle),
    )
    transform = trimesh.transformations.translation_matrix(center) @ trimesh.transformations.rotation_matrix(
        angle, (0.0, 1.0, 0.0)
    )
    return trimesh.creation.box((length, width, thickness), transform=transform)


def parkour_terrain(difficulty: float, cfg: MicroDuckParkourTerrainCfg):
    """Generate the continuous rumble, stair, bridge, and stepping-pad course."""
    scale = cfg.obstacle_scale_min + np.clip(difficulty, 0.0, 1.0) * (cfg.obstacle_scale_max - cfg.obstacle_scale_min)
    meshes = [_box(0.0, 1.0, COURSE_WIDTH, 0.0, COURSE_CENTER_Y)]
    for index, height in enumerate((0.0, 0.004, 0.008, 0.004, 0.0) * 2):
        x0 = 1.0 + 0.1 * index
        meshes.append(_box(x0, x0 + 0.1, COURSE_WIDTH, height * scale, COURSE_CENTER_Y))
    meshes.append(_box(2.0, 2.3, COURSE_WIDTH, 0.0, COURSE_CENTER_Y))
    for index, level in enumerate((1, 2, 3, 4, 5, 4, 3, 2, 1, 0)):
        x0 = 2.3 + 0.1 * index
        meshes.append(_box(x0, x0 + 0.1, COURSE_WIDTH, level * 0.006 * scale, COURSE_CENTER_Y))
    meshes.append(_box(3.3, 3.6, COURSE_WIDTH, 0.0, COURSE_CENTER_Y))
    bridge_z = BRIDGE_HEIGHT * scale
    meshes.extend(
        (
            _ramp(3.6, 4.4, 0.0, bridge_z, COURSE_WIDTH, COURSE_CENTER_Y),
            _box(4.4, 5.4, 0.22, bridge_z, COURSE_CENTER_Y),
            _ramp(5.4, 6.2, bridge_z, 0.0, COURSE_WIDTH, COURSE_CENTER_Y),
            _box(6.2, 6.5, COURSE_WIDTH, 0.0, COURSE_CENTER_Y),
        )
    )
    for index in range(8):
        x0 = 6.5 + index * 0.125
        y = COURSE_CENTER_Y + (0.025 if index % 2 == 0 else -0.025)
        meshes.append(_box(x0, x0 + 0.11, 0.28, 0.005 * scale, y))
    meshes.append(_box(7.5, COURSE_LENGTH, COURSE_WIDTH, 0.0, COURSE_CENTER_Y))
    return meshes, np.array([START_X, COURSE_CENTER_Y, 0.0])


@configclass
class MicroDuckParkourTerrainCfg(SubTerrainBaseCfg):
    function = parkour_terrain
    obstacle_scale_min: float = 0.45
    obstacle_scale_max: float = 1.0


def downhill_terrain(difficulty: float, cfg: MicroDuckDownhillTerrainCfg):
    """Generate a flat entry, descending ramp, and level runout."""
    angle = math.radians(cfg.deg_min + np.clip(difficulty, 0.0, 1.0) * (cfg.deg_max - cfg.deg_min))
    ramp_length = cfg.ramp_length_min + 0.5 * (cfg.ramp_length_max - cfg.ramp_length_min)
    drop = ramp_length * math.tan(angle)
    meshes = [
        _box(0.0, cfg.flat_length, cfg.size[1], 0.0, cfg.size[1] / 2.0, cfg.thickness),
        _ramp(
            cfg.flat_length,
            cfg.flat_length + ramp_length,
            0.0,
            -drop,
            cfg.size[1],
            cfg.size[1] / 2.0,
            cfg.thickness,
        ),
        _box(
            cfg.flat_length + ramp_length,
            cfg.flat_length + ramp_length + cfg.runout_length,
            cfg.size[1],
            -drop,
            cfg.size[1] / 2.0,
            cfg.thickness,
        ),
    ]
    spawn_x = cfg.flat_length + cfg.spawn_on_ramp
    spawn_z = -cfg.spawn_on_ramp * math.tan(angle)
    return meshes, np.array([spawn_x, cfg.size[1] / 2.0, spawn_z])


@configclass
class MicroDuckDownhillTerrainCfg(SubTerrainBaseCfg):
    function = downhill_terrain
    flat_length: float = 2.0
    ramp_length_min: float = 3.0
    ramp_length_max: float = 8.0
    runout_length: float = 4.0
    spawn_on_ramp: float = 0.3
    deg_min: float = 2.0
    deg_max: float = 20.0
    thickness: float = 0.5
