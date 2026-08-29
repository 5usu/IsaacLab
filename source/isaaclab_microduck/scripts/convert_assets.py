# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Convert all bundled MicroDuck MJCF models to layered USD assets."""

from __future__ import annotations

import argparse
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--force", action="store_true", help="Overwrite existing converted assets.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

from pxr import Sdf, Usd, UsdPhysics

from isaaclab.sim.converters import MjcfConverter, MjcfConverterCfg

EXTENSION_DIR = Path(__file__).resolve().parents[1]
ROBOT_DIR = EXTENSION_DIR / "isaaclab_microduck" / "data" / "robots"
OUTPUT_DIR = ROBOT_DIR / "usd"

MODELS = {
    ROBOT_DIR / "microduck" / "robot_walk.xml": False,
    ROBOT_DIR / "microduck" / "robot_allcollisions.xml": True,
    ROBOT_DIR / "microduck" / "robot_allcollisions_rollers.xml": True,
    ROBOT_DIR / "microduck" / "robot_walk_backlash.xml": False,
    ROBOT_DIR / "microduck" / "robot_allcollisions_backlash.xml": True,
    ROBOT_DIR / "microduck" / "robot_allcollisions_rollers_backlash.xml": True,
    ROBOT_DIR / "microduck" / "ball.xml": False,
    ROBOT_DIR / "xl330_test_bench" / "xl330_test_bench.xml": False,
}


def _normalize_articulation_root(usd_path: Path) -> None:
    """Normalize converter output for IsaacLab asset and sensor discovery."""
    stage = Usd.Stage.Open(str(usd_path))
    root = stage.GetDefaultPrim()
    if not root:
        raise RuntimeError(f"Converted asset has no default prim: {usd_path}")

    # The MJCF converter can give a link and one of its geometry children the
    # same name. PhysX views match recursively by leaf name, so contact sensors
    # then probe both paths and emit a warning for the non-rigid geometry prim.
    # Rename only the local geometry specs; link and joint paths remain stable.
    duplicate_geometry_paths = []
    for prim in stage.Traverse():
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            continue
        child = prim.GetChild(prim.GetName())
        if child and not child.HasAPI(UsdPhysics.RigidBodyAPI):
            duplicate_geometry_paths.append(child.GetPath())

    base_layer = Sdf.Layer.FindOrOpen(str(usd_path.parent / "payloads" / "base.usda"))
    namespace_renames = [(path, path.ReplaceName(path.name + "_geometry")) for path in duplicate_geometry_paths]
    if base_layer is not None and namespace_renames:
        edits = Sdf.BatchNamespaceEdit()
        for old_path, new_path in namespace_renames:
            if base_layer.GetPrimAtPath(old_path):
                edits.Add(old_path, new_path)
        if edits.edits:
            if not base_layer.Apply(edits):
                raise RuntimeError(f"Could not rename duplicate geometry prims in {base_layer.identifier}")
            base_layer.Save()
            stage.Reload()
            root = stage.GetDefaultPrim()

    articulation_prims = [prim for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)]
    if not articulation_prims:
        return

    stage.SetEditTarget(stage.GetRootLayer())
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            prim.AddAppliedSchema("PhysxContactReportAPI")
            prim.CreateAttribute("physxContactReport:threshold", Sdf.ValueTypeNames.Float).Set(0.0)
    if usd_path.stem == "ball":
        for prim in articulation_prims:
            prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        stage.GetRootLayer().Save()
    else:
        UsdPhysics.ArticulationRootAPI.Apply(root)
        for prim in articulation_prims:
            if prim != root:
                prim.RemoveAPI(UsdPhysics.ArticulationRootAPI)
        if usd_path.stem == "xl330_test_bench":
            base_body = next(
                prim
                for prim in stage.Traverse()
                if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim.GetName() == "bench_holder"
            )
            fixed_joint = UsdPhysics.FixedJoint.Define(stage, root.GetPath().AppendChild("world_fixed_joint"))
            fixed_joint.CreateBody1Rel().SetTargets([base_body.GetPath()])
        stage.GetRootLayer().Save()

    # The solver-common payload is selected independently by the physics
    # manager. Normalize its authored list directly instead of relying only on
    # a stronger interface-layer delete.
    physics_layer = Sdf.Layer.FindOrOpen(str(usd_path.parent / "payloads" / "Physics" / "physics.usda"))
    if physics_layer is None:
        return
    root_spec = physics_layer.GetPrimAtPath(f"/{root.GetName()}")

    def _walk_specs(spec):
        yield spec
        for child in spec.nameChildren.values():
            yield from _walk_specs(child)

    articulation_tokens = {"PhysicsArticulationRootAPI", "NewtonArticulationRootAPI"}
    for spec in _walk_specs(root_spec):
        schemas = spec.GetInfo("apiSchemas")
        schemas.prependedItems = [token for token in schemas.prependedItems if token not in articulation_tokens]
        spec.SetInfo("apiSchemas", schemas)
    if usd_path.stem != "ball":
        root_schemas = root_spec.GetInfo("apiSchemas")
        root_schemas.prependedItems = list(
            dict.fromkeys([*root_schemas.prependedItems, "PhysicsArticulationRootAPI", "NewtonArticulationRootAPI"])
        )
        root_spec.SetInfo("apiSchemas", root_schemas)
    physics_layer.Save()


def main() -> None:
    """Convert every source model and print the generated interface layer."""
    for source, self_collision in MODELS.items():
        output_path = OUTPUT_DIR / source.stem / f"{source.stem}.usda"
        if output_path.exists() and not args_cli.force:
            _normalize_articulation_root(output_path)
            print(f"[skip] {output_path}")
            continue
        cfg = MjcfConverterCfg(
            asset_path=str(source),
            usd_dir=str(OUTPUT_DIR),
            usd_file_name=output_path.name,
            force_usd_conversion=args_cli.force,
            self_collision=self_collision,
            import_physics_scene=False,
            run_asset_transformer=True,
            run_multi_physics_conversion=True,
        )
        converter = MjcfConverter(cfg)
        _normalize_articulation_root(Path(converter.usd_path))
        print(f"[converted] {source.name} -> {converter.usd_path}")


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
