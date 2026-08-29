# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Add physically serial gearbox-backlash joints to a MicroDuck MJCF file.

Each actuated body is split into a motor-side carrier and the original
load-side body. The actuated joint connects the parent to the carrier, and a
limited passive hinge connects the carrier to the load. This topology preserves
both degrees of freedom in MuJoCo, Newton, and PhysX.
"""

from __future__ import annotations

import argparse
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("xml", type=Path, help="MJCF file to modify in place")
    parser.add_argument("--backlash-deg", type=float, default=2.0, help="Total peak-to-peak play in degrees")
    parser.add_argument("--damping", type=float, default=0.01, help="Passive-joint damping [N m s/rad]")
    parser.add_argument("--armature", type=float, default=0.001, help="Passive-joint armature [kg m^2]")
    parser.add_argument("--frictionloss", type=float, default=0.0, help="Passive-joint dry friction [N m]")
    parser.add_argument("--joint-class", default="chosen_actuator", help="MJCF default class to augment")
    return parser.parse_args()


def _insert_backlash_defaults(
    root: ET.Element, half_range: float, damping: float, armature: float, frictionloss: float
) -> None:
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("MJCF file has no <worldbody> element.")
    defaults = ET.Element("default")
    backlash = ET.SubElement(defaults, "default", {"class": "backlash"})
    ET.SubElement(
        backlash,
        "joint",
        {
            "damping": f"{damping:g}",
            "frictionloss": f"{frictionloss:g}",
            "armature": f"{armature:g}",
            "limited": "true",
            "range": f"{-half_range:.17g} {half_range:.17g}",
            "solreflimit": "0.01 1",
            "solimplimit": "0.95 0.999 0.0001 0.5 2",
        },
    )
    root.insert(list(root).index(worldbody), defaults)


def _split_actuated_body(body: ET.Element, joint: ET.Element, half_range: float) -> str:
    joint_name = joint.get("name")
    body_name = body.get("name")
    if not joint_name or not body_name:
        raise ValueError("Every augmented body and joint must have a name.")

    original_children = list(body)
    body.set("name", f"{body_name}_{joint_name}_motor")
    for child in original_children:
        body.remove(child)
    body.append(joint)
    body.append(
        ET.Element(
            "inertial",
            {"pos": "0 0 0", "mass": "1e-8", "diaginertia": "1e-12 1e-12 1e-12"},
        )
    )

    load_body = ET.Element("body", {"name": body_name})
    passive_attributes = {
        "axis": joint.get("axis", "0 0 1"),
        "name": f"passive_{joint_name}_backlash",
        "type": "hinge",
        "class": "backlash",
        "range": f"{-half_range:.17g} {half_range:.17g}",
    }
    if joint.get("pos") is not None:
        passive_attributes["pos"] = joint.get("pos")
    load_body.append(ET.Element("joint", passive_attributes))
    for child in original_children:
        if child is not joint:
            load_body.append(child)
    body.append(load_body)
    return joint_name


def add_backlash(
    xml_path: Path,
    total_degrees: float,
    damping: float,
    armature: float,
    frictionloss: float,
    joint_class: str,
) -> list[str]:
    """Modify an MJCF file in place and return the augmented joint names."""
    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    tree = ET.parse(xml_path, parser=parser)
    root = tree.getroot()
    if any(joint.get("class") == "backlash" for joint in root.iter("joint")):
        raise ValueError(f"{xml_path} already contains backlash joints.")

    half_range = math.radians(total_degrees) / 2.0
    targets: list[tuple[ET.Element, ET.Element]] = []
    for body in list(root.iter("body")):
        joints = [child for child in list(body) if child.tag == "joint" and child.get("class") == joint_class]
        if len(joints) > 1:
            raise ValueError(f"Body {body.get('name')!r} has more than one {joint_class!r} joint.")
        if joints:
            targets.append((body, joints[0]))
    if not targets:
        raise ValueError(f"No joints with class {joint_class!r} were found.")

    _insert_backlash_defaults(root, half_range, damping, armature, frictionloss)
    names = [_split_actuated_body(body, joint, half_range) for body, joint in targets]
    ET.indent(tree, space="  ")
    tree.write(xml_path, encoding="unicode", xml_declaration=True)
    return names


def main() -> int:
    args = _parse_args()
    try:
        names = add_backlash(
            args.xml,
            args.backlash_deg,
            args.damping,
            args.armature,
            args.frictionloss,
            args.joint_class,
        )
    except ValueError as error:
        print(f"[add_backlash] {error}", file=sys.stderr)
        return 1
    print(f"[add_backlash] added serial play to {len(names)} joints in {args.xml}: {', '.join(names)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
