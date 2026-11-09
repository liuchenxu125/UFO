"""Apply the 2026-11-06 Casbot02 motor revision, preserving MJCF position limits.

URDF supplies effort and velocity, with explicit user corrections taking priority.
The reflected inertias below come from
user measurements; gains retain the project's 10-Hz, damping-ratio-2 convention.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import re
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "humanoidverse/data/robots/casbot02"
REVISION = "20261106"
FREQUENCY_HZ = 10.0
DAMPING_RATIO = 2.0
ACTION_SCALE_FACTOR = 0.25
LIMIT_OVERRIDES = {
    name: {"effort_limit": 120.0, "velocity_limit": 12.56}
    for name in [
        *(f"leg_{side}{number}_joint" for side in ("l", "r") for number in (3, 5, 6)),
        "waist_yaw_joint",
    ]
}


def armature_for_joint(name: str) -> float:
    if name.startswith("leg_"):
        return 0.07 if int(re.search(r"([1-6])_joint$", name).group(1)) in {1, 2, 4} else 0.065797
    if name == "waist_yaw_joint":
        return 0.065797
    if name.startswith("upper_"):
        return 0.03298 if int(re.search(r"([1-7])_joint$", name).group(1)) in {1, 2, 4} else 0.02453
    raise ValueError(f"No user-specified inertia for {name}")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def position_ranges(text: str) -> dict[str, str | None]:
    return {j.get("name"): j.get("range") for j in ET.fromstring(text).iter("joint") if j.get("name")}


def calculate_motors(urdf: Path, xml: Path, names: list[str]):
    urdf_by_child = {}
    for joint in ET.parse(urdf).getroot().findall("joint"):
        if joint.get("type") != "fixed" and joint.find("limit") is not None:
            child = joint.find("child").get("link")
            if child in urdf_by_child:
                raise ValueError(f"Ambiguous URDF child {child}")
            urdf_by_child[child] = joint
    body_by_joint = {j.get("name"): body.get("name") for body in ET.parse(xml).getroot().iter("body") for j in body.findall("joint")}
    omega = 2 * math.pi * FREQUENCY_HZ
    motors, mapping = {}, {}
    for name in names:
        joint = urdf_by_child[body_by_joint[name]]
        limit = joint.find("limit")
        effort, velocity = float(limit.get("effort")), float(limit.get("velocity"))
        if name in LIMIT_OVERRIDES:
            effort = LIMIT_OVERRIDES[name]["effort_limit"]
            velocity = LIMIT_OVERRIDES[name]["velocity_limit"]
        if not (effort > 0 and velocity > 0):
            raise ValueError(f"Invalid URDF actuator limits for {name}")
        inertia = armature_for_joint(name)
        kp = inertia * omega**2
        kd = 2 * DAMPING_RATIO * inertia * omega
        motors[name] = dict(
            stiffness=kp,
            damping=kd,
            effort_limit=effort,
            velocity_limit=velocity,
            armature=inertia,
            friction=0.0,
            action_scale=ACTION_SCALE_FACTOR * effort / kp,
        )
        mapping[name] = joint.get("name")
    if len(set(mapping.values())) != len(names):
        raise ValueError("URDF mapping must be one-to-one")
    return motors, mapping


def updated_xml(text: str, motors: dict) -> str:
    def update(match):
        tag = match.group()
        found = re.search(r'\bname="([^"]+)"', tag)
        if found is None or found.group(1) not in motors:
            return tag
        value = str(motors[found.group(1)]["armature"])
        if 'armature="' not in tag:
            raise ValueError(f"Expected explicit armature in {tag}")
        return re.sub(r'\barmature="[^"]*"', f'armature="{value}"', tag)

    result = re.sub(r"<joint\b[^>]*>", update, text)
    if position_ranges(result) != position_ranges(text):
        raise AssertionError("Position limits changed")
    return result


def apply(urdf: Path, *, write: bool):
    robot_path = ROOT / "configs/robots/casbot02.yaml"
    hydra_path = ROOT / "humanoidverse/config/robot/casbot02/casbot02.yaml"
    motor_path = ASSETS / "motor_parameters.json"
    provenance_path = ASSETS / "ufo_source_manifest.json"
    config = yaml.safe_load(robot_path.read_text())
    hydra = yaml.safe_load(hydra_path.read_text())
    names = list(config["control_joints"]["names"])
    motors, mapping = calculate_motors(urdf, ASSETS / "xml/casbot02_primitives.xml", names)
    previous = json.loads(motor_path.read_text())
    revision = dict(
        date="2026-11-06",
        urdf_path=str(urdf),
        urdf_sha256=sha256(urdf),
        urdf_joint_mapping=mapping,
        limit_overrides=LIMIT_OVERRIDES,
        limit_overrides_source="Explicit user correction: both legs 3/5/6 and waist, 120 Nm and 12.56 rad/s.",
        frequency_hz=FREQUENCY_HZ,
        damping_ratio=DAMPING_RATIO,
        stiffness_formula="armature * (2*pi*frequency_hz)^2",
        damping_formula="2 * damping_ratio * armature * (2*pi*frequency_hz)",
        action_scale_formula="0.25 * effort_limit / stiffness",
        inertia_units="kg*m^2 (reflected joint inertia / MuJoCo armature)",
        position_limits_source="Existing MJCF range attributes, unchanged; URDF lower/upper are not imported.",
    )
    params = copy.deepcopy(previous)
    if params.get("parameter_revision", {}).get("urdf_sha256") != revision["urdf_sha256"]:
        params["previous_parameter_metadata"] = {
            k: previous[k] for k in ("source", "source_sha256", "control_parameters_override", "armature_override") if k in previous
        }
    for key in ("control_parameters_override", "armature_override"):
        params.pop(key, None)
    params.update(source=str(urdf), source_sha256=revision["urdf_sha256"], motors=motors, parameter_revision=revision)
    config["training"]["control"].update(
        stiffness={n: motors[n]["stiffness"] for n in names},
        damping={n: motors[n]["damping"] for n in names},
        effort_limit=[motors[n]["effort_limit"] for n in names],
        velocity_limit=[motors[n]["velocity_limit"] for n in names],
    )
    for n in names:
        config["training"]["actuator"]["joints"][n].update(
            {k: motors[n][k] for k in ("armature", "friction", "effort_limit", "velocity_limit")}
        )
    config["metadata"].update(
        review_status="user_parameters_20261106",
        motor_parameter_revision=revision,
        notes=[
            "27 control joints; fixed head bodies retained.",
            "Effort/velocity limits from supplied URDF, overridden by user to 120 Nm / 12.56 rad/s for legs 3/5/6 and waist; MJCF position limits retained.",
            "Implicit PD, 10Hz natural frequency, damping ratio 2; normalized UFO action range [-5, 5].",
        ],
    )
    hydra["robot"]["control"].update(
        stiffness=dict(config["training"]["control"]["stiffness"]), damping=dict(config["training"]["control"]["damping"])
    )
    hydra["robot"]["dof_effort_limit_list"] = list(config["training"]["control"]["effort_limit"])
    hydra["robot"]["dof_vel_limit_list"] = list(config["training"]["control"]["velocity_limit"])
    changes = {
        robot_path: yaml.safe_dump(config, sort_keys=False, allow_unicode=True),
        hydra_path: "# @package _global_\n\n" + yaml.safe_dump(hydra, sort_keys=False, allow_unicode=True),
        motor_path: json.dumps(params, indent=2) + "\n",
    }
    ranges = {}
    for filename in ("casbot02.xml", "casbot02_primitives.xml"):
        path = ASSETS / "xml" / filename
        ranges[filename] = position_ranges(path.read_text())
        changes[path] = updated_xml(path.read_text(), motors)
    print("joint | URDF joint | effort | velocity | armature | Kp | Kd | action scale")
    for n, p in motors.items():
        print(
            f"{n} | {mapping[n]} | {p['effort_limit']:g} | {p['velocity_limit']:g} | {p['armature']:g} | "
            f"{p['stiffness']:.9f} | {p['damping']:.9f} | {p['action_scale']:.9f}"
        )
    if not write:
        return
    backup = ROOT / "runs" / f"casbot02_parameter_update_{REVISION}" / "before"
    for path in [*changes, provenance_path]:
        target = backup / path.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        if not target.exists():
            shutil.copy2(path, target)
    for path, text in changes.items():
        path.write_text(text)
    provenance = json.loads(provenance_path.read_text())
    provenance.setdefault("local_overrides", {}).update(
        {
            str(path.relative_to(ASSETS)): {
                "sha256": sha256(path),
                "revision": REVISION,
                "reason": "User motor revision; URDF limits plus user 120 Nm / 12.56 rad/s override for legs 3/5/6 and waist; prescribed armature, recomputed PD/action scales; position limits retained.",
            }
            for path in changes
            if path.is_relative_to(ASSETS)
        }
    )
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")
    report = dict(
        revision=revision,
        motors=motors,
        preserved_position_limits=ranges,
        backup_dir=str(backup),
        updated_files=[str(p.relative_to(ROOT)) for p in changes],
    )
    (ASSETS / f"motor_revision_{REVISION}.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Updated {len(changes)} files; previous values saved in {backup}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--urdf", required=True, type=Path)
    parser.add_argument("--apply", action="store_true", help="Write updated files; otherwise print calculations only.")
    args = parser.parse_args()
    apply(args.urdf.expanduser().resolve(), write=args.apply)
