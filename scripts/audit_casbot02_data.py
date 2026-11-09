"""Audit supplied Casbot02 CSVs against the current MJCF without editing them."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mujoco
import numpy as np
import torch

from humanoidverse.agents.envs.humanoidverse_mjlab import _compose_humanoidverse_config
from humanoidverse.sim2sim import NativeRobot, build_native_model
from humanoidverse.utils.robot_spec import load_robot_training_spec


def audit(source: Path, output: Path, *, ground: bool):
    torch.set_num_threads(2)
    training = load_robot_training_spec("configs/robots/casbot02.yaml")
    snapshot = training.to_env_dict()
    hv, _ = _compose_humanoidverse_config(
        num_envs=1,
        relative_config_path="exp/bfm_zero/bfm_zero",
        hydra_overrides=[f"robot={training.hydra_robot}", *training.hydra_overrides],
        lafan_tail_path="unused.pkl",
        data_mix_weights=None,
        disable_obs_noise=True,
        disable_domain_randomization=True,
        max_episode_length_s=None,
        root_height_obs=True,
        robot_training=snapshot,
    )
    model = build_native_model(hv, snapshot)
    robot = NativeRobot(model, hv, snapshot)
    floor = model.geom("ufo_floor").id
    joint_ids = [model.joint(name).id for name in robot.names]
    limits = model.jnt_range[joint_ids]
    rows = []
    paths = sorted(source.glob("*.csv"))
    if not paths:
        raise ValueError(f"No CSV files in {source}")
    for path in paths:
        values = np.loadtxt(path, delimiter=",", ndmin=2)
        if values.shape[1] != 34 or not np.isfinite(values).all():
            raise ValueError(f"Expected finite T×34 RobotState CSV: {path}, {values.shape}")
        norm = np.linalg.norm(values[:, 3:7], axis=1)
        if np.any(norm < 1e-6):
            raise ValueError(f"Zero root quaternion in {path}")
        excess = np.maximum(limits[:, 0] - values[:, 7:], values[:, 7:] - limits[:, 1])
        excess = np.maximum(excess, 0)
        frame, joint = np.unravel_index(excess.argmax(), excess.shape)
        row = dict(
            filename=path.name,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
            frames=len(values),
            fps=30.0,
            root_height_min_m=float(values[:, 2].min()),
            quaternion_norm_max_error=float(np.abs(norm - 1).max()),
            joint_limit_excess_max_rad=float(excess.max()),
            joint_limit_excess_gt_1e4_frames=int(np.any(excess > 1e-4, axis=1).sum()),
            worst_limit_joint=robot.names[joint],
            worst_limit_frame=int(frame),
        )
        if ground:
            depth = np.zeros(len(values))
            for t, value in enumerate(values):
                robot.data.qpos[:3] = value[:3]
                robot.data.qpos[3:7] = value[[6, 3, 4, 5]] / norm[t]
                robot.data.qpos[robot.qpos_ids] = value[7:]
                # Collision distances need kinematics, not a constraint solve.
                mujoco.mj_kinematics(model, robot.data)
                mujoco.mj_collision(model, robot.data)
                depth[t] = max([0.0, *[-float(c.dist) for c in robot.data.contact if floor in c.geom]])
            row.update(
                max_floor_penetration_m=float(depth.max()),
                floor_penetration_gt_2cm_frames=int((depth > 0.02).sum()),
                floor_penetration_gt_5cm_frames=int((depth > 0.05).sum()),
            )
        rows.append(row)
        print(json.dumps(row), flush=True)
    summary = dict(
        source=str(source.resolve()),
        robot_xml=training.robot.xml_path,
        csv_files=len(rows),
        total_frames=sum(r["frames"] for r in rows),
        fps=30,
        total_hours=sum(r["frames"] for r in rows) / 30 / 3600,
        max_joint_limit_excess_rad=max(r["joint_limit_excess_max_rad"] for r in rows),
        joint_limit_excess_gt_1e4_frames=sum(r["joint_limit_excess_gt_1e4_frames"] for r in rows),
        files=rows,
    )
    if ground:
        summary.update(
            max_floor_penetration_m=max(r["max_floor_penetration_m"] for r in rows),
            floor_penetration_gt_2cm_frames=sum(r["floor_penetration_gt_2cm_frames"] for r in rows),
            floor_penetration_gt_5cm_frames=sum(r["floor_penetration_gt_5cm_frames"] for r in rows),
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print("DATA_AUDIT_COMPLETE", json.dumps({k: v for k, v in summary.items() if k != "files"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=Path("data/lafan1_casbot02"))
    parser.add_argument("--output", type=Path, default=Path("cache/motion_data/casbot02_lafan1_20261106/source_audit.json"))
    parser.add_argument("--ground", action="store_true")
    args = parser.parse_args()
    audit(args.source, args.output, ground=args.ground)
