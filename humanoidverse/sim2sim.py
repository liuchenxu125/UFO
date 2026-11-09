"""Replay a trained UFO policy in native (CPU) MuJoCo, independent of MJLab.

The actor contract follows G1 tracking inference and UFO-Deploy's previous-frame
history. Physical parameters come from the training run's saved robot snapshot.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any

if "--headless" in sys.argv:
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import joblib
import mujoco
import numpy as np
import torch
from safetensors import safe_open
from torch import nn

from humanoidverse.agents.envs.humanoidverse_mjlab import (
    _action_target_scale,
    _compose_humanoidverse_config,
    _default_joint_pos,
    _match_joint_value,
)
from humanoidverse.agents.envs.utils.gym_spaces import json_to_space
from humanoidverse.agents.load_utils import MODEL_NAME_TO_CLASS
from humanoidverse.utils.helpers import get_backward_observation
from humanoidverse.utils.motion_data import prepare_manifest_dataset_path, prepare_manifest_robot_config_path
from humanoidverse.utils.motion_lib.motion_lib_robot import MotionLibRobot
from humanoidverse.utils.robot_spec import assert_robot_configs_compatible, load_robot_training_spec

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def project_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


class InferencePolicy(nn.Module):
    """Load actor, backward encoder and trained normalization, without critics.

    A full FB checkpoint also holds target networks, discriminators and critics;
    none of those are required for deterministic sim2sim execution.
    """

    def __init__(self, model_dir: Path, device: str = "cpu"):
        super().__init__()
        raw = json.loads((model_dir / "config.json").read_text())
        model_class = MODEL_NAME_TO_CLASS.get(raw["name"])
        if model_class is None:
            raise ValueError(f"Unsupported UFO model: {raw['name']}")
        raw["device"] = "cpu"
        self.cfg = model_class.config_class(**raw)
        kwargs = json.loads((model_dir / "init_kwargs.json").read_text())
        self.obs_space = json_to_space(kwargs["obs_space"])
        self.action_dim = int(kwargs["action_dim"])
        self._actor = self.cfg.archi.actor.build(self.obs_space, self.cfg.archi.z_dim, self.action_dim)
        self._backward_map = self.cfg.archi.b.build(self.obs_space, self.cfg.archi.z_dim)
        self._obs_normalizer = self.cfg.obs_normalizer.build(self.obs_space)
        with safe_open(model_dir / "model.safetensors", framework="pt", device="cpu") as checkpoint:
            keys = list(checkpoint.keys())
            for name in ("_actor", "_backward_map", "_obs_normalizer"):
                prefix = name + "."
                weights = {key[len(prefix) :]: checkpoint.get_tensor(key) for key in keys if key.startswith(prefix)}
                getattr(self, name).load_state_dict(weights, strict=True)
        self.to(device)
        self.eval()
        self.requires_grad_(False)

    @torch.no_grad()
    def backward_map(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        return self._backward_map(self._obs_normalizer(obs))

    def project_z(self, z: torch.Tensor) -> torch.Tensor:
        if self.cfg.archi.norm_z:
            return math.sqrt(z.shape[-1]) * torch.nn.functional.normalize(z, dim=-1)
        return z

    @torch.no_grad()
    def act(self, obs: dict[str, torch.Tensor], z: torch.Tensor) -> torch.Tensor:
        return self._actor(self._obs_normalizer(obs), z, self.cfg.actor_std).mean.float()


def load_training_config(model_folder: Path, robot_config: Path | None, data_path: Path):
    saved = json.loads((model_folder / "config.json").read_text())["env"]
    snapshot = saved.get("robot_training")
    if snapshot is None:
        raise ValueError("This sim2sim entrypoint needs a checkpoint with a saved robot_training snapshot.")
    if robot_config is not None:
        requested = load_robot_training_spec(robot_config)
        robot = snapshot["robot"]
        if requested.robot.name != robot["name"] or list(requested.robot.control_joint_names) != robot["control_joint_names"]:
            raise ValueError("--robot-config does not match the checkpoint robot/control-joint order")
    hv, _ = _compose_humanoidverse_config(
        num_envs=1,
        relative_config_path=saved.get("relative_config_path", "exp/bfm_zero/bfm_zero"),
        hydra_overrides=saved.get("hydra_overrides", []),
        lafan_tail_path=str(data_path),
        data_mix_weights=None,
        disable_obs_noise=True,
        disable_domain_randomization=True,
        max_episode_length_s=None,
        root_height_obs=bool(saved.get("root_height_obs", True)),
        robot_training=snapshot,
    )
    return hv, snapshot


def build_native_model(hv, snapshot: dict[str, Any]) -> mujoco.MjModel:
    """Replace XML motors with the same implicit position PD used for training."""
    actuator = snapshot["actuator"]
    if actuator.get("model") != "builtin_position" or actuator.get("source") != "yaml":
        raise ValueError("Native sim2sim currently supports YAML builtin_position actuators (Casbot02).")
    xml_path = project_path(snapshot["robot"]["xml_path"])
    spec = mujoco.MjSpec.from_file(str(xml_path))
    # Eliminate unused motor actuators, sensors and keyframes referencing old nu.
    for item in list(spec.actuators):
        spec.delete(item)
    for key in list(spec.keys):
        spec.delete(key)
    for name in snapshot["robot"]["control_joint_names"]:
        params = actuator["joints"][name]
        kp = _match_joint_value(name, snapshot["stiffness"])
        kd = _match_joint_value(name, snapshot["damping"])
        effort = float(params["effort_limit"])
        joint = spec.joint(name)
        joint.armature = float(params["armature"])
        joint.frictionloss = float(params.get("friction", 0.0))
        joint.damping[0] = float(actuator.get("viscous_damping", 0.0))
        motor = spec.add_actuator(name="ufo_" + name, target=name)
        motor.trntype = mujoco.mjtTrn.mjTRN_JOINT
        motor.dyntype = mujoco.mjtDyn.mjDYN_NONE
        motor.gaintype = mujoco.mjtGain.mjGAIN_FIXED
        motor.biastype = mujoco.mjtBias.mjBIAS_AFFINE
        motor.gainprm[0] = kp
        motor.biasprm[1:3] = [-kp, -kd]
        motor.gear[0] = 1.0
        motor.inheritrange = 0.0
        motor.ctrllimited = False
        motor.forcelimited = True
        motor.forcerange[:] = [-effort, effort]
    if not any(geom.type == mujoco.mjtGeom.mjGEOM_PLANE for geom in spec.geoms):
        spec.worldbody.add_geom(
            name="ufo_floor",
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[200, 200, 0.1],
            friction=[1.0, 0.005, 0.0001],
            rgba=[0.22, 0.27, 0.32, 1],
            contype=1,
            conaffinity=1,
        )
    if not list(spec.lights):
        spec.worldbody.add_light(pos=[0, -3, 6], dir=[0, 0, -1])
    model = spec.compile()
    # Match mjlab.sim.MujocoCfg defaults and the saved 200-Hz physics clock.
    model.opt.timestep = 1.0 / float(hv.simulator.config.sim.fps)
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON
    model.opt.jacobian = mujoco.mjtJacobian.mjJAC_AUTO
    model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    model.opt.impratio = 1.0
    model.opt.iterations = 100
    model.opt.tolerance = 1e-8
    model.opt.ls_iterations = 50
    model.opt.ls_tolerance = 0.01
    model.opt.ccd_iterations = 50
    model.opt.gravity[:] = [0, 0, -9.81]
    model.vis.global_.offwidth = 1920
    model.vis.global_.offheight = 1080
    return model


class NativeRobot:
    def __init__(self, model: mujoco.MjModel, hv, snapshot: dict[str, Any]):
        self.model = model
        self.data = mujoco.MjData(model)
        self.hv = hv
        self.names = list(snapshot["robot"]["control_joint_names"])
        joints = [model.joint(name).id for name in self.names]
        if any(model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE for j in joints):
            raise ValueError("Expected one hinge joint per controlled DOF")
        self.qpos_ids = model.jnt_qposadr[joints]
        self.qvel_ids = model.jnt_dofadr[joints]
        self.actuator_ids = np.asarray([model.actuator("ufo_" + name).id for name in self.names])
        self.root_id = model.body(snapshot["robot"]["base_body"]).id
        free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
        if len(free) != 1 or model.jnt_bodyid[free[0]] != self.root_id:
            raise ValueError("Expected exactly one free joint on the configured base body")
        self.root_qpos = int(model.jnt_qposadr[free[0]])
        self.root_qvel = int(model.jnt_dofadr[free[0]])
        self.default = _default_joint_pos(hv).numpy()
        self.action_scale = _action_target_scale(hv).numpy()
        self.decimation = int(hv.simulator.config.sim.control_decimation)
        self.dt = self.decimation * model.opt.timestep
        dims = {
            "actions": len(self.names),
            "base_ang_vel": 3,
            "dof_pos": len(self.names),
            "dof_vel": len(self.names),
            "projected_gravity": 3,
        }
        self.history = {key: np.zeros((int(n), dims[key]), np.float32) for key, n in hv.obs.obs_auxiliary.history_actor.items()}
        self.last_action = np.zeros(len(self.names), np.float32)

    def reset(self, qpos: np.ndarray, qvel: np.ndarray) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        self.data.ctrl[self.actuator_ids] = self.data.qpos[self.qpos_ids]
        mujoco.mj_forward(self.model, self.data)
        self.last_action.fill(0)
        for value in self.history.values():
            value.fill(0)

    def observation(self) -> dict[str, np.ndarray]:
        # Root angular velocity and gravity are expressed in the body frame.
        rotation = self.data.xmat[self.root_id].reshape(3, 3)
        spatial_vel = np.zeros(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.root_id, spatial_vel, 0)
        raw = {
            "actions": self.last_action.copy(),
            "base_ang_vel": rotation.T @ spatial_vel[:3],
            "dof_pos": self.data.qpos[self.qpos_ids] - self.default,
            "dof_vel": self.data.qvel[self.qvel_ids].copy(),
            "projected_gravity": rotation.T @ np.array([0.0, 0.0, -1.0]),
        }
        scaled = {key: np.asarray(value * float(self.hv.obs.obs_scales[key]), dtype=np.float32) for key, value in raw.items()}
        # Training concatenates each feature's four previous frames, newest first,
        # in sorted key order. Current values are inserted only AFTER reading it.
        history = np.concatenate([self.history[key].reshape(-1) for key in sorted(self.history)])
        result = {
            "state": np.concatenate([scaled[k] for k in ("dof_pos", "dof_vel", "projected_gravity", "base_ang_vel")]),
            "last_action": scaled["actions"],
            "history_actor": history * float(self.hv.obs.obs_scales.history_actor),
        }
        for key, buffer in self.history.items():
            buffer[1:] = buffer[:-1].copy()
            buffer[0] = scaled[key]
        return {key: value[None, :].astype(np.float32) for key, value in result.items()}

    def step(self, actor_action: np.ndarray) -> None:
        action = np.asarray(actor_action, np.float32).reshape(-1)
        if action.shape != self.last_action.shape or not np.isfinite(action).all():
            raise ValueError(f"Invalid policy action, expected finite shape {self.last_action.shape}")
        control = self.hv.robot.control
        if bool(control.normalize_action):
            action = action * float(control.normalize_action_to) / float(control.normalize_action_from)
        action = np.clip(action, -float(control.action_clip_value), float(control.action_clip_value))
        self.last_action[:] = action  # Store normalized environment action, NOT raw actor output.
        self.data.ctrl[self.actuator_ids] = self.default + self.action_scale * action
        for _ in range(self.decimation):
            mujoco.mj_step(self.model, self.data)
        # Refresh positions/velocities at t+dt before observations or rendering.
        mujoco.mj_forward(self.model, self.data)
        if not np.isfinite(self.data.qpos).all() or not np.isfinite(self.data.qvel).all():
            raise FloatingPointError("MuJoCo state contains NaN/Inf")
        if any(
            self.data.warning[i].number
            for i in (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL, mujoco.mjtWarning.mjWARN_BADQACC)
        ):
            raise FloatingPointError("MuJoCo reported numerical instability")


def select_motion(keys: list[str], motion_id: int | None, motion_name: str | None) -> int:
    if motion_id is not None:
        if not 0 <= motion_id < len(keys):
            raise ValueError(f"--motion-id must be in [0, {len(keys) - 1}]")
        return motion_id
    name = motion_name or "walk1_subject1"
    if name in keys:
        return keys.index(name)
    matches = [i for i, key in enumerate(keys) if name.lower() in key.lower()]
    if len(matches) != 1:
        raise ValueError(f"Motion {name!r} has {len(matches)} matches; use an exact name or --motion-id")
    return matches[0]


def prepare_reference(hv, motion_id: int, robot: NativeRobot):
    motion_cfg = hv.robot.motion
    motion_cfg.step_dt = robot.dt
    library = MotionLibRobot(motion_cfg, num_envs=1, device="cpu")
    library.load_motions(random_sample=False, start_idx=motion_id, num_motions_to_load=1)
    env = SimpleNamespace(
        _motion_lib=library,
        dt=robot.dt,
        device="cpu",
        config=hv,
        default_dof_pos=torch.from_numpy(robot.default)[None],
        gravity_vec=torch.tensor([[0.0, 0.0, -1.0]]),
        use_contact_in_obs_max=False,
    )
    backward, ref = get_backward_observation(env, 0, use_root_height_obs=bool(hv.obs.root_height_obs))
    frames = len(backward["state"])
    qpos = np.tile(robot.model.qpos0, (frames, 1))
    qvel = np.zeros((frames, robot.model.nv))
    root = ref["ref_body_pos"][:, 0].numpy()
    quat = ref["ref_body_rots"][:, 0].numpy()  # xyzw
    root_index = robot.root_qpos
    qpos[:, root_index : root_index + 3] = root
    qpos[:, root_index + 3 : root_index + 7] = quat[:, [3, 0, 1, 2]]
    qpos[:, robot.qpos_ids] = ref["dof_pos"].numpy()
    root_vel = robot.root_qvel
    qvel[:, root_vel : root_vel + 3] = ref["ref_body_vels"][:, 0].numpy()
    from humanoidverse.utils.torch_utils import quat_rotate_inverse

    qvel[:, root_vel + 3 : root_vel + 6] = quat_rotate_inverse(
        ref["ref_body_rots"][:, 0], ref["ref_body_angular_vels"][:, 0], w_last=True
    ).numpy()
    qvel[:, robot.qvel_ids] = ref["ref_dof_vel"].numpy()
    return backward, qpos, qvel


@torch.no_grad()
def encode_context(policy: InferencePolicy, backward, *, device: str, lookahead: int, batch_size: int = 512):
    chunks = []
    for start in range(1, len(backward["state"]), batch_size):
        obs = {k: v[start : start + batch_size].to(device) for k, v in backward.items()}
        chunks.append(policy.backward_map(obs).cpu())
    if not chunks:
        raise ValueError("Motion must contain at least two control frames")
    z = torch.cat(chunks)
    if lookahead > 1:
        z = torch.stack([z[t : t + lookahead].mean(0) for t in range(len(z))])
    return policy.project_z(z).numpy().astype(np.float32)


def validate_contract(policy: InferencePolicy, robot: NativeRobot) -> None:
    expected = {
        "state": 2 * len(robot.names) + 6,
        "last_action": len(robot.names),
        "history_actor": sum(value.size for value in robot.history.values()),
    }
    if policy.action_dim != len(robot.names):
        raise ValueError(f"Checkpoint has {policy.action_dim} actions; robot has {len(robot.names)}")
    keys = list(policy.cfg.archi.actor.input_filter.key)
    if set(keys) - set(expected):
        raise ValueError(f"Unsupported actor input filter: {keys}")
    for key, dim in expected.items():
        if policy.obs_space[key].shape != (dim,):
            raise ValueError(f"Checkpoint {key} shape {policy.obs_space[key].shape} != ({dim},)")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-folder", type=Path, default=Path("runs/ufo_fb_casbot02"))
    parser.add_argument("--robot-config", type=Path)
    data = parser.add_mutually_exclusive_group()
    data.add_argument("--data-path", type=Path)
    data.add_argument("--data-manifest", type=Path)
    parser.add_argument("--dataset", default="lafan1_casbot02")
    motion = parser.add_mutually_exclusive_group()
    motion.add_argument("--motion-id", type=int)
    motion.add_argument("--motion-name")
    parser.add_argument("--list-motions", action="store_true")
    parser.add_argument("--start-time", type=float, default=0.0, help="Start offset in seconds, at the 50-Hz control clock.")
    parser.add_argument("--duration", type=float, help="Playback duration in seconds; default: remaining full motion.")
    parser.add_argument("--max-steps", type=int, help="Optional total control-step cap, including loop repetitions.")
    parser.add_argument("--lookahead", type=int, default=1, help="Average future backward encodings (default: 1, matching G1 tracking).")
    parser.add_argument("--z-path", type=Path, help="Optional .npy/.pkl context, [T,z_dim] or [z_dim].")
    parser.add_argument("--device", default="cpu", help="Actor/encoder device; native MuJoCo physics always uses CPU.")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--loop", action="store_true", help="Reset and repeat the selected segment.")
    parser.add_argument("--no-realtime", action="store_true")
    parser.add_argument("--save-mp4", action="store_true")
    parser.add_argument("--render-size", type=int, default=640)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--export-onnx", action="store_true")
    args = parser.parse_args(argv)
    if not math.isfinite(args.start_time) or args.start_time < 0:
        parser.error("--start-time must be finite and non-negative")
    if args.duration is not None and (not math.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be finite and positive")
    for name in ("max_steps", "lookahead", "threads", "render_size"):
        value = getattr(args, name)
        if value is not None and value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    return args


def run(args) -> dict[str, Any] | None:
    model_folder = project_path(args.model_folder)
    manifest = args.data_manifest
    if manifest is None and args.data_path is None:
        manifest = PROJECT_ROOT / "configs/data/casbot02_lafan1.yaml"
    if manifest is not None:
        manifest = project_path(manifest)
        manifest_robot = prepare_manifest_robot_config_path(manifest)
        if args.robot_config is not None:
            assert_robot_configs_compatible(project_path(args.robot_config), manifest_robot)
        args.robot_config = manifest_robot
        data_path = Path(prepare_manifest_dataset_path(manifest, args.dataset, split="inference", rebuild_cache=False))
    else:
        data_path = project_path(args.data_path)
    data = joblib.load(data_path)
    keys = list(data)
    if args.list_motions:
        for i, key in enumerate(keys):
            print(f"{i:2d}  {key}")
        return None
    motion_id = select_motion(keys, args.motion_id, args.motion_name)
    motion_name = keys[motion_id]
    del data
    torch.set_num_threads(args.threads)
    hv, snapshot = load_training_config(model_folder, args.robot_config, data_path)
    model = build_native_model(hv, snapshot)
    robot = NativeRobot(model, hv, snapshot)
    policy = InferencePolicy(model_folder / "checkpoint/model", args.device)
    validate_contract(policy, robot)
    backward, qpos, qvel = prepare_reference(hv, motion_id, robot)
    # MotionLib temporarily uses one thread during FK; restore the chosen actor budget.
    torch.set_num_threads(args.threads)
    if args.z_path:
        path = project_path(args.z_path)
        z = np.load(path) if path.suffix == ".npy" else np.asarray(joblib.load(path))
        if z.ndim == 1:
            z = z[None]
        if z.ndim != 2 or z.shape[1] != policy.cfg.archi.z_dim or not np.isfinite(z).all() or len(z) == 0:
            raise ValueError(f"Invalid context shape/values: {z.shape}")
        if len(z) > 1 and len(z) != len(qpos) - 1:
            raise ValueError(f"Tracking context needs {len(qpos) - 1} frames, got {len(z)}")
        if len(z) == 1:
            z = np.repeat(z, len(qpos) - 1, axis=0)
    else:
        z = encode_context(policy, backward, device=args.device, lookahead=args.lookahead)
    start = int(round(args.start_time / robot.dt))
    end = len(z) if args.duration is None else min(len(z), start + max(1, int(round(args.duration / robot.dt))))
    if not 0 <= start < end:
        raise ValueError(f"Selected time segment is outside motion duration {len(z) * robot.dt:.2f}s")
    output = project_path(args.output_dir) if args.output_dir else model_folder / "sim2sim" / motion_name
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "contexts.npy", z)
    robot.reset(qpos[start], qvel[start])
    if args.export_onnx:
        from humanoidverse.utils.helpers import export_meta_policy_as_onnx

        metadata = export_meta_policy_as_onnx(policy, output, "policy.onnx")
        metadata.update(robot_name=snapshot["robot"]["name"], control_joint_names=robot.names, control_dt=robot.dt, robot_training=snapshot)
        (output / "policy.meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(
        f"[sim2sim] {snapshot['robot']['name']} | {len(robot.names)} joints | "
        f"physics={1 / model.opt.timestep:g}Hz policy={1 / robot.dt:g}Hz | {args.device}",
        flush=True,
    )
    print(f"[sim2sim] motion={motion_id}:{motion_name} segment={start * robot.dt:.2f}–{end * robot.dt:.2f}s", flush=True)
    print("[sim2sim] implicit PD; observation noise/domain randomization/command delay disabled", flush=True)
    print("[sim2sim] window keys: Space=pause/resume, R=reset, close window=stop", flush=True)
    state = {"paused": False, "reset": False}

    def key_callback(key):
        if key == 32:
            state["paused"] = not state["paused"]
        elif key in (ord("r"), ord("R")):
            state["reset"] = True

    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    camera.trackbodyid = robot.root_id
    camera.distance, camera.azimuth, camera.elevation = 3.0, 135.0, -18.0
    steps, resets, frame_idx = 0, 0, start
    joint_errors, heights, max_torques = [], [], []
    trajectory, actions = [], []
    stopped = "segment_complete"
    started = time.perf_counter()
    with ExitStack() as stack:
        viewer = None
        if not args.headless:
            from mujoco import viewer as mjviewer

            viewer = stack.enter_context(mjviewer.launch_passive(model, robot.data, key_callback=key_callback))
            viewer.cam.type, viewer.cam.trackbodyid = camera.type, camera.trackbodyid
            viewer.cam.distance, viewer.cam.azimuth, viewer.cam.elevation = camera.distance, camera.azimuth, camera.elevation
        renderer = None
        writer = None
        if args.save_mp4:
            import mediapy as media

            renderer = mujoco.Renderer(model, height=args.render_size, width=args.render_size)
            stack.callback(renderer.close)
            writer = stack.enter_context(
                media.VideoWriter(str(output / "sim2sim.mp4"), shape=(args.render_size, args.render_size * 2), fps=round(1 / robot.dt))
            )
            expert = mujoco.MjData(model)
        try:
            while True:
                tick = time.perf_counter()
                if viewer is not None and not viewer.is_running():
                    stopped = "viewer_closed"
                    break
                if state["reset"] or frame_idx >= end:
                    if frame_idx >= end and not args.loop and not state["reset"]:
                        break
                    robot.reset(qpos[start], qvel[start])
                    frame_idx, resets, state["reset"] = start, resets + 1, False
                if state["paused"]:
                    if viewer is not None:
                        viewer.sync()
                    time.sleep(0.02)
                    continue
                obs = {k: torch.from_numpy(v).to(args.device) for k, v in robot.observation().items()}
                action = policy.act(obs, torch.as_tensor(z[frame_idx : frame_idx + 1], device=args.device, dtype=torch.float32))
                robot.step(action.cpu().numpy()[0])
                joint_errors.append(float(np.mean(np.abs(robot.data.qpos[robot.qpos_ids] - qpos[frame_idx + 1, robot.qpos_ids]))))
                heights.append(float(robot.data.xpos[robot.root_id, 2]))
                max_torques.append(float(np.max(np.abs(robot.data.actuator_force[robot.actuator_ids]))))
                trajectory.append(robot.data.qpos.copy())
                actions.append(robot.last_action.copy())
                if writer is not None:
                    expert.qpos[:] = qpos[frame_idx + 1]
                    mujoco.mj_forward(model, expert)
                    renderer.update_scene(expert, camera=camera)
                    reference_image = renderer.render().copy()
                    renderer.update_scene(robot.data, camera=camera)
                    writer.add_image(np.concatenate([reference_image, renderer.render()], axis=1))
                if viewer is not None:
                    viewer.sync()
                steps, frame_idx = steps + 1, frame_idx + 1
                if steps % 100 == 0:
                    print(
                        f"[sim2sim] step={steps} t={steps * robot.dt:.2f}s height={heights[-1]:.3f} "
                        f"joint_mae={np.mean(joint_errors):.4f} rad",
                        flush=True,
                    )
                if args.max_steps is not None and steps >= args.max_steps:
                    stopped = "max_steps"
                    break
                if not args.no_realtime:
                    time.sleep(max(0.0, robot.dt - (time.perf_counter() - tick)))
        except KeyboardInterrupt:
            stopped = "keyboard_interrupt"
    summary = {
        "robot": snapshot["robot"]["name"],
        "model_folder": str(model_folder),
        "motion_id": motion_id,
        "motion_name": motion_name,
        "data_path": str(data_path),
        "control_dt": robot.dt,
        "physics_dt": model.opt.timestep,
        "steps": steps,
        "simulated_seconds": steps * robot.dt,
        "elapsed_seconds": time.perf_counter() - started,
        "resets": resets,
        "stop_reason": stopped,
        "mean_joint_mae_rad": float(np.mean(joint_errors)) if steps else None,
        "min_root_height_m": min(heights) if heights else None,
        "max_abs_actuator_torque_nm": max(max_torques) if max_torques else None,
        "actor_observation_dims": {k: list(policy.obs_space[k].shape) for k in ("state", "last_action", "history_actor")},
        "z_dim": policy.cfg.archi.z_dim,
        "lookahead": args.lookahead,
        "domain_randomization": False,
        "observation_noise": False,
        "command_delay": False,
        "video": str(output / "sim2sim.mp4") if args.save_mp4 else None,
    }
    np.savez_compressed(
        output / "rollout.npz",
        qpos=np.asarray(trajectory),
        actions=np.asarray(actions),
        joint_mae=np.asarray(joint_errors),
        root_height=np.asarray(heights),
    )
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[sim2sim] done: {steps} steps, mean joint MAE={summary['mean_joint_mae_rad']}; output={output}")
    return summary


def main():
    args = parse_args()
    if args.headless:
        os.environ.setdefault("MUJOCO_GL", "egl")
        os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    run(args)


if __name__ == "__main__":
    main()
