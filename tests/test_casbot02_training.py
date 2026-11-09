"""Verify the imported Casbot controls and the FK used for FB expert data."""
import hashlib
import json
import unittest
from pathlib import Path

import mujoco
import numpy as np
import torch
from omegaconf import OmegaConf

from humanoidverse.agents.envs.humanoidverse_mjlab import (
    _compose_humanoidverse_config,
    make_mjlab_ufo_env_cfg,
)
from humanoidverse.train import build_ufo_mjlab_config
from humanoidverse.utils.motion_data.robot_state import RobotStateMotion
from humanoidverse.utils.motion_data.robot_state_convert import robot_state_to_ufo_motion
from humanoidverse.utils.motion_lib.torch_humanoid_batch import Humanoid_Batch
from humanoidverse.utils.robot_spec import load_robot_training_spec

ASSETS = Path("humanoidverse/data/robots/casbot02")


def composed_config(disable_dr=False, robot_config="configs/robots/casbot02.yaml"):
    cfg = build_ufo_mjlab_config(
        device="cpu", work_dir="/tmp/casbot02-unit", num_envs=2,
        num_env_steps=32, seed=1, use_wandb=False, wandb_run_name=None,
        robot_config=robot_config, smoke=True,
    )
    env = cfg.env
    hv, _ = _compose_humanoidverse_config(
        num_envs=2, relative_config_path=env.relative_config_path,
        hydra_overrides=env.hydra_overrides, lafan_tail_path="unused.pkl",
        data_mix_weights=None, disable_obs_noise=True,
        disable_domain_randomization=disable_dr,
        max_episode_length_s=None, root_height_obs=True,
        robot_training=env.robot_training,
    )
    return cfg, hv


class Casbot02TrainingTest(unittest.TestCase):
    def test_assets_match_pinned_source(self):
        manifest = json.loads((ASSETS / "ufo_source_manifest.json").read_text())
        for filename, expected in manifest["files"].items():
            expected = manifest.get("local_overrides", {}).get(filename, {}).get("sha256", expected)
            self.assertEqual(hashlib.sha256((ASSETS / filename).read_bytes()).hexdigest(), expected, filename)

    def test_palm_free_training_geometry(self):
        revision = json.loads((ASSETS / "xml_revision_20261107.json").read_text())
        xml = ASSETS / "xml/casbot02_primitives.xml"
        self.assertEqual(hashlib.sha256(xml.read_bytes()).hexdigest(), revision["source_sha256"])
        model = mujoco.MjModel.from_xml_path(str(xml))
        self.assertEqual((model.nq, model.nv, model.nu), (34, 33, 27))
        for side, mass in (("left", 0.11076), ("right", 0.10176)):
            self.assertAlmostEqual(model.body_mass[model.body(f"{side}_wrist_roll_link").id], mass)
            self.assertEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, f"{side}_base_link"), -1)
            self.assertEqual(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"{side}_hand_collision"), -1)
            self.assertGreaterEqual(model.geom(f"{side}_wrist_collision").id, 0)
            self.assertAlmostEqual(model.geom_size[model.geom(f"{side}_shin_collision").id, 0], 0.05)

    def test_compiled_actuators_match_motor_table(self):
        from mjlab.actuator import BuiltinPositionActuatorCfg
        params = json.loads((ASSETS / "motor_parameters.json").read_text())
        cfg, hv = composed_config()
        env = make_mjlab_ufo_env_cfg(hv, num_envs=2, seed=1,
            mjcf_path=cfg.env.mjcf_path, auto_reset=False, robot_training=cfg.env.robot_training)
        robot = env.scene.entities["robot"]
        self.assertEqual(len(robot.articulation.actuators), 27)
        spec = robot.spec_fn()
        for act in robot.articulation.actuators:
            self.assertIsInstance(act, BuiltinPositionActuatorCfg)
            name = act.target_names_expr[0]
            p = params["motors"][name]
            for key in ("stiffness", "damping", "effort_limit", "armature"):
                self.assertEqual(getattr(act, key), p[key], (name, key))
            self.assertEqual(act.frictionloss, p["friction"])
            self.assertEqual(act.delay_max_lag, 2)
            self.assertEqual(act.delay_update_period, 2**40)
            self.assertAlmostEqual(env.actions["actions"].scale[name], 0.25*p["effort_limit"]/p["stiffness"])
            act.build(None, [0], [name]).edit_spec(spec, [name])
        model = spec.compile()
        self.assertEqual((model.nq, model.nv, model.nu), (34, 33, 27))
        for i in range(model.nu):
            joint_id = model.actuator_trnid[i, 0]
            p = params["motors"][model.joint(joint_id).name]
            np.testing.assert_allclose(model.actuator_gainprm[i, 0], p["stiffness"])
            np.testing.assert_allclose(model.actuator_biasprm[i, 1:3], [-p["stiffness"], -p["damping"]])
            np.testing.assert_allclose(model.actuator_forcerange[i], [-p["effort_limit"], p["effort_limit"]])
            self.assertEqual(model.dof_damping[model.jnt_dofadr[joint_id]], 0)
        self.assertEqual(robot.init_state.joint_pos, params["default_joint_pos"])
        self.assertEqual(env.decimation, 4)
        self.assertEqual(env.sim.mujoco.timestep, 0.005)

    def test_urdf_motor_revision_and_preserved_position_limits(self):
        import math
        import xml.etree.ElementTree as ET
        params = json.loads((ASSETS / "motor_parameters.json").read_text())
        revision = json.loads((ASSETS / "motor_revision_20261106.json").read_text())
        groups = [
            ([f"leg_{side}{number}_joint" for side in ("l", "r") for number in (1,2,4)], 0.07, 150.0, 14.0),
            ([f"leg_{side}{number}_joint" for side in ("l", "r") for number in (3,5,6)] + ["waist_yaw_joint"], 0.065797, 120.0, 12.56),
            ([f"upper_{side}_{number}_joint" for side in ("left", "right") for number in (1,2,4)], 0.03298, 75.0, 12.2),
            ([f"upper_{side}_{number}_joint" for side in ("left", "right") for number in (3,5,6,7)], 0.02453, 36.0, 9.3),
        ]
        for names, inertia, effort, speed in groups:
            for name in names:
                p = params["motors"][name]
                self.assertEqual(p["armature"], inertia)
                self.assertEqual(p["effort_limit"], effort)
                self.assertEqual(p["velocity_limit"], speed)
                self.assertAlmostEqual(p["stiffness"], inertia*(2*math.pi*10)**2)
                self.assertAlmostEqual(p["damping"], 4*inertia*(2*math.pi*10))
                self.assertAlmostEqual(p["action_scale"], 0.25*effort/p["stiffness"])
        for filename, expected in revision["preserved_position_limits"].items():
            tree = ET.parse(ASSETS / "xml" / filename)
            actual = {j.get("name"): j.get("range") for j in tree.getroot().iter("joint") if j.get("name")}
            self.assertEqual(actual, expected)
            model = mujoco.MjModel.from_xml_path(str(ASSETS / "xml" / filename))
            for name, p in params["motors"].items():
                joint = model.joint(name).id
                self.assertEqual(model.dof_armature[model.jnt_dofadr[joint]], p["armature"])

    def test_disable_dr_turns_off_command_latency(self):
        cfg, hv = composed_config(disable_dr=True)
        env = make_mjlab_ufo_env_cfg(hv, num_envs=2, seed=1,
            mjcf_path=cfg.env.mjcf_path, auto_reset=False, robot_training=cfg.env.robot_training)
        self.assertTrue(all(a.delay_max_lag == 0 for a in env.scene.entities["robot"].articulation.actuators))

    def test_g1_retains_dc_motor_actuators(self):
        from mjlab.actuator import DcMotorActuatorCfg
        cfg, hv = composed_config(robot_config="configs/robots/g1_29dof.yaml")
        env = make_mjlab_ufo_env_cfg(hv, num_envs=2, seed=1,
            mjcf_path=cfg.env.mjcf_path, auto_reset=False, robot_training=cfg.env.robot_training)
        self.assertTrue(all(isinstance(a, DcMotorActuatorCfg) for a in env.scene.entities["robot"].articulation.actuators))

    def test_motion_fk_matches_mujoco_including_fixed_head(self):
        training = load_robot_training_spec("configs/robots/casbot02.yaml")
        robot = training.robot
        model = mujoco.MjModel.from_xml_path(robot.xml_path)
        rng = np.random.default_rng(7)
        joint_pos = np.array([training.default_joint_angles[n] for n in robot.control_joint_names])[None, :] + rng.uniform(-0.1, 0.1, (4, 27))
        root = np.array([[0.0, 0.0, 0.9], [1, 2, 0.85], [-1, 2, 0.92], [0.2, -0.4, 0.9]], dtype=np.float32)
        quat = rng.normal(size=(4, 4)); quat /= np.linalg.norm(quat, axis=1, keepdims=True)
        motion = RobotStateMotion("test", root, quat, joint_pos, 30)
        converted = robot_state_to_ufo_motion(motion, robot)
        motion_cfg = OmegaConf.create({"asset": {"assetRoot": str(Path(robot.xml_path).parent), "assetFileName": Path(robot.xml_path).name}, "extend_config": [], "has_mesh": False})
        fk = Humanoid_Batch(motion_cfg).fk_batch(torch.tensor(converted["pose_aa"])[None], torch.tensor(root)[None], return_full=True)
        np.testing.assert_allclose(fk.dof_pos[0].numpy(), joint_pos, atol=1e-6)
        for frame in range(4):
            data = mujoco.MjData(model)
            data.qpos[:3] = root[frame]
            data.qpos[3:7] = quat[frame, [3, 0, 1, 2]]
            data.qpos[7:] = joint_pos[frame]
            mujoco.mj_forward(model, data)
            np.testing.assert_allclose(fk.global_translation[0, frame].numpy(), data.xpos[1:], atol=2e-6)
            np.testing.assert_allclose(fk.global_rotation_mat[0, frame].numpy(), data.xmat[1:].reshape(-1, 3, 3), atol=2e-6)


if __name__ == "__main__":
    unittest.main()
