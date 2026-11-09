"""Check native control/observation parity and compact checkpoint loading."""

import json
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np
import torch
from gymnasium import spaces

from humanoidverse.agents.envs.humanoidverse_mjlab import HumanoidVerseMjlabCore, _compose_humanoidverse_config
from humanoidverse.agents.presets.fb import build_fb_agent
from humanoidverse.envs.env_utils.history_handler import HistoryHandler
from humanoidverse.sim2sim import (
    InferencePolicy,
    NativeRobot,
    build_native_model,
    load_training_config,
    select_motion,
    validate_contract,
)
from humanoidverse.utils.robot_spec import load_robot_training_spec
from humanoidverse.utils.torch_utils import quat_rotate_inverse


class Casbot02Sim2simTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        training = load_robot_training_spec("configs/robots/casbot02.yaml")
        cls.snapshot = training.to_env_dict()
        cls.hv, _ = _compose_humanoidverse_config(
            num_envs=1,
            relative_config_path="exp/bfm_zero/bfm_zero",
            hydra_overrides=[f"robot={training.hydra_robot}", *training.hydra_overrides],
            lafan_tail_path="unused.pkl",
            data_mix_weights=None,
            disable_obs_noise=True,
            disable_domain_randomization=True,
            max_episode_length_s=None,
            root_height_obs=True,
            robot_training=cls.snapshot,
        )
        cls.model = build_native_model(cls.hv, cls.snapshot)

    def setUp(self):
        self.robot = NativeRobot(self.model, self.hv, self.snapshot)
        qpos = self.model.qpos0.copy()
        qpos[2] = 1.3
        qpos[self.robot.qpos_ids] = self.robot.default
        self.robot.reset(qpos, np.zeros(self.model.nv))

    def test_native_pd_and_action_mapping_match_training(self):
        r = self.robot
        self.assertEqual((r.model.nq, r.model.nv, r.model.nu), (34, 33, 27))
        self.assertAlmostEqual(r.dt, 0.02)
        for i, name in enumerate(r.names):
            j = r.model.joint(name).id
            aid = r.actuator_ids[i]
            params = self.snapshot["actuator"]["joints"][name]
            kp = self.snapshot["stiffness"][name]
            kd = self.snapshot["damping"][name]
            np.testing.assert_allclose(r.model.actuator_gainprm[aid, 0], kp)
            np.testing.assert_allclose(r.model.actuator_biasprm[aid, 1:3], [-kp, -kd])
            np.testing.assert_allclose(r.model.actuator_forcerange[aid], [-params["effort_limit"], params["effort_limit"]])
            self.assertEqual(r.model.dof_armature[r.model.jnt_dofadr[j]], params["armature"])
        raw = np.linspace(-2, 2, 27, dtype=np.float32)
        r.step(raw)
        expected = np.clip(raw * 5, -5, 5)
        np.testing.assert_allclose(r.last_action, expected)
        np.testing.assert_allclose(r.data.ctrl[r.actuator_ids], r.default + r.action_scale * expected)
        self.assertTrue(np.isfinite(r.data.qpos).all())
        r.reset(r.model.qpos0.copy(), np.zeros(r.model.nv))
        self.assertTrue(all(not x.any() for x in r.history.values()))
        self.assertFalse(r.last_action.any())

    def test_observations_and_history_match_training_adapter(self):
        r = self.robot
        # Exercise nontrivial orientation, angular velocity, q0 offsets, actions,
        # and more than four updates to detect both feature and temporal order.
        rng = np.random.default_rng(20)
        core = object.__new__(HumanoidVerseMjlabCore)
        core.config = self.hv
        core.episode_length_buf = torch.zeros(1, dtype=torch.long)
        core.is_evaluating = True
        core.add_noise_currculum = False
        core.default_dof_pos = torch.from_numpy(r.default)[None]
        core.default_dof_pos_offset = torch.zeros(1, 27)
        core._max_local_self = torch.zeros(1, 448)
        core._compute_reference_and_privileged_obs = lambda: None
        core.history_handler = HistoryHandler(1, self.hv.obs.obs_auxiliary, self.hv.obs.obs_dims, "cpu")
        for step in range(7):
            qpos = self.model.qpos0.copy()
            qpos[2] = 1.3
            quat = rng.normal(size=4)
            quat /= np.linalg.norm(quat)
            qpos[3:7] = quat
            qpos[r.qpos_ids] = r.default + rng.uniform(-0.1, 0.1, 27)
            r.data.qpos[:] = qpos
            r.data.qvel[:] = rng.uniform(-1, 1, self.model.nv)
            mujoco.mj_forward(self.model, r.data)
            r.last_action[:] = rng.uniform(-5, 5, 27)
            core.actions = torch.from_numpy(r.last_action.copy())[None]
            core.dof_pos = torch.tensor(r.data.qpos[r.qpos_ids], dtype=torch.float32)[None]
            core.dof_vel = torch.tensor(r.data.qvel[r.qvel_ids], dtype=torch.float32)[None]
            # MuJoCo free-joint rotational qvel is already body-local.
            core.base_ang_vel = torch.tensor(r.data.qvel[3:6], dtype=torch.float32)[None]
            core.projected_gravity = quat_rotate_inverse(
                torch.tensor(quat[[1, 2, 3, 0]], dtype=torch.float32)[None], torch.tensor([[0.0, 0.0, -1.0]]), w_last=True
            )
            actual = r.observation()
            expected = core.get_observation(to_numpy=True)
            for key in ("state", "last_action", "history_actor"):
                np.testing.assert_allclose(actual[key], expected[key], atol=2e-6, err_msg=f"{step}:{key}")
        self.assertEqual(actual["history_actor"].shape, (1, 348))

    def test_compact_model_preserves_actor_encoder_and_batchnorm(self):
        original_cfg = build_fb_agent(device="cpu", compile=False).model
        raw_cfg = original_cfg.model_dump()
        raw_cfg["archi"]["z_dim"] = 16
        for name in ("actor", "f", "critic", "aux_critic", "discriminator", "b"):
            raw_cfg["archi"][name]["hidden_dim"] = 32
            raw_cfg["archi"][name]["hidden_layers"] = 2
        cfg = type(original_cfg)(**raw_cfg)
        obs_space = spaces.Dict(
            {
                key: spaces.Box(-np.inf, np.inf, (dim,), np.float32)
                for key, dim in {"state": 60, "last_action": 27, "history_actor": 348, "privileged_state": 448}.items()
            }
        )
        full = cfg.build(obs_space, 27)
        full._prepare_for_train()
        with torch.no_grad():
            for norm in full._obs_normalizer._normalizers.values():
                norm._normalizer.running_mean.fill_(0.3)
                norm._normalizer.running_var.fill_(1.7)
        full.eval()
        obs = {k: torch.randn(3, *v.shape) for k, v in obs_space.items()}
        z = full.sample_z(3)
        with tempfile.TemporaryDirectory() as temp:
            full.save(temp)
            slim = InferencePolicy(Path(temp))
            validate_contract(slim, self.robot)
            np.testing.assert_allclose(slim.act(obs, z), full.act(obs, z), atol=1e-6)
            np.testing.assert_allclose(slim.backward_map(obs), full.backward_map(obs), atol=1e-6)
            actor_obs = {k: v for k, v in obs.items() if k != "privileged_state"}
            np.testing.assert_allclose(slim.act(actor_obs, z), full.act(actor_obs, z), atol=1e-6)
            self.assertFalse(any("critic" in k or "target" in k for k in slim.state_dict()))

    def test_checkpoint_snapshot_and_robot_mismatch(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)
            saved = {"env": {"robot_training": self.snapshot, "hydra_overrides": ["robot=casbot02/casbot02"], "root_height_obs": True}}
            (path / "config.json").write_text(json.dumps(saved))
            _, snapshot = load_training_config(path, Path("configs/robots/casbot02.yaml"), Path("unused.pkl"))
            self.assertEqual(snapshot["stiffness"], self.snapshot["stiffness"])
            with self.assertRaisesRegex(ValueError, "does not match"):
                load_training_config(path, Path("configs/robots/g1_29dof.yaml"), Path("unused.pkl"))

    def test_motion_selection_rejects_ambiguous_names_and_invalid_ids(self):
        keys = ["dance1_subject1", "walk1_subject1", "walk1_subject2"]
        self.assertEqual(select_motion(keys, None, None), 1)
        self.assertEqual(select_motion(keys, 0, None), 0)
        with self.assertRaises(ValueError):
            select_motion(keys, None, "walk1")
        with self.assertRaises(ValueError):
            select_motion(keys, -1, None)


if __name__ == "__main__":
    unittest.main()
