"""Bounded GPU integration check: Casbot evaluation, FB updates and checkpoint.

Run from the repository root with .venv/bin/python scripts/smoke_casbot02_fb.py.
Uses the production FB architecture with a small batch, two motion clips and
16 environments. Writes to a new temporary run directory on each invocation.
"""
from pathlib import Path
import argparse
import json
import tempfile

import joblib
import torch

from humanoidverse.train import build_ufo_mjlab_config
from humanoidverse.utils.motion_data import prepare_motion_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-manifest", default="configs/data/casbot02_lafan1_20261107_nopalms.yaml")
    parser.add_argument("--output-root", type=Path, default=Path("runs/casbot02_validation"))
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("This integration check requires a CUDA GPU")
    torch.set_num_threads(2)
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="fb_", dir=output_root)).resolve()
    manifest = prepare_motion_manifest(args.data_manifest)
    motions = joblib.load(manifest.train_data_paths[0])
    keys = [k for k in motions if k.startswith("walk1_subject1")][:2]
    assert len(keys) == 2, "Expected two walk1_subject1 clips"
    data_path = work / "smoke_motions.pkl"
    joblib.dump({k: motions[k] for k in keys}, data_path)
    del motions
    cfg = build_ufo_mjlab_config(
        device="cuda:0", work_dir=str(work), num_envs=16, num_env_steps=1024,
        seed=4728, use_wandb=False, wandb_run_name=None,
        robot_config="configs/robots/casbot02.yaml", data_path=str(data_path),
        buffer_size=4096, buffer_device="cpu", num_agent_updates=1,
        checkpoint_every_steps=1024,
    )
    cfg = cfg.model_copy(update={
        "agent": cfg.agent.model_copy(update={
            "compile": False,
            "train": cfg.agent.train.model_copy(update={"batch_size": 32}),
        }),
        "num_seed_steps": 256,
        "update_agent_every": 256,
        "log_every_updates": 256,
        "checkpoint_buffer": False,
        "nonfinite_check_model_every_updates": 1,
        "nonfinite_check_rollout_every_local_steps": 16,
    })
    workspace = cfg.build()
    workspace.train()
    assert workspace._optimizer_steps >= 2, "Smoke must exercise actual FB optimizer updates"
    status = json.loads((work / "checkpoint/train_status.json").read_text())
    assert status["global_time"] == cfg.num_env_steps
    assert status["optimizer_steps"] == workspace._optimizer_steps
    report = {"optimizer_steps": workspace._optimizer_steps,
              "env_steps": cfg.num_env_steps, "num_envs": 16,
              "batch_size": 32, "compile": False,
              "motion_clips": keys, "work_dir": str(work)}
    (work / "validation.json").write_text(json.dumps(report, indent=2) + "\n")
    print("CASBOT02_FB_SMOKE_PASSED", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
