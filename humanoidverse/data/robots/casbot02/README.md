# Casbot02 27 DOF asset

Copied from the user's local ScaleBFM/ScaleTrack Casbot02 assets. The copied
meshes and adapted MJCF are independent of that project's Python environment.
See `source_manifest.json` for the exact source and changes.

The source MJCF and URDF agree on common body transforms at zero joint angles.
Position limits now follow the user-confirmed hardware limits from
`/home/casbot/Downloads/casbot_02_27dof.xml` (2026-10-29), for both mesh and
primitive training models. Earlier versions used the narrower ScaleTrack URDF
elbow limits. Motors, PD gains, armature, and action scale
are configured in `protomotions/robot_configs/casbot02.py` and the Casbot02
tracking experiment. Embedded terrain/lights are removed. Passive joint
damping and frictionloss are zero, with active PD damping set by the config.

The limit alignment restores both elbows to `[-2.26, 0]` rad, both knees to
`[0, 2.53073]` rad and the right hip pitch to `[-1.91986, 1.57]` rad. All other
joint ranges are identical. The 0.9 soft-limit factor remains in effect for
reset clipping and the joint-limit reward. The eight SDK `.data` motions were
reconverted from their original sources with these hard limits; widening the
XML alone cannot restore angles already clipped in older NPZ files.
Existing checkpoints/deployment exports retain their learned behavior, and a
running training process does not reload edited XML or motion data. Use a new
run name for training with the revised model and references.

The training default joint pose is defined by `default_joint_pos` in
`motor_parameters.json`, which `RobotProfile.entity_cfg()` turns into the
model's initial keyframe. This pose was overridden with the user's requested
values on 2026-10-26; the PD gains and source MJCF are unchanged. The arm names
`upper_left1_joint` / `upper_right1_joint` in that request correspond to
`upper_left_1_joint` / `upper_right_1_joint` in this MJCF. The unspecified waist
joint remains at zero.

This changes both the `q - q0` observation offset and the
`q_des = q0 + action_scale * action` action offset for subsequent training.
Existing checkpoints were trained with the previous offset: use their original
matching environment/bundle when playing, resuming or exporting them. Existing
deployment manifests and exported scenes were not modified by that pose-only change.

## HANDOFF control-parameter override (2026-10-26)

Kp, Kd and torque limits now follow the actuator declarations in
`/home/casbot/Desktop/HANDOFF/src/wbc_mjlab/casbot02_constants.py`; the source
hash and mapping details are recorded in `motor_parameters.json` under
`control_parameters_override`. Action scales use the same formula as that file:
`0.25 * effort_limit / stiffness`. Other physical properties and motor delays
retain this project's existing configuration.

## Command latency and floor friction (updated 2026-10-28)

New Casbot02 training enables `env.actuator_delay` by default. Position targets
pass through mjlab's builtin actuator delay buffer before implicit PD; joint
feedback and the policy's previous environment-input action are not delayed. At 200 Hz
physics, each environment draws one lag uniformly from 0, 1, 2 substeps (0/5/10 ms)
on the first physics step of an episode and holds it until reset
(`env.actuator_delay_mode=episode`). The hold uses MorphMimic's update period,
`2**40` physics steps, with `delay_hold_prob=0`. The range stays 0/5/10 ms.
The 50 Hz policy clock and observation/action dimensions are unchanged.

All 27 matching builtin actuators share one command bus per environment in
mjlab. `--env.actuator-delay-mode step` keeps the previous draw on every physics
step. `--env.no-actuator-delay` disables the added latency. G1's actuator
configuration is unaffected.

The flag and mode are stored in new configs/checkpoints. Benchmark, drift
diagnosis and play turn the delay off, matching MorphMimic's eval mode; a
checkpoint trained with latency is still scored without it. A missing flag in
old checkpoints means training had no latency; an enabled flag without a mode
preserves the old episode-constant training latency. Native sim2sim playback ignores that flag and sends the current position
target, including bundles exported while training delay was enabled. This is training/simulation latency; no artificial wait is added to
the real robot controller or existing deployment bundles. Changing the flag
requires a new training run. To resume a legacy run, match its original delay
flag/mode and floor (e.g. `--env.actuator-delay-mode step` for a per-step run,
`--env.training-floor-friction 1.0`, or `--env.no-actuator-delay` if disabled).
An already running process does not pick up this code change.

Training with domain randomization now uses floor sliding friction 0.3
(`env.training_floor_friction`), following MorphMimic. Foot friction remains
uniform in [0.3, 1.2], sampled once at startup, as is the encoder bias in
[-0.01, 0.01] rad, following the local BeyondMimic event timing. These values
persist across episode resets. COM offsets remain startup-only. Equal-priority MuJoCo
contacts take the maximum of the floor/foot friction; lowering the floor
allows the low-friction portion of the foot distribution to affect contact.
With domain randomization disabled, the evaluation floor and feet remain 1.0.

The SuperMimic paper (Sections 3.1-3.2, Appendix A, Tables A.1-A.3) describes
friction/calibration/COM as a per-episode context but does not give explicit
startup-versus-reset event settings. It explicitly stores executed noisy actions
in replay. It lists previous action and action smoothness without specifying
whether execution noise is included in those two simulator quantities. The
new collection behavior follows the colleague/MorphMimic convention below;
Table A.1's missing sensor noise is not proof about execution noise.

## Collection noise and light-leg armature (2026-10-28)

New training defaults to `env.action_noise_mode=collection`: the collector adds
Gaussian noise N(0, 0.1^2) to normalized actions before `env.step`. The action
term's own noise is zero, so execution, previous-action observations, action-rate
penalties and replay share the same noisy action. No additional action clamp
was introduced. Imagined rollouts and deterministic evaluation do not add this
execution noise. The replay field `a_clean` retains its old name for interface
compatibility, but stores the already-noisy environment input in this mode.
`--env.action-noise-mode execution` retains the old execution-only noise path;
resume rejects a silent change between these semantics.

The armature of `leg_l3/l5/l6_joint` and `leg_r3/r5/r6_joint` is now 0.029 kg*m^2,
matching `yuanxz/yuanxz` branch `supermimic_casbot02_22dof` at `f38a49a`.
Both the motor table and source MJCF are updated; Kp, Kd, effort limits, action
scales, waist and arm inertias are unchanged. Provenance is recorded under
`armature_override` in the motor table. These changes apply to new training and
future exports; existing deployment bundles and datasets are unchanged. Start a
new run to train with the new physical parameters and noise semantics.

The source articulation only lists 22 arm/leg joints. This project retains
27 DOF: the waist uses its separately declared waist actuator, and wrist axes
6/7 on both arms use the declared light-arm actuator's parameters. These four
wrist assignments are an extension to the source file, not explicit entries
in that file.

The current walk1 hardware manifest has these new control parameters, while
retaining the original actor, reference data and legacy q0. Its old checkpoint
was not trained with the new parameters. A 1 kHz explicit-PD CPU closed-loop
test failed at 9.56 seconds; retrain and revalidate before treating this as a
working deployment policy. The original hardware bundle is backed up under
`outputs/casbot02_handoff_parameters/before/hardware_walk1/`.

## G1-style primitive collisions (2026-10-28)

New Casbot02 training uses `env.collision_model=primitive`. The separate
`xml/casbot02_primitives.xml` replaces the other active collision meshes with
capsules and one head sphere, while preserving visual meshes, joints, body
frames and explicit inertias. Each `leg_pelvic_pitch_link` has a hip capsule covering its visual mesh, with
the outer end pulled 5 cm medially so the zero pose does not touch the arms. Each foot
uses the same mesh collision as `casbot02.xml`: `left_foot_collision` and
`right_foot_collision` on `left/right_leg_ankle_roll_link`, with condim=3,
conaffinity=15, solref and friction taken from that file. Other contacts use
condim=1. Non-adjacent self contacts remain enabled. Parent-adjacent pairs,
the two hip capsules against each other and the torso, and each shin capsule
against its foot mesh are excluded so those approximations do not generate a
resting self-collision. The preparation pose has no false self intersections.

`primitive_collision.json` records dimensions, exclusions and the source hash.
Regenerate with `uv run python scripts/build_casbot02_primitive_collision.py`.
This is a new approximate contact model, not an identical convex hull. The
original mesh XML remains available with `--env.collision-model mesh`.
Old checkpoints without this field keep mesh collision in play/benchmark/export;
resuming with a different collision model is rejected. New primitive exports include the two source foot meshes in the shared
foot-friction interface.

For primitive training with >=32 environments, defaults are nconmax=48,
njmax=256; small viewers retain nconmax=128. Mesh Casbot02 stays at 128/500,
and G1 stays at 35/250. Full reference auditing, random stress and a 100-update
4096-environment/batch-4096 training test passed without capacity overflow or
OOM, with no Torch cache trimming. This is bounded validation, not proof of
arbitrary long-run or arbitrary-pose capacity. See `docs/CASBOT02_SIM_CAPACITY.md`
and `outputs/casbot02_primitive_collision/` at the repository root.


## UFO local motor revision (2026-11-06)

Current UFO training uses the user-supplied ENCOS URDF only for effort/velocity
limits, with the explicit user correction of 120 Nm / 12.56 rad/s for both legs
3/5/6 and waist taking priority; user-prescribed reflected armatures; and 10-Hz, damping-ratio-2
PD gains. Position ranges in both imported MJCFs are retained exactly. The
upstream historical motor values described above are superseded locally by
`motor_parameters.json` and `motor_revision_20261106.json`. See
`../../../../docs/casbot02_fb_training.md` for the active training/data settings.


## Active training XML revision (2026-11-07)

`xml/casbot02_primitives.xml` is an exact copy of the user-updated SuperMimic
primitive XML. Both uninstalled palms are removed; wrist inertials and wrist/shin
collision capsules follow that source. All 27 joint definitions, positional limits
and reflected armatures are unchanged. Motor caps and PD gains retain the
2026-11-06 user correction. See `xml_revision_20261107.json` for source hashes
and the pre-update backup. `xml/casbot02.xml` and `primitive_collision.json`
remain historical assets; training and native sim2sim use the active primitive XML.
