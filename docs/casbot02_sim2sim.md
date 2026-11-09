# Casbot02 FB sim2sim

在 UFO 根目录直接运行，使用已训练的 `runs/ufo_fb_casbot02/checkpoint/model/`：

```bash
./run_sim2sim_casbot02.sh --motion-name walk1_subject1 --duration 30
```

会打开原生 MuJoCo 窗口，按 50 Hz 执行策略、200 Hz 运行物理。默认策略和编码器也在
CPU 上运行，不要求 CUDA。窗口内空格暂停/继续，`R` 从选定片段的初始参考状态重置，
关闭窗口或 `Ctrl+C` 结束。

默认播放 `walk1_subject1`，不指定 `--duration` 时播放剩余完整动作。查看全部动作：

```bash
./run_sim2sim_casbot02.sh --list-motions
```

也可按缓存中的编号选择，例如 `--motion-id 28` 为 `walk1_subject1`。
舞蹈和起身等动作可直接按名称选择：

```bash
./run_sim2sim_casbot02.sh --motion-name dance1_subject1 --duration 20
./run_sim2sim_casbot02.sh --motion-name fallAndGetUp1_subject1 --duration 20
```

从动作第 30 秒开始，重复播放后续 10 秒；每次重复会重置机器人及观测历史：

```bash
./run_sim2sim_casbot02.sh --motion-name walk1_subject1 \
  --start-time 30 --duration 10 --loop
```

## 无窗口验证与视频

```bash
./run_sim2sim_casbot02.sh \
  --motion-name walk1_subject1 \
  --duration 10 \
  --headless \
  --no-realtime \
  --save-mp4 \
  --render-size 480 \
  --output-dir runs/casbot02_sim2sim_walk
```

视频左侧为参考动作，右侧为策略驱动的机器人。`--no-realtime` 取消实时等待，仿真步长不变。
`--max-steps 500` 可将总策略步数限制为 500，循环播放也受此限制。
默认输出位置为 `<model-folder>/sim2sim/<motion-name>/`，包含：

- `sim2sim.mp4`：启用 `--save-mp4` 时生成的参考/策略对照视频。
- `summary.json`：步数、基座最低高度、力矩和逐关节平均绝对误差。
- `rollout.npz`：实际 qpos、归一化后的环境动作、逐帧关节误差和基座高度。
- `contexts.npy`：完整动作在 50 Hz 下的潜变量序列（从第 1 个后继状态开始）。

同一输出目录重复执行会更新这些文件；保留多次结果时指定不同的 `--output-dir`。

## 完整 Python 指令

```bash
.venv/bin/python -m humanoidverse.sim2sim \
  --model-folder runs/ufo_fb_casbot02 \
  --robot-config configs/robots/casbot02.yaml \
  --data-manifest configs/data/casbot02_lafan1.yaml \
  --dataset lafan1_casbot02 \
  --motion-name walk1_subject1 \
  --device cpu \
  --duration 30
```

动作数据使用 manifest 的 **inference** 分支，即完整动作缓存
`cache/motion_data/casbot02_lafan1/lafan1_casbot02_full_ufo.pkl`。
也可以用 `--data-path /path/to/full_motions.pkl` 代替 `--data-manifest`。
若要使用另一轮 Casbot02 训练模型，在脚本后覆盖 `--model-folder`。
`--device cuda:0` 可将策略和编码器放到 GPU，物理仍在原生 CPU MuJoCo 中运行。

## 与 G1 推理及训练对齐的部分

实现参考本仓库 `humanoidverse/tracking_inference.py`，以及上游
[G1 UFO-Deploy 的观测历史实现](https://github.com/Roboparty/UFO/blob/c70f4342705a1c4630bd74ae18a830ff8025a4fc/rl_policy/observations.py)。
保留相同的“后继参考状态 → backward encoder → 归一化潜变量 → 确定性 actor”流程。

- 当前状态：`[q-q0, qdot, projected_gravity, 0.25*base_angular_velocity]`，60 维。
- 上一动作：27 维，使用训练环境实际接收的、映射及裁剪到 `[-5, 5]` 的动作。
- 历史：348 维；按 `actions, base_ang_vel, dof_pos, dof_vel, projected_gravity`
  分组，每组前 4 帧，新帧在前。先读取历史，再加入当前帧，重置时全部清零。
- 基座角速度使用基座坐标系；代码先读取世界角速度再转换，避开 MuJoCo 局部刚体
  速度所涉及的惯性坐标系。重力投影同样转换到基座坐标系。
- actor 输出先乘 `normalize_action_to/normalize_action_from = 5` 并裁剪，
  再生成 `q_target = q0 + (0.25 * effort_limit / stiffness) * action`。
- 原 XML 的 motor 被替换为训练时的隐式位置 PD；保持各关节 Kp、Kd、armature、
  摩擦、力矩限幅及碰撞模型，使用 implicitfast 积分器。不会把训练的隐式 PD 改成显式 PD。
- 控制参数优先读取训练目录 `config.json` 中的 `robot_training` 快照。
  传入的机器人 YAML 用于核对机器人与控制关节顺序。
- 推理关闭域随机化、观测噪声和额外命令延迟，初始状态取自选定的参考帧。
- 默认 `--lookahead 1` 对齐当前 G1 tracking 入口；可通过 `--lookahead 8`
  尝试平均未来 8 帧编码，这会改变推理行为。

仅加载 actor、backward encoder 和已训练的观测归一化参数；训练用的 critic、target
网络、判别器和优化器不会占用推理内存。所有所需权重按模块严格核对。

## 复用潜变量与导出 ONNX

可用 `--z-path` 读取 `.npy` 或 joblib `.pkl`。固定目标/奖励的向量可为 `[256]`
或 `[1, 256]`，会保持不变；跟踪序列必须是完整动作的 `[参考控制帧数-1, 256]`。
外部潜变量应来自同一个 checkpoint，并与所选动作的 50 Hz 时间轴对齐。

```bash
./run_sim2sim_casbot02.sh --motion-name walk1_subject1 --duration 10 \
  --headless --no-realtime --export-onnx
```

生成 `policy.onnx` 和 `policy.meta.json`。当前 Casbot02 模型的输入是 691 维
`[state(60), last_action(27), history_actor(348), z(256)]`，输出是 27 维原始 actor 动作。
动作缩放和 PD 位于仿真控制器中；导出的 actor 本身不执行这些控制变换。
元数据中记录控制关节顺序、输入分段、时钟及训练控制参数。

## 验证

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m unittest discover -s tests -p test_casbot02_sim2sim.py
```

测试覆盖编译后的隐式 PD、力矩和惯量、动作映射、带非平凡根旋转的观测及四帧历史、
重置行为、训练配置快照、错误机器人拒绝，以及包含已训练归一化参数的精简模型与
完整 FB 模型之间的 actor/encoder 数值一致性。

实际 checkpoint 的验证结果保存在 `runs/casbot02_sim2sim_validation/`。
其中 `walk/` 和 `dance/` 各包含 10 秒对照视频；`walk_full/` 为完整行走动作的闭环记录。
这些指标用于描述已测试片段，不代表所有动作都已经验证。

2026-11-06 实际验证（CPU 策略、原生 MuJoCo）：

| 动作 | 策略步数 | 仿真时长 | 逐关节 MAE（rad） | 基座最低高度（m） |
| --- | ---: | ---: | ---: | ---: |
| walk1_subject1（完整动作） | 13064 | 261.28 s | 0.17556 | 0.63912 |
| walk1_subject1（视频片段） | 500 | 10 s | 0.14313 | 0.79674 |
| dance1_subject1（视频片段） | 500 | 10 s | 0.14227 | 0.78216 |

完整动作正常到达结尾，未触发数值异常。另验证了固定 5 步片段循环执行 12 步、
发生 2 次重置，确认外部潜变量、直接数据路径、循环索引和历史重置能够一起工作。
5 项单元测试通过，导出的 ONNX 通过 `onnx.checker` 结构检查。

## 2026-11-06 新电机参数训练

新训练输出使用 `/srv/liuchenxu/UFO/runs/ufo_fb_casbot02_20261107_nopalms_4096`。训练完成后，显式选择新模型与新数据：

```bash
./run_sim2sim_casbot02.sh \
  --model-folder /srv/liuchenxu/UFO/runs/ufo_fb_casbot02_20261107_nopalms_4096 \
  --data-manifest configs/data/casbot02_lafan1_20261107_nopalms.yaml \
  --motion-name walk1_subject1 --duration 30
```

本页上方 2026-11-06 的行走/舞蹈验证表是旧电机参数模型的结果，不能用来表示新模型的性能。
新电机参数与原始 CSV 的使用决定见 [训练说明](casbot02_fb_training.md)。


## 2026-11-07 无手掌模型更新

训练和 sim2sim 使用的 `casbot02_primitives.xml` 已与用户提供的 SuperMimic XML 逐字节对齐。
同步移除双侧手掌，更新腕部惯量、腕部碰撞胶囊和小腿碰撞胶囊；27 个关节、位置限位、
转子惯量、120 N·m / 12.56 rad/s 等电机配置均保持已确认值。
使用同一份原始 CSV，另建 `casbot02_lafan1_20261107_nopalms` 动作缓存，正式训练从头启动，
旧模型保留。源 XML、校验和与备份见 `humanoidverse/data/robots/casbot02/xml_revision_20261107.json`。
