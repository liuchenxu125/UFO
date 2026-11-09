# Casbot02 FB 训练（2026-11-07 无手掌模型）

## 当前任务

```bash
cd /srv/liuchenxu/UFO
./run_train_casbot02_fb.sh
```

单卡、4096 个环境、192,000,000 全局环境样本、2,560,000 replay 容量，
每次触发执行 64 次更新，latent 每 100 步更新。项目迁移后结果保存至
`/srv/liuchenxu/UFO/runs/ufo_fb_casbot02_20261107_nopalms_4096`，运行缓存写入
`/srv/liuchenxu/UFO/cache/runtime_casbot02_20261106`。
训练进度见输出目录内的 `launch.log`、`train_log.txt` 与 `checkpoint/train_status.json`。
2026-11-07 已迁移至 `/srv/liuchenxu/UFO`；正式结果与运行缓存均已移出 `/tmp`。

完整指令：

```bash
UFO_CACHE_DIR=/srv/liuchenxu/UFO/cache/runtime_casbot02_20261106 ./run_train.sh \
  --agent fb \
  --robot-config configs/robots/casbot02.yaml \
  --data-manifest configs/data/casbot02_lafan1_20261107_nopalms.yaml \
  --gpu-ids single \
  --num-envs 4096 \
  --num-env-steps 192000000 \
  --buffer-size 2560000 \
  --num-agent-updates 64 \
  --update-z-every-step 100 \
  --work-dir /srv/liuchenxu/UFO/runs/ufo_fb_casbot02_20261107_nopalms_4096
```

脚本末尾参数可覆盖默认值。同一输出目录已有 checkpoint 时，UFO 会尝试恢复；
本轮使用新目录从头训练。此前使用 60 N·m 的任务已停止，输出保留在
`/srv/liuchenxu/UFO/runs/ufo_fb_casbot02_20261106_4096`。上一轮 120 N·m 模型保留在 `runs/ufo_fb_casbot02_20261106_4096_120nm`；本轮不加载旧 checkpoint。

## 电机参数

力矩上限和速度上限来自：
`/home/liuchenxu/Desktop/SuperMimic-Reproduced/src/supermimic/robots/assets/casbot02/CASBOT02_ENCOS_7dof_shell_20251015.urdf`。
URDF SHA-256：`52433d5928295c3c92e013f69785ad2854a54d260e6ece14cce8bb65974e794f`。
用户随后明确修正：双腿 3/5/6 号及腰关节采用 120 N·m、12.56 rad/s，优先于 URDF。
这 7 个关节的 action_scale 随力矩重算，Kp/Kd 与惯量保持本轮原值。
按 URDF 的 child link 与 MJCF 的 joint body 匹配，27 个控制关节一一对应。
URDF 中的头部、手指关节不增加到当前的 27 DOF 模型。

现有两个 MJCF 的关节位置 `range` 均保持原值，包括与该 URDF 不同的肘和右腕限位。
用户给出的转子/反射惯量写入 MuJoCo `armature`；本轮腕部质量和刚体惯量随用户 XML 更新。
沿用项目 10 Hz 自然频率、阻尼比 ζ=2：

```text
ω = 2π × 10
Kp = I × ω²
Kd = 2 × ζ × I × ω
逐关节 action_scale = 0.25 × effort_limit / Kp
```

| 关节组（左右相同） | armature (kg·m²) | 力矩 (N·m) | 速度 (rad/s) | Kp | Kd | action_scale |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 腿 1/2/4 | 0.070000 | 150 | 14 | 276.348923 | 17.592919 | 0.135698 |
| 腿 3/5/6、腰 | 0.065797 | 120 | 12.56 | 259.756144 | 16.536590 | 0.115493 |
| 臂 1/2/4 | 0.032980 | 75 | 12.2 | 130.199821 | 8.288778 | 0.144009 |
| 臂 3/5/6/7 | 0.024530 | 36 | 9.3 | 96.840558 | 6.165061 | 0.092936 |

UFO 先将 actor 原始动作乘 5 并裁剪到 `[-5,5]`，再用上表的逐关节系数生成
`q_target = q0 + action_scale * action`。配置中的全局 `training.control.action_scale=0.25`
是上述公式的基础系数；执行时按每个关节的 effort/Kp 自动计算。

控制实现仍使用 `BuiltinPositionActuatorCfg` 隐式位置 PD，物理步长 0.005 秒、
控制周期 0.02 秒。速度上限作为隐式位置执行器的机器人元数据保留，
没有额外引入 DC 电机力矩—速度曲线。被动阻尼和摩擦为零，软限位系数为 0.9。
训练命令延迟按每回合抽取 0/1/2 个物理子步；`--disable-dr` 会将其关闭。

更新文件：

- `configs/robots/casbot02.yaml` 与 Hydra 的 Casbot02 YAML。
- `humanoidverse/data/robots/casbot02/motor_parameters.json`。
- `xml/casbot02.xml` 和 `xml/casbot02_primitives.xml` 中的逐关节 armature。
- `motor_revision_20261106.json` 记录完整参数、URDF 映射与原始位置限位。

改动前的配置和资产备份位于
`runs/casbot02_parameter_update_20261106/before/`。
`ufo_source_manifest.json` 保留原始来源哈希，在 `local_overrides` 中另记当前改动。
原始资产来自 SuperMimic-Reproduced 的 `e8af1d8ff6a3771815f335f3940e6800336a40fc`。

可重算并查看参数；去掉 `--apply` 时只打印：

```bash
.venv/bin/python scripts/update_casbot02_motors.py \
  --urdf /home/liuchenxu/Desktop/SuperMimic-Reproduced/src/supermimic/robots/assets/casbot02/CASBOT02_ENCOS_7dof_shell_20251015.urdf \
  --apply
```

## 新数据

源目录 `data/lafan1_casbot02/`：41 个 CSV，共 266,927 帧、约 2.472 小时。
每帧为根位置 xyz（米）、根旋转 xyzw、27 个 MJCF 顺序的关节角（弧度）。
按 30 Hz 导入，MotionLib 在 50 Hz 时钟下插值。
包含新文件 `Form_1_stageii_g1.csv`，其实际数据格式为 34 列、27 DOF，已一并导入。

新 manifest：`configs/data/casbot02_lafan1_20261107_nopalms.yaml`。
10 秒长度/步幅切片，保留至少 1 秒的短片段，共 910 个片段。
独立缓存目录 `cache/motion_data/casbot02_lafan1_20261107_nopalms/` 包含完整动作、训练动作及
`source_audit.json`；旧缓存目录不覆盖。

数据按用户明确选择原样导入，不做关节裁剪或地面修正。审计记录：

- 所有数值有限，根四元数有效。
- 999 帧存在超过 1e-4 rad 的关节位置越界，最大 0.01556345 rad。
- 按当前训练碰撞模型，10,291 帧地面穿透超过 2 cm，最深约 0.4145 m。
- 审计包含每个源 CSV 的 SHA-256，便于核对本轮训练所用数据。

重新审计（只读数据）：

```bash
.venv/bin/python scripts/audit_casbot02_data.py --ground --output cache/motion_data/casbot02_lafan1_20261107_nopalms/source_audit.json
```

## 验证

```bash
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  .venv/bin/python -m unittest discover -s tests -p 'test_casbot02*.py'

OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MUJOCO_GL=egl \
  .venv/bin/python scripts/smoke_casbot02_fb.py \
  --output-root /tmp/casbot02_nopalms_validation
```

12 项单元/回归测试通过，核对编译后的 PD、力矩、惯量、速度配置、XML 限位、
G1 回归、FK、观测历史及精简推理模型。GPU 验证完成 16 环境、1024 样本、
3 次优化器更新，有限值与 checkpoint 写入检查通过。
本次验证结果：`/srv/liuchenxu/UFO/runs/casbot02_xml_update_20261107/validation/gpu_smoke.json`。
首次尝试写入 `/home` 时因空间不足而保存失败，该轮新建的失败输出已清理；
正式任务现保存至项目内的 `runs/`。

## 本轮训练后的 sim2sim

```bash
./run_sim2sim_casbot02.sh \
  --model-folder /srv/liuchenxu/UFO/runs/ufo_fb_casbot02_20261107_nopalms_4096 \
  --data-manifest configs/data/casbot02_lafan1_20261107_nopalms.yaml \
  --motion-name walk1_subject1 --duration 30
```

推理读取该轮保存的控制参数快照。更多播放、视频与 ONNX 参数见
[Casbot02 sim2sim 说明](casbot02_sim2sim.md)。


## 2026-11-07 无手掌模型更新

训练和 sim2sim 使用的 `casbot02_primitives.xml` 已与用户提供的 SuperMimic XML 逐字节对齐。
同步移除双侧手掌，更新腕部惯量、腕部碰撞胶囊和小腿碰撞胶囊；27 个关节、位置限位、
转子惯量、120 N·m / 12.56 rad/s 等电机配置均保持已确认值。
使用同一份原始 CSV，另建 `casbot02_lafan1_20261107_nopalms` 动作缓存，正式训练从头启动，
旧模型保留。源 XML、校验和与备份见 `humanoidverse/data/robots/casbot02/xml_revision_20261107.json`。
