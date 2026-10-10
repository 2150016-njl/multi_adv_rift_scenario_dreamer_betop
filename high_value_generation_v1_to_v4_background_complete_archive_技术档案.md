# 高价值交互状态与 Background 状态生成：V1-V4C 完整技术档案

> 最后整理：2026-10-10。本文是下一轮分析/实现的独立交接文档。
> 它只记录已观察到的事实、可复现实验和明确的推断；不会把训练 loss、
> oracle 指标、RIFT 颜色变化或未 spawn 的运行误写成模型成功。

## 1. 项目目标、输出契约与不可违背的语义

### 1.1 真正的目标

在 RIFT/CARLA 闭环中，以当前 ego、ego route、局部 HD Map 和交通上下文为条件，
生成两辆位于 ego 当前 64m FOV 前方或侧前方的关键周车初始 raw state。它们必须：

1. 可在 CARLA 正常 spawn 并由 Traffic Manager 正常演化；
2. 与道路拓扑、地图、朝向、速度和上下文一致；
3. 与 ego 存在强的 **route-level conflict interaction**，而非只是在空间上靠近；
4. 在正常的 `CBVs_recog()` 机制下，若确实成为 CBV，则被 RIFT 识别和接管。

高价值目标不是“让车变绿”，也不是最大化 ego 碰撞或降低 ego driving score。
orange->green 是正常 RIFT 识别后的观察指标；它不能作为训练标签、线上筛样规则、
状态修改规则或 reward。

### 1.2 raw-state 契约

HV 和普通 BV 的车辆状态统一为：

```text
[x_local_m, y_local_m, heading_rad, speed_mps, length_m, width_m, exists]
```

- 生成器的 `x/y/yaw/speed` 必须原样用于 CARLA spawn。
- road waypoint 只能提供 terrain `z`，不能把 x/y 投影到车道、替换 yaw、替换速度、
  clamp 输出或按 RIFT 分数 rerank。
- 任何 clamp/lane snap 的数值只能作为诊断 ablation，不能当作模型性能。
- 模型输出若自然无法 spawn，属于模型失败，不应由兜底逻辑掩盖。

### 1.3 颜色、控制权与数据角色

| 初始来源 | 初始显示/控制 | 被正常 `CBVs_recog()` 识别后 |
| --- | --- | --- |
| Scenario Dreamer ordinary BV | 蓝色，Traffic Manager | 紫色，RIFT/Pluto |
| HV generator 的关键 BV | 橙色，Traffic Manager | 绿色，RIFT/Pluto |

- RIFT label 只用于离线采集、target 选择、训练监督和评估。
- RIFT 的 post-spawn audit 只能观察，不得删掉未立即 positive 的车、不得改状态、不得直接注册 CBV。
- 在线生成仍固定输出两辆 HV。通用 graph 层技术上可分配不同节点数，但所有已训练 checkpoint
  的验证任务均为 `num_generate_agents=2`；不能把它说成自适应数量。

## 2. 公共环境、核心模型与正确理解

```bash
export ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
export PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
export HV_ROOT=$ROOT/data/hv_init
export AE_CKPT=$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_autoencoder_nuplan/last.ckpt
export LDM_CKPT=$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_ldm_large_nuplan/last.ckpt
export PYTHONPATH=$ROOT/RIFT:$ROOT/scenario-dreamer:${PYTHONPATH:-}
```

在线 `runuser -u riftuser -- bash -lc ...` 子 shell 还必须设置 CARLA PythonAPI、CARLA egg、
`$ROOT/scenario-dreamer` 和 `$ROOT/RIFT` 到 `PYTHONPATH`。缺失时会发生
`ModuleNotFoundError: high_value_init`，而 ego benchmark 仍可能跑完；该结果不能被当成 HV 实验。

### 2.1 共享的 Scenario Dreamer 架构事实

当前 HV V1-V4C 和 background 线均基于 NuPlan Scenario Dreamer：

```text
raw scene -> frozen AutoEncoder encoder -> latent diffusion model (LDM)
         -> frozen AutoEncoder decoder -> CARLA raw state
```

- `RouteConditionedHighValueLDM` 将 AutoEncoder 设为 eval 且 `requires_grad_(False)`；训练器返回的
  可优化参数是 LDM，不是 AE/decoder。
- 训练中的 decoded-state loss 是：真 target latent 加噪后预测 x0，再经**冻结 decoder**得到状态。
- 在线 `generate()` 是 condition-only reverse diffusion，之后经同一个冻结 decoder 输出 raw state。
- 因而“只用 AE encode->decode”只能在已知真实 target latent 时重建；它不能从 condition 生成缺失 target。
- 训练日志里的 `val_decoded_position_l1` 在**归一化状态空间**计算，不是米单位，也不是完整 reverse-diffusion
  采样误差，更不是 spawn/route-conflict 指标。

模型内部用于 raw physical audit 的边界是：

```text
speed [0, 30] m/s, length [2, 8] m, width [1.4, 3.2] m
```

## 3. 版本地图与资产总览

| 版本 | 实际模型任务 | 主要变化 | 当前结论 |
| --- | --- | --- | --- |
| V1 / Phase5 | 固定 2 HV inpainting | 首个 HV LDM 微调 | raw 采样误差很大；基线 |
| V2 / Phase7 | 固定 2 HV sparse inpainting | 54,936 多 seed 快照、稀疏 context | 已有最强的 HV 闭环基线，但双车强交互率仍低 |
| V3 / Phase8 | V2 上微调 | HD Map enrichment + 弱 lane/clearance loss | 部分离线字段改善，未证明闭环提升 |
| V4A | 前方 HV target | `forward_min_x=0`、弱几何约束 | 数据 split 弱、闭环弱；不能替代 V2 |
| V4B | data/split/control 版本 | group-stratified、interaction score/region 诊断 | 没有形成可信已训练 V4B checkpoint |
| V4C-C0 | V4C 数据对照 | target-only route sidecar 仅作为 metadata | 不是 route-loss 模型 |
| V4C-C2 | V3 上训练 | target-only route progress/anchor losses | route proxy 改善，但 raw validity 变差、clean online 无 spawn |
| Background-B1 | 普通 BV directional inpainting | 后方/当前 context -> 前方普通 BV，最大 8 target | 单步 raw 评估不足以上线 |

应保留、可用于消融的 checkpoint：

```text
hv_route_inpaint_ldm_sparse_phase5/best.ckpt
hv_route_inpaint_ldm_sparse_phase7_v2_full_b32_ft_from_phase5_20260922/best.ckpt
hv_route_inpaint_ldm_sparse_phase7_v2_full_b32_scratch_20260922/best.ckpt
hv_route_inpaint_ldm_sparse_phase8_v3_submission_ft_20261005/best.ckpt
hv_route_inpaint_ldm_v4a_23k_exploratory_gpu3_20261006/best.ckpt
hv_route_inpaint_ldm_v4c_c0_from_v3_20261010/best.ckpt
hv_route_inpaint_ldm_v4c_c2_route_sync_from_v3_20261010/best.ckpt
scenario_dreamer_background_directional_existing_raw_gpu2_20261006/best.ckpt
```

V4B 不是一个应被引用为“已训练模型”的版本：曾经的 V4B build 有 train split 为空的 bug，
且没有可接受的 V4B model result。

## 4. HV 数据：采集、合并、构建和质量结论

### 4.1 Phase5/V1 递归旧数据

- 旧递归目录：`rift_labeled_all_recursive_20260919_133508`。
- `find -L` 统计为 70,329 个 symbolic-linked JSON；不能用普通 `find` 漏计链接。
- 早期构建 dataset 保留 13,706 条：train 10,939、val 1,518、test 1,249。
- 早期真实 positive 快照的平均 observed actor 数约 3.63、平均 RIFT-positive 数约 2.57。

历史问题：最早训练命令指向不存在的 `inpainting_sparse_all_recursive`，这是 dataset path 错误，
不是模型训练失败。`PosixPath is not JSON serializable` 是 dataset summary 写文件错误，后续已修复。

### 4.2 Phase7/V2 多 seed 采集与 union

完成目录：

```text
rift_labeled_phase7_v2_uncovered_routes_seed0_7
```

实际完成：54,936 snapshots、220 routes、seed 0--7；所有 routes 至少有 10 snapshots。
使用多个 worker、互不重叠 routes、独立 CARLA/TM port。典型采集入口：

```bash
cd $ROOT/RIFT
CUDA_VISIBLE_DEVICES=0 $PY scripts/collect_high_value_init_scenes.py \
  --output_dir $COLLECT_ROOT/worker0_seed0 \
  --routes rift/scenario/route/bench2drive220.xml \
  --routes_subset <disjoint_route_ids> --seed 0 --repetitions 1 --no_resume \
  --base_port 32000 --base_tm_port 52000 \
  --rift_hd_map_dir $ROOT/RIFT/data/map_data/HD-Map \
  --rift_speed_limits_dir $ROOT/RIFT/data/speed_limits/Speed-Limits \
  --sample_interval_s 1.0 --actor_radius 80 --map_radius 64 --max_frames_per_episode 200
```

V2 数据构建实际使用旧递归数据与 Phase7 数据的 symlink union。最终 B32 V2 train/val 为
58,168 / 6,431；稀疏 context 样本约 78.6%，observed <=2 约占 train 66.7%。
这改善了稀疏上下文覆盖，但没有改变“固定两个 HV target”的定义。

### 4.3 V3 HD Map 数据集

输入仍为 54,936 Phase7 snapshots。旧 snapshot 没有可用 local lane 序列；最初 V3 直接用 snapshot lane
导致全部 lane reject。后续使用 RIFT HD Map enrichment，完整输出：

| 项目 | 数量 |
| --- | ---: |
| processed snapshots | 54,936 |
| HD Map errors | 0 |
| raw records | 57,580 |
| lane rejected | 1,565 |
| quality rejected | 2,125 |
| dedup removed | 16,702 |
| train / val / test | 30,468 / 4,558 / 3,727 |

Town12 在 train 中占主导；Town10HD 仅在 val、Town15 大量落入 test。拓扑为 connector/junction
38,185、multi-lane 12,485、opposing/parallel 6,905、single/sparse 仅 5。故 V3 全局均值无法证明跨 town 泛化。

### 4.4 V4A 高价值前方数据

V4A 相关 source：

```text
rift_labeled_v4a_union_phase7_r4
rift_labeled_v4a_recovery_35k_summaryonly_16workers_20260928_r4
rift_labeled_v4a_union_phase7_r4_plus_recovery35k_retry2_20261005
```

两 source 合并为 109,937 JSON：旧 V4A union 56,527，recovery 53,410。
recovery 数据全部是 forward-positive collection：53,410/53,410 snapshot 有至少两个前方 RIFT-positive；
Town06 19,208，Town05 7,983，Town10HD 6,050 等，Town12 不在该 recovery source。

V4A exploratory dataset 通过结构审计，23,667 records：

| split | records | 明显问题 |
| --- | ---: | --- |
| train | 21,199 | Town06 7,272、Town04 5,006，分布不均 |
| val | 541 | 太小，Town06 199，占比高 |
| test | 1,927 | Town10HD 834；`opposing_or_parallel` 仅 1 |

因此 V4A 的 val/test 不是可靠、均衡的跨 Town/拓扑比较集。Town01 总计只有 4 条，也绝不能作为泛化证据。

### 4.5 V4B 构建经历与数据问题

V4B 的目的本来是：在不重新采集的前提下，用 episode group split、Town/拓扑/目标区域 pair 的统计来改善
V4A 的不良 split，并用 interaction score 为强交互 target 加权。

第一个实现发生严重 split bug：`train=0, val=11,344, test=11,648`。原因是 group 分配逻辑错误，
不是数据不足；该 output 不可训练。后续修复 build 后，全局 `22,992` 记录或 route-supervision 过滤后的
`20,406` 记录可以产生非空 split。V4B 本身未得到可信训练/评估结论。

V4B/V4C builder 中 `source_file_sha1` 实际为：

```python
hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:12]
```

它是**路径 hash，不是文件内容 hash**。所以它不能验证相同 JSON 内容是否跨 split 出现；后续必须用 canonical
snapshot JSON/content hash 或采集期 immutable UUID 做该项审计。

### 4.6 V4C route-supervision 数据

V4C 的核心不是把 RIFT route trace 作为线上输入，而是在离线 target 上建立 target-only route sidecar：

```text
target state + target preview/conflict waypoint + target trace_route
-> ego/target 到 conflict anchor 的 route-relative quantity
```

全量 manifest：22,992 records、45,984 target actors。sidecar 结果：

| 项目 | 数量 |
| --- | ---: |
| manifest targets | 45,984 |
| route targets written | 43,611 |
| missing | 2,373 |
| preview/conflict waypoint missing | 770 |
| trace too short | 1,603 |
| sidecar coverage | 94.84% |

最终可用于 V4C、两个 target 都有 route supervision 的数据：20,406。
严格 episode-group split 后：train 16,362（108 episodes）、val 2,005（36）、test 2,039（43）；
无 episode leakage。为抑制 Town06 dominance，global balance 删除 555 条 Town06 records。

最终 split 的 connector/junction 仍明显偏多：train 11,259/16,362，val 1,429/2,005，test 1,400/2,039；
opposing/parallel 分别只有 216、51、28。这个长尾限制必须随所有 V4C 结论一起报告。

重要限制：`min_target_interaction_score=0`、`min_pair_interaction_score=0` 的 all-positive V4C 数据
没有把“两个 target 均强 route interaction”变成硬数据条件；interaction strength 只改变训练权重。
所以 V4C 不等价于“强交互专用数据集”。

## 5. HV 模型训练：配置、结果与含义

### 5.1 V1 / Phase5

任务是固定两 target 的 inpainting。Phase5 raw evaluation：

| metric | raw | clamp ablation |
| --- | ---: | ---: |
| position L2 | 15.70 m | 16.29 m |
| heading absolute | 0.189 rad | 0.209 rad |
| speed L1 | 2.61 | 2.49 |
| size L1 | 1.18 | 1.21 |
| physical invalid rate | 20.5% | 0% |

clamp 仅把错误藏在后处理内，并使位置/heading 更差；不能被视为改进。

### 5.2 V2 / Phase7

V2 B32 两个主要 run：

| run | 初始化 | best epoch | best normalized val decoded position L1 | val decoded state loss |
| --- | --- | ---: | ---: | ---: |
| scratch | base LDM | 12/60 | 0.14995 | 0.57911 |
| fine-tune | Phase5 best | 12/40 | 0.14638 | 0.56352 |

选定 baseline 是 fine-tune `best.ckpt`，不是最后 epoch；最后 epoch 的 position L1 均退化至约 0.161。
这说明提前停止是必要的，但不能说明 full diffusion raw sample 已经准确。

### 5.3 V3

V3 在 V2 上保留 AE/LDM 主干，不改 architecture；增加 HD Map 和弱项：

```text
diffusion loss
+ 0.4 * decoded state loss
+ 0.10 * lane corridor loss
+ 0.10 * initial clearance loss
+ 0.00 * lane heading loss
```

最初出现 CUDA autograd in-place error；修复为 `torch.where` 条件替换和 functional inverse scaling。
one-batch full forward/backward 已通过，故训练可执行。V3 submission run：

```text
best val_decoded_position_l1 = 0.1098499
last epoch 29 val_decoded_position_l1 = 0.1350719
```

again，数值在 normalized decoded-x0 路径，不能直接与米级 sample error 混淆。

### 5.4 V4A

V4A 采用前方 target、forward support、OBB clearance、lane/speed 等弱几何项。训练：

```text
checkpoint: hv_route_inpaint_ldm_v4a_23k_exploratory_gpu3_20261006
best val_decoded_position_l1: 0.0848370
last epoch 29 val_decoded_position_l1: 0.0910664
```

它有比 V3 更小的训练期 normalized metric，但 V4A dataset 的 val 小且 test 偏 Town10HD，
并不证明比 V3/V2 更好。在线 matched cross-town seed0 中，V4A 的 eventual orange->green 为
8/24 = 33.3%，低于 V2 和 V3；故 V4A 没有实现目标。

### 5.5 V4C：C0 数据对照与 C2 route supervision

两者都从 V3 best 微调，输出节点仍为固定 2：

| 模型 | best epoch | val position L1 | val state | route distance | sync error | route corridor |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| C0 metadata-only | 2 | 0.12426 | 0.56571 | 3.297 | 2.323 | 4.171 |
| C2 route-progress/sync | 2 | 0.12896 | 0.58059 | 2.757 | 1.887 | 4.797 |

两者均在 epoch 2 最好，epoch 29 均退化。C2 的直接 loss 为：

```text
0.03 * interaction_route_distance_loss
0.03 * interaction_route_sync_loss
0.015 * interaction_route_corridor_loss
0.01 * interaction_route_tangent_loss
```

其计算发生在训练期：对预测 decoded raw state 的 2m preview 点投影到**原 target 的** RIFT route/anchor，
约束预测保持原样本的 route-relative conflict 几何。它不是推理时优化、不是 rejection、不是 RIFT 在线干预。

但 target route/anchor 在部署时不存在；C2 只能从训练分布隐式学习该关系，无法由在线 input 指定
“这次要哪一条 target approach 和哪个 conflict anchor”。

## 6. HV 离线评估：原始结果、解释和不能做的推论

### 6.1 V2 与 V3 同一 V3 test split 的 S1 raw evaluation

均为 3,727 records、single sample、无 clamp：

| metric | V2 | V3 |
| --- | ---: | ---: |
| position L1 | 9.447 m | 9.345 m |
| position L2 | 15.244 m | 15.247 m |
| heading abs | 0.639 | 0.463 |
| speed L1 | 3.632 | 3.610 |
| size L1 | 1.078 | 1.015 |
| physical invalid | 10.48% | 7.04% |
| nearest lane distance | 3.03 m | 3.16 m |
| associated-lane distance | 9.61 m | 9.02 m |
| associated lane failure / event | 0.661 | 0.869 |
| OBB overlap / event | 0.131 | 0.124 |

V3 比 V2 主要改善 heading、size 和 physical invalid；position 没改善，associated lane failure 反而增加。
故 V3 是局部字段改善，不是高价值交互生成成功。

### 6.2 V4C full sampling-contract audit：最关键的诊断

对相同 V4C test records 比较三条链路：

1. `AE oracle`：真实完整 target scene `encode -> decode`；有真实 target latent，线上不可用；
2. `training-time x0`：真实 target latent 加噪后单次预测 x0；仍有 target latent；
3. `reverse diffusion S1`：只用线上同等 condition 的完整采样；这是唯一线上同构路径。

512-record C0/C2 审计的核心结果：

| 路径/指标 | C0 | C2 |
| --- | ---: | ---: |
| AE oracle position L2 | 1.397 m | 1.397 m |
| AE oracle physical invalid | 95.3% | 95.3% |
| x0 position L2 | 7.162 m | 5.824 m |
| x0 physical invalid | 10.1% | 16.6% |
| S1 reverse position L2 | 13.724 m | 9.112 m |
| S1 reverse physical invalid | 9.8% | 24.5% |
| S1 route-distance abs error | 12.326 m | 4.436 m |
| S1 route-sync gap | 10.429 m | 7.892 m |
| S1 nearest-lane failure / event | 0.539 | 0.750 |
| S1 associated-lane failure / event | 1.727 | 1.465 |
| S1 forward violation rate | 0.10% | 3.42% |

128-record C2 detailed distribution corroborates该结论：reverse sample length p90 约 9.38m、max 10.79m，
width 与 size/速度也出现失配；AE oracle 中 length/width 的越界更严重。由于 speed=0 合法，
AE oracle 的 95.3% invalid 主要由 decoder 输出的车身尺寸落在 CARLA 合同外造成，而不是“车速为零”。

正确解释：

- C2 相较 C0，确实将 sample 位置和 route-distance proxy 拉近了原标注 target。
- C2 同时显著恶化 physical validity、前方保持和最近车道一致性。
- `route_sync_within_threshold_rate` 可很高但平均 gap 仍约 8m，说明该阈值过松，不能作为强交互成功标准。
- 冻结 NuPlan AE 在 CARLA 的 length/width/speed 语义上已失配；仅增大 LDM route loss 不可能修复 decoder 输出域。
- LDM x0 和 full reverse diffusion 又继续放大位置/状态误差。这个问题不是 CARLA spawn、RIFT 或视频渲染造成的。

### 6.3 V4C target consistency / identifiability audit

20,406 records 的审计显示：

- 按 `scene_id` 可见大量不同 target pair，但同一 `scene_id` 并不代表同一物理快照；不同采集运行可复用文本 scene id。
- 按完整模型 condition 的量化签名，`same_model_condition_with_multiple_target_pairs = 0`。
- 按完整 local fleet 签名，存在少量多 target pair group（149 records），但它们的 model condition 不相同；
  target 被从 observed context masking 后，输入差异是合理的。

这排除了“完全相同模型输入配不同 label”的直接矛盾监督作为主要原因；它**没有**证明任务在统计意义上完全可识别。
真实问题仍是多模态：仅给 ego route、HD map 和实时车流时，可能存在多条合理的 target approach/conflict branch；
当前模型没有一个线上可用的 candidate branch/anchor condition 来指定要生成哪一种交互。

### 6.4 V4C clean online 结论

route1711、initial-only、C0/C2 的 clean smoke 都出现 0 HV spawn。dump 中生成 raw state 接近或越过
在线 30m target region 边界，road projection 也远；两个模型都没有进入生命周期/RIFT CBV 评估。

因此：

```text
C0/C2 没有 route1711 clean online 成功证据；禁止据此继续跑 cross-town 三 seed。
```

此前 rolling 场景中的少量 green 不能弥补该缺口：rolling 有不同触发时刻、已有交通、可能多次尝试和
不同上下文，不是“初始 FOV 前方固定两辆强交互 HV”的公平验证。

## 7. HV 在线实验：可用结论与无效结论

### 7.1 V2 12-route cross-town matched diagnostic

routes：

```text
24211,24224,24206,24785,24759,24367,24258,24240,24252,1711,3248,24333
```

V2 historical diagnostic（seed0、one repeat）：

| 指标 | 值 |
| --- | ---: |
| mean ego driving score | 86.2505 |
| mean route completion | 93.7508 |
| collision route ratio | 0.25 |
| generation calls | 19 |
| calls with spawn | 17/19 |
| immediate generated positive=2 | 1/19 |
| immediate positive=1 | 8/19 |
| immediate positive=0 | 10/19 |

它证明 V2 链路能被调用并 spawn，但不能证明 HV 模型质量；没有 matched no-HV 对照时，ego 分数不能做因果归因。

后续严格单 sample matched run 的 eventual lifecycle 统计：

| method | spawned HV | eventual orange->green | 事件最终 >=2 green | activation mean frame | green >=50 frames |
| --- | ---: | ---: | ---: | ---: | ---: |
| V2 | 22 | 59.1% | 5/11 | 21.5 | 9 |
| V3 | 22 | 50.0% | 1/11 | 18.4 | 4 |
| V4A | 24 | 33.3% | 3/12 | 27.3 | 5 |

这说明 V2 是当前已有 checkpoint 中最强的**演化为 RIFT CBV 的经验基线**；V3/V4A 没有超过它。
但这仍不是“两个强 route-conflict CBV”的成功率：RIFT positive/green 是系统识别结果，不等价于明确 conflict-anchor
条件达成，且 V2 事件最终双 green 只有 5/11。

### 7.2 ego driving score 的解释

HV 组比 no-HV 的 ego driving score 更低、碰撞更多，不自动说明生成器失败。目标本来是更具交互性。
但只有在以下条件满足时，攻击性才可以被解释为有效高价值性：

```text
same route/seed/config + successful raw spawn + route-level conflict metric
+ interaction sustained + no model postprocessing
```

V4C 尚不满足第一步 spawn，因此不能把碰撞、green 或 driving score 用作 V4C 结论。

## 8. Background 数据、模型和组合生成

### 8.1 为什么需要独立背景线

generic NuPlan LDM 不能直接用来生成 CARLA ordinary BV。800 个 one-BV audit：

| 指标 | 值 |
| --- | ---: |
| fully feasible | 111/800 = 13.875% |
| invalid size | 71.75% |
| lane distance >5m | 45.25% |
| FOV invalid | 10.50% |
| lane distance mean / p90 | 6.95m / 18.11m |

这证明需要单独的 CARLA background dataset/checkpoint；不能把 `sd6 + hv2` 说成已可用的全模型交通生成。

### 8.2 Background 采集和数据构建

主采集目录：

```text
rift_labeled_background_full_non_town12_50k_16workers
```

它是为 normal BV directional inpainting 构建的多 worker 数据，而非 HV target 数据。尾部 250 个 route-seed
任务曾连续两天无进度；recovery launch 最终 250/250 失败、0 snapshot。根因是 stale/orphan CARLA、失败 worker
和资源/启动环境问题。该 250 个任务只占计划 12,325 的约 2%，且已有背景数据规模足够完成当前训练，
因此没有必要为了这部分继续采集。

background dataset：

```text
scenario_dreamer_background_directional_existing_raw_20261005
```

任务定义为 directional ordinary BV：以既有/后方 context 预测前方 BV，最多 8 target，`max_num_agents=61`。
数据审计分布覆盖 connector/junction、multi-lane、opposing/parallel，但具体 split 仍受 town 和 topology 不均衡影响；
评估 test 实际有 5,184 records。所有后续结论应以该实际 test split 为准。

### 8.3 Background 模型训练和离线评估

checkpoint：

```text
scenario_dreamer_background_directional_existing_raw_gpu2_20261006/best.ckpt
```

训练：best epoch=0，best val position L1=0.134465；last epoch=39 为 0.151677。
这意味着现有 fine-tuning **没有在所选 validation metric 上超过初始化模型**；不能声称训练收敛后变好。

single-sample raw test：

| metric | 值 |
| --- | ---: |
| records | 5,184 |
| count exact rate | 1.0 |
| position L1 / L2 | 6.25 m / 10.27 m |
| physical invalid | 11.04% |
| nearest-lane distance | 3.03 m |
| associated-lane distance | 4.88 m |
| lane corridor failure | 0.461 / event |
| forward violation | 20.08% |
| OBB overlap | 13.48% |

输出 target 数量可以匹配，但 raw geometry、前方区域和碰撞约束均不足。它不能替代 CARLA/TM 真实背景，
更不能直接用于长时滚动。尚未完成可信的 1/2/3/5-step autoregressive rollout stability report。

### 8.4 HV + background composed V2 功能与限制

入口：

```text
high_value_init.scene_generator_composed_v2:ScenarioDreamerComposedHighValueSceneGeneratorV2
```

保留两个 mode：

```text
--sd_background_source carla
    现有 CARLA/TM ordinary BV 作为 HV context；不新增 model ordinary BV。

--sd_background_source scenario_dreamer
    Scenario Dreamer 先生成 K 辆 ordinary BV，再将其和 live context 交给固定两辆 HV model。
```

`--sd_background_model_only_traffic` 仅用于第二种严格实验：移除 CARLA-originated 非模型车辆，
但 CARLA 仍是物理/actor spawn 执行器。它没有删除 legacy CARLA background 路径。

已知 integration evidence：route1711 `sd1 + hv2` 成功 spawn 一辆 model ordinary BV 和两辆 V2 HV，
两辆 HV 在 frame 26 成为 CBV、frame 198 回到 TM。12-route composed diagnostic：10/12 full spawn，
blue->purple 2/10，orange->green 9/20。这只能证明链路接通；当时非模型 CARLA 交通仍可能出现，
generic/background 模型也未达到可部署质量。

当前实现不是原版 Scenario Dreamer 的完整 directional learned-count 流程：

- 当前 wrapper 可用固定 `--sd_background_num_agents K`，不是从原 `P(num_after | lanes, num_before)` count matrix 采样；
- 当前 HV wrapper 可将 full FOV context 放入条件，未严格实现“rear half -> front half”；
- rolling hook 是 HV replenishment，反复触发会追加 K 辆普通 BV，可能超过 61-agent capacity；
- 因而它不是连续 ordinary-traffic predictor，也不应进入正式性能结论。

## 9. 代码、运行与评估过程中的真实问题

| 类别 | 事实 | 处理/含义 |
| --- | --- | --- |
| 权限 | `online_debug` 被 root 创建导致 `PermissionError` | run 前保证 dump/RIFT output 归 `riftuser` |
| CARLA | root 直接启动 CARLA 被拒绝 | CARLA server/online eval 使用 `riftuser` |
| port | `bind: Address already in use` 后 CARLA 崩溃 | 每个 job 独立 CARLA/TM port，清 stale process 后再跑 |
| PATH | 被手动覆盖后 `bash/id/mkdir/runuser` command not found | 保留 `/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin`，调用 `/bin/bash` |
| V3 data | snapshot local map 空，所有样本 lane reject | 改为 RIFT HD Map enrichment |
| V3 train | in-place autograd error | V3 中改为 functional `torch.where`/inverse scaling，one-batch backward PASS |
| V4A audit | `PosixPath` JSON serialization | 审计写 JSON 时转换 path/string；不影响原始数据 |
| V4B split | train 为空 | group split bug，旧输出不可训练 |
| online summary | lifecycle activation count 有时为 int，被 `len()` | summary parser 已需兼容 count/list 两种 schema |
| row analysis | 某些 generated JSONL 无 `record_index` | 按行 fallback，不能假设所有 evaluator schema 相同 |
| source identity | `source_file_sha1` 是 path hash | 必须新增 content hash；现有 source-based leakage 结论无效 |

每次在线结论前必须先检查：generator 真被调用、proposal 有目标状态、spawn attempt 的 raw/world/road
transform、actor IDs、source、lifecycle events、RIFT statistics entry status。只看到 ego route finished 或视频生成，
并不能说明 HV generator 成功。

## 10. 当前失败的根因：基于证据的排序

### 已证实

1. **冻结 AE 的 CARLA raw-state decode contract 不合格。** AE oracle 即使用真 target latent 仍有 95.3% physical invalid，
   尤其 length/width 越界。这不是 LDM 单独能修复的问题。
2. **LDM 条件采样误差很大。** C2 的 target-known x0 已约 5.82m L2，condition-only reverse diffusion 到约 9.11m；
   C0 更差约 13.72m。
3. **V4C-C2 的 route proxy 改善以 raw feasibility 退化为代价。** 不能用它替换 V2，也不能部署。
4. **C0/C2 clean initial-only online 无 spawn。** 没有任何有效的 C0/C2 strong-interaction closed-loop result。
5. **background model 也未达到普通交通替换的最小标准。** 最优 epoch=0 且 raw directional violation/overlap 显著。

### 已排除或不支持

1. 完整量化 model condition 映射到多 target pair 的直接标签矛盾不是主因（计数为 0）。
2. “提高训练期 decoded position loss”不等价于米级采样、spawn 或 route conflict 成功。
3. 后处理 clamp、lane snap、RIFT rerank/删除车、直接注册 CBV 都不是允许的改进手段。
4. C2 的高 `within_threshold` 不足以证明 target 与 ego 强 conflict；其绝对 sync gap 仍大。

### 仍然存在但尚未定量证明为主因

1. **在线 intent 不可辨识。** 当前 input 没有指定 candidate target lane/approach 和 candidate conflict anchor；
   target-only RIFT route 在训练中可见、在推理时不可见。多模态交通下模型只能盲猜一种 mode。
2. all-positive V4C 将强/弱 interaction 混合，仅用权重；尚未硬性构造足够规模、强 route-conflict 的训练子集。
3. 采用 path hash 意味着尚未完成内容级重复/泄漏审计。
4. target route sidecar 的缺失 5.16% 是可解释的（trace short/waypoint missing），但过滤后各拓扑类别更稀疏。

## 11. 当前应停止、应保留、应如何报告

### 停止

- 不跑 V4C C0/C2 cross-town 三 seed；route1711 clean initial-only 已无 spawn。
- 不继续增加 C2 route loss 系数；已观察到 route proxy 与 physical validity 的负向 trade-off。
- 不把 V4A/V4C normalized validation loss、best-of-N oracle 或 green 本身报告为模型成功。
- 不将 background checkpoint 接入 model-only continuous traffic rollout。
- 不删除 V1/V2/V3/V4A/V4C/background 的 `best.ckpt`、数据集和关键 eval 证据；它们是失败分析和消融基线。

### 可报告的、诚实的结果

1. V2 完成在线集成，能 spawn，并有一定 eventual RIFT recognition；但事件级双强交互/双 green 的证据有限。
2. V3 在相同 test 上改善部分状态字段，但没有改善位置和闭环 high-value outcome。
3. V4A 前方重建/弱几何路线没有赢过 V2，且 split 质量不足。
4. V4C 第一次将 target-only route-progress/static supervision 加入训练；它改善了部分 static route proxy，
   但暴露出 frozen AE 和 reverse diffusion 不能承担 CARLA raw state generation 的核心瓶颈。
5. Background 从“generic NuPlan LDM 不可用”进展到有专用数据/训练/评估，但还未达到上线门槛。

## 12. 下一阶段的技术建议：先修问题定义，再训练

### 12.1 必须先做的表征决策

不应继续在冻结 NuPlan AE + 旧 LDM latent contract 上堆 loss。下一模型至少需要其一：

1. 训练/微调一个 CARLA-specific state AE，使 `x/y/yaw/speed/length/width` 的 encode->decode oracle 在 raw
   physical bounds、米级几何上先过关；随后在新 latent space 重新训练条件生成器；或
2. 采用单独的 CARLA raw-state head/generator，不依赖该冻结 decoder 输出速度和尺寸。

这不是推理时 clamp。可行的 output parameterization 应是模型架构的一部分，并在训练数据分布内学习，
例如以合适变量化表达 positive size 和有限 speed；仍需原始模型输出直接用于 CARLA，不能在采样后修车道状态。

在开始 LDM/生成器训练前，应设立硬 gate：

```text
AE oracle physical invalid rate 接近 0
AE oracle 位置/heading/speed/size raw metrics 明确报告
```

若该 gate 不过，任何 diffusion route loss 试验都只是在错误 decoder 表示上优化。

### 12.2 必须让“强交互意图”成为线上可得条件

从 ego route + HD Map 构造在线可得 candidate set：

```text
candidate target approach lane
candidate conflict anchor
ego route distance to anchor
target route distance / arrival-gap bucket
```

离线用 RIFT trace 仅把真实 target 标注到一个 candidate；线上不输入 RIFT target trace。
条件生成器必须接收该 candidate/intent，再生成两辆 raw state。否则同一类 junction 的多种合理
approach/冲突模式仍只能由无条件连续状态去猜，难以稳定获得指定强 conflict。

训练/验证应有三个分层门槛：

1. **representation**：AE/raw head oracle；
2. **conditional sample**：S1 raw physical/FOV/lane/clearance 和 conflict-anchor 误差；
3. **closed loop**：normal TM/RIFT 下 spawn、route conflict、持续交互、正常 RIFT recognition。

任何一层失败都不扩大到 cross-town 三 seed。

### 12.3 数据与评估的最小改造

1. 新增 canonical content hash，重新做跨 split 内容重复审计；group split 不得只依赖 `scene_id` 或路径 hash。
2. 将强 interaction 定义为可报告的 map/route quantity，而非 RIFT green：例如 anchor 可达、ego/target 到 anchor
   route distance、arrival/ETA gap、target approach lane/方向一致性。
3. 不要只以 all-positive weight 代替强 target 定义。若强样本数量足够，建立 strong/medium/weak strata，
   在 train 以比例采样，并在 test 分别报告。
4. 保留原始 distribution，不为得到好看指标而在推理时 best-of-N 选样或删样。
5. background 先完成 1/2/3/5 step open-loop rollout，分别统计 physical valid AND forward AND lane corridor
   AND no OBB overlap 的联合率；稳定前不进入 model-only online traffic。

## 13. 关键复现命令模板

以下模板使用新 output path；`resume_checkpoint` 是只读输入，绝不能输出到已有 checkpoint 目录。

### 13.0 历史数据/训练入口索引

下表将每个版本的实际脚本、数据目录、checkpoint 和是否仍建议执行分开列出。历史命令必须使用新的
`output_dir` 才可重复；不要向已有 `best.ckpt` 所在目录训练。

| 版本 | dataset builder | trainer | evaluator | 当前状态 |
| --- | --- | --- | --- | --- |
| V1/Phase5 | `build_high_value_scene_dataset.py` | `train_high_value_scene_generator.py` | `eval_high_value_scene_generator.py` | 仅 baseline 复现 |
| V2 | `build_high_value_scene_dataset_v2.py` | `train_high_value_scene_generator_v2.py` | `eval_high_value_scene_generator_v2.py` | 必须保留的闭环 baseline |
| V3 | `build_high_value_scene_dataset_v3.py` | `train_high_value_scene_generator_v3.py` | `eval_high_value_scene_generator_v3.py` | 可做 V2 对照，不应替换 V2 |
| V4A | `build_high_value_scene_dataset_v4.py` | `train_high_value_scene_generator_v4a.py` | `eval_high_value_scene_generator_v4a.py` | 历史实验，不应再扩展 |
| V4B | `build_high_value_scene_dataset_v4b.py` | `train_high_value_scene_generator_v4b.py` | V4A/V4B 审计 | builder/control；无可信模型 |
| V4C | `build_high_value_scene_dataset_v4c.py` | `train_high_value_scene_generator_v4c.py` | `eval_high_value_scene_generator_v4c.py` | C0/C2 均不应继续线上扩展 |
| background | `build_scenario_dreamer_background_dataset.py` | `train_scenario_dreamer_background.py` | `eval_scenario_dreamer_background.py` | 单步 baseline，未上线 |

历史 V2 全流程命令、各 B16/B32 比较、Phase5 数据路径的完整原文保留在
`high_value_generation_phase5_to_v3_experiment_archive_技术档案.md` 的 Appendix A；本文中的路径与数值
是该账本之后 V4/background/V4C 的增量总汇。

### 13.0.1 V2 数据与 B32 baseline 的历史复现契约

```bash
export OLD_COMBINED=$HV_ROOT/rift_labeled_all_recursive_20260919_133508
export COLLECT_ROOT=$HV_ROOT/rift_labeled_phase7_v2_uncovered_routes_seed0_7
export UNION_DIR=$HV_ROOT/rift_labeled_union_phase5_phase7_v2_uncovered_routes_seed0_7
export V2_DATA=$HV_ROOT/datasets/inpainting_sparse_phase7_v2_uncovered_routes_seed0_7

cd $ROOT/scenario-dreamer
$PY build_high_value_scene_dataset_v2.py \
  --input_dir "$UNION_DIR" --output_dir "$V2_DATA" --mode inpainting \
  --min_rift_positive_agents 2 --num_generate_agents 2 --actor_radius 80 --fov 64 \
  --positive_target_policy route_s_diversity --sparse_context_copies 4 \
  --sparse_context_min_observed 0 --sparse_context_max_observed 4 \
  --quality_filter_v2 --min_target_speed 0.5 --max_target_speed 25 \
  --min_target_ego_distance 3 --max_target_ego_distance 60 \
  --target_fov_margin 2 --min_target_pair_distance 2 --dedup_grid_xy 1 --seed 42

export PHASE5_BEST=$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase5/best.ckpt
export V2_REPRO=$HV_ROOT/checkpoints/<new_v2_reproduction>
CUDA_VISIBLE_DEVICES=3 $PY train_high_value_scene_generator_v2.py \
  --mode inpainting --train_jsonl "$V2_DATA/train.jsonl" --val_jsonl "$V2_DATA/val.jsonl" \
  --output_dir "$V2_REPRO" --autoencoder_checkpoint "$AE_CKPT" --ldm_checkpoint "$LDM_CKPT" \
  --resume_checkpoint "$PHASE5_BEST" --num_generate_agents 2 --epochs 40 --batch_size 32 --lr 2e-5 --seed 0 \
  --decoded_state_loss_weight 0.4 --decoded_position_loss_weight 2.0 \
  --decoded_heading_loss_weight 0.7 --decoded_speed_loss_weight 1.0 \
  --decoded_size_loss_weight 0.5 --physical_bounds_loss_weight 0.3 \
  --best_metric val_decoded_position_l1
```

这只用于复现/消融，不应覆盖当前 V2 baseline。

### 13.0.2 V3 数据与训练的历史复现契约

```bash
export RAW_SNAPSHOTS=$HV_ROOT/rift_labeled_phase7_v2_uncovered_routes_seed0_7
export V3_DATA=$HV_ROOT/datasets/inpainting_sparse_phase8_v3_fov64_hdmap_full
cd $ROOT/scenario-dreamer
$PY build_high_value_scene_dataset_v3.py \
  --input_dir "$RAW_SNAPSHOTS" --output_dir "$V3_DATA" --mode inpainting --fov 64 \
  --map_source hd_map --rift_root "$ROOT/RIFT" --rift_hd_map_dir "$ROOT/RIFT/data/map_data/HD-Map" \
  --num_generate_agents 2 --quality_filter_v2 --min_target_speed 0.0 \
  --max_target_lane_distance 5.0 --sparse_context_copies 4 --seed 42

export V3_REPRO=$HV_ROOT/checkpoints/<new_v3_reproduction>
CUDA_VISIBLE_DEVICES=0 $PY train_high_value_scene_generator_v3.py \
  --train_jsonl "$V3_DATA/train.jsonl" --val_jsonl "$V3_DATA/val.jsonl" \
  --output_dir "$V3_REPRO" --autoencoder_checkpoint "$AE_CKPT" --ldm_checkpoint "$LDM_CKPT" \
  --resume_checkpoint "$V2_BEST" --num_generate_agents 2 --batch_size 32 --epochs 30 --lr 1e-5 \
  --decoded_state_loss_weight 0.4 --decoded_position_loss_weight 2.0 \
  --decoded_heading_loss_weight 0.7 --decoded_speed_loss_weight 1.0 \
  --decoded_size_loss_weight 0.5 --physical_bounds_loss_weight 0.3 \
  --lane_corridor_loss_weight 0.10 --lane_heading_loss_weight 0 \
  --initial_clearance_loss_weight 0.10 --best_metric val_decoded_position_l1
```

### 13.0.3 V4C 的已完成 C0/C2 训练契约

此段仅为复现实验记录；在没有新 CARLA-specific representation 和 intent condition 前，不推荐启动新 C2 run。

```bash
export V4C_DATA=$HV_ROOT/datasets/v4c_route_weighted_allpositive_20261009
export V3_BEST=$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase8_v3_submission_ft_20261005/best.ckpt
export V4C_C2_REPRO=$HV_ROOT/checkpoints/<new_v4c_c2_reproduction>
cd $ROOT/scenario-dreamer
CUDA_VISIBLE_DEVICES=3 $PY train_high_value_scene_generator_v4c.py \
  --train_jsonl "$V4C_DATA/train.jsonl" --val_jsonl "$V4C_DATA/val.jsonl" \
  --output_dir "$V4C_C2_REPRO" --autoencoder_checkpoint "$AE_CKPT" --ldm_checkpoint "$LDM_CKPT" \
  --resume_checkpoint "$V3_BEST" --num_generate_agents 2 --batch_size 32 --epochs 30 --lr 1e-5 --seed 0 \
  --v4b_stage b0_data_only --v4c_stage c2_route_progress_sync \
  --expected_target_pair_policy high_value_pair --forward_min_x 0 --target_fov_margin 2 \
  --min_obb_axis_clearance 0.5 --context_dropout_prob 0 --interaction_strength_weight_power 1.0 \
  --decoded_state_loss_weight 0.4 --decoded_position_loss_weight 2.0 \
  --decoded_heading_loss_weight 0.7 --decoded_speed_loss_weight 1.0 \
  --decoded_size_loss_weight 0.5 --physical_bounds_loss_weight 0.3 \
  --lane_corridor_loss_weight 0.10 --lane_heading_loss_weight 0 --initial_clearance_loss_weight 0.10 \
  --interaction_route_distance_loss_weight 0.03 --interaction_route_sync_loss_weight 0.03 \
  --interaction_route_corridor_loss_weight 0.015 --interaction_route_tangent_loss_weight 0.01 \
  --best_metric val_decoded_position_l1
```

对应 C0 使用相同 V4C dataset、V3 initialization 和基础 decoded/geometry losses，但不启用四个
`interaction_route_*_loss_weight`。它是 data/metadata control，不是 V4B model。

### 13.0.4 Background 训练的历史复现契约

```bash
export BG_DATA=$HV_ROOT/datasets/scenario_dreamer_background_directional_existing_raw_20261005
export BG_REPRO=$HV_ROOT/checkpoints/<new_background_reproduction>
cd $ROOT/scenario-dreamer
CUDA_VISIBLE_DEVICES=2 $PY train_scenario_dreamer_background.py \
  --train_jsonl "$BG_DATA/train.jsonl" --val_jsonl "$BG_DATA/val.jsonl" \
  --output_dir "$BG_REPRO" --autoencoder_checkpoint "$AE_CKPT" --ldm_checkpoint "$LDM_CKPT" \
  --max_target_agents 8 --fov 64 --partition_boundary_x 0 --target_fov_margin 2 \
  --max_num_agents 61 --batch_size 32 --epochs 40 --lr 1e-5 --seed 0 \
  --best_metric val_decoded_position_l1
```

实际 background run 最优在 epoch 0；任何新的 background trainer 必须先和 epoch-0/base、best、last
在相同 S1 test 和多步 rollout 上比较，不能只报告最后一个 epoch。

### 13.1 V3 对照评估

```bash
export V2_BEST=$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase7_v2_full_b32_ft_from_phase5_20260922/best.ckpt
export V3_BEST=$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase8_v3_submission_ft_20261005/best.ckpt
export V3_DATA=$HV_ROOT/datasets/inpainting_sparse_phase8_v3_fov64_hdmap_full
export EVAL_DIR=$HV_ROOT/eval/<new_v2_v3_eval>
mkdir -p "$EVAL_DIR"
cd $ROOT/scenario-dreamer

CUDA_VISIBLE_DEVICES=2 $PY eval_high_value_scene_generator_v2.py \
  --mode inpainting --jsonl "$V3_DATA/test.jsonl" --checkpoint "$V2_BEST" \
  --autoencoder_checkpoint "$AE_CKPT" --ldm_checkpoint "$LDM_CKPT" \
  --num_generate_agents 2 --fov 64 --num_samples_per_condition 1 \
  --output_json "$EVAL_DIR/v2_raw_s1.json" \
  --output_generated_jsonl "$EVAL_DIR/v2_raw_s1_generated.jsonl"

CUDA_VISIBLE_DEVICES=3 $PY eval_high_value_scene_generator_v3.py \
  --jsonl "$V3_DATA/test.jsonl" --checkpoint "$V3_BEST" \
  --autoencoder_checkpoint "$AE_CKPT" --ldm_checkpoint "$LDM_CKPT" \
  --num_generate_agents 2 --num_samples_per_condition 1 --seed 0 \
  --output_json "$EVAL_DIR/v3_raw_s1.json" \
  --output_generated_jsonl "$EVAL_DIR/v3_raw_s1_generated.jsonl"
```

### 13.2 V4C sampling-contract audit

```bash
export V4C_DATA=$HV_ROOT/datasets/v4c_route_weighted_allpositive_20261009
export C0_BEST=$HV_ROOT/checkpoints/hv_route_inpaint_ldm_v4c_c0_from_v3_20261010/best.ckpt
export C2_BEST=$HV_ROOT/checkpoints/hv_route_inpaint_ldm_v4c_c2_route_sync_from_v3_20261010/best.ckpt
export AUDIT_DIR=$HV_ROOT/eval/<new_v4c_sampling_contract>
mkdir -p "$AUDIT_DIR"
cd $ROOT/scenario-dreamer

CUDA_VISIBLE_DEVICES=2 $PY audit_hv_sampling_contract.py \
  --jsonl "$V4C_DATA/test.jsonl" --checkpoint "$C0_BEST" \
  --autoencoder_checkpoint "$AE_CKPT" --ldm_checkpoint "$LDM_CKPT" \
  --max_records 512 --fov 64 --forward_min_x 0 --target_fov_margin 2 --seed 0 \
  --output_json "$AUDIT_DIR/c0_sampling_contract_512.json" \
  --output_rows_jsonl "$AUDIT_DIR/c0_sampling_contract_512_rows.jsonl"

CUDA_VISIBLE_DEVICES=3 $PY audit_hv_sampling_contract.py \
  --jsonl "$V4C_DATA/test.jsonl" --checkpoint "$C2_BEST" \
  --autoencoder_checkpoint "$AE_CKPT" --ldm_checkpoint "$LDM_CKPT" \
  --max_records 512 --fov 64 --forward_min_x 0 --target_fov_margin 2 --seed 0 \
  --output_json "$AUDIT_DIR/c2_sampling_contract_512.json" \
  --output_rows_jsonl "$AUDIT_DIR/c2_sampling_contract_512_rows.jsonl"
```

### 13.3 V4C target consistency audit

```bash
cd $ROOT/scenario-dreamer
$PY audit_v4c_snapshot_target_consistency.py \
  --jsonl "$V4C_DATA/train.jsonl" --jsonl "$V4C_DATA/val.jsonl" --jsonl "$V4C_DATA/test.jsonl" \
  --output_json "$AUDIT_DIR/v4c_snapshot_target_consistency_all_splits.json"
```

它只审计，不重写 dataset。解释时必须记住 `source_file_sha1` 是 path hash 的局限。

### 13.4 Background raw S1 baseline

```bash
export BG_DATA=$HV_ROOT/datasets/scenario_dreamer_background_directional_existing_raw_20261005
export BG_BEST=$HV_ROOT/checkpoints/scenario_dreamer_background_directional_existing_raw_gpu2_20261006/best.ckpt
export BG_EVAL=$HV_ROOT/eval/<new_background_eval>
mkdir -p "$BG_EVAL"
cd $ROOT/scenario-dreamer
CUDA_VISIBLE_DEVICES=2 $PY eval_scenario_dreamer_background.py \
  --jsonl "$BG_DATA/test.jsonl" --checkpoint "$BG_BEST" \
  --autoencoder_checkpoint "$AE_CKPT" --ldm_checkpoint "$LDM_CKPT" \
  --max_target_agents 8 --fov 64 --partition_boundary_x 0 --target_fov_margin 2 \
  --max_num_agents 61 --min_initial_clearance 2.5 --num_samples_per_condition 1 --seed 0 \
  --output_json "$BG_EVAL/test_raw_s1.json" \
  --output_generated_jsonl "$BG_EVAL/test_raw_s1_generated.jsonl"
```

### 13.5 在线运行前的硬检查

```bash
cd $ROOT/RIFT
$PY scripts/audit_high_value_generation_pipeline.py \
  --root "$ROOT" --scenario_dreamer_root "$ROOT/scenario-dreamer"
$PY -m py_compile \
  rift/carla_runner.py rift/cbv/initialization/hv_online_generation.py \
  scripts/summarize_high_value_generation_online.py
```

在线扩展评估只在 initial-only smoke 满足以下条件后执行：

```text
generator called > 0
exactly two raw HV proposals
two actors both spawn
raw x/y/yaw/speed unchanged except terrain z
no lane snap / no clamp / no audit delete
static FOV/road/clearance checks pass
```

## 14. 新聊天框的最短交接结论

```text
当前主线不是“继续调 V4C loss”。

V2 是唯一具有一定 closed-loop eventual RIFT-CBV 证据的基线，仍不够强。
V3、V4A 未超过 V2；V4B 没有可信模型；V4C-C2 只改善了部分离线 route proxy，
但在 frozen NuPlan AE 的 CARLA raw-state mismatch 和 LDM sampling error 下无法 clean spawn。
Background 也尚未达到上线条件。

下一步必须先验证/重建 CARLA raw-state representation，并把在线可得的 candidate conflict intent
作为生成条件。RIFT route trace 仅可离线监督，不能在推理期偷渡为 target 条件。
```

## 15. 相关历史文档与代码入口

- `high_value_generation_phase5_to_v3_experiment_archive_技术档案.md`：Phase5-V3 更长的历史命令账本。
- `high_value_generation_phase7_reproduction_复现_v2.md`：Phase7 多 worker 采集与 V2 B32 复现细节。
- `high_value_generation_v3_technical_route_技术路线和命令.md`：V3 HD Map/弱几何路线的历史设计。
- `high_value_init_generation_phase0_phase6_audit.md`：online wiring 和诊断演进。
- `scenario-dreamer/high_value_init/high_value_ldm.py`：冻结 AE、LDM 训练/采样、raw bounds。
- `scenario-dreamer/high_value_init/high_value_ldm_v4c.py`：C2 route supervision。
- `scenario-dreamer/audit_hv_sampling_contract.py`：AE/x0/reverse 三段采样链路审计。
- `scenario-dreamer/audit_v4c_snapshot_target_consistency.py`：target/condition consistency 审计。
- `scenario-dreamer/high_value_init/scene_generator_composed_v2.py`：CARLA-background 与 model-background 两种 composed mode。

## 16. 逐步命令、输入输出与结果证据路径账本

本节是给新聊天框/新实验者的追溯索引。路径均相对于：

```bash
export ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
export HV_ROOT=$ROOT/data/hv_init
export PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
```

### 16.1 统一统计与文件存在性检查

以下命令只读，不会修改结果。每次重新解读一个版本前先运行；不要通过终端退出码猜测训练是否成功。

```bash
check_path () {
  local p="$1"
  if [ -e "$p" ]; then
    printf 'EXISTS  '; du -sh "$p" 2>/dev/null || true
    printf 'PATH    %s\n' "$p"
  else
    printf 'MISSING %s\n' "$p"
  fi
}

check_json () {
  local p="$1"
  if [ -f "$p" ]; then
    echo "===== $p ====="
    "$PY" - "$p" <<'PY'
import json, sys
p=sys.argv[1]
try:
    d=json.load(open(p))
except Exception as e:
    print('JSON_READ_ERROR:', repr(e)); raise SystemExit(1)
for k in ('schema_version','checkpoint','output_dir','num_records','records','passed',
          'num_train_records','num_val_records','num_test_records','best_metric',
          'best_metric_value','entry_status'):
    if k in d: print(k, '=', d[k])
PY
  else
    echo "MISSING $p"
  fi
}
```

所有 JSONL 的真实记录数必须使用：

```bash
wc -l "$DATA/train.jsonl" "$DATA/val.jsonl" "$DATA/test.jsonl"
```

不要用目录大小或 `.json` 文件数量替代记录数。symlink source 则用：

```bash
find -L "$SOURCE_DIR" -type f -name '*.json' | wc -l
```

### 16.2 V1 / Phase5：数据、训练、评估证据

| 步骤 | 实际输入 | 实际输出/证据路径 | 已知结果 |
| --- | --- | --- | --- |
| 原始递归快照 | `$HV_ROOT/rift_labeled_all_recursive_20260919_133508` | `find -L` 结果 | 70,329 JSON links |
| dataset | `$HV_ROOT/datasets/inpainting_all_recursive` | `train.jsonl`, `val.jsonl`, `test.jsonl`, `audit.json` | 13,706 records；10,939 / 1,518 / 1,249 |
| Phase5 model | 上述 dataset | `$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase5/{best.ckpt,history.json}` | 当前 Phase5 reference |
| 早期全递归训练 | same | `$HV_ROOT/checkpoints/hv_route_inpaint_ldm_all_recursive/` | 历史 run，不能与 Phase5 best 混淆 |
| raw eval | `inpainting_all_recursive/test.jsonl` | `$HV_ROOT/eval/phase5_raw_s1.json`；历史 100-record 结果还在 `$HV_ROOT/eval/inpainting_all_recursive_test_eval_100_s1.json` | raw L2=15.70m，invalid=20.5% |

历史 dataset build、audit、训练、评估命令在
`high_value_init_generation_experiment_status_next_steps.md:451-505` 和
`high_value_generation_phase5_to_v3_experiment_archive_技术档案.md:668-731`。
其中第一条指向 `inpainting_sparse_all_recursive` 的训练命令是**历史失败命令**，路径不存在，不得复制使用。

### 16.3 V2 / Phase7：采集、union、训练、离线与在线证据

#### A. 采集与统计

| 步骤 | 命令/输入 | 输出路径 | 结果 |
| --- | --- | --- | --- |
| Phase7 collection | `RIFT/scripts/collect_high_value_init_scenes.py`，8 workers、disjoint routes、seed0--7 | `$HV_ROOT/rift_labeled_phase7_v2_uncovered_routes_seed0_7` | 54,936 snapshots、220 routes |
| source count | `find -L "$COLLECT_ROOT" -type f -name '*.json' \| wc -l` | worker data | 54,936 |
| union | `RIFT/scripts/merge_labeled_snapshot_sources.py` 或 symlink union loop | `$HV_ROOT/rift_labeled_union_phase5_phase7_v2_uncovered_routes_seed0_7` | Phase5 old + Phase7 merged |
| dataset build | `scenario-dreamer/build_high_value_scene_dataset_v2.py` | `$HV_ROOT/datasets/inpainting_sparse_phase7_v2_uncovered_routes_seed0_7` | B32 run 的唯一 dataset |
| data audit | `scenario-dreamer/audit_high_value_scene_dataset.py` | `$V2_DATA/audit.json` | mode/fixed-2/FOV audit |

实际 V2 build 命令在本文 `13.0.1`；完整 worker collection/union 命令在
`high_value_generation_phase7_reproduction_复现_v2.md:41-161`。

#### B. 训练、评估、线上路径

| run | 精确 checkpoint/输出路径 | 训练结果证据 | 离线/在线结果路径 |
| --- | --- | --- | --- |
| B32 scratch | `$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase7_v2_full_b32_scratch_20260922/` | `history.json`，best epoch12/60，val pos=0.1499498 | 仅对照 |
| B32 Phase5 fine-tune | `$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase7_v2_full_b32_ft_from_phase5_20260922/` | `history.json`，best epoch12/40，val pos=0.1463771 | 当前 V2 baseline |
| V2 raw S1 on V3 test | V2 best + `$HV_ROOT/datasets/inpainting_sparse_phase8_v3_fov64_hdmap_full/test.jsonl` | `$HV_ROOT/eval/v2_v3_same_v3test_full_20261006/v2_raw_s1.json` 和 `_generated.jsonl` | L2=15.244m，invalid=10.48% |
| V2 raw S4 oracle | same | `$HV_ROOT/eval/v2_v3_same_v3test_full_20261006/v2_raw_s4_oracle.json` | oracle，不可当线上性能 |
| V2 matched online | 12 routes、seed0、strict S1 | `$HV_ROOT/eval/hv_matched_cross_town_seed0_strict_s1_20261008/`；对应 simulator 输出 `$ROOT/RIFT/hv_v2_hv_matched_cross_town_seed0_strict_s1_20261008/` | 22 spawned HV，eventual green 13/22 |
| 早期 V2 online | historical | `$HV_ROOT/online_debug/hvgen_debug_cross_town_fov64_ftb32_seed0_rerun1/summary.json` | 19 calls，immediate two-positive 1/19 |

V2 B32 训练的完整实际命令在
`high_value_generation_phase7_reproduction_复现_v2.md:171-236`。V2 raw evaluation、clamp ablation、
online `runuser` 命令在 `:244-370`；clamp 输出仅做诊断。

### 16.4 V3：数据、训练、离线对照证据

| 步骤 | 输入/命令 | 输出证据路径 | 结果 |
| --- | --- | --- | --- |
| HD Map build | `build_high_value_scene_dataset_v3.py --map_source hd_map` | `$HV_ROOT/datasets/inpainting_sparse_phase8_v3_fov64_hdmap_full/{train,val,test}.jsonl`；`quality_report.json` | 30,468 / 4,558 / 3,727；HD map error=0 |
| preflight | 同 builder，`--max_input_files 5000` | `$HV_ROOT/datasets/inpainting_sparse_phase8_v3_fov64_hdmap_preflight/` | 5,915 raw records，验证 HD Map 读取 |
| in-place 修复 smoke | V3 model one-batch backward | 终端记录；无单独 checkpoint | `V3 one-batch backward PASS` |
| earlier V3 FT | V2 best initialization | `$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase8_v3_fov64_hdmap_full_ft_from_v2b32_noninplace_20260924/` | 历史 checkpoint，保留作消融 |
| submission V3 FT | V2 best initialization | `$HV_ROOT/checkpoints/hv_route_inpaint_ldm_sparse_phase8_v3_submission_ft_20261005/{best.ckpt,history.json}` | best val pos=0.1098499 |
| V3 raw S1 | V3 best，V3 test | `$HV_ROOT/eval/v2_v3_same_v3test_full_20261006/v3_raw_s1.json` 和 `_generated.jsonl` | L2=15.247m，invalid=7.04% |

完整 build/train/eval 命令在本文 `13.0.2` 和旧档案
`high_value_generation_phase5_to_v3_experiment_archive_技术档案.md:1028-1087`。V3 输出必须查看
`quality_report.json`，不是不存在的 `dataset_summary.json`。

### 16.5 V4A：数据、训练、评估证据

| 步骤 | 输入/输出路径 | 已知统计或结果 |
| --- | --- | --- |
| raw union | `$HV_ROOT/rift_labeled_v4a_union_phase7_r4_plus_recovery35k_retry2_20261005` | 109,937 JSON，来自 56,527 V4A union + 53,410 recovery |
| V4A dataset | `$HV_ROOT/datasets/hv_v4a_fullfov_forward_23k_exploratory_20261006/` | 23,667；21,199 / 541 / 1,927 |
| structural audit | `$HV_ROOT/eval/v4a_23k_exploratory_20261006/dataset_audit_structural.json` | passed，但 val/test town/topology 明显失衡 |
| trainer output | `$HV_ROOT/checkpoints/hv_route_inpaint_ldm_v4a_23k_exploratory_gpu3_20261006/` | `best.ckpt`, `history.json`, `train_config.json`；best val pos=0.0848370 |
| V4A raw eval | `$HV_ROOT/eval/v4a_23k_exploratory_eval_20261007/` | S1/S4 raw report 与 generated JSONL；不得与 V3 test 混为同分布比较 |
| matched online | `$HV_ROOT/eval/hv_matched_cross_town_seed0_strict_s1_20261008/`；`$ROOT/RIFT/hv_v4a_hv_matched_cross_town_seed0_strict_s1_20261008/` | 24 spawned，eventual green=8/24 |

V4A 的精确历史 CLI 若需要逐项追溯，以该 checkpoint 目录的 `train_config.json` 和数据目录的
`quality_report.json` 为准。聊天记录没有保留一条可证明与该 checkpoint 完全同参的单行启动命令；
不得用猜测参数重新训练后冒充该结果。

### 16.6 V4B：失败 build 与修复后的数据账本

| 项目 | 路径 | 状态 |
| --- | --- | --- |
| 初始错误 build | `$HV_ROOT/datasets/hv_v4b_scorefirst_grouped_20261009/` | 历史输出：train=0、val=11,344、test=11,648；不可训练 |
| diagnostics | `$HV_ROOT/eval/v4b_scorefirst_grouped_20261009/{build_diagnostics.json,dataset_audit.json}` | 能定位 split allocator bug |
| V4B code | `scenario-dreamer/build_high_value_scene_dataset_v4b.py`、`audit_high_value_scene_dataset_v4b.py`、`train_high_value_scene_generator_v4b.py` | 保留，不改写 V4A/V3 |

不存在可接受的 `V4B_BEST` 结果路径。任何新聊天若发现 V4B checkpoint，应先检查是否来自空 train split；
在未证实前不能纳入比较。

### 16.7 V4C：manifest、sidecar、dataset、训练、审计和 online 证据

#### A. route manifest 与 sidecar

| 步骤 | 命令入口 | 输入 | 输出路径 | 结果 |
| --- | --- | --- | --- | --- |
| manifest build | `build_high_value_scene_dataset_v4b.py --route_interaction_manifest_only` | 109,937 raw union | `$HV_ROOT/eval/v4c_route_weighted_allpositive_20261009/route_target_manifest.jsonl` | 22,992 records、45,984 target actors |
| CARLA route sidecar | `RIFT/scripts/enrich_rift_interaction_routes.py` | manifest + raw union + CARLA port 48600 | `$HV_ROOT/route_interaction_sidecars/v4c_route_weighted_allpositive_20261009/route_interaction_sidecar.jsonl` | 43,611/45,984 target routes written |
| sidecar summary | same script | same | `.../target_route_outcomes.jsonl` 和 summary JSON | missing 2,373：770 waypoint missing，1,603 trace short |
| formal V4C build | `build_high_value_scene_dataset_v4c.py` | raw union + route sidecar | `$HV_ROOT/datasets/v4c_route_weighted_allpositive_20261009/` | 20,406，16,362 / 2,005 / 2,039 |
| data audit | `audit_high_value_scene_dataset_v4c.py` | V4C split JSONL | `$HV_ROOT/eval/v4c_route_weighted_allpositive_20261009/v4c_dataset_audit.json` | passed，no episode leakage |

严格说明：启动 `enrich_rift_interaction_routes.py` 前 CARLA 不能以 root 运行；若 `CarlaUE4.sh` 报
`Refusing to run with root privileges`，应切换 `riftuser`。该 step 的输出不完整（94.84%）是已知数据事实，
不是可以忽略的 warning。

#### B. C0/C2 模型、评估和 online

| 步骤 | checkpoint / 输入 | 结果路径 | 结论 |
| --- | --- | --- | --- |
| C0 training | V3 best + V4C data | `$HV_ROOT/checkpoints/hv_route_inpaint_ldm_v4c_c0_from_v3_20261010/{best.ckpt,history.json,train_config.json}` | best epoch2，val pos=0.1242552 |
| C2 training | V3 best + V4C data + route losses | `$HV_ROOT/checkpoints/hv_route_inpaint_ldm_v4c_c2_route_sync_from_v3_20261010/{best.ckpt,history.json,train_config.json}` | best epoch2，val pos=0.1289569 |
| gradient smoke | V3 best + V4C train | `$HV_ROOT/eval/v4c_route_weighted_allpositive_20261009/` 的 smoke 输出/terminal log | direct route gradient L2=18.49，证明 loss 有梯度，不证明性能 |
| offline C0/C2 S1 | V4C test | `$HV_ROOT/eval/v4c_route_weighted_allpositive_20261009/offline_c0_c2/{c0_raw_s1.json,c2_raw_s1.json}` | C2 route proxy 较好、invalid 更差 |
| sampling contract | V4C test | `$HV_ROOT/eval/<contract audit dir>/c0_sampling_contract_512.json` 与 `c2_sampling_contract_512.json` | AE/x0/reverse 三段故障定位 |
| target consistency | V4C all splits | `$HV_ROOT/eval/<contract audit dir>/v4c_snapshot_target_consistency_all_splits.json` | 直接同 condition 多 target pair=0 |
| clean online smoke | initial-only route1711 C0/C2 | 对应 `$HV_ROOT/eval/...` 和 `$ROOT/RIFT/hv_v4c_*...` | 两者 0 spawn；不可升级到 cross-town |

`<contract audit dir>` 指实际使用的目录，例如：

```text
$HV_ROOT/eval/contract_and_joint_audit_20261008_retry1
$HV_ROOT/eval/v4c_route_weighted_allpositive_20261009/offline_c0_c2
```

不要用 `summary.json` 的空 lifecycle 来判断 C0/C2 模型“不交互”；正确的先验结论是更早一层：
它们没有 spawn 成 actor，因此 lifecycle/RIFT 指标不适用。

### 16.8 Background：采集、构建、训练、评估证据

| 步骤 | 输入/命令入口 | 输出路径 | 结果/解释 |
| --- | --- | --- | --- |
| generic base audit | `eval_scenario_dreamer_background_generator.py` | `$HV_ROOT/eval/sd_background_generic_ldm_fov64_n1_test200_s4/` | generic base 不可部署；800 sample fully feasible 13.875% |
| ordinary BV collection | `RIFT/scripts/plan_scenario_dreamer_background_collection.py`、`render_background_collection_launch.py` | `$HV_ROOT/rift_labeled_background_full_non_town12_50k_16workers/` | 主采集完成到 97.97%，余 250 任务 |
| failed recovery | `render_background_capacity_recovery_launch.py` | `$HV_ROOT/rift_labeled_background_full_non_town12_50k_16workers_recovery_remaining_20261005/` | 250/250 failed、0 JSON；不再采集 |
| dataset build | `build_scenario_dreamer_background_dataset.py` | `$HV_ROOT/datasets/scenario_dreamer_background_directional_existing_raw_20261005/` | directional max-8 ordinary BV task |
| dataset diagnostics | `audit_scenario_dreamer_background_dataset.py` | `$HV_ROOT/eval/background_existing_raw_build_20261005/background_build_diagnostics.json` | town/topology/no obvious distribution failure |
| trainer | `train_scenario_dreamer_background.py` | `$HV_ROOT/checkpoints/scenario_dreamer_background_directional_existing_raw_gpu2_20261006/{best.ckpt,history.json,train_config.json}` | best epoch0，val pos=0.1344652 |
| raw S1 eval | `eval_scenario_dreamer_background.py` | `$HV_ROOT/eval/background_directional_existing_raw_gpu2_20261006/{test_raw_s1.json,test_raw_s1_generated.jsonl}` | 5,184 records；forward violation 20.08% |
| S4 oracle | same evaluator | `.../test_raw_s4_oracle.json` | oracle，仅用于分布诊断 |

历史 collection launcher 的精确 worker/port plan 位于：

```text
$HV_ROOT/plans/background_full_non_town12_50k_16workers_launch.sh
$HV_ROOT/plans/background_full_non_town12_50k_16workers*_plan.json
```

它们是历史恢复证据，不应不加检查地重新执行。先读 plan、查看完成 marker 和 worker log，避免重采已有数据。

### 16.9 HV/background 在线输出的目录约定与结果汇总命令

每次 online experiment 同时有两类输出：

```text
$ROOT/RIFT/<OUT>/eval/.../simulation_results.json       # ego/RIFT evaluator statistics
$HV_ROOT/online_debug/<OUT>/*.json                      # generator proposal/spawn/audit dump
$HV_ROOT/eval/<TAG>/                                    # matched summary、辅助审计、视频/分析
```

历史 V2 regular online summary：

```bash
cd $ROOT/RIFT
$PY scripts/summarize_high_value_generation_online.py \
  --dump_dir "$HV_ROOT/online_debug/<OUT>" \
  --statistics_dir "$ROOT/RIFT/<OUT>" \
  --output_json "$HV_ROOT/online_debug/<OUT>/summary.json"
```

matched V2/V3/V4A result 的 event-level 统计使用：

```bash
cd $ROOT/RIFT
$PY scripts/analyze_hv_eventual_rift_cbv.py \
  --dump_dir "$HV_ROOT/eval/hv_matched_cross_town_seed0_strict_s1_20261008" \
  --output_json "$HV_ROOT/eval/hv_matched_cross_town_seed0_strict_s1_20261008/hv_eventual_rift_cbv_analysis.json"
```

`analyze_hv_eventual_rift_cbv.py` 输出的 orange->green、latency、sustained frames、terminal reason 只对
**确实 spawned 的 HV actor**有意义。generation call=0、spawn=0 或 actor 状态从未写入 lifecycle 时，
对应 ratio 必须记录为 `None`/not applicable，不能当 0% 模型交互率。

### 16.10 追溯某个 checkpoint 的最小操作顺序

```bash
CKPT=<absolute_checkpoint_path>
DATA=<absolute_dataset_dir>
EVAL=<absolute_eval_dir>

ls -lh "$CKPT" "$(dirname "$CKPT")/history.json" "$(dirname "$CKPT")/train_config.json"
wc -l "$DATA/train.jsonl" "$DATA/val.jsonl" "$DATA/test.jsonl"
find "$EVAL" -maxdepth 2 -type f \( -name '*.json' -o -name '*.jsonl' \) -printf '%TY-%Tm-%Td %TT %p\n' | sort
```

然后按以下优先级读取：

1. dataset `quality_report.json` / `audit.json` / build diagnostics；
2. `train_config.json` 和 `history.json` 的 best epoch；
3. raw S1 evaluator JSON，而非 S4 oracle；
4. online `summary.json` 中的 generator/spawn funnel；
5. 若有 spawned actor，再看 lifecycle/route statistics/video。

这样可以避免重复出现“训练已完成但 train JSONL 为 0”、“ego 路线跑完但 generator 未 import”、
“S4 oracle 被误当线上结果”或“未 spawn 却解读 green ratio”的历史错误。
