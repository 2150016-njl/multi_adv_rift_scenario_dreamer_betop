# V5：真实 CARLA 四维原始状态的单次联合生成实验（**代码交付版，不是已训练/已上线成功版**）

## 设计与严格约束

- **不修改** V2/V3/V4A/V4C 和 RIFT 原生 `CBVs_recog()`；不覆盖其 checkpoint。
- 输入：ego route、正确方向的 HD Map (`right_handed`)、FOV 内已有普通车辆。**不输入**在线不可见的 RIFT target route/score/interaction waypoint。
- 候选：由地图与 ego route **在生成前**确定候选有向车道和 route 邻近锚点；这是模型的条件构造，并非对生成结果的吸附或筛选。当前 route 邻近锚点是几何**代理**，不是原生 RIFT `trace_route`。
- 输出模型：新的 `CARLAJointRawGenerator`，没有冻结 NuPlan AE / LDM。模型**一次**采样一个有序双车候选组合和一个二维车辆联合 8D 高斯状态，直接通过模型内的可微分有向车道坐标头得到两个 `x,y,yaw_rad,speed_mps`。
- 不是 `generate 8 -> rerank 2`，不做 rejection、snap、clamp、reroute、CBV 直接注册、世界 tick，也不做二次模型调用。模型状态通过已有 RIFT 严格在线入口直接 spawn。
- 车型真实尺寸始终由 **CARLA 固定** `vehicle.tesla.model3` 蓝图决定。V5 序列化中仅提供两个兼容的尺寸占位字段，**不据此选择车型或改变 CARLA 车框**。
- 神经网络输出头中的 `sigmoid/tanh` 和有向车道 arclength 插值定义 **预测变量本身**，不是先自由预测世界坐标再后处理。这限制生成状态分布的支持域，但不保证真实 CARLA waypoint 或冲突认定成功。
- 当前不考虑推理时选取最佳 seed；不改变每个场景一次采样、两车原始状态的在线合同。

## 源码清单

```
scenario-dreamer/high_value_init/v5_candidates.py
scenario-dreamer/high_value_init/v5_model.py
scenario-dreamer/high_value_init/scene_generator_v5.py
scenario-dreamer/build_high_value_scene_dataset_v5.py
scenario-dreamer/audit_high_value_scene_dataset_v5.py
scenario-dreamer/train_high_value_scene_generator_v5.py
scenario-dreamer/eval_high_value_scene_generator_v5.py
scenario-dreamer/test_high_value_v5.py
scenario-dreamer/test_v5_dataset_smoke.py
run_v5_strict_online.sh
V5_README_CN.md
```

## 0. 安装：只新增 V5 文件

```bash
export ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
export PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
cd "$ROOT"
# 下载本次交付的 v5_patch.tar.gz 到 ROOT 后：
# tar -xzf v5_patch.tar.gz -C "$ROOT"  （压缩包内路径直接从 scenario-dreamer/ 开始）
"$PY" -m py_compile scenario-dreamer/high_value_init/v5_candidates.py scenario-dreamer/high_value_init/v5_model.py scenario-dreamer/high_value_init/scene_generator_v5.py
cd "$ROOT/scenario-dreamer"
"$PY" -m unittest -v test_high_value_v5.py
"$PY" test_v5_dataset_smoke.py  # CPU 合成测试，产物在临时目录；绝不能引用为 RIFT 训练结果
```

## 1. 是否需要重新采集数据？

**不立即采集。** 既有 V4C 训练集已包含原生 RIFT target-only 路径交互标签，优先复用。旧 V2 的 0 lane 数据不适合作为 V5 正确地图条件的训练依据。V4C 历史约 16,362 train/2,005 val/2,039 test（需以服务器实时文件为准）；其中包含弱正样本，所以 V5 默认要求两名 target 的 RIFT 交互强度均 >=0.4，且二者都能被地图候选 + 物理原始状态准确表示。

**明确可能失败：** 由于同时要求路口在 64m FOV 内、目标沿有向道路、对应锚点、真实车框无初始重叠、两名高强度 target，过滤后可能不足以训练。应先统计，不得悄悄放宽约束或构造假的交互标签。如果覆盖不足，才设计针对缺失 Town/route/interaction 模式的补采计划。

### 构建与统计

```bash
export ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
export PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
export V4C_DATA="$ROOT/data/hv_init/datasets/v4c_route_weighted_allpositive_20261009"
export V5_DATA="$ROOT/data/hv_init/datasets/v5_joint_raw_strong_20261012"
cd "$ROOT/scenario-dreamer"
for split in train val test; do test -s "$V4C_DATA/$split.jsonl" || { echo "Missing V4C $split data";exit 1;};done
"$PY" build_high_value_scene_dataset_v5.py \
  --train-jsonl "$V4C_DATA/train.jsonl" \
  --val-jsonl "$V4C_DATA/val.jsonl" \
  --test-jsonl "$V4C_DATA/test.jsonl" \
  --output-dir "$V5_DATA" --min-rift-score 0.4 \
  --max-gt-lane-distance 1.25 --max-anchor-distance 12 \
  --min-train-records 1000 --min-val-records 100 --min-test-records 100
"$PY" -m json.tool "$V5_DATA/dataset_audit.json" | less
```

报告涵盖原始量、接受率、拒绝原因、Town/拓扑/交互强度/候选数量分布、canonical 内容 hash 跨 split 检查和 episode 去重。只要 `ready_to_train=false`，构建脚本直接以非零状态退出，**训练不应开始**。

## 2. 第一门：表示正确性与条件一致性（无训练、无 CARLA）

```bash
"$PY" audit_high_value_scene_dataset_v5.py \
  --data-dir "$V5_DATA" --max-per-split 0 \
  --output-json "$V5_DATA/representation_gate.json"
"$PY" -m json.tool "$V5_DATA/representation_gate.json" | head -80
```

`passed=true` 的必要条件：GT 经 V5 chart 编码→解码，最大位置误差 <=0.03m、heading<=0.01rad、speed<=0.02m/s；所有审计记录的在线候选必须与重新从自身 `condition` 计算所得候选严格相同；不允许 RIFT 标签泄漏进 `condition`。**这只是表示 oracle，不是模型学好生成的证据**。

## 3. 新 V5 模型训练（GPU；不加载/覆盖 NuPlan AE、V2–V4 checkpoint）

```bash
export V5_CKPT_DIR="$ROOT/data/hv_init/checkpoints/hv_v5_joint_raw_strong_route_20261012"
test ! -e "$V5_CKPT_DIR" || { echo "Existing run dir, use a NEW unique name"; exit 1; }
cd "$ROOT/scenario-dreamer"
CUDA_VISIBLE_DEVICES=0 "$PY" train_high_value_scene_generator_v5.py \
  --data-dir "$V5_DATA" --output-dir "$V5_CKPT_DIR" \
  --device cuda --hidden 128 --epochs 12 --batch-size 8 \
  --lr 0.0003 --s1-val-records 128 --patience 4 --seed 42 \
  --min-joint-s1-rate 0.45
```

训练损失包括有监督的**联合 candidate pair 类别预测**、连续 chart Gaussian NLL、原始位置/航向/速度、clearance，以及小权重的**静态 ego/target route 同步代理**（不是 RIFT native trace-route）。不通过 GT 注入候选。checkpoint 根据固定 128 条 held-out validation **每条恰好一次 S1** 的联合可行性和 route 代理选择，非 x0 损失，非 best-of-N。训练日志保存 `history.json`, `best.ckpt`, `last.ckpt`, `training_readiness.json`。

注意：`training_readiness.json` 的数值由验证集随机数固定的 S1 计算；不是“保证线上变绿”。不得覆盖旧模型或复用同名 output dir。

## 4. 第二门：未参与训练的单次完整 S1

```bash
CUDA_VISIBLE_DEVICES=0 "$PY" eval_high_value_scene_generator_v5.py \
  --jsonl "$V5_DATA/test.jsonl" --checkpoint "$V5_CKPT_DIR/best.ckpt" \
  --output-json "$V5_CKPT_DIR/test_s1.json" \
  --device cuda --seed 2027 --max-records 0
"$PY" - <<'PY'
import json,os
r=json.load(open(os.path.join(os.environ['V5_CKPT_DIR'],'test_s1.json')))
print('records/samples:',r['num_conditions'],r['sampling_calls'])
for k,v in r['summary'].items():print(k,v)
PY
```

门槛：`both_joint_feasible` >=0.45 **并且** `joint_plus_route_proxy` >=0.25；与 V2+legacy 的成功演示比较时，必须将不同的训练/推理条件清楚标注。所有输出是原始 `generated_agent_states`，没有后处理。Offline 代理不等同于 RIFT native 交互；两个指标达标仍需第三门。

## 5. 第三门：严格 CARLA/RIFT 单次闭环

必须使用已有严格在线 `run_strict_hv_eval.sh` 和 `audit_hv_one_shot_results.py`，不能使用历史 `run_hv_v4c_matched_online_eval.sh` 中容许 rolling、多次尝试的配置。V5 wrapper 校验 checkpoint/map/generation mode；额外 runner 检查训练 S1 readiness。这里**只跑一个 scene seed 一次**。

```bash
cd "$ROOT"
ROOT="$ROOT" PY="$PY" V5_CKPT="$V5_CKPT_DIR/best.ckpt" \
  bash run_v5_strict_online.sh \
    --tag v5_strict_1711_seed1 --routes 1711 --seed 1 \
    --gpu 0 --base-port 31000 --base-tm-port 51000
```

RIFT 输出：`$ROOT/data/hv_init/eval/strict_hv_v5_strict_1711_seed1_dump/strict_audit_summary.json`。验收：`simulator_exit_code=0`、`num_generator_called=1`、`num_both_spawned=1`、`num_at_least_one_native_green=1`、`contract_errors=[]`；阅读对应 raw proposal 和生命周期中 `rift_cbv_activated`。**不直接通过 audit 注册 CBV，不 tick，不重采样，不修改 x/y/yaw/speed**。如果失败，应记录原始 proposal、首次 CARLA spawn error、路口关系，而非筛掉 seed 或多跑直到成功。

只有这一轮确证后才考虑独立路线/seed 的未筛选验证；优势 Seed 0/2 可用于演示，但不能报告为未筛选成功率。

## 6. 局限与修复条件

V5 是一个独立可训练原型，**不是承诺的强交互生成器**。候选当前通过局部 HD Map 与 ego route 的距离/方向构造，尚未等价于 RIFT `GlobalRoutePlanner.trace_route` 的 lane graph / distance；route-gap 只是距离代理；不能在离线审计宣称原生 RIFT green。条件图依赖完整且方向正确的地图中心线；曲线 lane 被限制在模型支持域内，遇上特异道路、交错路段会发生训练记录拒绝或在线 candidate 不足。

如 V5 的训练样本不足或测试 gate 失败，下一工程任务应是**在输入图中增加 CARLA/RIFT 同源可行驶 lane-connector 拓扑/规划路径度量**，而非推理时吸附、重复采样或增加重训轮次。离线 target-only RIFT sidecar 可用于监督，但绝不能作为在线 condition。如果一份新的训练数据覆盖了缺失路口和方向，再以新的固定 split 和未筛选 seed 复测。

## 7. 已执行与未执行

- 已执行：Python 语法检查、9 个 CPU 单测、合成标签数据构建→2 epoch CPU 训练→S1 评估、以及条件一致性和 chart oracle 检查。
- 合成标签的 CPU 训练 **未达到 S1 在线部署门槛**，这份数据不可用于科研性能结论。
- 未执行：真实 V4C 数据构建/训练、V5 真正 checkpoint 的离线性能、实际 CARLA/RIFT online。服务器不存在于本执行环境，不能称这些已通过。
