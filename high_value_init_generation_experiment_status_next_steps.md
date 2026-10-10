# 高价值初始状态生成：技术路线、实验状态与后续调整

## 1. 目标与边界

目标是在 RIFT/CARLA 场景运行中生成 ego 附近高价值交互初始状态，使至少 2 辆周车与 ego route 构成高价值交互。输入应以 ego 状态、ego route、局部地图为核心，可选使用 ego FOV 内已有部分周车状态；输出是若干周车初始状态，后续可将其中关键车注册为 CBV 并交给 RIFT-pluto 控制。

核心边界：

- 不重新实现 RIFT route/conflict/interaction 逻辑，RIFT 原始模块只用于采集标签和在线 audit。
- 不让神经网络学习“RIFT 规则打分器”作为核心任务；RIFT 标签用于筛选训练数据、评估生成结果是否高价值。
- Scenario Dreamer/高价值生成模型负责生成状态，而不是从现有车辆中用 mask 选择 CBV。
- 在线运行时应随 ego 运动持续更新 FOV 内可交互对象；当当前 FOV 内缺少高价值交互车时，触发 rolling generation / inpainting。

## 2. 期望方法

### 2.1 数据侧

在 CARLA/RIFT 中运行 route，采集多个 frame 的 RIFT-native snapshot：

- ego 状态。
- ego 剩余 route。
- 局部地图 lane graph。
- ego FOV/半径内周车状态。
- RIFT 原始 interaction labels，包括每辆周车是否与 ego route 构成路径级交互。

构建训练数据时，只保留至少 2 个 RIFT-positive actor 的 snapshot。训练样本采用 inpainting 形式：

- condition：ego + route + map + 非 target 周车状态。
- target：2 个 RIFT-positive 周车状态。
- RIFT 标签不作为推理输入，只用于决定训练 target 和评估。

### 2.2 模型侧

当前模型路线是 route-conditioned high-value LDM：

- 使用 Scenario Dreamer autoencoder / LDM 权重作为基础。
- 高价值模块把 RIFT snapshot 转成 Scenario Dreamer 风格的局部状态表达。
- 训练目标是给定 partial scene，生成 2 个高价值周车状态。
- 当前状态向量包含 `[x, y, heading, speed, length, width, valid]`。

### 2.3 在线推理侧

在线应分为两个模式。

模式 A：初始化阶段生成

- 在 reset 后，根据 ego 初始状态、route、map、局部已有周车，生成关键周车。
- 如果采用严格 inpainting，可以先隐藏已有 background actor 再补全。
- 但不能要求所有 route 初始帧都必须有 2 个可隐藏 actor，否则低密度场景会直接无法生成。

模式 B：rolling / 动态补全

- ego 每运行一段时间，重新检查 FOV 内已有周车和 RIFT-positive 数量。
- 当当前 FOV 内高价值交互车不足时，触发生成。
- 生成车离开生命周期范围后清理，避免无限累积。
- 若生成车通过 RIFT audit，可选择将 2 个 generated actor 注册为 CBV，交给 RIFT-pluto 控制；否则保留为普通 BV 或记录失败。

## 3. 当前实现状态

### 3.1 RIFT 侧关键脚本/模块

- `RIFT/scripts/run.py`
  - RIFT 主运行入口。
  - 已增加 high-value generation 参数，如 `--high_value_init_generation`、`--hv_generation_rolling`、`--hv_generation_hidden_slots` 等。

- `RIFT/scripts/collect_high_value_init_scenes.py`
  - 采集 RIFT-labeled snapshot 的 wrapper。
  - 用 `--mode collect_hv_generation` 调用 RIFT 主流程。
  - 支持 `--base_port`、`--base_tm_port`，便于多进程 CARLA 采集。

- `RIFT/rift/cbv/initialization/hv_scene_snapshot.py`
  - 从 live CARLA/RIFT 环境导出 snapshot。
  - 调用 RIFT 原始 interaction labeler，不重写 conflict 逻辑。

- `RIFT/rift/cbv/initialization/hv_online_generation.py`
  - 在线生成入口。
  - 当前实现包括 partial context、generation attempts、spawn、RIFT audit、rolling trigger、generated actor 生命周期管理。

- `RIFT/scripts/summarize_high_value_generation_online.py`
  - 汇总 online generation dump 和 RIFT evaluation 结果。

- `RIFT/scripts/visualize_high_value_generation.py`
  - 可视化 online generation dump。

### 3.2 Scenario Dreamer 侧关键脚本/模块

- `scenario-dreamer/build_high_value_scene_dataset.py`
  - 从 RIFT-labeled JSON 构建 inpainting/full_generation 数据集。

- `scenario-dreamer/audit_high_value_scene_dataset.py`
  - 检查数据集 schema、target 数量、split 泄漏等。

- `scenario-dreamer/train_high_value_scene_generator.py`
  - 训练 route-conditioned high-value scene generator。

- `scenario-dreamer/eval_high_value_scene_generator.py`
  - 离线生成评估。
  - 注意当前实现是跑完后才写 `output_generated_jsonl`，长时间运行中途可能看不到输出文件。

- `scenario-dreamer/visualize_high_value_scene_dataset.py`
  - 可视化训练/测试样本的 GT partial/target。

- `scenario-dreamer/high_value_init/high_value_ldm.py`
  - 高价值生成模型主体。

- `scenario-dreamer/high_value_init/scene_generator.py`
  - RIFT online 调用的 generator wrapper。

## 4. 当前数据与训练结果

### 4.1 数据采集与构建

正确合并方式必须保留目录结构，不能把 JSON 扁平软链接到同一个目录，否则同名文件会覆盖。

当前递归合并后：

```json
{
  "num_input_json": 70329,
  "num_records": 13706,
  "num_skipped": 56623,
  "num_train_records": 10939,
  "num_val_records": 1518,
  "num_test_records": 1249
}
```

数据 audit：

```json
{
  "passed": true,
  "num_records": 13706,
  "observed_count_mean": 3.6345,
  "rift_positive_count_original_mean": 2.5676,
  "errors_by_type": {},
  "episode_split_leakage_examples": {}
}
```

分析：

- 有效样本率约 19.5%，合理，因为要求至少 2 个 RIFT-positive actor 且 target 在 FOV 内。
- 平均 observed actor 数约 3.6，说明 partial context 较稀疏。
- 平均 RIFT-positive 数约 2.57，数据确实偏“刚好 2 个高价值车”的场景。

### 4.2 训练结果

使用 13706 条数据，基于已有 high-value checkpoint 继续训练：

```text
train records: 10939
val records: 1518
epochs: 20
best_epoch: 1
best_val_loss: 0.07553
last_val_loss: 0.09956
```

最佳 epoch 指标：

```text
val_decoded_state_loss: 0.3197
val_decoded_position_l1: 0.1711
val_decoded_heading_l1: 0.1252
val_decoded_speed_l1: 0.1289
```

分析：

- 新数据训练的 best val loss 优于之前 5635 条数据的 `0.08419`。
- 但 best 在 epoch 1，后续 train loss 继续下降、val loss 变差，存在过拟合/漂移。
- 后续使用 `best.ckpt`，不要使用 `last.ckpt`。

### 4.3 离线评估结果

快速测试 100 条、每条 1 sample：

```json
{
  "num_records": 100,
  "num_samples_per_condition": 1,
  "best_of_n_position_l2": 24.0989,
  "best_of_n_heading_abs": 0.5243,
  "best_of_n_speed_l1": 5.6738
}
```

生成样本中出现：

```text
x/y 贴近 FOV 边界，例如 [-32, 32]
length/width 异常，例如 length=15.7, width=3.95
```

分析：

- 离线单样本生成质量不稳定。
- 当前 loss 下降并不等价于最终 decoded state 生成质量好。
- 位置误差大说明模型还没有稳定学会“给定 partial scene 补全真实 2 个 target actor”的分布。
- 但离线 position L2 衡量的是重建 test 中的真实 target，不等价于“生成任意 RIFT-positive 高价值车”的成功率；最终仍需在线 RIFT audit。

## 5. 当前 online 实验结果

### 5.1 运行结果

Route `1711`，partial + rolling smoke：

```text
All scenarios are completed
route_completion: 100
driving_score: 100
collision_routes_ratio: 0
```

生成 dump：

```text
frame 0:   success=False, hidden=1, failure=partial context prepared 1 hidden slots; requires 2
frame 50:  success=False, hidden=1, failure=partial context prepared 1 hidden slots; requires 2
frame 100: success=False, hidden=0, failure=partial context prepared 0 hidden slots; requires 2
```

online summary：

```json
{
  "num_dumps": 3,
  "generation_success_rate": 0.0,
  "carla_spawn_success_rate": 0.0,
  "rift_audit_pass_rate": 0.0,
  "failure_reasons": {
    "partial context prepared 1 hidden slots; requires 2": 2,
    "partial context prepared 0 hidden slots; requires 2": 1
  }
}
```

### 5.2 关键分析

这次 online 失败不是模型生成后失败，而是模型根本没有被调用：

```text
proposal_null=True
audit_after_generation=None
```

原因是当前 online 入口要求 strict inpainting：

```text
必须先隐藏 2 个已有 background actor
隐藏不足则直接 failure
```

但 `RouteScenario_1711` 在 frame 0/50/100 附近可隐藏 background actor 不足 2 个，且原始 RIFT-positive 数为 0。因此 rolling 虽然正确触发，但被 hidden-slot 前置条件拦住。

这说明当前 online 不 work 的直接原因是推理任务定义和代码入口问题，而不是模型本身已经在线失败。

## 6. 当前不合理点

### 6.1 推理任务定义与代码入口不一致

离线训练的本质是：

```text
从高价值场景移除 2 个 target actor -> 用剩余 partial context 补全 target
```

当前 online strict inpainting 却要求：

```text
现场必须先有 2 个 background actor 可隐藏 -> 才允许生成
```

这会导致低交通密度场景无法生成高价值车，和“缺高价值交互车时主动生成”的目标冲突。

### 6.2 初始化生成与 rolling 生成逻辑需要分离

初始化阶段可以做一次生成，但不能要求 frame0 必然满足 2 个隐藏槽位。

rolling 阶段应作为重点：

- 随 ego 运动更新 FOV。
- 检查当前 RIFT-positive 数。
- 不足时生成关键交互车。
- 生成车离开生命周期范围后清理。

当前 rolling 已触发，但仍被 strict hidden-slot 条件拦住。

### 6.3 生成车未注册为 CBV

当前 `high_value_init_generation` 生成的车被注册为 `BVs`：

- 视频显示为蓝色。
- 由 CARLA Traffic Manager 控制。
- 不会显示为紫色 CBV。
- 不会由 RIFT-pluto 控制。

如果目标是多智能体对抗/协同交互，生成成功后需要把通过 RIFT audit 的关键 generated actors 注册为 CBV。

### 6.4 模型生成状态仍不稳定

离线评估显示：

- 单样本 position error 大。
- 速度误差大。
- 出现 FOV 边界饱和。
- length/width 异常。

这说明训练目标/状态约束需要调整，不能只依赖当前 latent loss。

### 6.5 dump/debug 信息不完整

当前 hidden slots 不足时提前返回，failure result 中应完整记录：

- `hidden_context_actor_ids`
- `hidden_context_actors`
- 当前 agent 数量。
- 当前 RIFT-positive 数。
- rolling 是否触发。

这不影响失败原因，但影响调试。

## 7. 建议调整方案

### 7.1 将 online generation 拆成两个明确模式

`strict_inpainting`：

- 必须隐藏足够 actor。
- 隐藏不足时记为 skipped，不记为模型生成失败。
- 用于验证训练/推理完全一致的受控实验。

`sparse_inpainting`：

- 不要求隐藏已有 actor。
- 输入 ego + route + map + 当前已有部分 actors。
- 直接生成 2 个新关键 actor。
- 更符合真实 online 高价值初始状态生成。

当前任务应以 `sparse_inpainting` / rolling generation 为主。

### 7.2 调整 online 入口

当 hidden slots 不足时：

- strict 模式：记录 skip，继续 route，不调用模型。
- sparse 模式：继续调用 generator，用当前 observed actors 作为 partial context。

生成后：

1. CARLA spawn。
2. 调用 RIFT 原始 interaction audit。
3. 若生成 actors 中至少 2 个 RIFT-positive：
   - 可将这 2 个 generated actors 注册为 CBV。
   - 调用 `_activate_CBV()`。
   - 交给 `rift_pluto` 控制。
4. 若不通过：
   - 记录失败。
   - 不让 RIFT 从所有 BV 里重新选，避免淹没模型输出。

### 7.3 调整训练数据/目标

继续使用 RIFT-positive 场景构建训练集，但区分两种训练任务：

strict inpainting 数据：

```text
observed = full scene - 2 target actors
target = removed 2 RIFT-positive actors
```

sparse generation 数据：

```text
observed = 当前可观测非 target actors，可以为 0 或很少
target = 2 RIFT-positive actors
```

需要让训练覆盖 online 会遇到的低 observed_count 场景，否则 rolling sparse generation 泛化不足。

### 7.4 调整模型状态输出约束

建议：

- 固定或弱学习 length/width，不让模型自由生成异常尺寸。
- 对 x/y 加合理范围约束，但不能简单硬过滤高交互远车。
- 对速度范围做物理约束。
- 加强 decoded state loss 或把 decoded state loss 作为主评价，而不是只看 latent loss。
- 增加 RIFT-audit-based offline reranking/eval，但 RIFT score 不作为推理输入。

### 7.5 在线评估指标

必须同时报告：

- generation_success_rate。
- carla_spawn_success_rate。
- rift_audit_pass_rate。
- generated actor count。
- generated actors 中 RIFT-positive 数。
- CBV registration count。
- route_completion。
- driving_score。
- collision_routes_ratio。
- rolling 触发次数和 frame 分布。
- 生成车离开 FOV 后的清理情况。

## 8. 常用实验命令

### 8.1 环境变量

```bash
export ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
export CARLA_ROOT=$ROOT/carla
export PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
export HV_ROOT=$ROOT/data/hv_init
export SD_AE_CKPT=$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_autoencoder_nuplan/last.ckpt
export SD_LDM_CKPT=$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_ldm_large_nuplan/last.ckpt
export PYTHONPATH=$ROOT/scenario-dreamer:$ROOT/RIFT:$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:$PYTHONPATH
```

### 8.2 构建递归合并目录

```bash
COMBINED=$HV_ROOT/rift_labeled_all_recursive_$(date +%Y%m%d_%H%M%S)
mkdir -p "$COMBINED"

link_tree () {
  tag="$1"
  src="$2"
  find "$src" -type f -name "*.json" -print0 | while IFS= read -r -d "" f; do
    rel="${f#$src/}"
    dst="$COMBINED/$tag/$rel"
    mkdir -p "$(dirname "$dst")"
    ln -s "$f" "$dst"
  done
}

link_tree base "$HV_ROOT/rift_labeled"
link_tree parallel "$HV_ROOT/rift_labeled_parallel"
link_tree seed4_8 "$HV_ROOT/rift_labeled_seed4_8_parallel"

find "$COMBINED" -type l -name "*.json" | wc -l
```

### 8.3 构建与审核数据集

```bash
cd $ROOT/scenario-dreamer

$PY build_high_value_scene_dataset.py \
  --input_dir "$COMBINED" \
  --output_dir $HV_ROOT/datasets/inpainting_all_recursive \
  --mode inpainting \
  --min_rift_positive_agents 2 \
  --num_generate_agents 2 \
  --actor_radius 80 \
  --fov 64 \
  --val_ratio 0.1 \
  --test_ratio 0.1 \
  --seed 0

$PY audit_high_value_scene_dataset.py \
  --train_jsonl $HV_ROOT/datasets/inpainting_all_recursive/train.jsonl \
  --val_jsonl $HV_ROOT/datasets/inpainting_all_recursive/val.jsonl \
  --test_jsonl $HV_ROOT/datasets/inpainting_all_recursive/test.jsonl \
  --expected_mode inpainting \
  --expected_num_generate_agents 2 \
  --fov 64 \
  --output_json $HV_ROOT/datasets/inpainting_all_recursive/audit.json
```

### 8.4 训练

```bash
cd $ROOT/scenario-dreamer

CUDA_VISIBLE_DEVICES=0 $PY train_high_value_scene_generator.py \
  --mode inpainting \
  --train_jsonl $HV_ROOT/datasets/inpainting_all_recursive/train.jsonl \
  --val_jsonl $HV_ROOT/datasets/inpainting_all_recursive/val.jsonl \
  --output_dir $HV_ROOT/checkpoints/hv_route_inpaint_ldm_all_recursive \
  --autoencoder_checkpoint $SD_AE_CKPT \
  --ldm_checkpoint $SD_LDM_CKPT \
  --resume_checkpoint $HV_ROOT/checkpoints/hv_route_inpaint_ldm_combined_unique/best.ckpt \
  --num_generate_agents 2 \
  --epochs 20 \
  --batch_size 16 \
  --lr 5e-5 \
  --decoded_state_loss_weight 0.1
```

### 8.5 快速离线评估

```bash
export HV_CKPT=$HV_ROOT/checkpoints/hv_route_inpaint_ldm_all_recursive/best.ckpt

CUDA_VISIBLE_DEVICES=0 $PY eval_high_value_scene_generator.py \
  --mode inpainting \
  --jsonl $HV_ROOT/datasets/inpainting_all_recursive/test.jsonl \
  --checkpoint $HV_CKPT \
  --autoencoder_checkpoint $SD_AE_CKPT \
  --ldm_checkpoint $SD_LDM_CKPT \
  --output_json $HV_ROOT/eval/inpainting_all_recursive_test_eval_100_s1.json \
  --output_generated_jsonl $HV_ROOT/eval/inpainting_all_recursive_test_generated_100_s1.jsonl \
  --num_generate_agents 2 \
  --num_samples_per_condition 1 \
  --max_records 100
```

### 8.6 当前 online smoke

```bash
runuser -u riftuser -- bash -lc '
export ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
export CARLA_ROOT=$ROOT/carla
export PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
export HV_ROOT=$ROOT/data/hv_init
export PYTHONPATH=$ROOT/RIFT:$ROOT/scenario-dreamer:$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg

cd $ROOT/RIFT
CUDA_VISIBLE_DEVICES=0 $PY scripts/run.py \
  --mode eval \
  --routes rift/scenario/route/bench2drive220.xml \
  --routes_subset 1711 \
  --ego_cfg pdm_lite.yaml \
  --cbv_cfg rift_pluto.yaml \
  --no_resume \
  -rep 1 \
  --save_bev \
  --rift_hd_map_dir $ROOT/RIFT/data/map_data/HD-Map \
  --rift_speed_limits_dir $ROOT/RIFT/data/speed_limits/Speed-Limits \
  --high_value_init_generation \
  --hv_generation_generator_entry high_value_init.scene_generator:ScenarioDreamerHighValueSceneGenerator \
  --hv_generation_checkpoint $HV_ROOT/checkpoints/hv_route_inpaint_ldm_all_recursive/best.ckpt \
  --hv_autoencoder_checkpoint $ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_autoencoder_nuplan/last.ckpt \
  --hv_ldm_checkpoint $ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_ldm_large_nuplan/last.ckpt \
  --hv_generation_mode inpainting \
  --hv_num_generate_agents 2 \
  --hv_generation_hidden_slots 2 \
  --hv_generation_max_attempts 16 \
  --hv_generation_rolling \
  --hv_generation_rolling_min_step 50 \
  --hv_generation_rolling_interval_steps 50 \
  --hv_generation_rolling_min_rift_positive 2 \
  --no_hv_generation_fail_on_invalid \
  --hv_generation_dump_dir $HV_ROOT/online_eval_partial_rolling_smoke_all_recursive
'
```

### 8.7 汇总 online 结果

```bash
cd $ROOT/RIFT

$PY scripts/summarize_high_value_generation_online.py \
  --dump_dir $HV_ROOT/online_eval_partial_rolling_smoke_all_recursive \
  --statistics_dir $ROOT/RIFT/log/eval \
  --output_json $HV_ROOT/online_eval_partial_rolling_smoke_all_recursive_summary.json

cat $HV_ROOT/online_eval_partial_rolling_smoke_all_recursive_summary.json
```

## 9. 后续优先级

1. 修正 online generation 入口：增加 `sparse_inpainting`，不要因 hidden slots 不足阻断生成。
2. 修正 failure dump：记录完整 hidden/context/debug 信息。
3. 生成成功后，将通过 RIFT audit 的 generated actors 注册为 CBV，并交给 RIFT-pluto 控制。
4. 调整训练数据，增加低 observed_count / sparse context 样本，使训练和 rolling online 更一致。
5. 约束 length/width/speed，减少异常状态和 FOV 边界饱和。
6. 重新跑离线 s1、小规模 online、多 route online，再汇总 RIFT audit/pass/collision/route completion。

