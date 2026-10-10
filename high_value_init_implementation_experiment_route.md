# 高价值初始状态生成实现与实验技术路线

本文是实现版技术路线，目标是指导后续代码开发、数据采集、训练、验证和在线评估。它不替代分析文档 `high_value_init_generation_route.md`。

## 0. 核心约束

本路线固定以下约束，后续实现不得偏离：

```text
1. 不训练神经网络选择 CBV。
2. 不训练神经网络在线预测 RIFT score 来选车。
3. 不做“扰动已有 actor 再预测 delta 恢复”的任务。
4. RIFT 只用于筛选高价值训练数据、记录标签、生成后评估。
5. 生成模型输出周车初始状态集合，不输出 CBV id。
6. 自由生成的新车是匿名周车，不绑定原 CARLA actor id。
7. inpainting 里的已知/未知标志只是条件输入，不是模型预测的 mask。
8. 第一阶段只做 reset-time 初始状态生成，不做运行中连续补车。
```

## 1. 总体目标

输入：

```text
ego_state
ego_route
local_map / lane_graph
optional_observed_agents
num_generate_agents
```

输出：

```text
generated_agent_states[num_generate_agents]
```

每辆周车状态建议统一为：

```text
[x, y, heading, speed, length, width, type]
```

坐标系第一版统一使用：

```text
ego-centric right-handed local frame, ego heading aligned with +x
```

生成结果落地到 RIFT/CARLA 后，再调用 RIFT 原始路径级交互逻辑做 audit：

```text
num_rift_positive_agents >= 2
```

该指标是评估结果，不是在线兜底筛选逻辑。

## 2. 系统结构

```text
CARLA/RIFT closed-loop data collection
        |
        v
scene snapshot dump
        |
        v
RIFT original interaction labeling
        |
        v
high-value scene database
        |
        +-----------------------+
        |                       |
        v                       v
full-generation dataset     inpainting dataset
        |                       |
        +-----------+-----------+
                    v
        Scenario Dreamer AE/LDM fine-tuning
                    |
                    v
        offline generation evaluation
                    |
                    v
        RIFT/CARLA reset-time online evaluation
```

## 3. 代码模块规划

### 3.1 RIFT 侧新增模块

#### 3.1.1 场景快照采集

新增：

```text
RIFT/rift/cbv/initialization/hv_scene_snapshot.py
RIFT/scripts/collect_high_value_init_scenes.py
```

职责：

```text
1. 在 CARLA/RIFT 闭环运行过程中采集 ego、route、map、周车状态。
2. 周车范围由 radius 控制，例如 80m。
3. 每隔 sample_interval_s 采样一次，避免相邻帧过密。
4. 保存原始快照，不在采集阶段做生成或优化。
```

输出目录：

```text
$ROOT/data/hv_init/raw_snapshots/{split}/
```

单个 snapshot 文件：

```text
{town}_{route_id}_{episode_seed}_{frame_idx}.json
```

#### 3.1.2 RIFT 原始标签导出

新增：

```text
RIFT/rift/cbv/initialization/hv_rift_labeler.py
RIFT/scripts/label_high_value_init_scenes.py
```

职责：

```text
1. 直接调用 RIFT 原始 route / conflict / interaction 逻辑。
2. 不重新实现简化版 route。
3. 不改变 RIFT 阈值含义。
4. 为每辆周车记录是否与 ego route 存在路径级交互。
```

输出目录：

```text
$ROOT/data/hv_init/rift_labeled/{split}/
```

每个样本必须保存：

```json
{
  "schema_version": "hv_init_rift_labeled_v1",
  "scene_id": "...",
  "town": "Town03",
  "frame_id": "ego_local_rh_x_forward",
  "ego_state": [0.0, 0.0, 0.0, 0.0, 4.8, 2.0],
  "ego_route": {
    "polyline": [[0.0, 0.0], [1.0, 0.0]],
    "s": [0.0, 1.0]
  },
  "map": {
    "lane_polylines": [],
    "lane_types": [],
    "lane_connections": []
  },
  "agents": [
    {
      "local_id": 0,
      "actor_id": 123,
      "state": [10.0, 3.5, 0.0, 5.0, 4.8, 2.0, 1],
      "source": "carla_actor"
    }
  ],
  "rift_labels": [
    {
      "local_id": 0,
      "actor_id": 123,
      "is_interactive": true,
      "score": 0.83,
      "distance_diff": 2.1,
      "ego_distance": 20.0,
      "actor_distance": 22.1,
      "interaction_point": [20.0, 0.0]
    }
  ],
  "diagnostics": {
    "num_rift_positive_agents": 2
  }
}
```

#### 3.1.3 生成结果在线落地

新增：

```text
RIFT/rift/cbv/initialization/hv_generated_scene_loader.py
RIFT/rift/cbv/initialization/hv_online_initializer.py
```

职责：

```text
1. 接收 Scenario Dreamer 生成的匿名周车状态集合。
2. 在 reset-time 把生成周车落地到 CARLA。
3. 不按旧 actor id 回写。
4. 不用 RIFT 分数改写生成结果。
5. 仅做 CARLA 必要合法性检查：spawn transform、初始碰撞、车道方向、速度范围。
```

注意：

```text
物理合法性检查不是高价值筛选。
若生成结果 RIFT audit 不通过，只记录失败，不用规则补救。
```

### 3.2 Scenario Dreamer 侧新增模块

#### 3.2.1 RIFT 数据集读取与转换

新增：

```text
scenario-dreamer/high_value_init/rift_dataset.py
scenario-dreamer/high_value_init/rift_to_sd_converter.py
```

职责：

```text
1. 读取 RIFT labeled json。
2. 转换成 Scenario Dreamer 使用的 agent/lane graph 数据结构。
3. 保持 ego、route、map、observed agents、generated target agents 的字段清晰分离。
4. 不产生 actor_scores / target_mask / CBV ids 作为训练目标。
```

#### 3.2.2 Route encoder

新增：

```text
scenario-dreamer/high_value_init/route_encoder.py
```

输入：

```text
ego_route.polyline[N, 2]
ego_route.s[N]
```

输出：

```text
route_context_embedding
route_point_embeddings
```

实现建议：

```text
1. route polyline 重采样到固定点数，例如 80。
2. 每个点使用 [x, y, cos_heading, sin_heading, normalized_s]。
3. 通过 PointNet / MLP + Transformer encoder 得到 route condition。
4. route condition 注入 LDM DiT blocks。
```

第一版不要修改 Scenario Dreamer AE 的输入维度，route condition 放在 LDM 侧。

#### 3.2.3 高价值 LDM 包装器

新增：

```text
scenario-dreamer/high_value_init/high_value_ldm.py
scenario-dreamer/train_high_value_scene_generator.py
scenario-dreamer/eval_high_value_scene_generator.py
scenario-dreamer/generate_high_value_scene.py
```

职责：

```text
1. 加载 Scenario Dreamer AE checkpoint。
2. 加载 Scenario Dreamer LDM checkpoint。
3. 在 LDM/DiT 中接入 route condition。
4. 支持 full-generation 和 inpainting 两种训练/推理模式。
5. 生成周车状态集合。
```

## 4. 数据集设计

### 4.1 数据目录

服务器路径：

```bash
export ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
export HV_ROOT=$ROOT/data/hv_init
```

建议目录：

```text
$HV_ROOT/raw_snapshots/train
$HV_ROOT/raw_snapshots/val
$HV_ROOT/raw_snapshots/test

$HV_ROOT/rift_labeled/train
$HV_ROOT/rift_labeled/val
$HV_ROOT/rift_labeled/test

$HV_ROOT/datasets/full_generation/train.jsonl
$HV_ROOT/datasets/full_generation/val.jsonl
$HV_ROOT/datasets/full_generation/test.jsonl

$HV_ROOT/datasets/inpainting/train.jsonl
$HV_ROOT/datasets/inpainting/val.jsonl
$HV_ROOT/datasets/inpainting/test.jsonl

$HV_ROOT/checkpoints
$HV_ROOT/offline_eval
$HV_ROOT/online_eval
$HV_ROOT/viz
```

### 4.2 Full-generation 样本

用于训练：

```text
ego + route + map -> all nearby agents
```

样本 schema：

```json
{
  "schema_version": "hv_init_full_generation_v1",
  "scene_id": "...",
  "split": "train",
  "condition": {
    "ego_state": [],
    "ego_route": {},
    "map": {}
  },
  "target": {
    "agent_states": [],
    "agent_types": []
  },
  "rift_audit": {
    "num_rift_positive_agents": 3,
    "positive_local_ids": [1, 5, 8]
  },
  "source": "real_positive"
}
```

筛选规则：

```text
只把 num_rift_positive_agents >= 2 的 real_positive 样本放入第一版训练集。
```

### 4.3 Inpainting 样本

用于训练：

```text
ego + route + map + observed agents -> missing/generated key agents
```

样本 schema：

```json
{
  "schema_version": "hv_init_inpainting_v1",
  "scene_id": "...",
  "split": "train",
  "condition": {
    "ego_state": [],
    "ego_route": {},
    "map": {},
    "observed_agent_states": [],
    "observed_agent_types": [],
    "num_generate_agents": 2
  },
  "target": {
    "generated_agent_states": [],
    "generated_agent_types": []
  },
  "rift_audit": {
    "removed_positive_local_ids": [1, 5],
    "num_rift_positive_agents_original": 3
  },
  "source": "real_positive"
}
```

构造方式：

```text
1. 从 real_positive 场景中找 RIFT-positive 周车。
2. 随机或按 interaction route-s 选择 2 辆作为 generated target。
3. 从 context 中移除这 2 辆。
4. 剩余周车作为 observed agents。
5. target 是被移除的 2 辆车状态。
```

这里使用的“移除/补全”是训练构造，不是在线预测 CBV。

### 4.4 训练、验证、测试切分

必须按 route/episode 级别切分，不能按 frame 随机切分。

推荐：

```text
train: 80%
val: 10%
test: 10%
```

切分键：

```text
town + route_id + repetition + seed
```

同一 episode 的所有帧必须落在同一 split。

## 5. 数据采集流程

### 5.1 环境变量

```bash
export ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
export CARLA_ROOT=/zeron-vepfs/tjqc/njl/carla
export PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
export PYTHONPATH=$ROOT/RIFT:$ROOT/scenario-dreamer:$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg

export RIFT_HD_MAP_DIR=$ROOT/RIFT/data/map_data/HD-Map
export RIFT_SPEED_LIMITS_DIR=$ROOT/RIFT/data/speed_limits/Speed-Limits
export HV_ROOT=$ROOT/data/hv_init
```

### 5.2 采集 raw snapshots

命令设计：

```bash
runuser -u riftuser -- bash -lc '
export ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
export CARLA_ROOT=/zeron-vepfs/tjqc/njl/carla
export PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
export PYTHONPATH=$ROOT/RIFT:$ROOT/scenario-dreamer:$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
export HV_ROOT=$ROOT/data/hv_init

cd $ROOT/RIFT

CUDA_VISIBLE_DEVICES=0 $PY scripts/collect_high_value_init_scenes.py \
  --routes rift/scenario/route/bench2drive220.xml \
  --ego_cfg pdm_lite.yaml \
  --cbv_cfg rift_pluto.yaml \
  --output_dir $HV_ROOT/raw_snapshots/train \
  --rift_hd_map_dir $ROOT/RIFT/data/map_data/HD-Map \
  --rift_speed_limits_dir $ROOT/RIFT/data/speed_limits/Speed-Limits \
  --sample_interval_s 1.0 \
  --actor_radius 80 \
  --max_episodes 200 \
  --no_resume
'
```

第一轮 smoke 建议：

```text
max_episodes = 5
sample_interval_s = 1.0
```

通过后再扩大。

### 5.3 RIFT 标注

命令设计：

```bash
cd $ROOT/RIFT

$PY scripts/label_high_value_init_scenes.py \
  --input_dir $HV_ROOT/raw_snapshots/train \
  --output_dir $HV_ROOT/rift_labeled/train \
  --rift_hd_map_dir $ROOT/RIFT/data/map_data/HD-Map \
  --rift_speed_limits_dir $ROOT/RIFT/data/speed_limits/Speed-Limits
```

输出统计：

```text
num_scenes
num_real_positive_scenes
positive_scene_ratio
positive_agents_per_scene_mean
positive_agents_per_scene_p90
distance_diff_distribution
interaction_point_distribution
```

## 6. 数据集构建流程

### 6.1 构建 full-generation 数据

命令设计：

```bash
cd $ROOT/scenario-dreamer

$PY build_high_value_scene_dataset.py \
  --input_dir $HV_ROOT/rift_labeled/train \
  --output_jsonl $HV_ROOT/datasets/full_generation/train.jsonl \
  --mode full_generation \
  --min_rift_positive_agents 2 \
  --actor_radius 80 \
  --num_generate_agents 16 \
  --source real_positive
```

说明：

```text
num_generate_agents=16 是外部配置，不是模型预测。
若真实场景超过 16 辆，按距离 ego 最近优先，同时必须保留 RIFT-positive 车辆。
若真实场景少于 16 辆，第一版丢弃该样本，不做 padding 生成。
```

### 6.2 构建 inpainting 数据

命令设计：

```bash
cd $ROOT/scenario-dreamer

$PY build_high_value_scene_dataset.py \
  --input_dir $HV_ROOT/rift_labeled/train \
  --output_jsonl $HV_ROOT/datasets/inpainting/train.jsonl \
  --mode inpainting \
  --min_rift_positive_agents 2 \
  --num_generate_agents 2 \
  --positive_target_policy route_s_diversity \
  --source real_positive
```

`positive_target_policy`：

```text
route_s_diversity:
从 RIFT-positive 周车中选择 ego route-s 分布不同的 target agents。
若有多个等价组合，随机采样。
```

注意：

```text
这个策略只用于构造训练 target，不是推理时模型选 CBV。
RIFT score 只作为 metadata 保存，不作为在线选择逻辑。
```

## 7. 模型实现路线

### 7.1 复用权重

服务器 checkpoint：

```bash
export SD_AE_CKPT=$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_autoencoder_nuplan/last.ckpt
export SD_LDM_CKPT=$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_ldm_large_nuplan/last.ckpt
```

### 7.2 第一版模型：Route-conditioned Inpainting LDM

优先实现 inpainting，而不是 full generation。

原因：

```text
1. 更贴近 Scenario Dreamer 原始 inpainting 机制。
2. 有 observed agents 作为上下文，分布更稳。
3. 不需要模型决定周车数量。
4. 更容易评估“补全关键交互车”是否成功。
```

输入：

```text
ego_state
ego_route
lane_graph
observed_agent_states
num_generate_agents = 2
```

输出：

```text
generated_agent_states[2]
```

实现要点：

```text
1. observed agents 通过 AE encoder 得到 latents。
2. lane graph 通过 AE encoder 得到 lane latents。
3. generated agent slots 从 Gaussian noise 开始。
4. 反向扩散每一步保持 observed agent latents 和 lane latents 固定。
5. route encoder 输出 route condition，注入 DiT。
6. AE decoder 解码 generated agent latents。
```

loss：

```text
diffusion_noise_loss on generated agent latents
decoded_state_loss on generated agent states
set_matching_loss for generated agents
```

不加入第一版：

```text
RIFT score prediction
CBV classifier
actor mask prediction
delta regression
```

### 7.3 第二版模型：Route-conditioned Full-generation LDM

输入：

```text
ego_state
ego_route
lane_graph
num_generate_agents = 16
```

输出：

```text
generated_agent_states[16]
```

实现方式：

```text
1. lane latents 固定。
2. route condition 固定。
3. 全部 agent latents 从 Gaussian noise 开始。
4. decoder 输出 16 辆匿名周车。
```

训练难点：

```text
target 是集合，需要 Hungarian matching。
多模态更强，样本量要求更高。
```

### 7.4 可选 ablation：RIFT score auxiliary loss

第一版不做。

后续若生成结果真实但 RIFT audit 通过率低，可以加一个训练时辅助任务：

```text
decoded generated agents -> auxiliary score head -> predict normalized RIFT score
```

限制：

```text
1. loss_weight <= 0.1。
2. 只在训练时使用。
3. 推理时不读取该 head。
4. 在线不依据该 head 选车。
```

如果该辅助任务提升 RIFT audit pass rate，同时不恶化 realism/map validity，才保留。

## 8. 训练命令设计

### 8.1 训练 inpainting LDM

```bash
cd $ROOT/scenario-dreamer

CUDA_VISIBLE_DEVICES=0 $PY train_high_value_scene_generator.py \
  --mode inpainting \
  --train_jsonl $HV_ROOT/datasets/inpainting/train.jsonl \
  --val_jsonl $HV_ROOT/datasets/inpainting/val.jsonl \
  --output_dir $HV_ROOT/checkpoints/hv_route_inpaint_ldm \
  --autoencoder_checkpoint $SD_AE_CKPT \
  --ldm_checkpoint $SD_LDM_CKPT \
  --num_generate_agents 2 \
  --route_num_points 80 \
  --batch_size 16 \
  --epochs 50 \
  --lr 1e-4 \
  --freeze_autoencoder \
  --no_rift_score_aux
```

训练日志必须包含：

```text
train_loss
val_loss
decoded_position_l1
decoded_heading_error
decoded_speed_l1
map_validity_rate
initial_collision_rate
```

### 8.2 训练 full-generation LDM

```bash
cd $ROOT/scenario-dreamer

CUDA_VISIBLE_DEVICES=0 $PY train_high_value_scene_generator.py \
  --mode full_generation \
  --train_jsonl $HV_ROOT/datasets/full_generation/train.jsonl \
  --val_jsonl $HV_ROOT/datasets/full_generation/val.jsonl \
  --output_dir $HV_ROOT/checkpoints/hv_route_full_ldm \
  --autoencoder_checkpoint $SD_AE_CKPT \
  --ldm_checkpoint $SD_LDM_CKPT \
  --num_generate_agents 16 \
  --route_num_points 80 \
  --batch_size 16 \
  --epochs 80 \
  --lr 1e-4 \
  --freeze_autoencoder \
  --no_rift_score_aux
```

建议只有 inpainting 跑通后再做 full generation。

## 9. 离线评估流程

### 9.1 生成离线样本

```bash
cd $ROOT/scenario-dreamer

CUDA_VISIBLE_DEVICES=0 $PY generate_high_value_scene.py \
  --mode inpainting \
  --jsonl $HV_ROOT/datasets/inpainting/test.jsonl \
  --checkpoint $HV_ROOT/checkpoints/hv_route_inpaint_ldm/best.ckpt \
  --autoencoder_checkpoint $SD_AE_CKPT \
  --num_samples_per_condition 8 \
  --output_dir $HV_ROOT/offline_eval/inpainting/generated
```

### 9.2 RIFT 离线 audit

```bash
cd $ROOT/RIFT

$PY scripts/audit_generated_high_value_scenes.py \
  --generated_dir $HV_ROOT/offline_eval/inpainting/generated \
  --output_json $HV_ROOT/offline_eval/inpainting/rift_audit.json \
  --rift_hd_map_dir $ROOT/RIFT/data/map_data/HD-Map \
  --rift_speed_limits_dir $ROOT/RIFT/data/speed_limits/Speed-Limits
```

输出指标：

```text
num_conditions
num_generated_samples
rift_pass_rate_at_1
rift_pass_rate_at_8
positive_agents_mean
positive_agents_p50
positive_agents_p90
initial_collision_rate
off_lane_rate
wrong_way_rate
speed_out_of_range_rate
```

说明：

```text
pass_at_8 表示同一条件采样 8 次中至少 1 次通过。
这是评估采样多样性，不是在线兜底。
```

## 10. 在线评估流程

### 10.1 reset-time online generation

第一版在线只做 reset-time。

命令设计：

```bash
runuser -u riftuser -- bash -lc '
export ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
export CARLA_ROOT=/zeron-vepfs/tjqc/njl/carla
export PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
export PYTHONPATH=$ROOT/RIFT:$ROOT/scenario-dreamer:$CARLA_ROOT/PythonAPI:$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg
export HV_ROOT=$ROOT/data/hv_init
export SD_AE_CKPT=$ROOT/scratch/scenario_dreamer/checkpoints/scenario_dreamer_autoencoder_nuplan/last.ckpt

cd $ROOT/RIFT

CUDA_VISIBLE_DEVICES=0 $PY scripts/run.py \
  --mode eval \
  --routes rift/scenario/route/bench2drive220.xml \
  --ego_cfg pdm_lite.yaml \
  --cbv_cfg rift_pluto.yaml \
  --no_resume \
  -rep 1 \
  --rift_hd_map_dir $ROOT/RIFT/data/map_data/HD-Map \
  --rift_speed_limits_dir $ROOT/RIFT/data/speed_limits/Speed-Limits \
  --high_value_init_generation \
  --hv_generation_mode inpainting \
  --hv_generation_checkpoint $HV_ROOT/checkpoints/hv_route_inpaint_ldm/best.ckpt \
  --hv_autoencoder_checkpoint $SD_AE_CKPT \
  --hv_num_generate_agents 2 \
  --hv_actor_radius 80 \
  --hv_dump_dir $HV_ROOT/online_eval/reset_time_inpainting
'
```

注意：

```text
该命令参数是目标接口设计。
实现时若 RIFT 当前 run.py 没有这些参数，需要新增 argparse 和 env config 传递。
```

### 10.2 在线统计

新增：

```text
RIFT/scripts/summarize_high_value_generation_online.py
```

命令：

```bash
cd $ROOT/RIFT

$PY scripts/summarize_high_value_generation_online.py \
  --dump_dir $HV_ROOT/online_eval/reset_time_inpainting \
  --output_json $HV_ROOT/online_eval/reset_time_inpainting/summary.json
```

指标：

```text
num_episodes
generation_success_rate
carla_spawn_success_rate
rift_audit_pass_rate
num_rift_positive_agents_mean
ego_collision_rate
route_completion_mean
driving_score_mean
generated_agent_lifetime_mean
```

## 11. 可视化要求

新增：

```text
RIFT/scripts/visualize_high_value_generation.py
scenario-dreamer/visualize_high_value_scene_dataset.py
```

每张图至少包含：

```text
ego pose
ego route
lane graph
observed agents
generated agents
RIFT-positive agents after audit
interaction points
```

输出目录：

```text
$HV_ROOT/viz/dataset
$HV_ROOT/viz/offline_generated
$HV_ROOT/viz/online_eval
```

可视化只用于审核，不参与训练和筛选。

## 12. 阶段验收标准

### 12.1 数据采集验收

```text
[ ] raw snapshot 数量 >= 1000
[ ] 每个 snapshot 有 ego_state、ego_route、map、agents
[ ] ego_route 长度合理
[ ] agents 均在 actor_radius 内
[ ] 坐标系统一到 ego local
```

### 12.2 RIFT 标注验收

```text
[ ] 标注代码直接调用 RIFT 原始 interaction 逻辑
[ ] 每辆 agent 有 is_interactive / score / distance_diff
[ ] 生成 dataset_stats.json
[ ] 可视化抽查 50 个 positive scene
[ ] real_positive 与 real_negative 分开保存
```

### 12.3 Dataset 验收

```text
[ ] train/val/test 按 episode 切分
[ ] inpainting target 来自 RIFT-positive agents
[ ] full-generation target 是完整附近周车集合
[ ] 没有 target_mask / actor_scores / CBV ids 训练字段
[ ] 没有 delta target
```

### 12.4 模型训练验收

```text
[ ] 成功加载 AE checkpoint
[ ] 成功加载 LDM checkpoint
[ ] route encoder 确实参与 forward
[ ] LDM 从 Gaussian noise 开始生成 target agent latent
[ ] observed agents 在 inpainting 中保持固定
[ ] loss 只监督生成目标状态
[ ] 不使用 RIFT score 作为推理输入
```

### 12.5 离线评估验收

```text
[ ] 能批量生成 test samples
[ ] 能对 generated samples 跑 RIFT audit
[ ] 报告 RIFT pass rate
[ ] 报告 realism/map validity/collision 指标
[ ] 保存可视化图
```

### 12.6 在线评估验收

```text
[ ] CARLA reset-time 能调用生成器
[ ] 生成周车能成功 spawn
[ ] RIFT audit 只记录，不改写生成结果
[ ] episode 能正常 rollout
[ ] 保存在线 summary.json
[ ] 保存 BEV/trajectory 可视化
```

## 13. 推荐实验顺序

### Experiment A: 数据可用性

```text
目标：确认 CARLA/RIFT 中真实高价值初始状态比例。
输入：闭环采集 1000 snapshots。
输出：positive ratio、positive agents 分布、可视化。
```

### Experiment B: AE 域适配审计

```text
目标：确认 NuPlan AE checkpoint 能否重建 CARLA/RIFT scene。
输入：RIFT labeled snapshots。
输出：agent reconstruction error、lane reconstruction error、可视化。
```

若 AE 重建失败：

```text
先 fine-tune AE，再训练 LDM。
```

### Experiment C: Route-conditioned inpainting

```text
目标：补全 2 辆 RIFT-positive 周车。
输入：observed agents + ego + route + map。
输出：2 generated agents。
评估：offline RIFT pass rate。
```

### Experiment D: Reset-time online rollout

```text
目标：把 inpainting 生成结果落地 CARLA。
评估：RIFT audit pass rate、collision、route completion。
```

### Experiment E: Full generation

```text
目标：只基于 ego + route + map 生成 16 辆周车。
前提：Experiment C/D 已跑通。
```

### Experiment F: RIFT score auxiliary ablation

```text
目标：验证辅助 score loss 是否真的提升生成分布。
前提：baseline 已稳定。
对比：aux_weight=0.0 / 0.05 / 0.1。
```

## 14. 当前最小可行版本

MVP 只做：

```text
1. CARLA/RIFT 闭环采集 snapshots。
2. RIFT 原始逻辑标注。
3. 构造 inpainting 数据集。
4. Route-conditioned inpainting LDM fine-tuning。
5. 离线 RIFT audit。
6. reset-time 在线评估。
```

MVP 不做：

```text
1. full-generation。
2. 运行中补车。
3. RIFT score auxiliary head。
4. 神经网络 CBV selector。
5. delta recovery。
6. 规则优化生成 positive 样本。
```

## 15. 后续代码实现优先级

按以下顺序实现：

```text
P0: RIFT snapshot dump schema
P1: RIFT original label export
P2: dataset builder for inpainting
P3: Scenario Dreamer converter and dataloader
P4: route encoder
P5: high-value inpainting LDM fine-tuning
P6: offline generation and RIFT audit
P7: reset-time online generation adapter
P8: online summary and visualization
P9: full-generation model
P10: episode-level dynamic inpainting
```

每完成一个 P 阶段，都必须跑对应 smoke test 和可视化抽查，不能直接跳到在线闭环。
