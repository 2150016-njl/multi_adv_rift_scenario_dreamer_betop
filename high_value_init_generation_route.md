# 高价值初始状态生成技术路线

## 1. 结论

本任务应该定义为一个场景初始状态生成问题，而不是 CBV 选择问题。

目标模型的必要输入是：

```text
ego 当前状态
ego route
局部地图 / lane graph
可选：ego 附近已有部分周车状态
```

目标模型的输出是：

```text
ego 附近周车的初始状态集合
```

其中训练数据应保证输出场景中至少存在两辆与 ego route 构成 RIFT 路径级冲突的周车。RIFT 在这里不应作为神经网络要拟合的在线选择器，而应作为：

```text
1. 数据筛选器：筛出高价值初始状态样本
2. 数据标注器：记录每辆车与 ego route 的冲突标签、score、distance_diff、交互点等
3. 评估器：评估模型生成结果是否仍能被 RIFT 判为高价值
```

因此，技术路线不应包含“模型预测 CBV mask 并注册 CBV”这种设计。模型不需要学习 RIFT 选车规则。若后续 RIFT 在线运行需要 CBV 控制对象，应在生成场景落地后，由 RIFT 原始识别和运行逻辑自然处理，或者只在实验统计中记录 RIFT 识别到的高价值车辆数量。

## 2. 必须避免的错误路线

### 2.1 不做 delta 恢复任务

不应把任务设计成：

```text
输入原始场景
扰动某两个 actor
模型预测 delta 恢复原始 actor 状态
```

原因：

1. 这本质是去噪重建，不是高价值初始状态生成。
2. 推理时没有“被扰动的真值目标”，训练和推理语义不一致。
3. 如果训练时扰动的是 RIFT top2，而模型推理时并不知道这两个 actor，恢复对象会错位。
4. 该设计会把模型限制在原始 actor identity 上，而 Scenario Dreamer 的 LDM 更适合生成匿名场景元素。

### 2.2 不让模型预测 RIFT 选择结果

不应把模型输出设计成：

```text
actor_scores
target_mask
CBV ids
```

这会变成学习 RIFT 规则打分，和当前目标相悖。RIFT 标签可以进入训练数据管理和评估，不应成为在线选择器的神经网络替代品。

### 2.3 不把位移阈值当作生成核心逻辑

不应通过“生成后如果相对原 actor 位移过大就跳过”来解决错位问题。对于自由生成或 inpainting 生成的新关键车来说，生成车没有必要和原 actor 一一对应。位移过大不是根本问题，根本问题是是否把“匿名生成的新车”错误地回写到了“已有 actor id”上。

## 3. 对 Scenario Dreamer 代码能力的实事求是判断

### 3.1 AutoEncoder 的能力边界

Scenario Dreamer 的 AE 编码 vectorized scene。根据 `scenario-dreamer/nn_modules/autoencoder.py` 和 `scenario-dreamer/datasets/nuplan/dataset_autoencoder_nuplan.py`，agent 状态在进入模型前被规范化为：

```text
[x, y, speed, cos(heading), sin(heading), length, width]
```

ego 在 NuPlan preprocessing 中被放到局部坐标原点，通常作为 agent 列表中的第一个 vehicle。周车和静态对象也作为 agent 节点输入。lane graph 由 lane polyline、lane type、lane-to-lane connection 和 lane-to-agent edge 构成。

AE 做的是：

```text
agent/lane state + graph -> agent/lane latent -> reconstructed agent/lane state
```

它是 VAE 风格的重建模型。直接把一个场景输入 AE 再 decoder 输出，主要得到重建结果，不会自动变成“更高价值”的初始状态。

结论：

```text
AE 可以作为状态编码器和 decoder 先验。
AE 不能单独承担高价值生成目标。
必须加入 ego route 条件和高价值数据训练。
```

### 3.2 LDM 的能力边界

根据 `scenario-dreamer/nn_modules/ldm.py`，LDM 推理时确实是从高斯噪声开始：

```text
x_agent = torch.randn(agent_shape)
x_lane = torch.randn(lane_shape)
```

在 `lane_conditioned` 模式下，lane latent 固定，只生成 agent latent。在 `inpainting` 模式下，`data['agent'].mask` 和 `data['lane'].mask` 为 True 的 latent 会在每一步反向扩散后被写回，从而保持已观测部分不变。

这说明 Scenario Dreamer 的 LDM 机制上支持：

```text
1. 基于地图生成 agent
2. 基于部分已知 agent/lane 补全未知 agent/lane
```

但原始 LDM 的条件主要是：

```text
map_id / scene_type
num_agents / num_lanes
lane latent
部分已有 agent latent
```

它没有天然显式使用 ego route，也没有高价值交互目标。

结论：

```text
LDM 的生成机制符合本任务方向。
原始权重不能直接保证至少两辆 RIFT 高价值交互车。
需要对 LDM 进行 route-conditioned high-value fine-tuning。
```

### 3.3 为什么生成前后 agent 对应不上

之前实验中，Scenario Dreamer AE/LDM 生成初始状态后，不能保证生成前后的 agent 对应。原因不是简单 bug，而是模型任务设定导致的。

Scenario Dreamer 把 agent 更接近看成一个集合，预处理阶段会按空间位置排序，LDM 生成的是一组 agent latent 槽位。自由生成槽位本身不携带 CARLA actor id，也没有“这个 latent 必须对应原 actor 122”的约束。

因此：

```text
自由生成模式：
生成的是匿名新车集合，不能直接回写到原 actor id。

inpainting 模式：
条件部分 agent 可以保持不变；
未观测槽位生成的是缺失/新增 agent，也不应强行对应原 actor id。
```

这对当前任务反而是合理的，因为现在目标是生成附近周车初始状态，而不是修改原始周车身份。在线落地时应把生成出的周车作为新的场景初始状态或新 spawn 车辆状态，而不是按旧 actor id 做一一更新。

## 4. 当前任务是否可行

### 4.1 只基于 ego、route、map 生成全部附近周车

可行，但需要满足三个条件：

```text
1. 训练数据必须来自高价值场景分布
2. 模型必须显式条件化 ego route
3. 在线输出应作为匿名周车集合落地，而不是回写到已有 actor
```

难点是“至少两辆高价值车”的保证。若不在推理时调用 RIFT 做 rejection sampling，也不使用规则投影，那么模型只能在分布意义上提高生成成功率，不能数学保证每次生成都被 RIFT 判为 >=2 高价值冲突车。

所以更准确的目标应写成：

```text
训练一个在 RIFT audit 下高通过率的高价值初始状态生成器。
```

而不是承诺 100% 硬保证。若论文或实验需要硬保证，只能引入在线 RIFT 检查、重采样或约束优化；这属于另一个实验设置，不能混成模型能力。

### 4.2 基于部分已有周车补全关键周车

这比全量生成更贴近 Scenario Dreamer 的 inpainting 能力。

推荐定义为：

```text
输入：
ego state + ego route + map + observed surrounding agents

输出：
M 辆新增/补全关键周车状态
```

这里的 `observed agents` 是条件，`generated agents` 是匿名新车，不需要对应原 actor id。训练时可以用 RIFT 筛出的高价值场景，把 RIFT-positive 车辆从 context 中移除，作为需要补全的目标车辆。这样使用了 RIFT 标签来构造训练任务，但模型没有学习在线选 CBV。

### 4.3 运行中反复补全关键车

概念上可行，但不建议作为第一阶段。

reset 时生成初始周车，只涉及场景初始化。运行中当旧关键车离开 ego 附近后再生成新关键车，则还涉及：

```text
CARLA 动态 spawn / destroy / traffic manager 接管
新车初始碰撞检查
速度、朝向、车道一致性
RIFT/Pluto 控制对象生命周期
历史轨迹缓存初始化
传感器、route、actor registry 更新
```

这些是闭环系统工程问题，不是 LDM 单独能解决的问题。

建议阶段顺序：

```text
第一阶段：reset-time 高价值初始状态生成
第二阶段：episode 内低频 inpainting 补车
```

## 5. RIFT score 作为辅助任务是否有意义

### 5.1 可以有帮助，但不是必要项

预测 RIFT score 作为训练时辅助任务，理论上可能有帮助：

```text
1. 迫使 latent 表达 route-relative interaction 几何
2. 在小数据下帮助模型关注 ego route 附近的冲突结构
3. 可作为训练诊断，观察模型是否理解 route conflict 分布
```

但它不是必要项，也有明显风险：

```text
1. 它本质上是在蒸馏 RIFT 规则分数，容易偏离“生成真实高价值状态分布”
2. 如果 loss 权重大，模型会优先拟合规则 score，而不是学习自然交通状态
3. 若在线使用这个 score 排序或选车，就又回到“神经网络学 RIFT 选车”的路线
```

### 5.2 推荐处理

第一版不加入 RIFT score prediction head。

RIFT 标签只用于：

```text
1. 筛选正样本：保留至少两辆 RIFT-positive 周车的状态
2. 构造 inpainting target：把 RIFT-positive 周车作为被补全目标
3. 可选 loss weighting：对 RIFT-positive 车辆的状态生成 loss 加权
4. 评估：生成后重新跑 RIFT audit，统计 >=2 positive 通过率
```

如果后续实验发现模型生成的车虽然真实但不靠近 route conflict，可以做一个 ablation：

```text
aux_score_loss_weight in {0.0, 0.05, 0.1}
```

并且必须保证该 score head 只在训练中使用，不参与在线选择和在线控制。

## 6. 数据路线

### 6.1 主数据来源：CARLA/RIFT 闭环采样

优先使用 CARLA/RIFT 闭环跑出来的数据，而不是纯随机状态采样。

闭环采样的优势：

```text
1. 与最终在线运行域一致
2. 周车状态来自交通流和控制器，速度、朝向、车距更真实
3. ego route、地图、周车状态天然同步
4. 能得到后续 rollout 指标，便于评估高价值初始状态是否真的产生交互
```

不闭环随机采样的优势是快，但缺点很重：

```text
1. 容易产生不真实车流密度
2. 容易出现车道朝向、速度、车距分布偏差
3. 生成模型会学到规则采样痕迹
```

建议：

```text
闭环采样作为主训练数据。
规则采样/优化只作为补充增强，并单独标记 source=synthetic。
```

### 6.2 数据采集粒度

每个 episode 不只采 reset 帧，也可以每隔固定时间采样一帧：

```text
t = 0s, 1s, 2s, ...
```

每个样本保存：

```text
scene_id
town / route_id / weather / seed
ego_state
ego_rest_route polyline
local lane graph
nearby actor states within radius
RIFT per-actor labels
RIFT diagnostics
rollout context metadata
```

采样频率不应太密，否则相邻帧高度重复。建议初期每 1s 或 2s 采一帧，按 route 或 episode 切分 train/val/test，避免相邻帧泄露。

### 6.3 正样本、负样本和增强样本

推荐把数据分三类：

```text
real_positive:
原始闭环状态中 RIFT-positive 周车数量 >= 2

real_negative:
原始闭环状态中 RIFT-positive 周车数量 < 2

synthetic_positive:
通过规则优化或采样补车后，经 RIFT 重新标注满足 >= 2 positive
```

第一版训练高价值生成器，优先只用 `real_positive`。如果数量不足，再引入 `synthetic_positive`，但训练和评估必须分开统计，不能把 synthetic 的提升误认为模型从真实数据分布中学会了。

对于 `<2 positive` 的原始状态：

```text
不建议直接规则优化后混入真实正样本。
可以保留为 negative context，用于后续对比或 inpainting 条件。
```

## 7. 训练任务设计

### 7.1 Full Generation

任务定义：

```text
输入：
ego state + ego route + map/lane graph + desired_num_agents

输出：
ego 周围 N 辆周车的初始状态集合
```

这里不需要模型输出 existence。因为不预测 existence 的前提下，必须由外部给定 `desired_num_agents`。这个数量可以来自：

```text
1. 训练集高价值场景的经验分布采样
2. 人工配置，例如每次生成 8、12、16 辆
3. 当前 CARLA 场景已有 traffic density
```

如果未来要求模型自己决定生成多少辆车，那就不可避免需要 count/existence 建模。但这不是第一版必要目标。

训练监督：

```text
LDM diffusion noise loss on generated agent latents
AE decoder reconstruction loss for decoded agent states
可选：RIFT-positive agents 更高 loss weight
```

RIFT-positive 加权不是让模型选择 CBV，而是让生成分布更重视高价值交互车辆的几何质量。

### 7.2 Inpainting

任务定义：

```text
输入：
ego state + ego route + map/lane graph + observed agents + num_missing_agents=M

输出：
M 辆缺失/新增关键周车状态
```

训练样本构造：

```text
从 real_positive 场景中取完整周车集合
移除 2 辆或 M 辆 RIFT-positive 周车
剩余周车作为 observed agents
被移除周车作为 generated target
```

这里会用到 inpainting 条件掩码，但它是输入控制信号，不是模型预测的 CBV mask。含义只是：

```text
这些 agent 已观测，保持不变；
这些 slot 是待生成的新车。
```

它不表示“模型选择了哪些车是 CBV”。

### 7.3 Agent slot 和集合对齐

生成的是集合，不是 actor id 序列。训练 loss 不能简单假设第 i 个生成 slot 必须对应某个 actor id。

推荐：

```text
1. ego 固定为条件，不作为生成目标
2. observed agents 固定排序作为条件
3. generated agents 作为无序集合
4. target matching 使用 Hungarian matching
```

matching cost 可由以下项组成：

```text
位置误差
heading 误差
speed 误差
车辆尺寸误差
```

这样避免 C1/C2 或多辆生成车因为排序变化导致训练不稳定。

## 8. 模型结构路线

### 8.1 复用权重

已有权重：

```text
AE:
/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop/scratch/scenario_dreamer/checkpoints/scenario_dreamer_autoencoder_nuplan/last.ckpt

LDM:
/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop/scratch/scenario_dreamer/checkpoints/scenario_dreamer_ldm_large_nuplan/last.ckpt
```

使用方式：

```text
AE 作为 agent/lane latent encoder 和 decoder 初始化。
LDM 作为 latent diffusion 初始化。
```

但由于最终数据域是 CARLA/RIFT，而权重来自 NuPlan，必须做域适配检查：

```text
1. AE 在 CARLA/RIFT vector scene 上的 reconstruction error
2. decoded 车辆是否落在合理局部范围
3. heading/speed/size 分布是否正常
4. lane graph 坐标和 NuPlan preprocessing 是否一致
```

如果 AE 重建 CARLA/RIFT 场景明显失真，不能直接冻结 AE，需要先 fine-tune AE 或加 adapter。

### 8.2 Route Conditioner

必须显式加入 ego route。

推荐把 ego route 编成 polyline tokens：

```text
route_points: [N, 2]
route_s: [N]
route_heading: [N]
```

然后通过 route encoder 得到 route embedding，并注入 LDM/DiT 条件。

不要把 ego route 简单混进 lane 坐标通道，因为这会破坏 AE 预训练输入维度。更合理的是：

```text
保留 AE 输入格式
在 LDM/DiT 中新增 route condition branch
```

条件形式：

```text
agent latent noising path:
agent latent + timestep + map/lane context + route context
```

### 8.3 Full Generation LDM

生成模式：

```text
lane latent 固定
ego/route condition 固定
agent latents 从高斯噪声开始
decoder 输出 N 辆周车状态
```

这里的 N 由外部给定，不让模型预测 existence。

### 8.4 Inpainting LDM

生成模式：

```text
lane latent 固定
ego/route condition 固定
observed agent latents 固定
missing/generated agent latents 从高斯噪声开始
每一步反向扩散后写回 observed agent latents
decoder 输出 missing/generated agent states
```

这是最贴近 Scenario Dreamer 原始 inpainting 的路线，但需要把原始“空间分区 inpainting”改成“任意 observed/generated agent 条件 inpainting”。

## 9. 在线运行路线

### 9.1 Reset-time generation

第一阶段只做 reset-time。

流程：

```text
RIFT 加载 route scenario
获取 ego 初始状态、ego rest route、CARLA map/lane graph
构建 Scenario Dreamer 条件输入
模型生成周车初始状态集合
将生成状态落地到 CARLA 场景
运行 RIFT 原始流程
记录 RIFT audit：num_positive、distance_diff、score、interaction point
```

如果生成结果中 RIFT audit 少于两辆 positive，第一版应记录失败，而不是用规则修改模型输出。是否重采样可以作为单独实验设置，不能混入模型主结果。

### 9.2 Episode 内 inpainting 补车

第二阶段再做。

触发条件：

```text
关键交互车离开 ego 附近范围
或 RIFT audit 显示当前附近无足够高价值交互车
```

输入：

```text
当前 ego state
ego rest route
当前局部 map
当前 observed agents
num_missing_agents=M
```

输出：

```text
M 辆新增关键周车初始状态
```

落地前必须做工程一致性检查：

```text
车道匹配
初始碰撞
spawn transform 合法性
速度方向与车道方向一致
历史状态初始化
控制器接管
```

这些检查是物理和仿真接口约束，不是 RIFT 高价值筛选。

## 10. 评估指标

### 10.1 离线生成指标

```text
RIFT pass rate:
生成场景中 RIFT-positive 周车数量 >= 2 的比例

state realism:
位置、速度、heading、车距、密度分布和真实 positive 数据的距离

map validity:
车辆是否在可行驶区域、是否贴近车道、是否逆行

initial collision rate:
生成车辆与 ego/其他车辆的初始碰撞率

diversity:
同一 ego/route/map 条件下多次采样结果的多样性
```

### 10.2 在线闭环指标

```text
RIFT audit pass rate
ego collision rate
route completion
driving score
生成车辆存活时长
有效交互持续时间
RIFT-recognized positive actor count over time
```

RIFT audit 是辅助记录和评估，不是模型输出修正器。

## 11. 分阶段实现计划

### Phase 0: 数据与预处理审计

目标：

```text
确认 CARLA/RIFT 场景能稳定转换为 Scenario Dreamer vector scene
确认 AE 对 CARLA/RIFT scene 的重建误差可接受
```

产物：

```text
CARLA/RIFT scene dump schema
RIFT label dump schema
Scenario Dreamer adapter
AE reconstruction audit report
```

### Phase 1: real_positive 数据集

目标：

```text
闭环跑 CARLA/RIFT
每隔固定时间采样状态
RIFT 标注每辆周车
保留 >=2 positive 的 real_positive 样本
```

产物：

```text
train.jsonl
val.jsonl
test.jsonl
dataset_stats.json
可视化样本
```

### Phase 2: Route-conditioned inpainting baseline

目标：

```text
从 real_positive 中移除 RIFT-positive 车辆
训练模型根据 ego/route/map/observed agents 补全这些关键车
```

这是最建议先做的模型，因为它最贴近原始 LDM inpainting 能力，也不需要模型决定生成多少车。

### Phase 3: Route-conditioned full generation

目标：

```text
只基于 ego/route/map 生成 N 辆周车
N 由外部给定或从训练分布采样
```

该阶段难度高于 inpainting，因为没有 observed traffic context。

### Phase 4: Online reset-time rollout

目标：

```text
模型在 RIFT reset 时生成初始周车
CARLA 落地运行
统计 RIFT audit 和闭环指标
```

### Phase 5: Episode 内补车

目标：

```text
关键车离开后，低频调用 inpainting 生成新关键车
```

这是最后做的闭环增强，不应作为第一版目标。

## 12. 严格审核清单

后续实现代码时，必须逐项检查：

```text
[ ] 模型没有输出 target_mask / CBV ids 作为在线选择结果
[ ] 模型没有用 RIFT score 作为推理输入
[ ] RIFT score prediction 若存在，只作为 ablation 辅助 loss，不参与在线选择
[ ] 没有把生成 slot 强行绑定到原 CARLA actor id
[ ] free-generation 输出被当作匿名周车集合
[ ] inpainting 中 observed agents 保持条件固定，generated slots 是新车/缺失车
[ ] ego route 作为显式条件进入模型
[ ] AE 输入维度不被随意破坏
[ ] CARLA/RIFT 到 Scenario Dreamer 的坐标、heading、speed 规范一致
[ ] train/val/test 按 episode/route/log 切分，避免相邻帧泄露
[ ] synthetic_positive 与 real_positive 分开统计
[ ] 在线失败只记录 audit，不用规则强行修改生成结果
[ ] reset-time 先跑通，再做 episode 内补车
```

## 13. 最终建议

最合理的第一版技术路线是：

```text
RIFT/CARLA 闭环采样
-> RIFT 筛选 real_positive 高价值初始状态
-> 构造 route-conditioned inpainting 数据
-> 复用 Scenario Dreamer AE/LDM 权重做 fine-tuning
-> 模型补全/生成匿名关键周车状态
-> RIFT 只做生成后 audit 与闭环评估
```

不要先做模型选 CBV，不要先做 delta 恢复，不要先做 episode 内连续补车。先把 reset-time 的 route-conditioned high-value inpainting 做扎实，再扩展到 full generation 和运行中补车。
