# 前方 FOV 内地图一致且具有 RIFT route interaction 的 CBV：任务澄清与实证诊断

日期：2026-10-13。本文件补充前一份版本诊断，按用户最新明确的 `ego FOV、x>0、地图约束、RIFT route conflict interaction` 重新判断。历史数据重统计不等于重新运行模型或 CARLA。

## 1. 任务应如何精确定义

当前项目 `fov=64` 表示以 ego 为中心的 64×64m 方形，因此车辆中心的前方生成区是 `0<x<=32, |y|<=32`；不是 `0<x<64`。若改成前向 64m 感知区，需要同时修改数据、图裁剪、模型和评估配置。

输出车辆应同时满足：

1. 位于约定前方 FOV；x>0 是位置条件，不要求车头朝 ego 的前向。
2. 原始状态与真实可行驶几何、道路连接和场景占位一致，可直接在 CARLA spawn。
3. 用相同原生配置，在初始状态下可得到 RIFT interaction waypoint：未来 route 的交互潜力已经存在，不是要求两车初始就位于同一点，也不是等待交通演化后再创造交互。
4. 继续正常运行时，是否成为 native green 另外统计。

不应额外要求输出重建某一辆 GT、位于原 GT 的同一条 lane、固定速度大于 0，或所有车都具有相同强交互分数。若继续保持双车输出，至少一车和双车交互率分列；不要悄悄把“至少一车高价值”变成“双车都强交互”的训练筛选条件。

## 2. 道路外车辆是否可接受

需要区分道路外、车道中心线外和原 GT lane 外。三者不能混用。

| 情形 | 对完整背景场景 | 对本任务的目标 CBV |
| --- | --- | --- |
| 距离某条中心线较远，但仍在宽车道、另一车道、可行驶路口区域 | 可以合法 | 可以接受，应检查真实可行驶区域、preview 和 native route |
| 离原 GT 关联 lane 很远，但处于另一条合法、有交互路径的 lane | 可以合法 | 可以接受，不应以 GT lane 重建失败否定 |
| 合法停车区/肩道上的静止车辆 | 可作为真实背景或 observed context | 不能仅凭真实性算目标成功；仍须满足本任务的可行驶/原生交互条件 |
| 真正道路外，且前方 preview 不落在 Driving lane | 是否真实、可 spawn 需另查 | 当前原生识别链无法得到有效 driving preview，不能算本任务成功 |
| 人行道、建筑内、初始车框重叠、错误高程 | 通常是错误或需明确特殊场景建模 | 本任务应拒绝为不符合目标，而非依靠后续修正 |

本任务建议主目标从 Driving lane/合法 connector/可行驶 junction 区域中生成，保留真实横向偏移、转弯和跨车道行为。车体几何应检查可行驶区域的并集和障碍占位，不能要求整个 OBB 始终留在单独一条 lane 内，否则合法换道会被错误惩罚。

也不能说 RIFT 严格要求“车辆中心和整车都在 Driving lane”。源码只先检查沿 yaw 向前 2m 的 preview，再做 junction 朝向和 route 可达性检查；车中心稍在边界外、preview 在内的情况，仍可能通过其地图入口。这种边界案例是否接受应由任务的地图约束和真实物理几何决定。

CARLA 官方文档说明，`get_waypoint(project_to_road=True)` 查询最近 lane；非投影查询才可用来判断位置是否属于所选类型的道路。当前 strict spawn 的查询用 True 获取 z，仍保留原始 x/y/yaw，因此 **查询到 waypoint 并不证明生成车在路上**。原生 RIFT 的 preview 查询使用 `project_to_road=None`、`lane_type=Driving`；服务器微测试应核对该绑定调用与显式 False 的行为，而不改写原生规则。

来源：[CARLA 0.9.15 Python API](https://carla.readthedocs.io/en/0.9.15/python_api/#carla.Map)，本地 `base_cbv.py:273`、`hv_online_generation.py:636,695`。

## 3. 不能用 NuPlan 预训练为任意路外输出作解释

通用模型学习的场景分布不自动满足前方交互 CBV 的条件分布。但本项目还有一个更具体的证据：

`scenario-dreamer/datasets/nuplan/dataset_autoencoder_nuplan.py:530` 明确写明，输入已由 SLEDGE 做 ego 坐标处理，且 off-driveable agents 已被移除；随后处理有效 tracks 与 FOV。

因此，依据当前源码，不能把“NuPlan 预训练天然包含大量真道路外车，因此生成道路外车就是正确行为”当作已证实事实。这是入口对上游缓存的假设，不是本轮对训练缓存的实测；实际缓存、checkpoint 是否遵循该过滤，还需要原数据审计。

即使真实缓存包含合法路外停车车，那也只说明它们可用于背景场景建模，不证明其适合当前 native interaction target。真正有意义的比较是：同一地图条件下，GT 与生成状态的真实可行驶/preview/native positive 分布是否一致。

## 4. 当前确实把哪些东西记成了“失败”

- `evaluation_geometry.py:179`：最近中心线距离超过默认 half-width 1.75m，记 `lane_corridor_failure_count`。
- `eval_high_value_scene_generator_v4a.py:169`：离原 GT 关联 lane 过远，记 associated-lane corridor failure。这对重建任务有意义，对任意有效交互生成只能作为诊断。
- `high_value_ldm_v3.py`：训练 lane corridor loss 针对 GT 关联 lane，可能抑制合法的其他分支。
- `v5_model.py`：intrinsic chart 限制横向偏移 1.35m。
- V5 builder：GT 到 lane 距离默认超过 1.25m 就无法匹配；这比评估默认 1.75m 更严，且不是原生 RIFT 判据。
- V5 eval：相对选定 chart 的距离≤1.75m、heading≤0.45rad等构成 feasibility 代理；没有真实道路 polygon/native preview 检查，也未强制 `x>0`。
- 旧 `physical_invalid_rate` 主要检查 speed/length/width，不是 off-road 指标。固定蓝图 spawn 下，模型输出尺寸不等于实际 CARLA 车框。

这些代理需要校准和改名，不能把代理失败率直接写成真实非法率。地图缺失应记 unknown，不能因未累加 corridor failure 而默认为合法。

## 5. 本轮已完成的全量统计：问题不是主要由 1.75m 太严造成

新增工具 `scenario-dreamer/audit_hv_forward_map_proxy.py`。对相同 V3 test 全 3727 条记录、V2/V3 已保存 S1 输出，逐条校验 index 与 scene_id 对齐；不修改状态、不运行生成器、不访问 CARLA。比较 GT 和模型在**所有已提供中心线的并集**上的最近距离，避免只对原 GT lane 判断。

| 指标 | GT | V2 S1 | V3 S1 |
| --- | ---: | ---: | ---: |
| 车辆数 | 7454 | 7454 | 7454 |
| x≤0 的车辆比例 | 36.01% | 41.02% | 30.87% |
| 两车均在前方 FOV 的事件比例 | 37.89% | 24.71% | 41.62% |
| 前方 FOV 内车辆数 | 4770 | 4396 | 5153 |
| 前方车辆中心线距离>1.75m | 15/4770=0.31% | 1706/4396=38.81% | 2486/5153=48.24% |
| 前方车辆中心线距离>5m | 0% | 20.11% | 24.43% |
| 前方车辆中心线距离 p90 | 1.11m | 10.50m | 8.52m |
| 至少一车在前方且中心线距离≤1.75m，占所有请求 | 90.10% | 62.73% | 59.65% |

解释：

1. 旧 V3 test 的目标并非统一前方目标，不能直接作为当前双车前方任务的训练/评估合同。
2. 1.75m 确实不是道路边界，但 GT 在同条件下只有 0.31% 超出；模型不仅轻微偏移，还有大量超过 5m 的输出。现有证据不支持将主要失败解释为“模型生成合理停车车，却被错误判非法”。
3. 以上仍然是中心线代理，不能把 38.81%/48.24% 称为真实 off-road rate。路口面积、地图缺失、道路宽度与高度，需要真实 CARLA 地图查询进一步区分。
4. V3 的 p90 改善但 1.75m 超限比例更高，说明不同误差区间有取舍，不能只看一个平均数。

结果与源文件 SHA256：[forward_map_proxy_full3727.json](forward_map_proxy_full3727.json)。新增工具的 4 项纯内存测试及语法检查通过。

补充核对：本地 V5 的 519/36/38 条已筛选 GT 均满足 x>0，但 V5 head 在上游车道片段上取 sigmoid progress，未裁掉 x≤0 的部分；“GT 都在前方”仍不等于 head 保证前方输出。

## 6. 不 work 的根源，按版本分开

### 6.1 V1/V2/V3：任务和输入表达不够对应目标

早期目标采样覆盖完整 FOV，当前要求只生成前方交互车。预训练和早期微调的“双侧场景补全”先验，会在缺少强交互意图约束时继续生成后方或其他模式。

本轮还核对到一个明确接口事实：`HighValueSceneGraphBuilder._lane_connections()` (`rift_dataset.py:308`) 对非自身 lane pair 一律写 `none`；NuPlan 预处理 (`dataset_autoencoder_nuplan.py:583`) 则编码真实 pred/succ。

也就是说，V3 虽然 enrichment 后有拓扑 metadata 和几何 loss，主 HV graph builder 仍主要喂 polyline 和无连接图，不能等同于“模型拿到了正确道路拓扑”。这是与预训练图语义的差异；其对性能的具体贡献需保持其他条件相同的连接图消融确定，不能凭代码断言全部失败都由此造成。

### 6.2 V3/V4A/V4C：loss 下降不代表 condition-only 采样成功

这些版本大多在真实 target latent 加噪的训练路径上施加重建/地图/route loss，上线从随机噪声做完整采样。V4A 加前方 loss、V4C 加 target route supervision，仍未让上线条件显式指定可达交互分支。

V4C 512 条结果中，C2 完整 S1 L2 9.11m、physical invalid 24.51%，不同于 AE oracle 的 1.40m 位置重建。已有 evidence 显示完整采样和输出字段有问题，但未确定全部来自模型容量、冻结 AE 域迁移，还是图/配置/归一化接口；应先按字段和路径做对照。对固定车型无需继续自由生成长度/宽度再据此推断 spawn。

### 6.3 V5：支持域错了，训练无法补救

V5 的 chart roundtrip 准确，说明“进入 chart 的 GT”可以表示；但候选仅限局部 lane 与 5–31m ego route 的几何近邻，不表达跨 connector 到未来锚点的 approach。

历史全量 build 最终只有 519/36/38 条；在先前 score、speed、OBB 过滤后进入候选阶段的记录中，候选匹配通过率约 5–7%。这是修候选/图条件的优先信号，不是立即补采十倍同类数据或继续训练的理由。

V5 的周车统计量编码又丢失 occupancy 布局；修 candidate 之后，还需 actor tokens。head 的前方支持、预览朝向、实际道路宽度、固定蓝图 OBB 和双车间距仍需要统一。

### 6.4 全版本：评估还没有直接测最终任务

GT 重建误差、中心线代理、spawn、initial-native-positive、eventual-green 是不同指标。当前 lifecycle green 可能在车辆演化后才出现；原生识别前 25 step 不运行，不能据后来变绿证明初始路线交互。

RIFT 的 interaction score 是路线距离差的函数，native 判据本身不使用目标速度计算 TTC。不要把路线距离同步代理误写成真实到达时间同步。

## 7. 下一步按什么顺序做

| 顺序 | 具体动作 | 目的/判断 | 何时进入下一步 |
| --- | --- | --- | --- |
| 1：统计/校准 | 复用原始 snapshots，统计前方 native-positive 池、目标选择、实际 center/preview Driving、真实宽度/高程、OBB、速度和冲突锚点/路径距离；GT 和 raw 生成输出同口径 | 分清真道路外、代理误判、地图缺失、后方目标、不可达分支 | 得到按 Town/route/topology 分层的明确失败漏斗 |
| 2：测试原生接口 | 同状态比较非投影中心查询、2m preview 查询、原生 trace_route 与初始 labels；隔离 live planner/cache；验证 z、固定蓝图、raw spawn | 证明任务与评估同源，避免改变状态后才通过 | GT 应能稳定复现其标签；失败例逐个归因 |
| 3：修表示/条件 | 保留有向 lane 身份与 pred/succ/connector、道路宽度、高程；从未来 ego route 反向找可达 approach；目标 spawn 域限制 x>0，地图/锚点查询范围独立 | 修复缺失支持域，不等同于只增加候选 cap | 同条件 online/offline candidate parity、GT coverage 和候选可行性 oracle 通过 |
| 4：补采，按缺口触发 | 只补缺失的前方交互拓扑、Town、ego 初始 frame/稀疏车流；保留正/负/无可行解场景与原因 | 解决真实信息缺失与分布偏差 | 不以“恰好两辆全局 positive”替代前方目标池；split 按 episode/route 组隔离 |
| 5：小训练/消融 | 先 32–64 条真实样本过拟合；比较地图图修复、actor tokens、显式 approach condition；完整 S1 选 checkpoint | 判断模型是否真正能从 condition 生成，而非借 target latent 重建 | 小样本 S1 仍失败则查接口/目标，不扩大训练掩盖 |
| 6：全量训练与测试 | 数据分布/可行域通过后训练；固定未筛选 route×seed，raw one-shot；分列前方率、地图/native preview、spawn、initial interaction、eventual green | 证明完整任务，不用 proxy 或单例代替 | 对失败按阶段修一个因素、同条件复测 |

**统计之后不一定先采集。** 如果 GT 已有足够前方原生交互车，但 current candidates 不覆盖，应先修表示；如果旧目标选择选了后方车，而原 snapshot 还有前方 positive，应先重选目标重建 dataset。只有原数据没有所需模式/拓扑或缺少在线必要字段，才需要补采/重导出。

每个场景还应记录“在该 FOV、占位和原生阈值下是否存在可行的交互初始状态”。全部请求成功率和有可行解场景上的条件成功率同时报告；不可行 scene 不能删掉后只报高成功率，也不能把结构上不可行的样本当训练不足。

## 8. 推荐的下一版模型最小方案

输入：ego state、完整未来 route、同源有向道路图、actor tokens。生成意图是 `approach + 可达 connector/path + ego route anchor`，全部由生成前可得信息构造；离线 GT/RIFT 路径只用于匹配和监督。

输出：在前方可行 approach 区间内生成 arclength/lateral/yaw/speed，通过模型内几何头得到 raw state。可行域不等于沿某条中心线固定 1.25m corridor：应以实际 Driving 区域、真实宽度与合法路口几何定义。两车采用联合支持或条件输出，固定真实车型尺寸。

学习式模型之前先有一个同条件的简单几何构造基线：在生成前确定的可行域里直接构造一次状态，不生成后筛选，不 retry。其用途是确认任务可实现、条件足够；基线失败先修图/语义，基线成功而学习模型失败才主要进入训练/模型诊断。

可以保留 Scenario Dreamer 的图编码/生成思想，但不必保留冻结 NuPlan decoder 和自由七维状态输出。若研究必须使用 LDM，应在通过上述基线的同一任务条件与输出支持域下比较，必要时适配 CARLA AE；不要以“模型是 NuPlan 训练”接受不符合本任务的输出。

## 9. 可复现本轮前方地图代理比较

项目根目录：

```powershell
$py = 'C:\Users\Lenovo\anaconda3\python.exe'
& $py scenario-dreamer/audit_hv_forward_map_proxy.py `
  --dataset-jsonl data/hv_init/datasets/inpainting_sparse_phase8_v3_fov64_hdmap_full/test.jsonl `
  --generated-jsonl data/hv_init/eval/v2_v3_same_v3test_full_20261006/v2_raw_s1_generated.jsonl `
  --generated-jsonl data/hv_init/eval/v2_v3_same_v3test_full_20261006/v3_raw_s1_generated.jsonl `
  --output-json research_analysis_new/forward_map_proxy_full3727.json
```

工具拒绝覆盖已有证据；JSON 包含输入 SHA256、所有请求/车辆分母、缺地图数量及明确限制。它不提供真实 off-road/native interaction 结论。后续服务器的 CARLA 同源审计仍是必要步骤。

## 10. 本轮结论

不建议为了让指标好看而直接允许目标车生成在真正道路外；也不建议继续用固定中心线距离把所有偏移车称为非法。

现有证据支持的主要根源是：早期前方任务不一致、道路拓扑/交互意图没有充分进入生成条件、完整采样不满足几何约束，后来 V5 又把可达交互错误缩成了局部车道 chart。第一行动是**统计校准 + 同源初始审计 + 表示修复**，随后依据缺口决定补采，再训练、再严格测试。
