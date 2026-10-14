# 原始 snapshot、原生交互审计与静态 spawn 回放：运行说明

## 1. 当前能力与验证状态

此前的 `audit_hv_forward_map_proxy.py`、`audit_hv_support_chain.py` 只能提供部分离线诊断。此前未实现统一原始 snapshot 审计、隔离 native 判定和静态车队回放。本轮新增：

- 根目录 `audit_hv_snapshot_experiment.py`：三个模式的统一入口。
- 根目录 `build_hv_snapshot_audit_manifest.py`：将 test JSONL、保存的 S1 输出与原始 snapshot 严格配对。
- `scenario-dreamer/high_value_init/native_state_audit.py`：原生候选函数的隔离适配器和静态 spawn。
- `test_hv_snapshot_experiment.py`：CPU/内存测试。

| 检查项 | snapshot 模式 | native 模式 | spawn 模式 |
| --- | --- | --- | --- |
| 原始车辆池、x>0 FOV、已保存 positive/score/路径 | 支持 | 支持 | 支持 |
| V5 候选几何/上游/标注 anchor 覆盖 | 支持；地图缺失为 unknown | 支持 | 支持 |
| GT 与生成车在同一 context 下的 2D OBB 占位 | 支持近似 | 支持近似 | 支持近似，另记录实际蓝图 box |
| CARLA 非投影 center/2m preview Driving 查询、真实 lane width | 无 | 支持入口 | 支持入口 |
| 原生 route interaction 判定、目标 route/距离、失败阶段 | 无 | 支持入口 | 支持入口 |
| 原 GT 类型和原始 pose/velocity 请求重建 | 无 | 无实际 spawn | 支持入口 |
| 实际生成车型、蓝图尺寸、raw spawn command、接受情况 | 无 | 无实际 spawn | 支持入口 |
| 整车道路 polygon 包含、静态 mesh/完整 3D 碰撞审计 | 未实现 | 未实现 | spawn 结果提供部分物理证据，仍未完整实现 |
| 原 episode 的动态轨迹、信号灯和行人精确回放 | 无 | 无 | 未实现；是零 tick 静态接受实验 |
| 正常 attention 排序、native green 接管与持续交互 | 无 | 未评估 | 未评估；仍用既有 strict online runner |

本机已通过 16 个 CPU/内存测试和语法检查；真实历史 dump 的 12 条 snapshot 模式已运行。这里的原生算法测试复用了实际 class 方法体，CARLA/map/GRP 用测试替身验证控制流与隔离；**不是 CARLA 实测**。本机没有 CARLA，native/spawn 两种模式尚待服务器验证。

另已完成一次配对器 CLI→snapshot CLI 的文件格式联通检查，位于 `paired_io_fixture_v1/`：引用真实历史 dump 构造参考输入，只验证数据配对/序列化，不是实际 test dataset 或新增模型性能结果。

本地 smoke 产物位于 `snapshot_probe12_v2/`：12 个历史请求、82 个原始周车、10 个前方周车，其中保存标签为 positive 的有 2 个；10 个请求带 proposal，共 20 个生成车，8 个在前方。12 个旧 dump 都缺少候选所需 map 输入，因此候选覆盖是 null/unknown，不能写成 V5 覆盖率 0。该文件前缀只验证工具运行，不是版本性能结论。

## 2. 三种输入怎么用

### A. 原始采集目录

使用采集器输出的 JSON snapshot，字段至少包含：

```text
coordinate_frame=carla_world
scene_id / town / ego / ego_route / agents / map_lanes / rift_labels
每个 actor 的 actor_id/type_id/x/y/z/yaw_deg/length/width
```

`--input-dir` 会递归读取 JSON，记录所有有效/无效请求。不要传转换后的 V3/V4C dataset 目录，也不要传 summary/quality report 集合。输出目录必须在输入树外，以免后续扫到自己的报告。

没有 proposal 时，统计原始池；spawn 模式回放 ego+整个已序列化车辆池。没有保存 label 的 actor 记 unknown，不默认为 negative。

### B. 既有 online generation dump

自动读取 dump 的 `scene` 和 `proposal.generated_agent_states`。GT 参考是整个原始池，生成车按“向当前场景添加车辆”评估。匹配 spawn attempt 时优先使用保存的请求 z，并校验 x/y/yaw 与原始 proposal 的 float32 一致性；无保存请求则复现 strict runner 的最近道路 z+0.3 规则。

### C. 离线 test GT 与 S1 输出的严格配对

使用 manifest。每行示意结构（actor ID 必须来自真实 snapshot，不能照抄）：

```json
{"snapshot_path":"/absolute/original_snapshot.json","target_actor_ids":[101,102],"context_actor_ids":[103,104],"reference_target_states":[[12,0,0,3,4.6,1.9,1],[20,0,0,3,4.6,1.9,1]],"generated_agent_states":[[13,0,0,3,4.6,1.9,1],[22,0,0,3,4.6,1.9,1]],"generated_blueprint":"vehicle.tesla.model3"}
```

配对器会自动写入真实 `snapshot_sha256`、target/context ID、reference targets、test record 的 `candidate_condition` 和保存的 raw S1 输出。检查原 target/observed 状态、ego 状态、**有序路线点**，不能只按 `scene_id` 配对。不同内容的多个 snapshot 同时匹配时拒绝；同内容的路径副本按 SHA256 去重。

为何需要 `context_actor_ids`：早期 sparse dataset 可能只保留部分 observed actors。GT 和生成车必须使用相同 observed context 做 occupancy/spawn 对照，不能一边恢复完整车队，另一边只看稀疏车辆。`raw_pool` 始终保留完整 snapshot 池作为另一个统计组。

为何需要 `candidate_condition`：旧 snapshot 可能没有正确地图，但 V3/V4C dataset 做过 HD Map enrichment。配对时应使用模型实际见到的条件构造 candidates；CARLA 原生道路/route 判定仍来自原始 world pose 和真实 CARLA 地图，不来自 GT 标签。

自动配对器当前支持 V3/V4C 的原始七维 target JSONL 和 V2/V3/V4 系列保存的 `generated.generated_agent_states` S1 JSONL。V5 的派生四维 `targets` 和单个 eval JSON 不直接支持自动配对；应使用对应原始 V4C 记录，或显式写上述 manifest。

## 3. 服务器环境与预检

沿用已经能运行 RIFT 的 Python/CARLA 环境。以下是项目历史服务器路径模板，需要与服务器实际路径一致：

```bash
export ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
export PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
export CARLA_ROOT="$ROOT/carla"
export PYTHONPATH="$ROOT/RIFT:$ROOT/scenario-dreamer:$CARLA_ROOT/PythonAPI/carla:${PYTHONPATH:-}"
# CARLA Python 包/egg 使用既有 RIFT 的同版本安装，不要混用另一版本。
cd "$ROOT"

"$PY" -m py_compile audit_hv_snapshot_experiment.py \
  build_hv_snapshot_audit_manifest.py scenario-dreamer/high_value_init/native_state_audit.py
"$PY" -m unittest -v test_hv_snapshot_experiment.py
"$PY" audit_hv_snapshot_experiment.py --help
```

snapshot 模式只需要 NumPy。native/spawn 需要完整 RIFT 依赖、CARLA Python API、相同地图资产和可连接的 CARLA server；无需模型 checkpoint，也不调用生成器。

## 4. 实验一：原始前方 positive 池与条件覆盖

设置 `RAW` 为原始采集根目录，`OUT` 为一个新的实验目录，不能覆盖已有结果：

```bash
export RAW="$ROOT/data/hv_init/实际原始采集目录"
export OUT="$ROOT/data/hv_init/eval/snapshot_audit_新的唯一实验名"

# 首先只做 100 条前缀预检；这个结果不能代表总体分布。
"$PY" audit_hv_snapshot_experiment.py \
  --input-dir "$RAW" --mode snapshot --max-records 100 \
  --output-dir "$OUT/pool_preflight"
```

检查无错误后，全量统计使用新的输出目录并设置 `--max-records 0`。原始池应保留无 positive 的场景，不能先筛成全正样本再推断原始可行性。

每次输出：

- `jobs.jsonl`：实际请求清单，可直接复用于 native/spawn，固定同一批输入。
- `rows.jsonl`：逐场景/逐 actor 原始 pose、前方状态、标签、候选、占位和错误证据。
- `summary.json`：原始池/selected GT/generated 分组统计、按 Town 统计、分母、缺失/unknown。

优先读 `groups.raw_pool.counts` 的 `forward_actors`、`saved_forward_positive_actors`、`saved_forward_label_unknown_actors` 和 `saved_positive_candidate_evaluated_actors`。`saved_positive_candidate_coverage` 的分母只包括候选输入存在的 positive；缺地图场景另列，不能用缺地图后的子集冒充所有请求。

若缺地图，优先复用已有 enriched dataset 的 condition 配对，不要仅因旧 dump 缺地图就立即重采。新采集确有必要时，既有 collector 的 `--min_forward_rift_positive 0` 可以保留负例；地图模式通过 `--extra_run_arg=--hv_snapshot_map_coordinate_mode --extra_run_arg=right_handed` 传入。当前前方工具严格 x>0，collector 的默认边界 x>=0 与 margin=2m 统计不同，要分清口径。

## 5. 实验二：配对同一 test、同一 S1 和同一 context

以下命令示例使用已有 V3 test 与 V2 S1；将 generated 路径换成 V3 即可再生成一个独立 manifest。

```bash
export TEST="$ROOT/data/hv_init/datasets/inpainting_sparse_phase8_v3_fov64_hdmap_full/test.jsonl"
export S1="$ROOT/data/hv_init/eval/v2_v3_same_v3test_full_20261006/v2_raw_s1_generated.jsonl"

"$PY" build_hv_snapshot_audit_manifest.py \
  --dataset-jsonl "$TEST" --generated-jsonl "$S1" \
  --snapshot-dir "$RAW" --max-records 100 \
  --output-manifest "$OUT/v2_pairs100.jsonl"

"$PY" audit_hv_snapshot_experiment.py \
  --manifest "$OUT/v2_pairs100.jsonl" --mode snapshot \
  --output-dir "$OUT/v2_pair_snapshot"
```

先看 `v2_pairs100.jsonl.audit.json` 的 `passed`，必须 true；missing/ambiguous 不能悄悄删除后再报告为全 100 条。配对器会扫描 snapshot 树，100 条指 dataset 的固定前缀，不是随机抽样。正式比较应使用事先冻结的分层清单或全量 `--max-records 0`。

V4C test 和 C0/C2 S1 也使用同样流程。**当前本机缺原始 V4C JSONL，无法替服务器完成这个配对实验。**

## 6. 实验三：真实 CARLA 道路/preview 与隔离原生判定

先启动一个独立审计 CARLA server。不要把允许 load_world 的实验接到正在运行的训练/benchmark 端口。以下命令在独立终端、以能运行 CARLA 的普通用户启动；不需要另外的 RIFT runner：

```bash
"$CARLA_ROOT/CarlaUE4.sh" -RenderOffScreen -nosound -carla-rpc-port=31000
```

在另一终端沿用上述环境，先做小批次；这里的 recognition config 必须与原实验一致。现有 collector 默认 `rule`，eval 参数与当前 attention eval 基础候选参数相同，但仍应明确指定并保存：

```bash
export RECOG="$ROOT/RIFT/rift/cbv/recognition/config/rule.yaml"

"$PY" audit_hv_snapshot_experiment.py \
  --manifest "$OUT/v2_pairs100.jsonl" --mode native --max-records 20 \
  --host 127.0.0.1 --port 31000 --load-maps \
  --recognition-config "$RECOG" --recognition-mode eval \
  --sampling-resolution 1.0 --output-dir "$OUT/v2_native20"
```

也可以把 manifest 替换为第一阶段输出的 `pool_preflight/jobs.jsonl`，审计同一原始池。无 `--load-maps` 时，只接受当前 server 的同名 Town；有此参数时会切换 world，依赖独占服务器。当前按输入顺序执行，多 Town 全量实验建议先按 Town 组织 manifest，减少重复 load_world。

隔离方式：调用原生 `BaseCBVRecog.get_CBV_candidates` 的原方法代码，只将该调用的 `CarlaDataProvider` 全局绑定换成私有 snapshot provider；不修改真实模块/provider。每个 actor 使用独立 policy、`inter_info` 和 GRP 的可变转向状态。保留原始 distance、junction heading、preview、route distance 阈值，不手工注册 CBV。

保留已有 CBV/reached IDs 的 snapshot debug；缺少这些信息时明确记录 `empty_slots_assumed`。这是基础候选资格审计，不包含 attention 排序或正常接管；不要直接写成 green success。

逐 actor 证据包括：

- `road_geometry.center_driving`、`preview_driving`、原生 waypoint road/lane/section/s/width。
- `preview_none_false_agree`：原生 None 参数与显式 False 的查询是否一致。
- `native.native_candidate`、`interaction_meta`、`target_route_to_interaction`。
- `candidate_debug.rejection_counts`：原生拒绝阶段。
- `saved_config_mismatches`：审计参数与 snapshot 保存参数是否有已知差异。
- `saved_native_label_disagreements` 的汇总；不同参数或缺容量上下文时先排除这些原因。

ego route 按保存的 road/lane/s 恢复，并验证有序位置和 section/lane 身份，位置误差超过 0.25m 则 unknown/error，不强制投影修正。报告保存 CARLA server 版本、OpenDRIVE SHA256、原生源码 SHA256 和参数。

## 7. 实验四：GT 与生成状态的静态 raw spawn 对照

在同一独立服务器执行；使用新的目录：

```bash
"$PY" audit_hv_snapshot_experiment.py \
  --manifest "$OUT/v2_pairs100.jsonl" --mode spawn --max-records 20 \
  --host 127.0.0.1 --port 31000 --load-maps \
  --recognition-config "$RECOG" --recognition-mode eval \
  --sampling-resolution 1.0 --output-dir "$OUT/v2_spawn20"
```

spawn 模式先做 native 检查，然后分两组：

1. ego + 同一 observed context + 原 GT targets；GT 使用保存的原车型、x/y/z/roll/pitch/yaw 和原 velocity vector（有 vx/vy/vz 时）。
2. 清理本工具创建的第一组车辆，再建立 ego + 相同 observed context + raw generated targets；生成车固定 Tesla Model3，x/y/yaw/speed 不改。

每个 actor 只 try_spawn 一次，没有替换蓝图、移动兜底、生成重试。现有 vehicle actor 非空时拒绝实验，不删除其他客户端的车。临时启用同步模式，零 tick，保持正常物理/碰撞设置；退出恢复设置并仅销毁本次创建的 actor。

检查 `spawn.selected_gt` 与 `spawn.generated`：

- `context_complete`：ego/context 是否完整恢复。若 false，不能把 target 成功率当可比的模型性能。
- `targets_all_accepted`、`whole_scene_accepted`。
- 每次 attempt 的 `requested_type_id`、`actual_type_id`、真实蓝图 box 尺寸/offset。
- `requested_pose`、`spawn_command`、`raw_request_fidelity`、`requested_velocity`。
- `cleanup_errors` 应为空；`spawn_frames_advanced` 应为 0。

原 GT 的车型缺失、source z 不匹配、场景静态障碍或 context 恢复失败，都会单独留证，不允许车型 fallback。

**零 tick 能验证请求接受和静态重建，不能验证车辆实际执行了目标速度、落地稳定或后续路线控制。** pre-tick 的实际 transform/velocity 可能 stale，工具不把这些读数称为 fidelity。动态验证仍需独立的正常 strict online 实验，不能把本工具改成生成后 tick 再调整状态。

## 8. 如何读统计，决定下一步

先确保 manifest audit `passed=true`，experiment `native_error_jobs=0`。不要只看退出码：没有异常也不要求模型成功率高；无 native 查询的 snapshot 模式中相关率是 null。

```bash
export AUDIT_SUMMARY="$OUT/v2_native20/summary.json"
"$PY" - <<'PY'
import json,os
r=json.load(open(os.environ['AUDIT_SUMMARY']))
print('planned/error:',r['planned_jobs'],r['native_error_jobs'])
for group in ('raw_pool','selected_gt','generated','spawn_selected_gt','spawn_generated'):
    if group not in r['groups']:continue
    s=r['groups'][group];c=s['counts']
    print('\n',group)
    print(c)
    print({k:v for k,v in s.items() if k!='counts'})
print('missing candidate inputs:',r['candidate_inputs_missing_jobs'])
print('without two generated states:',r['jobs_without_two_generated_states'])
PY
```

`forward_actor_fraction` 使用全部有效 cohort actor 为分母；`at_least_one_front_native_scene_rate` 使用该 cohort 的全部有效场景为分母，包含生成在后方而失败的场景。`native_positive_rate_given_evaluated` 只是前方已评估 actor 的条件比例，不能替代全任务成功率。存在 native unknown 时 scene rate 为 null；还必须保留所有计划请求和失败 ledger。

| 观察 | 下一步 |
| --- | --- |
| GT 原生标签无法复现 | 先查 config/mode、map SHA、路线恢复、None/False、容量上下文；不要训练 |
| GT 原生 positive 很多，候选覆盖低 | 修 candidate/道路连接/锚点 horizon；不先扩采同类数据 |
| 已有原始池缺少前方 positive，或确实缺道路/状态字段 | 定向补采或重导出，保留负例和不可行场景 |
| GT 可 spawn，生成车大量无 driving preview 或 native distance/route 失败 | 修输出支持域和交互条件，再做小样本 S1 训练 |
| native 通过，生成 spawn 失败 | 看真实蓝图 OBB、context/peer 占位、地形 z 和静态障碍 |
| context 本身无法重建 | 先解决回放条件/车型/地图版本；这不是模型 target 的有效对照 |
| 初始 native 与 raw spawn 都好，实际 online green 仍差 | 查正常 recognition 的时机、容量、attention 排序与后续控制 |

最后按 Town/route/episode 分层复核；同 episode 相邻帧不能当独立样本。当前自动汇总到 Town，route/episode/frame 保存在 rows；置信区间、完整 3D 道路包含和动态 episode 回放尚未实现，应在报告中明确。
