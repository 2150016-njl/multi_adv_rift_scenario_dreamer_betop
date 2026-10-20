# P1 科学候选 V2 / P2 有向交互 approach V1（2026-10-19）

## 已固定的科学证据与边界

- **真实历史基线**：95/95 被执行；81/95 结果已写；71/95 满足独立调度器的严格 native 综合子门槛；14 条在 Town11/Town13 重合路口的 exact road-ID 反查处中断；另 6 条 RIFT index parity、3 条候选 heading NaN、1 条历史 positive/native-negative 冲突；81 条 full-scene occupancy 全为 UNKNOWN。P2 Town15 lane support **14/26**，非正式 110 GT approach recall。
- **不可变父源码 SHA256**：`b6036d51d849d0e0aa46c1dc35754e4be9f81ec35dd504146f6cf877ec88a6bb`。
- **新候选主审计源码 SHA256**：`62b4bbc1bae838cd13185586c546d756a394cd39ae57b35a63242e96aa56609a`。
- 不改变原始 95 JSONL、snapshot、RIFT 源码、attention.yaml、CARLA XODR、spawn pose；不添加失败后的 z 调整、投影吸附、出生重试。新源码**不是**已有 P1 PASS。
- P1 仍可能 FAIL：特别是 CARLA 有向连续性不能独立证实的连接处、6 条 parity 不能复现、NaN、record 1349 的历史标签冲突以及完整场景占位 UNKNOWN。不能为了把 95 条改成全绿而关掉这些门槛。

## 包内源代码（无隐藏依赖安装步骤）

- `hv_v8/p1_native_audit.py`：以冻结父源码为基础的完整主模块；路由使用下面的源锚点优先函数；RIFT 3D ahead、`//2` 半视角；候选 heading 弧度转角度、NaN 不可当 native-negative；raw spawn 验证完整 XYZ/pitch/roll/yaw。
- `hv_v8/p1_science_contract_v2.py`：XODR 原始身份、Driving、s、XYZ、yaw 严格校验；仅对 XODR 与 exact 空间重叠的真路口承认反查非唯一；逐段有向站距/`next()` 连通性证明，失败则 FAIL，**不使用 exact 错路覆盖原始 road**。路口同车道的 s 单调性与连续距离属于必要局部连贯性检查，不等于全路线动力学可行。
- `hv_v8/p1_physics_independent.py`：独立 episode 的 raw 与 full-scene replay 两相实验，保持 UNKNOWN，不能自动提升为物理训练正例。
- `hv_v8/p1_world_guard.py`：受 SHA、Town 和 episode ID 约束的真实 world fence。
- `hv_v8/common.py`：本地测试版同名辅助函数；**服务器安装时从之前经真实审计使用的 depcomplete capsule 复制 SHA 锁定的原版**，避免改变共同依赖语义。
- `hv_v8/p1_isolated95_science_v2.py`、`hv_v8/p1_route_failure_witness.py`：95 条逐条隔离、可选择受影响样本、错误/UNKNOWN 留存、不遮蔽其他独立记录。
- `hv_v8/p2_prepare_town.py`：独占实例显式切图、验证 episode 和 OpenDRIVE。
- `hv_v8/p2_geometry_sampler.py`：CARLA GT-free 道路三维采样与几何交叉证据，拒绝平面不同高度的假交叉。
- `hv_v8/p2_lateral_legal_segments.py`：标线权限/同向性及 s 区间只读采样。
- `hv_v8/p2_topology_probe.py`：真实 CARLA 有向拓扑提取（已有 Town15 报告可复用）。
- `hv_v8/p2_approach_v1.py`：GT-free 地图交互锚点、纵向+有条件的横向前驱 BFS，输出真实 s 采样支持区间和上游 lane-path；事后独立 evaluation，不伪造动态路径或出生可行性。
- `prepare_science_capsule.py`、`build_candidate.py`、`parent_p1_native_audit.py`：原始父源码 SHA 锁定与可复现构建；**不要在服务器原目录直接执行 build_candidate.py**，它是独立源代码复现入口。
- `analyze_science_v2.py`：已写 JSON 的逐目标分项结果；不把 native-negative、NaN UNKNOWN 或 occupancy UNKNOWN 算成物理 PASS。
- `test_science_v2_offline.py`：直接模块导入、真实 capsule 构建 `python -m`、Python 3.8 AST、正反例断言、GT 隔离、不可覆盖测试；仅代表离线契约测试。

## 0. 服务器安装与完整入口测试（**无 CARLA 操作**）

确认 ZIP 解压后是 `$ROOT/p1_science_v2_release/...`；不覆盖旧 capsule。

```bash
bash <<'BASH'
set -e
ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
BASE="$ROOT/data/hv_init/v7"
unzip -n /path/to/p1_science_v2_release_20261019.zip -d "$ROOT"
REL="$ROOT/p1_science_v2_release"
echo '62b4bbc1bae838cd13185586c546d756a394cd39ae57b35a63242e96aa56609a  '"$REL/hv_v8/p1_native_audit.py" | sha256sum -c -
"$PY" "$REL/test_science_v2_offline.py"
"$PY" "$REL/prepare_science_capsule.py" --root "$ROOT" --source "$REL"
CAP="$BASE/p1_frozen_capsule_science_v2_20261019"
EGG="$ROOT/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
PYTHONPATH="$CAP:$EGG:$ROOT/carla/PythonAPI/carla:$ROOT/RIFT:$ROOT" "$PY" -m hv_v8.p1_native_audit --help >/dev/null
echo P1_CANDIDATE_MODULE_ENTRYPOINT_OK_NOT_CARLA
BASH
```

`prepare_science_capsule.py` 在目标已存在时**拒绝覆盖**。仅在第一次安装执行一次。失败时不要自行删除旧 capsule。

## 1. 真实 CARLA P1 受影响记录优先验证（不要先重跑全量）

以下阶段会 `load_world`、spawn/destroy，因此必须确认 31120 为本项目独占、允许切换五 Town 的实例。只读科学子门槛和物理 spawn 仍由同一个真实审计模块完成；本步骤用于**候选验收**，不是宣称 P1 完整 PASS。

```bash
ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
BASE="$ROOT/data/hv_init/v7"
REL="$ROOT/p1_science_v2_release"
TARGETS='0,100,155,160,165,342,592,597,839,844,1222,1227,1232,1237,1242,1247,1039,2376,3417,3422,3490,3505,1279,1284,1289,1349'
RUN="$BASE/p1_science_v2_targeted_$(date +%Y%m%d_%H%M%S)_$$"
"$PY" "$REL/hv_v8/p1_isolated95_science_v2.py" run \
  --root "$ROOT" --python "$PY" --host 127.0.0.1 --port 31120 \
  --only-indices "$TARGETS" --output-dir "$RUN" --ack-dedicated >"${RUN}.log" 2>&1
RC=$?
echo "P1_V2_TARGETED_EXIT=$RC" "P1_V2_TARGETED_RUN=$RUN"
tail -n 22 "${RUN}.log"
"$PY" "$REL/analyze_science_v2.py" --run-dir "$RUN" --output-dir "${RUN}_review"
```

注意：退出码 3 表示**科学子门槛尚有 FAIL**，不代表又是 Python 导入错误；`--only-indices` 未选中的记录在 95 分母中明确标 `not_run`。出现退出码 4 时立即检查 CARLA 世界安全条件，不继续。

重点对照：

```bash
"$PY" - "$RUN" <<'PY'
import json,sys,pathlib
run=pathlib.Path(sys.argv[1])
for i in (155,1222,1039,2376,1279,1284,1289,1349):
 p=run/'cases'/('record_%04d'%i)/'audit'/('record_%04d.json'%i)
 if not p.exists():
  print(i,'NOT_WRITTEN', (run/'cases'/('record_%04d'%i)/'result.json').read_text()[:650]);continue
 r=json.loads(p.read_text()); print(i,'route',len(r['route_checks']),
   'exact_overlap',sum(x.get('exact_lookup_status')=='verified_junction_overlap_other_road' for x in r['route_checks']),
   'parity',[x.get('rift_reference_parity_status') for x in r['native_rows']],
   'native',[x.get('native_positive_status') for x in r['native_rows']])
PY
```

该段直接传递 `RUN` 路径，不依赖额外导出环境变量。不上传整个服务器仓库。

## 2. 有针对性的 native 复验验收规则

- 14 条旧 exact 冲突记录：至少必须全部生成结果 JSON（否则分别保留 FAIL）并且具备 XODR/Driving/s/XYZ/yaw 和有向连贯证据。不能仅因为它们都是 junction 就 PASS。关注 `route_checks[].exact_lookup_status` 和 `directed_link_to_previous`。
- 6 条 index 不一致：参考 `selected_ego_route_index` 必须逐目标完全一致；如果仍有差异，继续调查原生 `trace_route` 距离/平局，而不是调阈值。
- 3 条 NaN：应为 `heading_numeric_status=unknown_nonfinite`、`candidate_filter_pass=null`、`native_positive_status=unknown`，**不能**计为干净 negative。
- 1349：历史 positive 与参考 native-negative 仍为冲突，不能翻为 positive；完整 P1 provenance/标签门槛未过。
- 未解决跨 seed 身份标签仍在（此前的17个重名成员 + `:sparse0`）。即便上述候选运行良好也不能由此宣布全部 P1 最终通过。

受影响记录的判定规则成立后，才进行**一次**无 `--only-indices` 的 95 条候选真实回归：

```bash
FULL="$BASE/p1_science_v2_full95_$(date +%Y%m%d_%H%M%S)_$$"
"$PY" "$REL/hv_v8/p1_isolated95_science_v2.py" run --root "$ROOT" --python "$PY" \
  --host 127.0.0.1 --port 31120 --output-dir "$FULL" --ack-dedicated >"${FULL}.log" 2>&1
RC=$?; echo P1_V2_FULL95_EXIT="$RC"; echo P1_V2_FULL95_DIR="$FULL"; tail -n 20 "${FULL}.log"
"$PY" "$REL/analyze_science_v2.py" --run-dir "$FULL" --output-dir "${FULL}_review"
```

## 3. Model3 物理证据：与 native 审计分开执行

仅在独占 CARLA 实例、空车辆/行人 world 上执行。每次一个新 episode，不连续复用旧 native world；`--phase raw` 和 `--phase occupancy` 的 raw pose 均不修改。单车接受不代表完整联合场景可行，UNKNOWN 不计数为成功。

```bash
CAP="$BASE/p1_frozen_capsule_science_v2_20261019"
EGG="$ROOT/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
PHYS="$BASE/p1_phy100_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$PHYS"
PYTHONPATH="$CAP:$EGG:$ROOT/carla/PythonAPI/carla:$ROOT/RIFT:$ROOT" "$PY" \
  -m hv_v8.p1_physics_independent --root "$ROOT" --record-index 100 --phase raw \
  --host 127.0.0.1 --port 31120 --output "$PHYS/raw.json" --ack-dedicated
PYTHONPATH="$CAP:$EGG:$ROOT/carla/PythonAPI/carla:$ROOT/RIFT:$ROOT" "$PY" \
  -m hv_v8.p1_physics_independent --root "$ROOT" --record-index 100 --phase occupancy \
  --host 127.0.0.1 --port 31120 --output "$PHYS/occupancy.json" --ack-dedicated
```

若本独立物理实验过程中发现污染或清理无法验证，立即停止，保存证据。不要连续针对所有 95 条运行物理相位直到完成独立控制验证。

## 4. P2 GT-free 地图交互候选（真实 CARLA Town15）

Town15 的旧 `14/26` 是**纵向 lane 身份支持率**，不是正式 approach recall。下述代码会尝试生成可追溯的横向权限区间、三维道路交叉候选、merge / following 锚点及上游有向 lane path。**GT 只在独立 `evaluate` 命令读取**；未形成连续换道轨迹、已通过原生 RIFT 的动态交互、真实 Model3 出生或 110 GT 正式召回前，一律不得称 P2 PASS。

```bash
ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
BASE="$ROOT/data/hv_init/v7"
REL="$ROOT/p1_science_v2_release"
CAP="$BASE/p1_frozen_capsule_science_v2_20261019"
EGG="$ROOT/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
P2="$BASE/p2_approach_v1_town15_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$P2"
COND="$BASE/p2_baseline95_v2_20261019_042433_1291/p2_conditions_95.jsonl"
GRAPH="$BASE/p2_topology_town15_20261019_041323_1291.json"
# 仅在允许切图、无其他用户的 CARLA 31120 上运行一次。
PYTHONPATH="$CAP:$EGG:$ROOT/RIFT:$ROOT" "$PY" -m hv_v8.p2_prepare_town \
  --root "$ROOT" --town Town15 --host 127.0.0.1 --port 31120 \
  --output "$P2/map_transition.json" --ack-dedicated
# 新候选源码的 P2 实用工具是独立脚本；不用修改冻结审计源码。
"$PY" "$REL/hv_v8/p2_lateral_legal_segments.py" --root "$ROOT" --town Town15 \
  --host 127.0.0.1 --port 31120 --step-m 2.0 --output "$P2/lateral.json"
"$PY" "$REL/hv_v8/p2_geometry_sampler.py" --root "$ROOT" --town Town15 \
  --host 127.0.0.1 --port 31120 --conditions "$COND" --output "$P2/geometry.json"
"$PY" "$REL/hv_v8/p2_approach_v1.py" make --topology "$GRAPH" \
  --conditions "$COND" --lateral "$P2/lateral.json" --geometry "$P2/geometry.json" \
  --max-hops 12 --output "$P2/approaches.json"
"$PY" "$REL/hv_v8/p2_approach_v1.py" evaluate --candidates "$P2/approaches.json" \
  --probe "$GRAPH" --output "$P2/gt26_diagnostic.json"
```

检查：

```bash
"$PY" - "$P2/approaches.json" "$P2/gt26_diagnostic.json" <<'PY'
import json,sys,collections
p=json.load(open(sys.argv[1]));r=json.load(open(sys.argv[2]))
print('P2_RECORDS',len(p['records']))
print('P2_ANCHOR_TYPES',dict(collections.Counter(a['type'] for x in p['records'] for a in x['anchors'])))
print('P2_CANDIDATE_LANES',sum(len(x['approaches']) for x in p['records']))
print('P2_26_LANE_DIAGNOSTIC',r['counts'])
print('P2_FORMAL_APPROACH_RECALL_PROVEN',r['formal_GT_approach_recall_proven'])
PY
```

若 12 个 Town15 miss 仍未覆盖，看 `lateral.json` 是否真的存在 road140 `-2 → -3`、road137 `-1 → -2` 的有证据支持区间，以及三维 `geometry.json` 是否存在足够几何锚点。**缺证据时保持 miss，禁止添加任意横向边**。

如果另有来源锁定的固定 **110 GT**：按已有实验 ID 和 `actor_lane` 生成 JSONL（110 个独立 `record_index + actor_id`，不能自行替换目标），再运行 `evaluate --fixed110 GT110.jsonl`。此版本仍只统计**支持车道命中**，不把它伪装为“合法初始化 + 动态原生交互”的最终 approach recall；正式通过还需要 P3/P4 动态验收。

## 5. 哪些结果上传回来用于下一轮精确分析

不需要服务器再上传整个仓库。仅压缩**候选 targeted 运行** `ledger.json`、`cases/`、`*_review` 的 summary/CSV，以及 P2 的 `map_transition.json`、`lateral.json`、`geometry.json`、`approaches.json`、`gt26_diagnostic.json` 和运行日志。候选是否科学通过由真实记录和 gate 决定，绝不因为 `--help`、mock、spawn accepted 或 lane 命中自动放行。
