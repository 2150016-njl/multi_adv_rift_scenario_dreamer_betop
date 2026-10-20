# P1/P2 不再逐记录打补丁：一次性图输入与主线开发契约

## 事实 / 已知问题

- P1 原冻结 95 条实际完成 95/95 次独立尝试，81 条产出审计文件、14 条原始 exact 反查失败；候选 V3 的 record 155/100 在 native 子门槛通过、record 0 在 route[20] 仍被临时连通证明器拦住。不能宣布 P1 完成，也不能宣布所有原始路线均合法。
- Town15 P2 图约 918 lane keys、1044 有向边；12 条条件有 852 个候选和 **124 条唯一必需纵向边**；最新 CARLA 128 个局部接触样本只有 3 个与候选必需边相交，0 条候选全部纵向边被取证。Town15 26/26 是 lane-key 诊断，**不是 GT110 approach recall**。
- 导致持续打补丁的核心原因：P1 `next()` 局部证明、P2 topology BFS 和局部 `next(2m)` 各自使用不同的连通性定义。需要改成 **同一 OpenDRIVE directed lane-section graph**，以 map SHA 为身份证据。

## 这次代码做什么（不声称完成科学修复）

1. `collect_existing.py`：**无 CARLA**。从真实冻结 `p1_calibration95.jsonl` 读取 95 个原始 snapshot，按 SHA 校验，枚举 **全部 route lane-key 相邻转换**；从 GT-free P2 approach 中枚举**实际使用的**纵向有向边与次数；将关键原始证据一起打包。**如果任何冻结 hash 错误则拒绝输出完整结果**。
2. `export_opendrive_once.py`：**会切换地图，需要独占、空 CARLA 实例和 `--ack-dedicated`**。读取实际 `world.get_map().to_opendrive()`，逐 Town 验证 episode 和 map SHA，每个 Town 仅加载一次，输出 five-Town XODR 压缩包。**不 spawn、不 tick、不调用 RIFT**。
3. `test_contracts.py`：离线原始 SHA 拒绝、95 条转换枚举、真实 Town15 852/124 输入、地图 episode 负例。**mock 不等于真实 CARLA PASS**。

## 安装（不要在交互终端执行 `set -e` 或 `exit`）

需先上传 `p1p2_core_graph_inputs_20261019.zip` 到 `$ROOT`（不要假设已自动上传到服务器）。

```bash
ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
BASE="$ROOT/data/hv_init/v7"
ZIP="$ROOT/p1p2_core_graph_inputs_20261019.zip"
TOOLS="$ROOT/p1p2_core_graph_inputs_20261019"

# 不依赖 unzip（此前服务器缺少该命令）
"$PY" -m zipfile -e "$ZIP" "$ROOT"
"$PY" "$TOOLS/test_contracts.py"
"$PY" -m py_compile "$TOOLS/collect_existing.py" "$TOOLS/export_opendrive_once.py"
```

## A. 只用现有数据一次性收集实际需要的边（优先执行）

```bash
A="$BASE/p1p2_graph_inputs_$(date +%Y%m%d_%H%M%S)_$$"
"$PY" "$TOOLS/collect_existing.py" --root "$ROOT" --out "$A" >"${A}.log" 2>&1
RC=$?
echo "P1P2_OFFLINE_EXIT=$RC"
tail -n 30 "${A}.log"
# 若 EXIT=0：上传 ${A}.tar.gz（脚本也会输出 UPLOAD_FILE）
```

输出：

- `p1_all95_route_transitions.json`：各 Town 所有原始相邻 lane-key 转接（包括 4894→4922，不只四组）。
- `p2_candidate_124_edge_demand.json`：GT-free 候选真实用到的边、引用次数和已有局部接触样本。
- `manifest.json` 和 `source/*`：SHA、冻结 manifest、P2 topology/approaches/geometry/lateral 和已有 CARLA contacts。

注意这一步**不检查 CARLA 真实图连通性**，只确定算法必须要验证的精确问题全集。

## B. 独占 CARLA 时只切五次地图，导出实际 OpenDRIVE

**仅确认 31120 是专用空服务器，且同意 `load_world` 可能清空现有场景时执行。** 严格验证 OpenDRIVE SHA；失败自动停止，不会把 stale world 的 XML 用作证据。

```bash
M="$BASE/p1p2_five_town_xodr_$(date +%Y%m%d_%H%M%S)_$$"
"$PY" "$TOOLS/export_opendrive_once.py" \
  --root "$ROOT" --out "$M" \
  --towns Town15 Town12 Town11 Town13 Town04 \
  --host 127.0.0.1 --port 31120 --ack-dedicated \
  >"${M}.log" 2>&1
RC=$?
echo "XODR_EXPORT_EXIT=$RC"
tail -n 30 "${M}.log"
# 若 EXIT=0：上传 ${M}.tar.gz（脚本也会输出 UPLOAD_FILE）
```

对于不能保证独占的服务器：**不要运行 B**。可只运行 A 并提供现有可信 Town XODR 文件供离线查验；不能伪造地图连通 PASS。

## 后续真正要修改的源码/数据契约

### P1 核心修复（与 P2 使用同一图）

- 新的 `directed_map.py`：解析 `road/link/predecessor/successor`、`junction/connection/laneLink`、跨 laneSection 身份与行驶方向，输出每个转接的精确 `entry/exit s`、连接类型、来源 XODR SHA；三值 `VERIFIED / INVALID / UNPROVEN`。**不再用 `next(2m)` 采样缺失当作道路无边**。允许合法中间 connector 的多跳检验，但仍证明各步 station 顺序；不能只凭 XYZ 相近通过。
- `p1_route_contract.py`：对冻结所有 route 的每一个相邻点（包括未失败路点）验证源锚点、路端 exact 例外、station 和图路径；逐点持久化完整状态。报告 95 条的明确分项结果；不能单条失败就终止全量证据。
- `p1_native_oracle.py`：对齐完整 RIFT 候选筛选 + 交互索引，3D ahead 与 `//2`，对 NaN 保留 UNKNOWN，record 1349 记录历史标签冲突，不篡改原始数据。
- `p1_physics.py`：隔离的单车原位、完整背景、联合 occupancy 三套实验；检查原始 6DoF 及 actor ownership，必须有真实 full-scene 成功才准入 Model3 物理正例。

### P2 核心修复

- `p2_directed_support.py`：**只消费上述统一 OpenDRIVE 图**，把状态变成 `(laneSection, reachable s interval, direction)`；有向纵向边有实际 road/laneLink 证据，横向边受标线许可区间约束，严格传播 station 有序性。
- `p2_conflict_anchors.py`：基于未来 ego route index/xyz/s 与另一条有向路径的实际合流、交叉、追赶机制生成具体锚点；不再把整条 ego-route lane 都当一个宽泛锚点。
- `p2_evaluate.py`：GT-free 候选落盘且哈希固定后，再读取原固定 110 前方 GT，逐例检查原始 actor lane/s/yaw 是否位于正确路径+锚点支持域。分开报 lane hit、拓扑有向可达、station 可达、物理 spawn、原生 RIFT 交互。

### 放行门槛

- P1：95 条完整**分类覆盖**，不要求伪造 95 条全 PASS；`LABEL_CONFLICT/ROUTE_UNPROVEN/NUMERIC_UNKNOWN` 隔离。可信 supervision 子集与 Model3 物理合格子集分开维护。
- P2：优先修 **124 条真实候选边的图证据**，不是继续采与之无关的另 125 条边；完成 Town15 的 GT26 完整路径实验，再五 Town 固定 GT110 召回。
- P3/P4：只消费已通过对应 native/路线校准资格的监督；真正生成 Model3 初始状态需另通过 spawn 和全场景 occupancy 的物理 gate。

**这两个脚本仅解决开发 shared graph 前的必要取证，不是 P1/P2 完成版本。** 完整输入拿到后，下一份真正应交付的是上述共享有向地图实现、逐路点批量分类和 P2 可达性求解器，不是又一套 case-specific patch。
