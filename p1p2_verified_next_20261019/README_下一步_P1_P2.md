# P1/P2 2026-10-19 下一步：根因驱动、最小变更

## 严格现状（本包未在服务器 31120 上运行）

- P1 老基线：真实 95/95 已执行；81 条写结果，71 条达到独立调度器的综合 native 子门槛；14 条 exact road-ID 反查分歧，另 10 条 native/数值/标签问题；81 条 full-scene occupancy 均 UNKNOWN。
- P1 V2 26 条针对性回归：13 条写结果（12 native 通过），另 13 条被 `DIRECTED_CONTINUITY_UNPROVEN` 拦截，其中旧基线通过的 record 0 发生回归。这不是 13 条“路不通”的证明。受影响路由恰有四类链接：Town12 299→4894、Town11 269→2555、Town12 4662→4623、Town13 19894→19897。
- P2 v1 真实 Town15：`LATERAL_SEGMENTS=2`、`SAMPLED=169`、`CROSSING_ANCHORS=0`，12 条有向候选，GT lane 评价 `miss=26`；旧基线为 14/26。GT26 仅是诊断，不是固定 GT110 正式召回。

## 包文件

- `p1_od_link_guard.py`：解析真正的 OpenDRIVE road `link`、junction `connection/laneLink`、行驶方向、进出边界站位；无法证明则 `unproven`，没有硬编码 road ID 放行。
- `p1_verify_four_connections.py`：只针对历史四种有向连接，按 Town 一次获取实际 CARLA 的 XODR 源锚点与显式路网链接证据，输出完整 JSON。不会把某条失败自动改为 PASS。
- `p1_build_linkfix_candidate.py`：**仅在证据支持时**克隆原 `p1_frozen_capsule_science_v2_20261019`，更换固定 7 步 next() 的不足；保留原 P1 审计主源码 SHA `62b4bbc1…`，创建新调度器绑定、依赖哈希和实测 `python -m` 导入，不修改原 capsule。
- `p2_full_lane_census_v2.py`：关键修复——直接用 CARLA `map.generate_waypoints(step_m)` 采样**所有 Driving 车道**的真实 s 范围，代替旧版仅选择 `get_topology()` 起终点同 lane key 的 169 个点；同时收集无 GT 的换道采样区间/三维交叉候选。不制造“连续换道合法”或出生合法的结论。
- `p2_candidate_triage.py`：对照真实 topology、geometry、lateral、candidates 和事后 GT，逐目标解释 lane 未命中，并统计拓扑车道缺少 s 采样的覆盖缺口。
- `test_local_contracts.py`：std-lib/假 CARLA 对象正反例。不能替代真实 CARLA 验收。

## A. P2 优先：当前已有 Town15，先分析为什么从 14/26 降到 0/26

在服务器同一 shell 运行（旧 v1 四个文件都保留，不修改）：

```bash
set -euo pipefail
ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
BASE="$ROOT/data/hv_init/v7"
TOOLS="$ROOT/p1p2_verified_next_20261019"
REL="$ROOT/p1_science_v2_release"
OLD="$BASE/p2_approach_v1_town15_fixed_20261019_114834_163"
GRAPH="$BASE/p2_topology_town15_20261019_041323_1291.json"
COND="$BASE/p2_baseline95_v2_20261019_042433_1291/p2_conditions_95.jsonl"

"$PY" "$TOOLS/p2_candidate_triage.py" \
 --topology "$GRAPH" --geometry "$OLD/geometry.json" \
 --lateral "$OLD/lateral.json" --approaches "$OLD/approaches.json" \
 --out "$OLD/triage_before.json"
cat "$OLD/triage_before.json" | head -n 25
```

**重要：**这一步不连接 CARLA。它会显示当前 26 个 miss 中，有多少由于 lane 没有 station 采样，有多少是确实没有有向路径或横向许可；二者科学含义不同。

### 重新做一次不含 GT 的完整车道采样（只读）

仅在 31120 **当前确实是 Town15 且独占，不会被别的客户端切图**时运行：

```bash
NEW="$BASE/p2_town15_full_lane_v2_$(date +%Y%m%d_%H%M%S)_$$"
"$PY" "$TOOLS/p2_full_lane_census_v2.py" \
  --root "$ROOT" --conditions "$COND" --topology "$GRAPH" --town Town15 \
  --host 127.0.0.1 --port 31120 --step-m 2.0 \
  --output-dir "$NEW"

"$PY" "$REL/hv_v8/p2_approach_v1.py" make \
  --topology "$GRAPH" --conditions "$COND" \
  --lateral "$NEW/lateral.json" --geometry "$NEW/geometry.json" \
  --max-hops 12 --output "$NEW/approaches.json"

"$PY" "$REL/hv_v8/p2_approach_v1.py" evaluate \
  --candidates "$NEW/approaches.json" --probe "$GRAPH" \
  --output "$NEW/gt26_diagnostic.json"

"$PY" "$TOOLS/p2_candidate_triage.py" \
  --topology "$GRAPH" --geometry "$NEW/geometry.json" \
  --lateral "$NEW/lateral.json" --approaches "$NEW/approaches.json" \
  --out "$NEW/triage_after.json"

echo "P2_RESULT_DIR=$NEW"
cat "$NEW/census_summary.json"
cat "$NEW/triage_after.json" | head -n 35
cat "$NEW/gt26_diagnostic.json" | head -n 30
```

评价：`graph_lanes_without_station_samples` 应显著减少；`triage_before/after` 区分**采样盲区被修复**和真正**有向/合法换道候选不足**。`GT26` 命中只是 lane 级 GT 后验诊断，不是 approach recall，也不证明 Model3 出生。若仍 0/26，以 `triage_after` 分类继续追查，不得增加 GT 指向的边。

## B. P1：先为四组连接取得真实 OpenDRIVE 的显式有向证据

**先完成 P2 Town15 取证。** 以下命令可能切换 Town12、Town11、Town13，必须独占 CARLA 并允许切图；不会创建车辆。

```bash
set -euo pipefail
ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
BASE="$ROOT/data/hv_init/v7"
TOOLS="$ROOT/p1p2_verified_next_20261019"
EVID="$BASE/p1_4_connections_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$EVID"
for TOWN in Town12 Town11 Town13; do
  "$PY" "$TOOLS/p1_verify_four_connections.py" \
    --root "$ROOT" --town "$TOWN" \
    --host 127.0.0.1 --port 31120 --ack-dedicated \
    --output "$EVID/$TOWN.json"
done

echo "P1_OD_LINK_EVIDENCE=$EVID"
"$PY" - "$EVID" <<'PY'
import json,sys
from pathlib import Path
rows=[]
for p in sorted(Path(sys.argv[1]).glob('Town*.json')):
 for item in json.loads(p.read_text())['cases']:
  v=item['directed_od_link'];rows.append((item['record_index'],v['status'],v['reason']))
  print('P1_OD_LINK',item['record_index'],v['status'],v['reason'])
print('P1_4_LINKS_EXPLICIT_ALL_VERIFIED=',len(rows)==4 and all(v[1]=='verified' for v in rows))
PY
```

必须检查 JSON 中 `source_anchors.status` 和 `directed_od_link.status`。显式有向 `road.link` / `junction.connection.laneLink` 缺失、车道方向不对、road boundary s 距离不合理时，**保留 UNPROVEN**，不能以历史 GT 为理由直接开绿灯。`status=verified` 也只证明这对路线点的 road/lane 连接；整体 route 和 native RIFT 仍需原审计器验证。

### 只有 4/4 被独立证据验证后，才创建候选、开展小范围真实复验

```bash
# 如果任何一组不是 verified，不执行这个阶段：提交 EVID 三个 JSON 以判断其他合法连接结构。
CAP_PARENT="$BASE/p1_frozen_capsule_science_v2_20261019"
CAP_NEW="$BASE/p1_frozen_capsule_explicit_xodr_link_v3_20261019"
"$PY" "$TOOLS/p1_build_linkfix_candidate.py" \
  --parent "$CAP_PARENT" --output "$CAP_NEW" --python "$PY"

# 真正的主模块入口验证（新 capsule 工作目录，避免仓库 hv_v8 遮蔽）
(
  cd "$CAP_NEW"
  PYTHONPATH="$CAP_NEW:$ROOT/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:$ROOT/RIFT:$ROOT" \
    "$PY" -m hv_v8.p1_native_audit --help >/dev/null
)

RUN="$BASE/p1_linkfix_small_real_$(date +%Y%m%d_%H%M%S)_$$"
set +e  # rc=3 means scientific fail; preserve it rather than suppress the report
"$PY" "$CAP_NEW/p1_isolated95_linkfix.py" run \
 --root "$ROOT" --python "$PY" \
 --host 127.0.0.1 --port 31120 \
 --only-indices "0,155,1039,1279,100" \
 --output-dir "$RUN" --ack-dedicated > "$RUN.log" 2>&1
RC=$?
set -e
echo "P1_SMALL_EXIT=$RC"
echo "P1_SMALL_RUN=$RUN"
tail -n 25 "$RUN.log"
```

**注意：**已经用 `set +e` / `set -e` 保留了退出码。`RC=3` 可以是科学 FAIL，不是运行入口错误。小样本真实验证 0/155/1039/1279/100 后，再决定是否继续受影响的 13 条和固定 95 条，不应直接重跑 95。

P1 的其他独立遗留项：6 条 native 索引差异中 5 条已复验通过，1039 受路线阻断；1279/1284/1289 的 NaN 真实再测待路线门槛通过；1349 保留历史标签 vs native-negative 冲突；完整场景 occupancy 在历史运行中仍 0 个确证正例。此次包**不修改**这些语义，也不放行 Model3 训练。

## C. 上传分析所需最少证据

只上传目录 `P2_RESULT_DIR` 下的：`census_summary.json`、`triage_before.json`（在 OLD 目录）、`triage_after.json`、`gt26_diagnostic.json`，如需要逐点排查则加 `lateral.json`、`geometry.json`、`approaches.json`。P1 只上传 `P1_OD_LINK_EVIDENCE` 三个 Town JSON。不要重复提交 95 个 snapshot 或整个仓库。
