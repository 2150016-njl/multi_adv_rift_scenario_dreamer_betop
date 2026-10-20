# P1 / P2 定向修复与真实证据检查（2026-10-19）

## 为什么这版不再直接修改 P1 主判定

- P1 V2 已验证 Town12 record 0 `299→4894` 和 Town11 record 155 `269→2555` 的**明确有向 junction laneLink**。V3 现有审计包应该用这两条做真正的回归（还要保留 record 100 对照）。
- Town12 record 1039 `4662→4623`：源 waypoint XODR 缺失，但原 `route_waypoints_v2` **早已有道路终点精确例外分支**。旧 `p1_verify_four_connections.py` 没复用该例外，于是得到假性的 `source_anchor_not_verified`。本包使用以前独立的 `p1_route_geometry95_forensic_v1.route_probe` 的精确分类、原始 SHA 验证，输出 road length、exact 查询、原始 road/junction/laneLink 等证据；不自动改 PASS。
- Town13 record 1279 `19894→19897`：真实相邻 XYZ 约 1m，但沿 road 连接边界的站位推进约 80m；显式 OpenDRIVE laneLink 需单独查。这个不合理的物理站位不能因视觉临近而跳过。没有 XML 连接/中间链证据则保持 UNPROVEN。
- P1 original frozen SHA、snapshot、地图、RIFT 阈值、Model3 pose 没有改动。

## 已经验证的 P2 数据与本版修正边界

- Town15: 19,313 个真实 CARLA 采样点、899 个有站位样本的 lane keys、140 个**标线采样**横向区间，26/26 仅为 GT lane-key 支持域命中。
- `p2_station_order_v2.py` 为现有 **GT-free** approach 候选绑定准确的 ego-route `index/s`，检查同 lane 的有向 s 单调性、横向换道采样区间的 s 次序，并保留 longitudinal station contact UNKNOWN。
- `p2_directed_contact_census_v1.py` 可选：真实 Town15 只读 CARLA，采集 `get_topology()` 端点的 `waypoint.next(2m)` 局部接触站位。不是每条纵向边都能被这种离散采样证明。缺少 contact 时 **UNKNOWN**，绝不据此判定断路。
- 本版本不把 `26/26` 升级为正式 approach recall。GT-only 的拓扑 probe 仅在候选固定后 `evaluate` 时读取。GT probe 缺少真实 actor 初始 s/速度，所以它只能测试**necessary station screen**，不能证明 GT 从其初始 s 到达 conflict anchor。
- 下一阶段真正的 full P2 还需要精确 OpenDRIVE/CARLA 接触点、ego-route 上的冲突区绑定、连续换道轨迹、时间/距离约束，最后按**固定 110 GT**评价。

## 0. 安装（不会在交互终端使用 `set -e` 或 `exit`）

把 `p1p2_station_evidence_v2_20261019.zip` 上传到服务器 `$ROOT`；如果放到其他位置，修改 `ZIP`。

```bash
ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
BASE="$ROOT/data/hv_init/v7"
ZIP="$ROOT/p1p2_station_evidence_v2_20261019.zip"
TOOLS="$ROOT/p1p2_station_release_20261019"

unzip -n "$ZIP" -d "$ROOT" && \
  (cd "$TOOLS" && sha256sum -c SHA256SUMS.txt) && \
  "$PY" "$TOOLS/test_release_offline.py" && \
  echo "P1P2_STATION_CODE_READY"
```

所有代码都是 Python 3.8 标准库或 CARLA 0.9.15 egg；**不写入旧冻结 capsule**。若任何一步失败，不要执行下一步；上面是用 `&&` 连接的，当前交互终端不会自动关闭。

## 1. P2 先运行完全离线 station-order 检查

```bash
GRAPH="$BASE/p2_topology_town15_20261019_041323_1291.json"
COND="$BASE/p2_baseline95_v2_20261019_042433_1291/p2_conditions_95.jsonl"
NEW="$BASE/p2_town15_full_lane_v2_20261019_122546_5560"
REVIEW="$BASE/p2_station_screen_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$REVIEW"

"$PY" "$TOOLS/p2_station_order_v2.py" make \
  --topology "$GRAPH" --conditions "$COND" \
  --geometry "$NEW/geometry.json" --lateral "$NEW/lateral.json" \
  --approaches "$NEW/approaches.json" --output "$REVIEW/station_candidates.json" && \
"$PY" "$TOOLS/p2_station_order_v2.py" evaluate \
  --screen "$REVIEW/station_candidates.json" --probe "$GRAPH" \
  --output "$REVIEW/gt26_station_diagnostic.json" && \
"$PY" "$TOOLS/analyze_outputs.py" \
  --p2-screen "$REVIEW/station_candidates.json" \
  --p2-gt26 "$REVIEW/gt26_station_diagnostic.json" \
  --out-dir "$REVIEW/analysis" && \
echo "P2_STATION_REVIEW=$REVIEW"

cat "$REVIEW/analysis/summary.json"
```

离线真实数据的预期：852 条原 lane approach；12 条直接同 lane 支持、12 条仅需采样横向边、828 条存在至少一条尚未核实接触站位的纵向边。26 个 GT record-actor 诊断全部仍满足**必要条件**，不是 native-positive 或连续轨迹 PASS。

## 2. 可选但建议的 P2 真实 CARLA 纵向接触取证

上一轮 P1 检查切换到了 Town13，因此这一步**不能假定服务器仍是 Town15**。若 31120 是独占、允许切图的服务器，使用已验证的 P2 World Guard 切到 Town15：

```bash
CAP="$BASE/p1_frozen_capsule_science_v2_20261019"
EGG="$ROOT/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg"
PREP="$BASE/p2_town15_prepared_station_$(date +%Y%m%d_%H%M%S)_$$.json"

(
  cd "$CAP" &&
  PYTHONPATH="$CAP:$EGG:$ROOT/carla/PythonAPI/carla:$ROOT/RIFT:$ROOT" \
    "$PY" -m hv_v8.p2_prepare_town \
    --root "$ROOT" --town Town15 --host 127.0.0.1 --port 31120 \
    --output "$PREP" --ack-dedicated
)
```

上面括号内是**子 shell**；它不会改变当前 shell。若服务实例不是你的独占实例，**不要执行这个切图命令**。若 `PREP` 已存在，工具会拒绝覆盖。

接着，原地只读 CARLA 实际 `waypoint.next()` 局部有向边：

```bash
CONTACT="$REVIEW/directed_contact_samples.json"
"$PY" "$TOOLS/p2_directed_contact_census_v1.py" \
  --root "$ROOT" --town Town15 --topology "$GRAPH" \
  --host 127.0.0.1 --port 31120 --output "$CONTACT"
```

如果 contact census 成功，重新运行**新输出文件**，不覆盖第一次 station 筛查：

```bash
"$PY" "$TOOLS/p2_station_order_v2.py" make \
  --topology "$GRAPH" --conditions "$COND" \
  --geometry "$NEW/geometry.json" --lateral "$NEW/lateral.json" \
  --approaches "$NEW/approaches.json" --contacts "$CONTACT" \
  --output "$REVIEW/station_with_contacts.json" && \
"$PY" "$TOOLS/p2_station_order_v2.py" evaluate \
  --screen "$REVIEW/station_with_contacts.json" --probe "$GRAPH" \
  --output "$REVIEW/gt26_with_contacts.json" && \
"$PY" "$TOOLS/analyze_outputs.py" \
  --p2-screen "$REVIEW/station_with_contacts.json" \
  --p2-gt26 "$REVIEW/gt26_with_contacts.json" \
  --out-dir "$REVIEW/analysis_with_contacts"

cat "$REVIEW/analysis_with_contacts/summary.json"
```

**分析重点**：`directed_contact_samples.json` 的 `num_unique_frozen_cross_lane_edges_with_samples`；对所有 `necessary_station_order_with_unverified_longitudinal`，检查是否补到了有方向的局部 s-contact。没有采到的边保持 UNKNOWN；不能以零 crossing 或空 sample 直接判为非法。需要进一步核对连续 OpenDRIVE lane link 和车道 s 方向。

## 3. P1：1039 / 1279 只读、严格道路终点 + XML 连接证据

与上面 CARLA 步骤互斥执行；仍要求独占服务器。两 Town 共用同一份冻结 P1 manifest，单次地图切换后只读取数据，不做 spawn/tick：

```bash
P1_OUT="$BASE/p1_endpoint_xml_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$P1_OUT"

"$PY" "$TOOLS/p1_od_endpoint_probe_v2.py" \
  --root "$ROOT" --town Town12 --host 127.0.0.1 --port 31120 \
  --ack-dedicated --output "$P1_OUT/Town12_1039.json"

"$PY" "$TOOLS/p1_od_endpoint_probe_v2.py" \
  --root "$ROOT" --town Town13 --host 127.0.0.1 --port 31120 \
  --ack-dedicated --output "$P1_OUT/Town13_1279.json"
```

注意看 `P1_OLD_GEOMETRY_STATUSES`、`P1_ORIGINAL_ANCHORS_VERIFIED`、`P1_EXPLICIT_XML_JUNCTION_CANDIDATES`。

- 1039 若严格 road-end exact 分支验证，那么**源锚点问题被排除**，仍要查从 4662 到 4623 的确切 junction connection/laneLink；不能直接判 PASS。
- 1279 要核对 source/target 各自 road length、laneSection 转换、road predecessor/successor、junction connection；若无连续链，输出 UNKNOWN，不能用小 XYZ 距离代替有向可达证据。

完成后可以生成包含 P1 两例的联合简报：

```bash
"$PY" "$TOOLS/analyze_outputs.py" \
 --p1-town12 "$P1_OUT/Town12_1039.json" \
 --p1-town13 "$P1_OUT/Town13_1279.json" \
 --p2-screen "$REVIEW/station_candidates.json" \
 --p2-gt26 "$REVIEW/gt26_station_diagnostic.json" \
 --out-dir "$REVIEW/analysis_with_P1"
cat "$REVIEW/analysis_with_P1/summary.json"
```

## 4. P1 V3：只跑 0 / 155 / 100 作为可证伪回归

V3 capsule 已经由你创建，**不用再构建、也不要覆盖**：

```bash
CAP_V3="$BASE/p1_frozen_capsule_explicit_xodr_link_v3_20261019"
SUP="$CAP_V3/p1_isolated95_linkfix.py"
P1_RUN="$BASE/p1_link_v3_controls_$(date +%Y%m%d_%H%M%S)_$$"

"$PY" "$SUP" run \
  --root "$ROOT" --python "$PY" \
  --host 127.0.0.1 --port 31120 \
  --only-indices '0,155,100' \
  --output-dir "$P1_RUN" --ack-dedicated \
  > "${P1_RUN}.log" 2>&1
RC=$?
echo "P1_V3_CONTROLS_EXIT=$RC"
echo "P1_V3_CONTROLS_DIR=$P1_RUN"
tail -n 30 "${P1_RUN}.log"
```

审计运行可能返回 3（科学门槛不全 PASS）或 4（服务安全问题）。不要用 `set -e` 关闭交互终端。查看账本与记录失败详情；若 record 0 在新 guard 下仍 FAIL，必须保留回归失败。

## 5. 最小打包回传（不需要再次传 95 条）

```bash
PACK="$BASE/p1p2_station_evidence_$(date +%Y%m%d_%H%M%S)_$$.tar.gz"
(
  cd "$BASE" &&
  tar -czf "$PACK" \
    "$(basename "$REVIEW")/station_candidates.json" \
    "$(basename "$REVIEW")/gt26_station_diagnostic.json" \
    "$(basename "$REVIEW")/analysis/summary.json" \
    "$(basename "$REVIEW")/analysis/p2_station_candidates.csv" \
    "$(basename "$REVIEW")/analysis/p2_gt26_station.csv"
)
sha256sum "$PACK"
stat -c '%s bytes' "$PACK"
echo "UPLOAD_FILE=$PACK"
```

可按实际文件存在情况补充 `P1_OUT/*.json`、`directed_contact_samples.json` 和 `analysis_with_contacts/summary.json`。这份打包脚本在括号内运行，即使失败也不会退出外层终端。

## 实验边界与停止条件

- Offline unit PASS ≠ CARLA PASS。`P1_OD_LINK verified` ≠ full-scene P1 PASS。
- Station necessary-condition PASS ≠ continuous trajectory PASS；图中路径可能忽略合法 s 顺序、进路口时序或碰撞，因此不能拿作训练正例。
- 不更改原始地图 SHA、manifest、GT 编号、RIFT 配置、车型姿态；不施加 z 偏移；不重跑 95 条。
- **正式 P2 110 GT**：现有 GT probe 仅含 actor lane，不能据此计算源 `(lane,s,yaw,v)` 的原生 interaction recall。下一步必须从原始冻结 snapshots 提取 source states，且只用于**make 完成后**的独立评价。
