# P1 / P2 主线实施：真实运行与科学结果分层（2026-10-19）

## 一、这次不改冻结科学代码

本包 **不包含也不覆盖** `p1_native_audit.py`。它只使用服务器已创建且已验证模块导入的 capsule：

- `data/hv_init/v7/p1_frozen_capsule_world_transition_depcomplete_v2_20261019/`
- 原主审计 SHA256 `b6036d51d849d0e0aa46c1dc35754e4be9f81ec35dd504146f6cf877ec88a6bb`
- 95 manifest SHA256 `92c16bec23132d44a28de9d1d97894ba2e50ea9bb2f1e53fa41fbe5182664c56`
- map manifest SHA256 `9bdb620998c9e67676c90edadf5c45cbe7bc6e12c1b661b12f2a6bb62405aadf`

文件：

- `hv_v8/p1_isolated95.py`：95 条逐条启动、科学失败继续、安全失败终止、逐条持久化。
- `hv_v8/p1_route_failure_witness.py`：同地图 episode 的 XODR/XYZ 真实失败现场取证，**不改变 FAIL**。
- `hv_v8/p2_directional_support_v0.py`：基于 CARLA 有向纵向图和 ego route 的**无 GT** 上游 lane 候选；评价单独执行。
- `hv_v8/p2_lateral_legal_segments.py`：可选的 CARLA 当前 Town **只读**合法换道采样取证；尚未整合为连续可行轨迹。
- `hv_v8/test_mainline_offline.py`：源码级负例与真实 Town15 输入隔离验证；不等于 CARLA PASS。

## 二、P1 一次真实全量：单记录科学失败不遮蔽全局

仅当 31120 为独占且允许 `load_world` 和真实 spawn/destroy 的项目实例时执行：

```bash
ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
BASE="$ROOT/data/hv_init/v7"

# 将本 ZIP 上传服务器，再解压：
unzip -n /path/to/p1_p2_mainline_isolated_real_20261019.zip -d "$ROOT"

RUN="$BASE/p1_real95_isolated_$(date +%Y%m%d_%H%M%S)_$$"
"$PY" "$ROOT/hv_v8/p1_isolated95.py" run \
  --root "$ROOT" --python "$PY" --host 127.0.0.1 --port 31120 \
  --output-dir "$RUN" --ack-dedicated \
  > "${RUN}.log" 2>&1
RC=$?
echo "P1_ISOLATED95_EXIT=$RC"
echo "P1_ISOLATED95_RUN=$RUN"
tail -n 15 "${RUN}.log"

# 可在运行完成之后离线汇总，绝不覆盖结果
"$PY" "$ROOT/hv_v8/p1_isolated95.py" report --root "$ROOT" --run-dir "$RUN"
```

`run` 自带冻结主模块真实导入预检、py3.7 egg 路径证明、95 条源 SHA 检查。**不必重复运行以前的 8 组 mock 或 worldfix 包准备脚本。** 如果第 155 条失败，它会产生如下证据，不修改科学判定：

```text
$RUN/cases/record_0155/process.log
$RUN/cases/record_0155/result.json
$RUN/cases/record_0155/audit/map_transition_record_0155.json
$RUN/cases/record_0155/route_failure_witness.json
$RUN/cases/record_0155/route_failure_witness.log
```

运行期间每条前后检查 CARLA 版本与 0 车辆/行人。科学级 route/native 报错继续其他记录；CARLA 污染、超时、进程未知崩溃/不安全 world **停止余下运行**，不得掩盖。严格区分：`record_written`、`record_failed_without_result`、`timeout`、`safety_abort`、`not_run`。即使 95 条均产生 `record_*.json`，**`overall_P1_scientific_pass` 当前仍为 False**：重复场景家族来源和固定 Model3 full-scene physics 尚未解决。

**如何交付证据**：上传 `RUN/ledger.json`，以及所有 `RUN/cases/record_*/route_failure_witness.json`，必要时上传少量失败 `process.log`。无需整仓打包。

## 三、P1 处理 record 155 的准确规则

当前事实：Town11 map transition 新 episode、SHA 全通过；155 `ego_route[9]` 是 road 2555 / lane +1 / s 17.9473、junction=true，`get_waypoint(False, Driving)` 返回不同 lane。**这不是地图就绪问题，也不能在没有 XODR 源锚点现场证据时直接判 GT 错误或审计误判。**

`route_failure_witness.json` 会记录：XODR 在原始 s 位置返回的 road/section/lane/s 与 XYZ/yaw 残差；exact XYZ 查询返回的另一 lane；地图和 episode 绑定。仅当独立 XODR 源锚点严格吻合且排除 overlapping junction 几何歧义后，才能另外提出、冻结并验收一个新版地图查询语义契约。在此之前不删断言、不调阈值。

## 四、P2 主线第一步：Town15 条件生成候选 + 独立评价

前提：已经有用户真实生成的：

- `$BASE/p2_topology_town15_20261019_041323_1291.json`
- `$BASE/p2_baseline95_v2_20261019_042433_1291/p2_conditions_95.jsonl`

只使用 `node_keys` + `directed_edges` + `p2_conditions_95.jsonl`，即**GT 不可见的生成输入**。

```bash
P2="$BASE/p2_directional_town15_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$P2"
GRAPH="$BASE/p2_topology_town15_20261019_041323_1291.json"
COND="$BASE/p2_baseline95_v2_20261019_042433_1291/p2_conditions_95.jsonl"

"$PY" "$ROOT/hv_v8/p2_directional_support_v0.py" make \
  --topology "$GRAPH" --conditions "$COND" --town Town15 \
  --max-hops 8 --output "$P2/candidates.json"

"$PY" "$ROOT/hv_v8/p2_directional_support_v0.py" evaluate \
  --topology "$GRAPH" --candidates "$P2/candidates.json" \
  --output "$P2/diagnostic.json"
```

**已使用你上传的真实 Town15 图和 12 条条件执行验证：候选生成 12 条，独立 lane ID 命中仍是 14/26**。这没有改善历史召回；不能宣称 P2 PASS。未命中的 12 个与合法横向换道相关；但必须先取得合法的有向换道 s 区间，避免在图上无条件添加邻道边。

## 五、P2 合法换道段取证（只读、可选）

只有 CARLA 当前已经是 Town15 且实例稳定时运行，不自动换图：

```bash
"$PY" "$ROOT/hv_v8/p2_lateral_legal_segments.py" \
  --root "$ROOT" --host 127.0.0.1 --port 31120 --town Town15 \
  --output "$P2/lateral_segments_town15.json"
```

只允许 `Waypoint.lane_change` 的合法 Left/Right/Both、Driving 邻道、同向航向、局部高程、非路口采样连续段输出。每段存来源 lane / 目标 lane 和来源 `s_min/s_max`。**这是离散采样许可的候选证据，不是连续路径证明或车辆碰撞通过**。如果 CARLA 已切到 Town11 则不运行此命令，也不为此切图。

## 六、P2 真正要实现的后续核心（不偷换验收）

1. **有向 segment 图**：以 `(town, road, section, lane, s_start, s_end)` 而不是一个 lane ID 表示几何；以拓扑 `next` 构建有向连接。与上述 `lateral_segments` 的来源 s 区间求交，才允许带 s 的合法横向边。
2. **ego-route interaction anchors**：由未来 ego 路线同向重合、其他入边汇合、实际空间冲突路口和高度分离检查求锚点；不能用历史 GT 来决定锚点。
3. **逆向 approach + 初始化支持域**：从 anchor 沿图逆向求有界 upstream interval；和车辆生成区域、Model3 尺寸、前景背景占用求交；保留类型（following/merging/crossing）与候选数量。
4. **固定 110 GT 验收**：输入必须是先前冻结的 110 前方 GT 清单。GT 只能用于独立匹配/漏解；不能用 95 条导出的 221 个正标签改变验收分母。要按路线家族划分训练和测试，避免跨 seed 重名泄漏。
5. **P3/P4**：将 P2 approach、s、d、Δyaw、v 解码到真实 3D spawn transform，利用 CARLA 一次性原 pose 真实评估；碰撞失败不得做 z 事后修复。

## 七、质量边界与代码版本

此包里的外层调度器已经过本地语法解析及实际源数据的离线逻辑测试，未连接服务器 CARLA；它不声称新 P1 科学 PASS。特别注意：95 条逐进程审计每条都执行一次完整 world 加载和真实检查，运行开销明显大于一次 95 条进程，但可以逐条归档、不再被首个科学错误遮蔽。若 CARLA 进程崩溃/污染，余下未运行记录保留 UNKNOWN，并停止而不是盲目继续。
