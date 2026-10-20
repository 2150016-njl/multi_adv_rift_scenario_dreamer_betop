# P1 95 条地图 episode 切换缺陷：证据、修复和复验

## 已证实的真实事实（2026-10-19）

真实 `p1_route_geometry95_real_20261019_053209_1291` 总结：已扫描 Town12 31 条 + Town15 12 条，合计 **43/95 records、1969 route points**，其中 `verified_xodr_and_exact=1945`、`verified_xodr_road_end_exception=24`、`unresolved=0`；余下 **52 条未扫描，不能计为通过**。`anomalies.jsonl` 是空文件。原工具 `summary.json` 的 Town15 分组 `status_counts={}` 为保存时机上的展示缺陷，逐记录统计和总计显示已验证 487 个 Town15 点；新普查修复每条记录后实时保存分组计数。

中断点：准备切到 Town11 时，`client.load_world('Town11')` 后立即使用 `world.get_map()` 得到 Town15，OpenDRIVE SHA 仍是 Town15 的正确 SHA。这是**地图世界未稳定或拿到旧 world/episode**的确证；无法单靠这份日志在 CARLA 内部异步加载与其他客户端切图之间作唯一归因。此前正式审计在首条 Town11 `record_index=155` 的严格查询中断，与同一缺陷高度吻合，但并非证明 Town11 路线非法。

## 改动

- `p1_world_guard.py`：一次 `load_world()`；轮询新的 `world.id`（必须区别于切图前 episode）、Town 名称、可信 OpenDRIVE SHA，要求同一新 episode 连续两次通过；设超时、保存身份轨迹；从未重试切图、更改源 pose、执行 tick 或 spawn。可在记录结束时检查 episode 未变。
- `p1_native_audit_worldfix.py`：仅新增 guard import，将 `audit_record()` 的旧 `load_world()/get_map()` 换成 guard，在路线检查前验证地图，在结果落盘前检查 episode 未发生变化。**其余已有函数 AST 不变；RIFT、95 条 manifest、road140、XODR road-end 和物理拒绝政策完全未变**。
- `p1p2_pipeline_worldfix.py`：继承修正 py3.7 CARLA egg 的启动器，指向全新冻结 capsule 和新 SHA，保留真实 95 条、identity flag 和子进程 import probe。
- `p1_route_geometry95_forensic_v2.py`：同样使用 guard，并修复 Town 分组计数部分保存时机，仅作为可选诊断；不要求重复前 43 条才能开展正式审计。

### 锁定 SHA

- 父源码：`ed27a2b78ffc7c3e6159db85d5756ff2ad9bfef754e79043225d343f5840854f`
- 新审计源码：`b6036d51d849d0e0aa46c1dc35754e4be9f81ec35dd504146f6cf877ec88a6bb`
- 原始 manifest：`92c16bec23132d44a28de9d1d97894ba2e50ea9bb2f1e53fa41fbe5182664c56`
- map manifest：`9bdb620998c9e67676c90edadf5c45cbe7bc6e12c1b661b12f2a6bb62405aadf`

## 服务器：只需运行一次离线门槛，之后是真实 P1 主任务

```bash
set -euo pipefail
ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
BASE="$ROOT/data/hv_init/v7"

# 将下载的 ZIP 路径替换为服务器实际位置；只新增工具文件。
unzip -n /path/to/p1_world_transition_resolution_20261019.zip -d "$ROOT"
"$PY" "$ROOT/hv_v8/test_p1_world_guard_offline.py"
"$PY" "$ROOT/hv_v8/p1_prepare_worldfix.py" "$ROOT"

# 真实全量，只在 31120 为本项目独占、允许切图/生成/清理 actor 时执行。
TOOL="$ROOT/hv_v8/p1p2_pipeline_worldfix.py"
RUNLOG="$BASE/p1_worldfix_full95_$(date +%Y%m%d_%H%M%S)_$$.log"
set +e
"$PY" "$TOOL" run95 \
  --root "$ROOT" --python "$PY" --host 127.0.0.1 \
  --port 31120 --ack-dedicated > "$RUNLOG" 2>&1
RC=$?
set -e
printf 'P1_WORLDFIX_LAUNCH_EXIT=%s\nP1_WORLDFIX_LOG=%s\n' "$RC" "$RUNLOG"
grep -E '^(P1_CHILD_IMPORT_|P1_FULL95_(RUN_DIR|AUDIT_OUT|PROCESS_EXIT|RESULT_DIR)|P1_FAIL)' "$RUNLOG" | tail -n 25 || true
tail -n 20 "$RUNLOG"
```

**重要：**`run95` 仍可能因首次真正的路线/native/物理验收异常而停止；新地图 guard 不会将科学失败改成 PASS。每条切图生成独立 `map_transition_record_NNNN.json`，能够识别期望 Town、实际新 episode、实际 SHA。若进程失败，上传新 `audit.log`、新 `map_transition_record_*.json`（至少异常记录及上一条）、和已产出的 `record_*.json` 统计即可。不要无限次重试。

## 生成独立账本（存在新审计目录时）

```bash
RUN_DIR=$(sed -n 's/^P1_FULL95_RUN_DIR=//p' "$RUNLOG" | tail -n 1)
if [ -n "$RUN_DIR" ] && [ -d "$RUN_DIR/audit" ]; then
  REPORT="$BASE/p1_worldfix_ledger_$(date +%Y%m%d_%H%M%S)_$$"
  "$PY" "$TOOL" report --root "$ROOT" \
    --audit "$RUN_DIR/audit" --report-dir "$REPORT"
  echo "P1_WORLDFIX_LEDGER=$REPORT"
fi
```

不要把 `p1_pass=true` 等同于合法 Model3 初始化状态充足。historical native 与 raw spawn、occupancy UNKNOWN 仍分层记录。旧 `scene_id` 重名 17 条、ordinal50 `:sparse0` 身份歧义仍留待 provenance 审计；这次修复不会掩盖它们。

## 如果只想继续完成路线普查（可选，不是 P1 全量的前置）

```bash
OUT="$BASE/p1_route95_worldfix_$(date +%Y%m%d_%H%M%S)_$$"
"$PY" "$ROOT/hv_v8/p1_route_geometry95_forensic_v2.py" \
  --root "$ROOT" --host 127.0.0.1 --port 31120 \
  --output-dir "$OUT" --ack-dedicated
```

这个新普查不会复用原 43 条作为“本轮扫描”，但原 43 条保留为独立证据。当前更建议优先运行上面的 P1 95 条正式审计，不用再重复做路线普查。

## P2 与最终目标

P2 可以同步推进有向 lane/connector、合法换道和 route-conditioned conflict anchor；但 Town15 `14/26` 仅为旧纵向拓扑诊断，**不是** 110 固定前方 GT 的 approach recall。正式生成训练需要可信的原生交互监督与固定 Model3 合法初始化支持域，不能把已测量的原始 spawn 拒绝重命名为正例。
