# P1 V3 同源冻结主线执行包

## 内容

- `hv_v8/p1_frozen_mainline_runner_v3.py` — 从服务器活动版创建单一冻结 capsule；在 capsule 内合并可信地图身份、严格道路端点、严格目标选择、CARLA world-snapshot 位姿验证；真实 CARLA 审计 1 条，并仅在严格通过时用**同一份冻结源码**审计 95 条。
- `hv_v8/p1_fixes/p1_target_actor_selection_contract_fix.py` — 历史 V1 目标选择接口引导；它只用于将旧版源码转换成历史中间版本，不作为最终身份依据。
- `hv_v8/p1_fixes/p1_native_audit_schema_contract_fix_v2.py` — 最终目标选择依据为 `agents` 与 `rift_labels.is_interactive`，不存在规则回退。
- `hv_v8/p1_fixes/p1_raw_pose_snapshot_fix.py` — 原始 `vehicle.tesla.model3` 生成后观察 CARLA world snapshot；不修改生成位置。
- `hv_v8/p1_v3_integration_preflight.py` — 合并回归测试，空目标/RIFT 判失败。
- `hv_v8/p1_v3_one_shot.sh` — 单条命令运行上述流程，不退出用户交互式终端。

## 前提

服务器已经存在并曾被 V2 运行器使用的：

- `hv_v8/p1_route_carla_scope_fix_v1.py`
- `hv_v8/p1_map_identity_manifest_runtime_v2_patch.py`
- `data/hv_init/v7/manifests/p1_calibration95.jsonl`
- `data/hv_init/v7/manifests/p1_map_identity_manifest_v2_20261017_154737.json`
- `RIFT/rift/cbv/recognition/config/attention.yaml`
- CARLA 0.9.15: `127.0.0.1:31120`

## 安装（不覆盖 p1_native_audit.py）

```bash
cd /zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
unzip -n /path/to/p1_mainline_v3_bundle.zip -d "$PWD"
bash hv_v8/p1_v3_one_shot.sh
```

## 真实验收

- 单条 P1 严格证据：`P1_SINGLE_RECORD_STRICT_ACCEPTANCE_PASS`；必须是四条 native/parity/pose 验证、零规则兜底，而不仅是 `P1_PROCESS_EXIT=0`。
- 如果单条失败，95 条不会启动；日志中会保存具体失败键、`record_0000.json` 和 `summary.json`（如果生成）。
- 单条严格通过后才会 `=== REAL CARLA P1 FULL 95, SAME FROZEN EXECUTABLE ===`，输出 `P1_FULL95_OUTPUT`。
- 全量必须 `P1_FULL95_PROCESS_EXIT=0` 且 `P1_FULL95_STRICT_ACCEPTANCE_PASS`、`summary.records=95`、每个 record 的非空目标、坐标、RIFT parity、地图 hash 和原始位姿均真实通过。
- 历史保存的审计只证明第一条有四个交互目标；其他 94 条按各自标签核验，不强行要求每条都恰好四辆。
- `fixed_model3_scene_occupancy.status` 单独计数输出，不替代正式判定；若仍为 unknown/incomplete，不能声称占用场景重放已通过，需要独立修复和验收。

## 可复现证据

`RUN_LOG`、`CAPSULE/.../p1_source_provenance.json`、`p1_runtime_changes.diff`、`P1_OUTPUT/record_0000.json`、`P1_OUTPUT/summary.json`；全量则 `P1_FULL95_OUTPUT/summary.json` 及 95 条 record。整个实际审计使用同一份 `FROZEN_EXECUTABLE_SHA256`，活动的 `hv_v8/p1_native_audit.py` 不改动。

**注意：本包仅进行了离线集成/负例测试，没有连接你的 CARLA 服务器；真实 P1 结论必须来自你的服务器输出。**
