# V7 第一轮：P0、P1、P2 前半部分

执行依据：1015技术路线.md。没有实现 lane_chart、模型、loss、训练、几何生成器或在线接管；没有冻结 1.35m 横向限幅、0.40rad 输出航向限幅或 12m/s 速度上限。

## 已实现的独立代码

- `v7/schema.py`：三文件协议、标签隔离、唯一ID、有效mask、前方第一槽与全FOV第二槽校验。
- `v7/coordinates.py`：CARLA world与ego-local位置、yaw、速度契约；局部+y向右，不额外镜像。
- `v7/condition_builder.py`：同一观测入口供离线/未来在线调用；只白名单读取ego、route、地图和观测车辆。
- `v7/directed_map.py`：保留road/section/lane、宽度、高程、边界/未知边；不依据XY距离补边。
- `v7/approach_builder.py`：独立ConflictAnchorBuilder和ApproachBuilder，构造全ego未来route的初步锚点与上游路径。锚点不限制在生成方框内。
- `v7/diagnostics.py`：候选冻结后才读GT oracle，逐GT检查preview lane、原生路线、anchor、纵向支持、导图方向和旧V5/V6漏因。
- `v7/pipeline.py`、`v7/io.py`：P0输入/代码SHA、3727条S1索引、95条绑定、三文件构建与禁止覆盖。
- `v7/carla_map.py`：独立CARLA导图，保留真实station/yaw；仅从同一lane的next/previous确定station方向，不跨road的s重置推断方向。采集RIFT GRP中由真实地图标线产生的换道边，独立保存，暂不粗暴合并到lane级next图。
- `v7/calibration.py`：定点GT/固定轿车校准，原生cache隔离、地图SHA、真实蓝图、零tick批量清理。未调用模型。
- `v7_cli.py`：`build`、`audit-map`、`audit-approaches`、`calibrate`。

三文件详细契约见 `v7/PROTOCOL.md`。训练标签和oracle不能进入condition；未保存前方GT不是k=0样本。历史车型与固定轿车兼容性仍是独立未知字段。

## 本地实际完成的实验

最终数据目录：

`data/hv_init/v7/diagnostics/p0_p2_round1_final_20261015/`

原始snapshot尚未同步到本地，因此本轮本地运行明确使用`--allow-ledger-reconstruction`。ego变换来自世界/局部地图点对；Town11空图例由非目标的已观测actor恢复。没有用GT pose拟合模型条件。所有95条的snapshot原始字节状态仍为missing，不能冒称通过原始snapshot P1验收。

生成了：

- `conditions.jsonl`、`targets.jsonl`、`oracles.jsonl`：各95条，一一关联。
- `full_s1_source_index.jsonl`：3727条V2/V3 S1与test行索引对齐；只有95条有已验证的历史snapshot绑定，其他3632条仍为unverified。
- `source_manifest.json`：旧checkpoint声明SHA、实际可读性、训练配置状态；本地权重未同步，V3 metadata未验证。
- `map_alignment_rows.jsonl`：95条坐标/地图诊断。
- `rows.jsonl`：309条逐actor报告，分别为110前方选中GT、80非前方选中GT、119原始池前方正例，分组统计，不混为独立样本。
- `cases/record_2325_{condition,target,oracle,approach_report}.json`与155对应文件。
- `calibration_plan.json`：2325、155、1532、160、40，尚未实跑。
- `summary.json`、输入/代码/协议SHA、git commit、config与错误日志。

局部坐标与历史actor状态的最大XY/回环误差约5.24e-14m，yaw约1.78e-15rad。它证明ledger数值契约一致，不证明缺失的原始snapshot已验证，也不替代新CARLA原生回放。

三文件SHA验证后的独立`audit-map`与`audit-approaches`也已运行，输出位于`map_recheck_20261015`和`approach_recheck_20261015`。

## 逐GT发现

|检查|前方选中GT（110）|原始池前方正例（119）|选中非前方GT（80）|
|---|---:|---:|---:|
|preview lane在历史图中|109|118|79|
|预备拓扑/anchor/初始进度支持|82|87|55|
|preview方向与导出折线相反|33|35|21|
|同时通过当前方向与支持检查|50|55|34|

这不是新模型成功率，也不是已通过P2的正式召回。初步支持没有检查lane_chart、station约束下的换道时机、完整路线代价、轿车footprint或spawn。更严格地检查了导图方向，不能直接将50/110和旧V6的54/110当同口径模型比较。

110个前方GT中，历史原生路线有25条出现同一road/section的跨lane转移，历史next-only导图没有这些边；另有1条为空地图。它们保留为map support unknown，未删除分母。

旧V5审计几何覆盖70/110，40个漏例精确复现：

- 28个：原始地图有1.25m内的近车道，但该车道位置未进入旧fragment/anchor候选。
- 12个：旧1.25m距离门限之外、原生preview lane半宽范围内。

上述是旧审计的几何门限，不是完整V5训练筛选链；不应将40个全解释为模型拟合失败。

旧V6覆盖54/110，55个false和1个unknown精确复现：

- 30个heading门限拒绝，其中29个同时存在preview lane导出方向相反。
- 24个preview lane未进入旧approach集合。
- 1个lane宽度匹配失败。
- 1个Town11空图/未知route roots。

旧V5和V6的布尔重算与原ledger均为0条不一致。

这里的33个反向问题来自历史CARLA审计导图，不等于已经证明V2/V3训练用的HD-map也反向。两份地图来源必须分开。原生preview/native判断直接查询CARLA API与GRP，不依赖该导出折线；不能因此推翻已完成的V2/V3原生成功率，也不能将导图反向算成模型yaw预测失败。

record2325 actor567：preview lane `179:0:1`与interaction lane `51:0:-1`均存在，原生路径拓扑可对齐；但preview朝向和导出折线切线相差约3.1326rad。不是把模型yaw约束放宽到π就能修好的问题，应先修导图方向。actor573的preview lane `231:0:-1`，旧V6可覆盖，中心到preview lane折线距离约1.545m，展示固定1.25m与实际lane宽度之间的差别。

record155：历史audit_map_lanes为空，但原生preview和interaction route有证据。它必须是地图输入缺失，不是无解或模型失败。

另外record160/actor2205和record40/actor6138的交互lane身份存在，但原生交互点到导出折线的距离约0.94m，投影落在折线端部；需检查截断与端点归属，不能直接扩大匹配阈值掩盖。

## 当前停止点

P0/P1/P2的代码与首版诊断可用，但验收门均未声明通过。当前最早失败层是地图方向、换道边和局部导图缺失。下一轮先用独立CARLA的地图证据校准这些问题，再决定P2后半部分和lane_chart；不进入P3/P4训练。

## 上传到服务器

直接同步源文件，不需要zip或服务器解压：

1. 新目录`v7/`（不必上传__pycache__）。
2. `v7_cli.py`、`test_hv_v7.py`。
3. `run_hv_v7_calibration_headless.sh`。
4. 更新的`run_hv_native_diagnostics.py`、`run_hv_native_headless.sh`。
5. 确保此前更新的`scenario-dreamer/high_value_init/native_state_audit.py`已包含`destroy_owned_without_tick`。
6. `1015技术路线.md`和本说明。

## 服务器运行：原始snapshot严格构建

在root终端执行CPU构建，不需要CARLA；不能加`--allow-ledger-reconstruction`来绕过服务器缺失文件：

```bash
export ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
export PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
export OLD="$ROOT/data/hv_init/eval/native_headless_20261014_130841_599893"
export V7="$ROOT/data/hv_init/v7/datasets/round1_$(date +%Y%m%d_%H%M%S)"
cd "$ROOT"

"$PY" -m unittest -q test_hv_v7 test_hv_mainline_batch test_hv_owned_cleanup test_hv_runtime_audit

"$PY" -u "$ROOT/v7_cli.py" build \
  --experiment "$OLD/experiment" \
  --dataset-jsonl "$ROOT/data/hv_init/datasets/inpainting_sparse_phase8_v3_fov64_hdmap_full/test.jsonl" \
  --snapshot-root "$ROOT/data/hv_init/rift_labeled_phase7_v2_uncovered_routes_seed0_7" \
  --output-dir "$V7"

printf 'V7_DATASET=%s\n' "$V7"
```

输出目录必须不存在，程序拒绝覆盖。可以加`--record-indices 2325,155`做定点小构建；这不会改变P0完整3727条输出索引的冻结范围。所有模型/训练验收状态仍按证据设置，build退出0表示诊断完成，不表示研究门已通过。

## 独立CARLA校准

构建生成`calibration_plan.json`后，可在另一终端独立运行校准分支。它不需要等待P2结论，也不更改三文件：

```bash
export ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
# 新终端务必把V7设为上一条命令实际打印的V7_DATASET路径。
test -f "$V7/calibration_plan.json" || { echo "Set V7 to the printed dataset directory"; exit 1; }
bash "$ROOT/run_hv_v7_calibration_headless.sh" \
  --plan "$V7/calibration_plan.json" --port 31120
```

wrapper由root启动，实际审计和CARLA由riftuser运行，使用-nullrhi，新服务端口自动避占用，保留38000–38002；新结果在`data/hv_init/v7/eval/native_headless_时间/`。

校准顺序：原始snapshot SHA -> OpenDRIVE SHA -> 新地图条件冻结 -> 原生GT回放比较 -> 独立官方spawn point测蓝图尺寸 -> GT原位单次spawn -> 同原始pose固定轿车单次spawn -> ID RPC清理。地图加载有独立300秒上限，其他RPC默认30秒；不运行physics tick、不中途抬z或重试，不用零tick观测宣称速度已执行。

查看`v7_calibration/rows.jsonl`中的`native_comparisons[].fresh_map_alignment`、`fixed_sedan_labels`、`cohorts`和`cleanup_evidence`。固定轿车几何兼容只有整组实际占位完整时才给布尔值，否则为unknown。

## 回传与验收

CPU分支：`summary.json`、`source_manifest.json`、`map_alignment_rows.jsonl`、`rows.jsonl`、`cases/`、输入/代码/协议SHA；保留三文件原始数据便于必要时复查。

CARLA分支：外层`status.json`、`server.log`、`experiment.log`和完整`v7_calibration/`。包括新导出的地图和原生换道边，不能只传汇总。

本地已通过16项V7 CPU测试及包含旧主线/清理/原生适配的120项回归；技术路线要求的完整runtime回归和旧V5/support-chain测试另外通过。两份shell启动脚本通过bash语法检查，运行代码通过Python3.8语法检查。CARLA实跑未在本地进行。
