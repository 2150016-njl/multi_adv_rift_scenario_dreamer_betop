# CARLA启动、重复差异读取及完整补丁

## 此次日志的结论

agents路径已修复，check_hv_native_environment成功导入真实CARLA与RIFT依赖。server_contacted=false表示它只查依赖，不代表31000服务就绪。后续client.get_server_version等待120秒超时，说明该地址没有及时返回RPC；尚未执行地图/native/spawn/physics，不能据此评价生成模型。常见原因是服务没启动、还未就绪或已退出，需要server日志确认。

diagnose_hv_pair_ambiguity的FileExistsError只是重复调用碰到既有结果；不用删除、重算或修改原报告。check_hv_attention_assets.py缺失则说明服务器当前代码不完整或不在对应工程目录；本地打包清单包含该文件。它与初始native审计无关，不应先用缺PlanT权重挡住route/preview实验。

## 本轮修复

- run_hv_native_diagnostics.py管理Linux专用CARLA：先验证既有partial证据和依赖，再启动服务，轮询RPC+world map就绪，执行原始池及成对native/spawn/physics。root使用riftuser启动；单独记录server.log、preflight.log、experiment.log、status.json。
- 只停止自己创建的进程组。端口占用时默认停止并报错，不接管其它实验。--use-existing-server仅用于用户自己准备好的专用空服务，并且不会清理该服务。
- 启动等待默认最多600秒，期间打印进度；服务进程退出会立即报错并指向server.log。服务就绪后的audit RPC超时独立默认300秒，不靠增加超时掩盖服务不存在。
- 既有differences文件重复调用时直接读取，打印具体字段和数值，不改写结果；新结果记录源report SHA。源report变化时要求另指定输出文件，避免误用旧结果。
- verify_hv_patch.py核对补丁文件清单和SHA，明确发现缺失/不同文件，包含attention资产检查脚本。

63项CPU/内存测试通过，包含RPC逐步就绪、启动失败、等待上限、只清理自有进程、既有差异重复读取及源变化拦截。尚非服务器CARLA实测。

## 一次完成同步与续跑

把hv_managed_native_20261014.zip上传工程根目录，然后执行：

```bash
export ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
export PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
cd "$ROOT"
unzip -o hv_managed_native_20261014.zip
"$PY" verify_hv_patch.py --manifest hv_managed_native_manifest.json

"$PY" run_hv_native_diagnostics.py \
  --pair-report "$ROOT/data/hv_init/eval/snapshot_audit_20261013_160509_936449/v2_pairs100.jsonl.audit.json" \
  --port 31000 --native-records 20 --probe-ticks 20
```

这条新命令自行启动服务，无需另开终端启动CARLA。默认使用ROOT/carla；CARLA在其它位置时先export CARLA_ROOT为真实目录。不下载软件或模型，也不需要PlanT权重。原始池100条native、95条成对snapshot、前20条成对native/spawn/physics分别记录；95条仍仅为有明确排除项的诊断子集，不代表完整100条成绩。

如果31000被占用，先确认服务归属；可以明确改--port 31010启动新的自有服务。若确认31000就是自己准备的专用空服务，用同一命令加--use-existing-server，入口会先查RPC，不会另起或关闭该服务。

日志目录为终端RESULT_DIR=.../native_managed_时间戳，其中experiment/是具体审计产物。成功或失败都上传整个目录。server.log可判断CARLA启动失败；status.json区分启动失败与实验失败；真实native结果看experiment中逐阶段rows/summary。

## 读取现有ego差异

```bash
for RUN in snapshot_audit_20261013_160019_107973 snapshot_audit_20261013_160509_936449; do
  "$PY" diagnose_hv_pair_ambiguity.py \
    --pair-report "$ROOT/data/hv_init/eval/$RUN/v2_pairs100.jsonl.audit.json"
done
```

现在不会因原结果存在而失败，直接打印已有first_differences的字段、a/b和absolute_delta。完整数据仍在原来的两个.audit.json.differences.json，上传这两个小文件即可先分析ego差异量级。这里只读结果，尚未消除歧义。

## attention资产另行检查

```bash
"$PY" check_hv_attention_assets.py --find
```

若补丁核对已通过，该脚本应存在；若模型文件仍缺失，它会明确报告资产缺失，属于与本次native独立的问题。正确权重恢复后再运行strict attention在线实验，不用把state encoder问题带入初始route实验。
