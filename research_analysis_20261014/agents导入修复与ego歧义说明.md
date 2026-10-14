# agents导入修复与ego歧义

## 本次错误

carla二进制已能导入，但RIFT的GlobalRoutePlanner还导入agents.navigation.local_planner和agents.tools.misc。这些源文件通常位于CARLA_ROOT/PythonAPI/carla/agents，不能仅靠已安装的carla二进制获得。审计入口此前缺少这段路径引导；本次还没有查询地图、运行native、spawn或physics。

新增carla_audit_environment自动使用CARLA_ROOT，缺省ROOT/carla，将PythonAPI/carla和PythonAPI加入模块路径。保留已有CARLA二进制；只有不存在时才尝试匹配当前Python/平台的egg。不会创建假的agents模块、替换原生规划器或自动下载依赖。

新增check_hv_native_environment.py在不连接服务、不加载模型权重的前提下导入CARLA、agents和真实RIFT native依赖，并打印来源路径。wrapper会在实验开始前把检查结果写入logs/native_environment.log。直接调用audit和NativeAudit也有引导。

55个CPU/内存测试通过，新增检查“已有CARLA二进制仍补agents路径”、缺二进制的显式egg选择、缺agents时具体目录提示。尚非CARLA实测。

## 立即续跑

即使不更新代码，也可以先手动补路径解决当前导入错误：

```bash
export ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
export PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
export CARLA_ROOT="$ROOT/carla"
export PYTHONPATH="$CARLA_ROOT/PythonAPI/carla:$CARLA_ROOT/PythonAPI:$ROOT/RIFT:$ROOT/scenario-dreamer:${PYTHONPATH:-}"
cd "$ROOT"
"$PY" -c 'import carla; import agents.navigation.local_planner as p; print(p.__file__)'
```

若检查仍失败，先确认源文件存在：

```bash
ls -l "$CARLA_ROOT/PythonAPI/carla/agents/navigation/local_planner.py"
```

如果CARLA安装在其它位置，设置真实CARLA_ROOT；若整个agents目录缺失，需要恢复与服务器版本匹配的PythonAPI源文件，不能pip安装一个同名无关agents包。

持久修复：将hv_agents_fix_20261014.zip上传工程根目录后解压，并执行预检：

```bash
unzip -o hv_agents_fix_20261014.zip
"$PY" check_hv_native_environment.py
```

预检通过且31000专用CARLA服务就绪后，复用原95条唯一配对：

```bash
"$PY" run_hv_snapshot_diagnostics.py \
  --diagnostic-pair-report "$ROOT/data/hv_init/eval/snapshot_audit_20261013_160509_936449/v2_pairs100.jsonl.audit.json" \
  --with-native --with-spawn --with-physics \
  --native-records 20 --probe-ticks 20 --port 31000
```

输出使用新目录，不覆盖040705失败证据；不重新配对扫描54936条。原始池native仍会读取其固定100条。此阶段不需要PlanT attention权重。

## ego歧义意味着什么

prefix 50–54的候选只在标准化ego输入上不同；95–99部分仅ego不同，另一个候选同时在ego/actors不同。stratified 5条也均是ego差异。unavailable_sources=[]表示差异工具成功读到了这些候选，不代表已找回唯一历史来源。

当前配对依据ego-local状态。数据集ego_state通常为[0,0,0,speed,length,width,1]，不保存世界xyz、pitch/roll、分量velocity；所以多个源可以通过局部状态检查，却具有不同原始ego物理状态。route与其它actor相同也不能自动推出这些差异无影响。

需要看first_differences中的具体路径、a/b值和absolute_delta，才能判断是位置、高程、姿态、速度、尺寸、车型或box offset。世界位置/高程会影响native距离与地图判断；pitch/roll/尺寸/offset会影响整车占位和spawn；分量速度影响物理回放。当前终端只给“ego”类别，无法判断差异量级。

先上传这两个小文件即可继续判断：

```text
snapshot_audit_20261013_160019_107973/v2_pairs100.jsonl.audit.json.differences.json
snapshot_audit_20261013_160509_936449/v2_pairs100.jsonl.audit.json.differences.json
```

必要时再补.audit.json及相关debug snapshot。暂不放宽容差、任意取首个源或将歧义当作模型失败。95条诊断和全100条评价继续明确区分。
