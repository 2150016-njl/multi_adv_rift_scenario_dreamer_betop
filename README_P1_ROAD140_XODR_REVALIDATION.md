# HV-V8 P1：road140 原始 XODR 锚点修复与真实 CARLA 分层验收

状态：**仅候选补丁、离线正反例测试通过；没有新源码真实 CARLA PASS。**

## 0. 已有真实证据

- 2026-10-18 CARLA 0.9.15 Town15 SHA `8e9c69cfb3e750ace3e5d6b057e46786952a073c3924f319bbabb1a166dc2c17` 与独立 manifest 一致。
- ordinal 8、`record_index=100`、`ego_route[7]`，原 `s=128.94887733459467`。
- XODR source：`abs(s_diff)=0.000005722045955280919m`；XYZ 差 0，yaw 差 0.0000038147°。
- XYZ->exact source：`abs(s_diff)=0.05094718933105469m`；3D 位置差 0.05080163299m；仍同 road/section/lane。
- 测量文件：`data/hv_init/v7/p1_town15_route7_forensics_20261018_130342_2012.json`（服务器已实际生成）。

## 1. 补丁范围和契约

**仅修改从 SHA `34a80666...` 创建的新 capsule 中 `hv_v8/p1_native_audit.py`**，不修改活动源码、历史快照、标签、RIFT 配置、车辆生成位置和旧输出。

- 正常 XODR FOUND：作为原始 road/lane/s/XYZ/yaw 的来源锚点，lane 三字段全一致，s 误差 `<=1e-4m`，3D 位置 `<=0.05m`，yaw `<=0.5°`。
- exact-location 查询：仍强制 lane 一致、XY 和 XYZ 与输入位置差 `<=0.30m`、yaw `<=2°`。其 `s` 偏差原值单独保留，无候选/标签兜底。
- XODR/exact 比较仍维持原 `<=0.30m`, `<=2°` 两项。
- XODR NONE：仅原冻结源码严格验证的 road-end 例外可通过，未验证的一律拒绝。
- native replay route：XODR FOUND 时使用源 `s` 对应的 XODR waypoint；端点例外才沿用严格验证的 exact waypoint。这是必须做真实 RIFT parity 回归的行为变化。

这些约束目前为**待真实 CARLA 验证的候选契约**，不代表任何科学 PASS。不得单纯把 `.05` 改成 `.10`。

## 2. 在服务器验证补丁与创建新源码 capsule（此阶段不连接 CARLA）

将随附两份 `.py` 文件放到 `$ROOT/hv_v8/`（不覆盖活动 `p1_native_audit.py`）。

```bash
ROOT=/zeron-vepfs/tjqc/njl/multi_adv_rift_scenario_dreamer_betop
PY=/zeron-vepfs/tjqc/njl/conda_envs/rift/bin/python
PATCH="$ROOT/hv_v8/p1_route140_xodr_anchor_patch.py"
TEST="$ROOT/hv_v8/test_p1_route140_xodr_contract.py"
sha256sum "$PATCH" "$TEST"
"$PY" "$TEST"
"$PY" "$PATCH" --root "$ROOT"  # dry-run; original SHA, manifest/map SHA and exact source anchor checks
PATCH_LOG="$ROOT/data/hv_init/v7/p1_route140_patch_$(date +%Y%m%d_%H%M%S)_$$.log"
"$PY" "$PATCH" --root "$ROOT" --create >"$PATCH_LOG" 2>&1
PATCH_RC=$?
printf 'PATCH_EXIT=%s\nPATCH_LOG=%s\n' "$PATCH_RC" "$PATCH_LOG"
cat "$PATCH_LOG"
CAPSULE=$(sed -n 's/^P1_ROUTE140_NEW_CAPSULE=//p' "$PATCH_LOG" | tail -n 1)
if [[ -n "$CAPSULE" && -f "$CAPSULE/hv_v8/p1_native_audit.py" ]]; then
  sha256sum "$CAPSULE/hv_v8/p1_native_audit.py"
  cat "$CAPSULE/route140_xodr_anchor.diff"
fi
```

如果 dry-run 的 exact 源码块匹配失败，**停止**，不要去掉 SHA/替换 guard，也不要改旧活动源码。需先对服务器原 capsule 做只读 diff 审查。patch 脚本会生成新的、带新 provenance 的 capsule；旧 capsule 仍然保留。`py_compile`、mock 自测、dry-run 均不是真实 CARLA PASS。

## 3. 准备两个单条 manifest（只写新诊断目录，原始 95 manifest 不动）

```bash
STAGE="$ROOT/data/hv_init/v7/p1_route140_single_manifests_$(date +%Y%m%d_%H%M%S)_$$"
mkdir -p "$STAGE"
"$PY" - "$ROOT" "$STAGE" <<'PY'
import hashlib,json,sys
from pathlib import Path
root, stage = map(Path,sys.argv[1:])
p = root/'data/hv_init/v7/manifests/p1_calibration95.jsonl'
assert hashlib.sha256(p.read_bytes()).hexdigest() == '92c16bec23132d44a28de9d1d97894ba2e50ea9bb2f1e53fa41fbe5182664c56'
lines=[x for x in p.read_text(encoding='utf-8').splitlines() if x.strip()]
assert len(lines)==95
for ordinal, index in ((8,100),(0,0)):
    record=json.loads(lines[ordinal]); assert int(record['record_index']) == index
    path=stage/('record_{}.jsonl'.format(index))
    with path.open('x',encoding='utf-8') as f: f.write(lines[ordinal]+'\n')
    print('P1_SINGLE_MANIFEST_{}={}'.format(index,path))
PY
```

## 4. 真实 CARLA 单条（先 100，再 0000）

执行前人工确认这是可供本实验专用的 CARLA 实例、没有其他用户。审计本身会 `load_world`、spawn，**不是 world 只读操作**。原程序在开头拒绝车辆非空，严禁在共享 world 自动清理其他 actor。

```bash
export PYTHONPATH="$CAPSULE:$ROOT:$ROOT/RIFT:$ROOT/scenario-dreamer:$ROOT/carla/PythonAPI:$ROOT/carla/PythonAPI/carla:$ROOT/carla/PythonAPI/carla/dist/carla-0.9.15-py3.7-linux-x86_64.egg:${PYTHONPATH:-}"

# 将 100 改为 0，即可执行第二次 Town12 回归；每次运行都要使用新输出路径。
RID=100
INPUT="$STAGE/record_${RID}.jsonl"
OUT="$ROOT/data/hv_init/v7/p1_route140_new_real_${RID}_$(date +%Y%m%d_%H%M%S)_$$"
LOG="$OUT.log"
if [[ -d "$CAPSULE" && -f "$INPUT" && ! -e "$OUT" ]]; then
  (cd "$CAPSULE" && "$PY" -m hv_v8.p1_native_audit \
      --manifest "$INPUT" \
      --rift-root "$ROOT/RIFT" \
      --rift-recognition-config "$ROOT/RIFT/rift/cbv/recognition/config/attention.yaml" \
      --rift-mode eval \
      --carla-pythonapi "$ROOT/carla/PythonAPI/carla" \
      --map-identity-manifest "$ROOT/data/hv_init/v7/manifests/p1_map_identity_manifest_v2_20261017_154737.json" \
      --host 127.0.0.1 --port 31120 \
      --startup-timeout 600 --startup-rpc-timeout 5 --startup-poll-interval 2 \
      --rpc-timeout 120 --expected-carla-version 0.9.15 \
      --limit 1 --output "$OUT") >"$LOG" 2>&1
  RUN_RC=$?
  printf 'P1_SINGLE_RC=%s\nP1_SINGLE_LOG=%s\nP1_SINGLE_OUT=%s\n' "$RUN_RC" "$LOG" "$OUT"
  tail -n 45 "$LOG"
else
  echo 'P1_NOT_STARTED: missing capsule/manifest or output already exists'
fi
```

检查结果，不要仅凭退出码或 summary：

```bash
"$PY" - "$ROOT" "$CAPSULE" "$OUT" <<'PY'
import hashlib,json,sys
from pathlib import Path
root,capsule,out=map(Path,sys.argv[1:])
meta=json.loads((capsule/'p1_source_provenance.json').read_text())
sha=hashlib.sha256((capsule/'hv_v8/p1_native_audit.py').read_bytes()).hexdigest()
assert sha == meta['frozen_sha256'], 'new frozen sha drift'
sys.path.insert(0,str(root))
from hv_v8.p1_frozen_mainline_runner_v3 import _strict_acceptance
ok,errors,evidence=_strict_acceptance(out,1)
print('P1_NEW_SINGLE_STRICT_RESULT',ok)
print('P1_NEW_SINGLE_STRICT_ERRORS',json.dumps(errors,ensure_ascii=False))
print('P1_NEW_SINGLE_STRICT_EVIDENCE',json.dumps(evidence,ensure_ascii=False))
if ok:
  record_files=list(out.glob('record_*.json'))
  assert len(record_files)==1
  r=json.loads(record_files[0].read_text())
  assert r['route_checks'] and all(x.get('pass') is True for x in r['route_checks'])
  assert all('xodr_source_anchor_pass' in x for x in r['route_checks'])
PY
```

如运行在 100 后留下 vehicle actor，**不要直接继续 0000**。确认独占实例后再按已授权方式重置专用 world 或重启独立 CARLA 服务。不要在共享 world 上执行这些操作。

每次记录还必须查新字段 `route_reference_kind`, `exact_location_s_abs_error_m`, `xodr_source_anchor_pass`；第一条 0000 的历史 native parity 应为 4/4、目标 IDs 为 6121/6125/6137/6138、原始 Model3 spawn accepted 4/4、occupancy complete。

## 5. 95 条（只有两个真实单条严格验收通过后）

使用同一个**新** capsule / sha，新输出路径，不修改 `p1_calibration95.jsonl`。audit 命令与单条相同，区别：

```bash
--manifest "$ROOT/data/hv_init/v7/manifests/p1_calibration95.jsonl" \
--limit 95 --output "$ROOT/data/hv_init/v7/p1_route140_new_full95_<unique-new-id>"
```

勿将字面 `<unique-new-id>` 直接作为实际输出路径；请用新的时间戳和 PID 构造 `OUT`。注意旧 `p1_resume_full95_dedicated_v2.py` 硬编码了原 SHA，不可未经审核就套在新 capsule 上；更不能删 SHA guard 来迎合执行。需保留实际 full95 进程退出码和日志，并用 `hv_v8.p1_frozen_mainline_runner_v3._strict_acceptance(out,95)` 独立验收，**仅当进程 RC=0、摘要 records=95、p1_pass=True、严格验收器通过，且执行前后 capsule SHA 均等于新 sha 时**，输出 `P1_FULL95_STRICT_ACCEPTANCE_PASS`。

若任何阶段失败，保持原日志、record 和 capsule；STOP，不要让新修复变成连续隐式兜底。P1 全量通过后仍须独立核对旧 Round3/4 P1/P2 gates。
