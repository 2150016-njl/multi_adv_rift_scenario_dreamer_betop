# HV V8 — P0/P1/P2 clean-room implementation

This directory intentionally does **not** reuse the old V5/V6/V7 generator code.

It implements only the dependency chain required by `1015技术路线.md`:

`P0 evidence -> P1 CARLA/RIFT semantic closure -> P2 map-only directed support`.

No neural model is trained here.

## Non-negotiable rules

- Main experiment is **one-shot**.
- best-of-N is an **ablation only**.
- Generator target is A: initial native-positive state.
- B is downstream closed-loop value.
- Generated variables are structured: `approach_id, s, d, delta_yaw, speed_mps`.
- Final application state is `[x,y,yaw,speed]`.
- Vehicle dimensions are not generated; the fixed blueprint is `vehicle.tesla.model3`.
- No `snap`, `clamp`, `retry`, `rerank`, oracle replacement, post-hoc pose correction, or forced `k=0/1/2`.
- `unknown` is never converted to negative.
- RIFT native logic is used only as a read-only oracle and as a map-only support compiler; it never receives GT during support construction.

## Scene input contract

Each `snapshot_path` in the P0 pair manifest must be a JSON object with at least:

```json
{
  "schema_version": "hv_v8_scene_v1",
  "record_index": 2325,
  "scene_id": "stable-scene-id",
  "town": "Town11",
  "opendrive_sha256": "sha256-of-CARLA-to_opendrive",
  "ego": {
    "x": 0.0, "y": 0.0, "z": 0.0,
    "yaw_deg": 0.0,
    "speed_mps": 0.0
  },
  "ego_route": [
    {
      "x": 1.0, "y": 2.0, "z": 0.0,
      "yaw_deg": 0.0,
      "road_id": 1, "section_id": 0, "lane_id": -1, "s": 12.0
    }
  ],
  "actors": [
    {
      "actor_id": 123,
      "type_id": "vehicle.tesla.model3",
      "x": 10.0, "y": 1.0, "z": 0.0,
      "yaw_deg": 0.0,
      "speed_mps": 5.0,
      "is_target": true
    }
  ]
}
```

If the real snapshot uses a different schema, write an explicit converter once.
Do not add field-name discovery or heuristic reconstruction to these auditors.

## P0 manifest rows

`pairs.jsonl` must contain exactly 95 rows:

```json
{"record_index":2325,"scene_id":"...","snapshot_path":"/abs/scene.json","v2_output_path":"/abs/v2.jsonl","v3_output_path":"/abs/v3.jsonl"}
```

`test_index.jsonl` must contain exactly 3727 rows:

```json
{"record_index":0,"scene_id":"...","v2_output_path":"/abs/v2.jsonl","v3_output_path":"/abs/v3.jsonl"}
```

The tool refuses to infer checkpoints, pairings, snapshots, or output files.

## P0

Run from the repository root:

```bash
PYTHONPATH=. python -m hv_v8.p0_freeze \
  --pairs /ABS/pairs.jsonl \
  --test-index /ABS/test_index.jsonl \
  --v2-checkpoint /ABS/V2_CHECKPOINT \
  --v3-checkpoint /ABS/V3_CHECKPOINT \
  --rift-recognition-config /ABS/RIFT_RECOGNITION_CONFIG.yaml \
  --rift-root /ABS/repo/RIFT \
  --rift-source-file /ABS/repo/RIFT/rift/cbv/recognition/base_cbv.py \
  --rift-source-file /ABS/repo/RIFT/rift/scenario/tools/global_route_planner.py \
  --carla-version 0.9.15 \
  --repo-root /ABS/repo \
  --output /ABS/p0_manifest.json
```

P0 PASS requires:
- 95 unique paired records;
- 3727 unique test records;
- every pair is a member of the 3727 index;
- every referenced source/output/checkpoint exists;
- SHA256 is recorded;
- current git commit is recorded;
- no filename/checkpoint guessing occurred.

## Scene schema gate

Before touching CARLA:

```bash
PYTHONPATH=. python -m hv_v8.schema_check \
  --manifest /ABS/pairs.jsonl \
  --output /ABS/scene_schema_audit.json
```

This is a pure-data gate. If it fails, P1/P2 are not run.

## P1

Start a **dedicated empty CARLA 0.9.15**. The script refuses to run on a world containing vehicles.

```bash
PYTHONPATH=. python -m hv_v8.p1_native_audit \
  --manifest /ABS/pairs.jsonl \
  --rift-root /ABS/repo/RIFT \
  --rift-recognition-config /ABS/repo/RIFT/rift/cbv/recognition/config/attention.yaml \
  --rift-mode eval \
  --carla-pythonapi /ABS/CARLA/PythonAPI \
  --host 127.0.0.1 \
  --port 31120 \
  --output /ABS/p1_$(date +%Y%m%d_%H%M%S)
```

P1 checks, independently:
1. map/town identity;
2. OpenDRIVE SHA;
3. ego route road/section/lane/s identity;
4. world -> ego-local -> world roundtrip;
5. exact RIFT preview waypoint semantics;
6. exact RIFT interaction predicate;
7. parity against the repository's `BaseCBVRecog.find_interaction_waypoint`;
8. fixed `vehicle.tesla.model3` blueprint box;
9. **one raw-pose sedan spawn attempt** per selected target, with no z/yaw/xy correction.
10. one-shot scene-occupancy replay when the exact ego/background blueprint IDs are present; otherwise scene occupancy is `unknown`, never guessed.

The output distinguishes:
- `historical native positive`;
- `fixed model3 raw-pose spawn accepted/rejected`;
- infrastructure/semantic errors.

A raw spawn rejection is not repaired and is not turned into a model failure.

## P2

```bash
PYTHONPATH=. python -m hv_v8.p2_support \
  --manifest /ABS/pairs.jsonl \
  --rift-root /ABS/repo/RIFT \
  --rift-recognition-config /ABS/repo/RIFT/rift/cbv/recognition/config/attention.yaml \
  --rift-mode eval \
  --carla-pythonapi /ABS/CARLA/PythonAPI \
  --host 127.0.0.1 \
  --port 31120 \
  --sample-m 1.0 \
  --map-radius-m 140 \
  --output /ABS/p2_$(date +%Y%m%d_%H%M%S)
```

P2 compiler inputs are **only**:
- CARLA map;
- ego pose;
- ego future route;
- slot geometry.

It does not read GT actors while compiling support.

For every driving-lane waypoint in the map near the scene, it checks whether the directed route can reach an ego future-route anchor under the same native routing semantics. It then groups those states into deterministic `approach_id`s and stores:
- directed lane identity;
- anchor route index;
- lane path;
- longitudinal `s` support;
- local slot support;
- piecewise lateral `d` intervals;
- route-distance evidence.

After support compilation, a separate GT audit uses target actors to answer:

`native-positive -> preview lane -> anchor -> approach -> s support`

and reports the first missing layer.

P2 PASS requires every known native-positive GT row to map to an approach, or every unmapped row to have an explicit, independently verified non-model cause. The script does not delete unmapped rows.

## Visualization

For any P2 record:

```bash
PYTHONPATH=. python -m hv_v8.visualize \
  --report /ABS/p2_.../record_2325.json \
  --scene /ABS/scene_2325.json \
  --output /ABS/p2_2325.png
```

The figure shows:
- ego-local route;
- map-only approach support;
- GT native-positive states;
- mapped vs unmapped GT;
- slot-0 front region.

## Acceptance order

Do not train until:

### P0
`p0_manifest.json` is PASS.

### Scene schema gate

Before touching CARLA:

```bash
PYTHONPATH=. python -m hv_v8.schema_check \
  --manifest /ABS/pairs.jsonl \
  --output /ABS/scene_schema_audit.json
```

This is a pure-data gate. If it fails, P1/P2 are not run.

## P1
All required records pass map identity, coordinate roundtrip, RIFT-reference parity and fixed Model3 blueprint calibration. Raw-pose spawn results are separately reported; rejected raw poses remain rejected.

### P2
Known native-positive GT support recall is >= 95% as the first engineering gate, then >= 99% on the final training cohort. The final target is **100% on all verified native-positive rows**; any exception must be explained row-by-row before P3.

### Only after P2
Implement P3:
`GT [x,y,yaw,v] -> [approach,s,d,delta_yaw,v] -> decode -> GT`.
Then P4 structured conditional one-shot model.

## Why this is not a fallback system

The support compiler does not change a model proposal.

It defines the **output space before generation**:

`condition -> approach chart -> model chooses (approach,s,d,delta_yaw,v) -> one raw state`

Evaluation is downstream:

`raw state -> CARLA/RIFT oracle -> pass/fail`

There is no path:

`model -> invalid -> rule repair -> success`.
