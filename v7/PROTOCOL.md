# HV V7 three-file protocol v1 — first round

Scope: P0/P1 and preliminary P2 only. No lane_chart, neural training, geometric generator, resampling or online model integration is implemented.

## Frames and units

- World: `carla_world_left_handed`, X/Y/Z in metres, actor yaw/pitch/roll in degrees.
- Local: `ego_local_carla_x_forward_y_right`, yaw in radians, speed in m/s, +Z up.
- The declared transform is a planar yaw rotation with Z translation. It does not flatten road elevation; pitch/roll are retained in historical world poses but not used to rotate the model's planar axes.
- Local +Y is right, with no additional reflection. World/local velocities use rotation only, without position translation.
- The historical RIFT HD-map right-handed adapter is outside this contract; CARLA world lanes must not be reflected a second time.
- First valid slot: `0 < x <= 32`, `-32 <= y <= 32`. Second slot: `-32 <= x,y <= 32`. Map extent and interaction-anchor extent are independent of these initial-position domains.

## Files

`conditions.jsonl`, `targets.jsonl`, `oracles.jsonl` share unique, ordered `record_id`. `record_index` is the zero-based line in the fixed test JSONL; `dataset_internal_idx` preserves its original internal `idx`. They are different identifiers.

### conditions.jsonl

`schema_version=hv_v7_condition_v1`; contains provenance, event and the `condition` object only. Model consumers read only `condition` plus event eligibility/capacity, never target/oracle files or provenance hashes as features.

Observation fields are allowlisted. The map converter receives ego, ego future route, observed actors and exported CARLA lanes; it never receives GT native routes, interaction points, scores, generated states, or labels. Conflict anchors and approaches are computed before oracle diagnostics.

`condition.directed_map` preserves road/section/lane identity, world/local XYZ, widths, successor/predecessor identities, boundary edges and unknown topology. `opendrive_s_samples=null` is not replaced with geometric arc length: `directed_progress_m` is a different field. Legacy centreline order remains unverified; diagnostics must not reverse it based on a target actor.

`event.mode=historical_unknown`, capacity/eligibility null means the historical source did not serialize event semantics. This does not authorize k=0 or an online generation event. Calibration does not turn these into initialization labels by assumption.

First-round supervised view withholds only the historical selected actors and retains every other actor recorded in the snapshot/ledger. Online uses `normal_online_snapshot` and removes no observed actor. The old sparse conditioning view remains oracle-side comparison evidence and is not silently reused as full online traffic.

### targets.jsonl

`schema_version=hv_v7_target_v1`. Keep all historical selected actors, original vehicle types and poses. Put a known front actor first if available, pack up to two valid historical states into slots, and retain actor-ID bindings. Type migration to sedan is explicitly unverified.

No saved front target means `num_targets`, `valid_mask`, `target_states`, and slot actor IDs are null, rather than a k=0 label. Actual k=0 supervision requires a verified allowed zero reason. Current targets all have `training_eligible=false` until support, sedan and chart gates are passed.

No speed maximum, fixed lateral residual limit or model yaw residual limit is frozen in v1. Speed must be finite and nonnegative; legitimate values above 12m/s remain representable in the data protocol.

### oracles.jsonl

`schema_version=hv_v7_oracle_v1`. Contains historical selected GT/raw pool native reports, original preview/interaction identity, native route and distance metadata, old V2/V3 generated states, historical candidate condition and V6 construction for reproduction. These fields are prohibited from online conditions.

Historical native truth and a newly executed native replay are distinct. Sedan geometry, raw spawn, physics and final green are separate statuses. `not_run`/null is unknown, never a model failure.

## Preliminary P2 support

DirectedMapBuilder -> ConflictAnchorBuilder -> ApproachBuilder are independent modules. Anchors are map-associated ego future-route lane envelopes, not native-certified conflicts. Approach search follows exported successor edges upstream of anchors. Initial intervals clip the centreline to each slot domain, with no learned chart constraints.

`preliminary_topology_support` checks graph reachability/anchor/initial progress. `preliminary_directed_support` also rejects export direction opposed to the native preview waypoint. Neither certifies lane-chart expressiveness, exact route costs, lane-change feasibility, fixed-sedan footprint or raw spawn. Opposed direction is a map-integrity diagnostic (greater than 90 degrees), not a new model heading constraint.

GT native paths are used only to identify missing nodes/edges and candidate failures after the conditions are frozen. In particular, missing lane-change transitions are not copied from oracle routes into the condition graph.

## Reproducibility

Every run is a new directory, refuses overwrite, and records config, input SHA256, code SHA256, git commit, rows, summary and errors. `protocol_sha256.json` binds the three files. Diagnostic re-audits validate it before reading labels.

Default build requires original snapshot bytes and their paired SHA. `--allow-ledger-reconstruction` explicitly enables a separate diagnostic view: ego frame is fitted from map pairs, or a non-withheld observed actor if the map is empty. Such data has snapshot_status=missing_explicit_ledger_reconstruction and cannot pass the original-snapshot P1 gate.

Full S1 index binds 3727 dataset lines to exactly one V2 and one V3 S1 output with matching scene identity. Only the 95 historically paired sources have original snapshot bindings; remaining source bindings and missing checkpoint metadata stay unverified. The P0 gate is not claimed complete.

## Independent CARLA calibration

The prepared plan pins records 2325 and 155, source SHA, map SHA, recognizer SHA and CARLA 0.9.15. Calibration runs only on an explicitly dedicated local service, never 38000. The wrapper starts a new -nullrhi CARLA under riftuser.

It reruns the original native implementation with isolated caches; exports a map seeded only from global map samples and the ego future route; and tests historical GT versus fixed Tesla at the same raw poses. A single official map spawn point is used separately to measure the blueprint box, not to retry a rejected GT pose. Cleanup uses ID-based batch RPC without tick.

The first-round calibration has zero physics ticks. It cannot verify actual velocity execution or settling. Failed/unfinished calibration is stored as infrastructure/calibration evidence, with model_failure_count=null. No V7 model is called.
