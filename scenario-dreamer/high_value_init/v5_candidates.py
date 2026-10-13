"""V5 online-available, ego-local CARLA-HD-map interaction candidates.

Pure geometry: NO RIFT labels, actor IDs, target states or online selection scores.
The same deterministic function is used by training and online generation.
Each candidate contains a directed lane fragment and a point near ego's future route.
Geometry here constructs *model conditions*, NEVER modifies a generated state.
"""
import math
import numpy as np

FOV = 64.0
MAX_CANDIDATES = 32
MAX_LANE_POINTS = 48


def _xy(point):
    if isinstance(point, dict):
        return [float(point['x']), float(point['y'])]
    return [float(point[0]), float(point[1])]


def _array(polyline):
    try:
        a = np.asarray([_xy(p) for p in polyline], dtype=np.float64)
        return a if a.ndim == 2 and a.shape[1] == 2 and len(a) > 1 and np.isfinite(a).all() else None
    except (TypeError, ValueError, KeyError, IndexError):
        return None


def _cum(points):
    return np.r_[0., np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))]


def _resample(points, step=1.5, max_samples=300):
    s = _cum(points)
    unique = np.r_[True, np.diff(s) > 1.e-7]
    if unique.sum() < 2:
        return None
    s, points = s[unique], points[unique]
    n = max(2, min(max_samples, int(math.ceil(s[-1] / step)) + 1))
    q = np.linspace(0., s[-1], n)
    return np.column_stack([np.interp(q, s, points[:, k]) for k in range(2)])


def closest(poly, xy):
    """Point-to-directed-polyline projection: (distance, arc distance, tangent, projected_xy)."""
    xy = np.asarray(xy, dtype=np.float64)
    a, b = poly[:-1], poly[1:]
    d = b-a
    denom = np.maximum(np.sum(d*d, axis=1), 1.e-10)
    t = np.clip(np.einsum('ij,ij->i', xy-a, d) / denom, 0., 1.)
    q = a+t[:, None]*d
    distance = np.linalg.norm(q-xy, axis=1)
    index = int(np.argmin(distance))
    angle = math.atan2(float(d[index, 1]), float(d[index, 0]))
    progress = _cum(poly)[index] + t[index]*math.sqrt(denom[index])
    return float(distance[index]), float(progress), angle, q[index].tolist()


def get_lanes(condition):
    m = condition.get('map') or {}
    if m.get('lane_polylines'):
        return list(m['lane_polylines'])
    return [row.get('centerline', []) for row in m.get('lanes_v3', [])]


def make_candidates(condition, max_candidates=MAX_CANDIDATES, fov=FOV):
    """Generate a *fixed map-only candidate set* before any model sampling.

    Future ego-route points (s>5m) that approach each directed lane are
    potential conflict/merge/longitudinal interaction anchors. Every approach
    is considered geometrically; no generated state or RIFT label is inspected.
    """
    route = _array((condition.get('ego_route') or {}).get('polyline') or [])
    if route is None:
        return []
    route = _resample(route, step=1.5)
    if route is None:
        return []
    rs = _cum(route)
    valid_r = [i for i in range(len(route)) if 5.0 <= rs[i] <= fov/2.-1.0 and np.max(np.abs(route[i])) <= fov/2.]
    if not valid_r:
        return []
    candidates = []
    for lane_id, row in enumerate(get_lanes(condition)):
        line = _array(row)
        if line is None:
            continue
        dense = _resample(line, step=1.5)
        if dense is None:
            continue
        # Split into contiguous visible fragments. This affects *input conditions*
        # only; output x/y will be computed by the model's differentiable head.
        inside = np.max(np.abs(dense), axis=1) <= fov/2.-1.0
        idx = np.flatnonzero(inside)
        if len(idx) < 2:
            continue
        groups = np.split(idx, np.where(np.diff(idx) > 1)[0]+1)
        for frag_no, ids in enumerate(groups):
            if len(ids) < 2:
                continue
            # Cap to 48 along-line points (not nearest-to-ego sorting).
            frag = dense[ids]
            if len(frag) > MAX_LANE_POINTS:
                indexes = np.linspace(0, len(frag)-1, MAX_LANE_POINTS).astype(int)
                frag = frag[indexes]
            arc = _cum(frag)
            if arc[-1] < 1.0:
                continue
            # Each 7m route-progress cell yields at most one anchor per lane.
            best_by_bin = {}
            for r_idx in valid_r:
                dist, lane_s, angle, projected = closest(frag, route[r_idx])
                if dist > 5.0:
                    continue
                bucket = int(rs[r_idx] // 7.0)
                candidate = (dist, r_idx, lane_s, angle, projected)
                if bucket not in best_by_bin or dist < best_by_bin[bucket][0]:
                    best_by_bin[bucket] = candidate
            for bucket, (dist, ri, anchor_s, angle, point) in best_by_bin.items():
                # Longitudinal approach: target must lie behind the anchor
                # on the directed lane (enforced by the learned chart).
                if anchor_s < 3.0:
                    continue
                anchor_route_s = float(rs[ri])
                candidates.append({
                    'lane_id': int(lane_id), 'fragment': int(frag_no),
                    'polyline': frag.astype(np.float32).tolist(),
                    'lane_length': float(arc[-1]),
                    'anchor_s': float(anchor_s),
                    'anchor_xy': [float(point[0]), float(point[1])],
                    'ego_route_s': anchor_route_s,
                    'distance_to_ego_route': dist,
                    'anchor_heading': angle,
                })
    # Diverse deterministic coverage: first best point per directed lane,
    # then other anchors; order must be identical online and offline.
    candidates.sort(key=lambda c: (round(c['distance_to_ego_route'], 5),
                                   c['ego_route_s'], c['lane_id'], c['fragment']))
    chosen, overflow, seen = [], [], set()
    for c in candidates:
        key = (c['lane_id'], c['fragment'])
        if key not in seen:
            chosen.append(c)
            seen.add(key)
        else:
            overflow.append(c)
    return (chosen+overflow)[:int(max_candidates)]


def assign_pair(candidates, target_states, supervision=None, max_pos_m=3.0,
                max_heading_rad=0.5, max_anchor_m=12.0):
    """OFFLINE ONLY: match RIFT-positive GT actors to online-available candidate charts.

    GT labels may choose training targets, never construct candidate geometries.
    Reject samples when candidate mapping is infeasible rather than relabeling.
    """
    if len(target_states) != 2 or not candidates:
        return None
    assigned = []
    for slot, state in enumerate(target_states):
        x, y, yaw = [float(v) for v in state[:3]]
        anchor = None
        if supervision and slot < len(supervision):
            anchor = (supervision[slot] or {}).get('conflict_point_local_xy')
        options = []
        for i, candidate in enumerate(candidates):
            dist, s, angle, _ = closest(np.asarray(candidate['polyline']), [x,y])
            err = abs(math.atan2(math.sin(yaw-angle), math.cos(yaw-angle)))
            if dist > max_pos_m or err > max_heading_rad or s >= candidate['anchor_s']-0.25:
                continue
            anchor_err = math.dist(candidate['anchor_xy'], anchor[:2]) if anchor else 0.
            if anchor is not None and anchor_err > max_anchor_m:
                continue
            options.append((dist+2.*err+0.1*anchor_err, i, s))
        if not options:
            return None
        options.sort()
        assigned.append((options[0][1], options[0][2]))
    return assigned


def signature(record):
    import hashlib, json
    condition = record.get('condition') or {}
    # A canonical content-based representation (not source path or scene-id hash).
    payload = {'town': record.get('town'), 'condition': condition,
               'targets': (record.get('target') or {}).get('generated_agent_states')}
    raw = json.dumps(payload, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()
