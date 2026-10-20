#!/usr/bin/env python3
"""Fail-closed CARLA 0.9.15 world episode/map readiness fence.

One world load; poll current world for a NEW episode and exact Town/XODR.
No CARLA tick, actor spawn, pose change, map reload retry or scientific fallback.
"""
import hashlib
import json
import time
from pathlib import Path


class MapTransitionFailure(RuntimeError):
    pass


def _town(name):
    return str(name).replace('\\', '/').rstrip('/').split('/')[-1]


def _episode(world):
    ep = getattr(world, 'id', None)
    if ep is None or isinstance(ep, bool):
        raise MapTransitionFailure('world.id episode ID unavailable: fail-closed')
    return int(ep)


def _save(path, evidence):
    if path is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('x', encoding='utf-8') as f:
        json.dump(evidence, f, ensure_ascii=False, sort_keys=True, indent=2)
        f.write('\n')


def load_verified_world(client, town, expected_sha256, record_id=None,
                        evidence_path=None, timeout_s=120.0, interval_s=0.5,
                        clock=None, sleep=None):
    """Return (current_world, current_map, verified_sha256, transition_evidence).

    The episode ID MUST change after load_world even if map name matches.
    Require two consecutive reads of the same NEW episode/map/XODR hash.
    NEVER retry load_world: readiness polling is not a scene/pose retry.
    """
    clock = clock or time.monotonic
    sleep = sleep or time.sleep
    if not (0 < float(timeout_s) <= 600 and 0 < float(interval_s) <= float(timeout_s)):
        raise ValueError('invalid world transition timeout/poll interval')
    if not isinstance(expected_sha256, str) or len(expected_sha256) != 64 or any(c not in '0123456789abcdef' for c in expected_sha256.lower()):
        raise MapTransitionFailure('trusted OpenDRIVE sha256 required before world load')
    expected_sha256 = expected_sha256.lower()
    name = _town(town)
    prior = client.get_world()
    before_id = _episode(prior)
    before_town = _town(prior.get_map().name)
    evidence = {
        'schema_version': 'p1_world_episode_fence_v1',
        'record_id': record_id, 'requested_town': name,
        'expected_sha256': expected_sha256, 'before_episode_id': before_id,
        'before_town': before_town, 'load_calls': 0, 'observations': [],
        'status': 'in_progress'
    }
    started = clock()
    try:
        client.load_world(str(town))
        evidence['load_calls'] = 1
        deadline = started + timeout_s
        good_episode = None
        consecutive_good = 0
        while True:
            if clock() >= deadline:
                raise MapTransitionFailure('new episode / correct Town / OpenDRIVE SHA not ready before timeout')
            try:
                w = client.get_world()
                ep = _episode(w)
                cm = w.get_map()
                observed_name = _town(cm.name)
                current = {'episode_id': ep, 'town': observed_name}
                if observed_name == name and ep != before_id:
                    sha = hashlib.sha256(cm.to_opendrive().encode('utf-8')).hexdigest()
                    current['opendrive_sha256'] = sha
                    if sha != expected_sha256:
                        raise MapTransitionFailure('Town matches but OpenDRIVE SHA mismatch: '+sha)
                    if good_episode == ep:
                        consecutive_good += 1
                    else:
                        good_episode = ep
                        consecutive_good = 1
                    if consecutive_good >= 2:
                        evidence['observations'].append(current)
                        evidence['status'] = 'verified'
                        evidence['verified_episode_id'] = ep
                        evidence['verified_town'] = observed_name
                        evidence['verified_sha256'] = sha
                        evidence['elapsed_seconds'] = round(clock()-started, 3)
                        _save(evidence_path, evidence)
                        return w, cm, sha, evidence
                else:
                    good_episode = None
                    consecutive_good = 0
            except MapTransitionFailure:
                raise
            except Exception as exc:
                current = {'query_exception_type': type(exc).__name__,
                           'query_exception': str(exc)[:240]}
                good_episode = None
                consecutive_good = 0
            # Cap evidence and report aggregate rather than hanging or storing an unlimited stream.
            if len(evidence['observations']) < 100:
                evidence['observations'].append(current)
            else:
                evidence['omitted_observation_count'] = evidence.get('omitted_observation_count', 0) + 1
            sleep(min(interval_s, max(0, deadline-clock())))
    except Exception as exc:
        evidence['status'] = 'fail_closed'
        evidence['failure_type'] = type(exc).__name__
        evidence['failure_reason'] = str(exc)
        evidence['elapsed_seconds'] = round(clock()-started, 3)
        _save(evidence_path, evidence)
        if isinstance(exc, MapTransitionFailure):
            raise
        raise MapTransitionFailure(str(exc)) from exc


def assert_same_episode_map(client, episode_id, town, sha256):
    """Fail if an external client changed/reloaded the world since the fence."""
    w = client.get_world()
    ep = _episode(w)
    cm = w.get_map()
    name = _town(cm.name)
    if ep != int(episode_id) or name != _town(town):
        raise MapTransitionFailure('world changed during record: episode={} expected={} town={} expected_town={}'.format(
            ep, episode_id, name, _town(town)))
    actual_sha = hashlib.sha256(cm.to_opendrive().encode('utf-8')).hexdigest()
    if actual_sha != sha256:
        raise MapTransitionFailure('world OpenDRIVE SHA changed during record: {}'.format(actual_sha))
    return True
