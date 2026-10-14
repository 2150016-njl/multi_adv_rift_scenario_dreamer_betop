"""Frozen three-file protocol. Unknown supervision is never a k=0 label."""
import math
from .coordinates import LOCAL_FRAME,WORLD_FRAME,in_domain

PROTOCOL='hv_v7_three_file_v1'
FORBIDDEN={'rift_labels','is_interactive','saved_positive','saved_score','native',
           'interaction_meta','interaction_waypoint','target_route_to_interaction',
           'generated_agent_states','reference_target_states','target_states','target_actor_ids',
           'oracles','targets','candidate_debug','training_weight'}


def finite_json(value):
    if isinstance(value,float) and not math.isfinite(value):raise ValueError('Nonfinite protocol number')
    if isinstance(value,dict):
        for v in value.values():finite_json(v)
    elif isinstance(value,list):
        for v in value:finite_json(v)


def validate_condition(record):
    finite_json(record)
    if record.get('schema_version')!='hv_v7_condition_v1':raise ValueError('Wrong condition schema')
    if record['condition']['coordinate_frame']!=LOCAL_FRAME:raise ValueError('Wrong local coordinate frame')
    if record['condition']['world_frame']!=WORLD_FRAME:raise ValueError('Wrong world frame')
    def walk(obj):
        if isinstance(obj,dict):
            for key,value in obj.items():
                if key in FORBIDDEN:raise ValueError('Label leaked into condition: '+key)
                walk(value)
        elif isinstance(obj,list):
            for value in obj:walk(value)
    walk(record)
    tokens=record['condition']['observed_actor_tokens']
    ids=[t['actor_token'] for t in tokens]
    if len(ids)!=len(set(ids)):raise ValueError('Duplicate actor tokens')
    if record['event']['mode'] not in ('initialization','rolling','historical_unknown'):
        raise ValueError('Unknown generation event mode')


def validate_target(record):
    finite_json(record)
    if record.get('schema_version')!='hv_v7_target_v1':raise ValueError('Wrong target schema')
    k=record['num_targets'];mask=record['valid_mask'];states=record['target_states']
    if k is None:
        if mask is not None or states is not None:raise ValueError('Unknown target must not imply zero slots')
        return
    if type(k) is not int or k not in (0,1,2) or not isinstance(mask,list) or len(mask)!=2 or len(states)!=2:
        raise ValueError('Invalid slot count')
    if any(type(m) is not bool for m in mask) or mask!=[i<k for i in range(2)]:
        raise ValueError('Valid slots must be packed')
    if k==0 and record.get('zero_reason') not in ('capacity_exhausted','no_valid_future_route_verified'):
        raise ValueError('Unproven zero-target label')
    for i,(valid,state) in enumerate(zip(mask,states)):
        if valid and (not isinstance(state,list) or len(state)!=4 or not in_domain(state,front=i==0)):
            raise ValueError('Slot violates output domain')
        if valid and (any(type(v) not in (int,float) for v in state) or state[3]<0):
            raise ValueError('State needs finite x/y/yaw and nonnegative speed')
        if not valid and state is not None:raise ValueError('Inactive slot has state')


def validate_triplets(conditions,targets,oracles):
    for records in (conditions,targets,oracles):
        ids=[r['record_id'] for r in records]
        if len(ids)!=len(set(ids)):raise ValueError('Duplicate record IDs')
    ids=[r['record_id'] for r in conditions]
    if any([r['record_id'] for r in records]!=ids for records in (targets,oracles)):
        raise ValueError('Three-file record order/identity mismatch')
    for c,t,o in zip(conditions,targets,oracles):
        validate_condition(c);validate_target(t);finite_json(o)
        if o.get('schema_version')!='hv_v7_oracle_v1':raise ValueError('Wrong oracle schema')
        visible={x['actor_token'] for x in c['condition']['observed_actor_tokens']}
        if visible & {str(a['actor_id']) for a in t['historical_targets']}:
            raise ValueError('Withheld target remains visible')
