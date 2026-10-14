#!/usr/bin/env python3
"""Summarize a run without treating sparse copies or adjacent frames as IID."""
import argparse
from collections import Counter
import json
from pathlib import Path


def summarize_rows(rows):
    hist=Counter();routes=Counter();towns=Counter();types=Counter()
    for r in rows:
        if 'raw_pool' not in r:continue
        actors=r['raw_pool'];front=[a for a in actors if a['forward_fov']]
        positive=[a for a in front if a['saved_positive'] is True]
        hist[len(positive)]+=1
        routes['%s:%s:%s' % (r.get('town'),r.get('route_id'),r.get('route_index'))]+=1
        towns[str(r.get('town'))]+=1
        types.update(a['type_id'] for a in positive)
    return {'saved_forward_positive_scene_histogram':dict(sorted(hist.items())),
            'unique_routes':len(routes),'scene_counts_by_route':dict(routes),
            'scene_counts_by_town':dict(towns),'saved_forward_positive_types':dict(types),
            'independent_sample_unit':'episode/route, not sparse copy or adjacent frame'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--experiment-dir',type=Path,required=True)
    p.add_argument('--output-json',type=Path)
    a=p.parse_args();report={'experiment_dir':str(a.experiment_dir),'stages':{}}
    for summary in sorted(a.experiment_dir.glob('*/summary.json')):
        stage=summary.parent
        s=json.loads(summary.read_text(encoding='utf-8'))
        rows=[json.loads(line) for line in (stage/'rows.jsonl').read_text(encoding='utf-8').splitlines() if line.strip()]
        report['stages'][stage.name]={'summary':s,'inventory':summarize_rows(rows)}
    for pairing in sorted(a.experiment_dir.glob('*.audit.json')):
        report[pairing.name]=json.loads(pairing.read_text(encoding='utf-8'))
    manifest=a.experiment_dir/'v2_pairs100.jsonl'
    if manifest.exists():
        jobs=[json.loads(x) for x in manifest.read_text(encoding='utf-8').splitlines() if x.strip()]
        report['paired_selection']={'records':len(jobs),
            'base_snapshot_shas':len({j['snapshot_sha256'] for j in jobs}),
            'episodes':dict(Counter(str(j.get('episode_key')) for j in jobs))}
    output=a.output_json or a.experiment_dir/'analysis.json'
    if output.exists():raise FileExistsError('Refuse to overwrite analysis: %s' % output)
    output.write_text(json.dumps(report,indent=2,ensure_ascii=False),encoding='utf-8')
    print('Analysis:',output)
    for name,s in report['stages'].items():
        print(name,json.dumps(s['inventory'],ensure_ascii=False))


if __name__=='__main__':main()
