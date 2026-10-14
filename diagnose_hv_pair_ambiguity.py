#!/usr/bin/env python3
"""Compare unresolved source candidates using the exact pairing fingerprint inputs."""
import argparse
import hashlib
import json
from pathlib import Path
from build_hv_snapshot_audit_manifest import replay_inputs


def differences(a,b,path='',limit=40):
    result=[]
    def visit(x,y,p):
        if len(result)>=limit or x==y:return
        if isinstance(x,dict) and isinstance(y,dict):
            for k in sorted(set(x)|set(y)):
                visit(x.get(k,'<MISSING>'),y.get(k,'<MISSING>'),p+'.'+str(k))
        elif isinstance(x,list) and isinstance(y,list) and len(x)==len(y):
            for i,(u,v) in enumerate(zip(x,y)):visit(u,v,p+'[%d]'%i)
        else:
            row={'path':p,'a':x,'b':y}
            if isinstance(x,(int,float)) and isinstance(y,(int,float)):row['absolute_delta']=abs(x-y)
            result.append(row)
    visit(a,b,path)
    return result


def saved_result(output,report_path,report_bytes):
    if not output.exists():return None
    result=json.loads(output.read_text(encoding='utf-8'))
    sha=result.get('source_pair_report_sha256')
    if sha is not None:
        if sha!=hashlib.sha256(report_bytes).hexdigest():
            raise FileExistsError('Source report changed; preserve this result and choose --output-json with a new filename')
    elif Path(result['source_pair_report']).resolve()!=report_path.resolve():
        raise FileExistsError('Legacy cached result belongs to another report path; choose a new --output-json')
    return result


def show_result(result,output,cached=False):
    print('Using saved differences:' if cached else 'Source differences:',output)
    for index,row in result['records'].items():
        print(index,[c['different_components'] for c in row['comparisons']],row['unavailable_sources'])
        seen=set()
        for comparison in row['comparisons']:
            for diff in comparison['first_differences']:
                if diff['path'] in seen or len(seen)>=8:continue
                seen.add(diff['path'])
                value=json.dumps(diff,ensure_ascii=False)
                print(' ',value if len(value)<=500 else value[:500]+' ... (full values in JSON)')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--pair-report',type=Path,required=True)
    p.add_argument('--output-json',type=Path)
    a=p.parse_args();report_bytes=a.pair_report.read_bytes();r=json.loads(report_bytes)
    output=a.output_json or Path(str(a.pair_report)+'.differences.json')
    previous=saved_result(output,a.pair_report,report_bytes)
    if previous is not None:
        show_result(previous,output,cached=True)
        return
    cache={};records={}
    for index,candidates in r.get('ambiguity_details',{}).items():
        loaded=[];errors=[]
        for candidate in candidates:
            sha=candidate['snapshot_sha256']
            if sha not in cache:
                debug=(r.get('debug_snapshot_files') or {}).get(sha)
                paths=([a.pair_report.parent/Path(debug).parent.name/Path(debug).name,Path(debug)] if debug else [])
                paths.append(Path(candidate['snapshot_path']))
                source=next((v for v in paths if v.is_file()),None)
                if source is None:
                    errors.append('Source not in bundle: '+sha);continue
                data=source.read_bytes()
                if hashlib.sha256(data).hexdigest()!=sha:raise ValueError('Source SHA differs: %s' % source)
                payload=json.loads(data);cache[sha]=replay_inputs(payload.get('scene',payload))
            loaded.append((sha,cache[sha]))
        comparisons=[]
        if loaded:
            first,base=loaded[0]
            for sha,value in loaded[1:]:
                comparisons.append({'a_sha256':first,'b_sha256':sha,
                    'different_components':[k for k in base if base[k]!=value.get(k)],
                    'first_differences':differences(base,value)})
        records[index]={'comparisons':comparisons,'unavailable_sources':errors}
    result={'source_pair_report':str(a.pair_report),'records':records,
            'source_pair_report_sha256':hashlib.sha256(report_bytes).hexdigest(),
            'historical_source_resolved':False,'tolerances_relaxed':False}
    output.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf-8')
    show_result(result,output)


if __name__=='__main__':main()
