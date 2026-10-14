#!/usr/bin/env python3
"""Verify all files from a supplied patch manifest; read only."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True)
    a=p.parse_args();root=Path(__file__).resolve().parent
    report=json.loads(a.manifest.read_text(encoding='utf-8'));errors=[]
    for name,expected in report['files'].items():
        path=root/name
        if not path.is_file():errors.append('Missing: '+name)
        elif hashlib.sha256(path.read_bytes()).hexdigest()!=expected:errors.append('Different: '+name)
    print(json.dumps({'checked_files':len(report['files']),'passed':not errors,'errors':errors},indent=2))
    if errors:raise SystemExit('Patch files incomplete/different; extract the intended archive into ROOT')


if __name__=='__main__':main()
