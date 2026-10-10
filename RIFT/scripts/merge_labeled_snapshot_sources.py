#!/usr/bin/env python
"""Create a collision-free, read-only symlink union of labeled snapshots."""
import argparse
import json
import os
from pathlib import Path


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--input_dir", type=Path, action="append", required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    p.add_argument("--summary_json", type=Path, required=True)
    args=p.parse_args()
    if args.output_dir.exists(): raise FileExistsError("refusing to overwrite union: %s" % args.output_dir)
    for root in args.input_dir:
        if not root.is_dir(): raise ValueError("input_dir is not a directory: %s" % root)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    count=0; sources={}
    for source_index, root in enumerate(args.input_dir):
        local=0
        for path in sorted(root.rglob("*.json")):
            if not path.is_file(): continue
            count += 1; local += 1
            link=args.output_dir/("%02d_%09d_%s" % (source_index, local, path.name))
            os.symlink(str(path.resolve()), str(link))
        sources[str(root)]=local
    summary={"schema_version":"labeled_snapshot_symlink_union_v1","output_dir":str(args.output_dir),"linked_json":count,"sources":sources}
    args.summary_json.parent.mkdir(parents=True,exist_ok=True); args.summary_json.write_text(json.dumps(summary,indent=2),encoding="utf-8")
    print(json.dumps(summary,indent=2))


if __name__=="__main__": main()
