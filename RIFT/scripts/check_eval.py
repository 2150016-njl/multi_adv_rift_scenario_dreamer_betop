#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@File    : check_eval.py
@Date    : 2025/03/29
'''

import argparse
import json
import ast
import sys
from pathlib import Path

def parse_value(v):
    """
    Parse a value from string to appropriate data type (int, float, or original string).
    """
    if not isinstance(v, str):
        return v
    try:
        return ast.literal_eval(v)
    except (SyntaxError, ValueError):
        pass
    try:
        if '.' in v:
            return float(v)
        else:
            return int(v)
    except ValueError:
        pass
    return v

def load_labels_values_as_dict(json_path: str):
    """
    Load JSON file and parse labels and values into a dictionary.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    entry_status = data.get("entry_status", None)
    return entry_status


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_dir', type=str, default='log/eval')
    parser.add_argument('--ego_name', type=str, default='pdm_lite.yaml')
    parser.add_argument('--cbv_name', type=str, default='grpo_pluto.yaml')
    parser.add_argument('--cbv_recog', type=str, default='rule')
    parser.add_argument('--seed', type=int, default=0)

    args = parser.parse_args()
    
    base_dir = Path(args.base_dir)

    ego_name = args.ego_name.split('.')[0]
    cbv_name = args.cbv_name.split('.')[0]
    foler_name = f'{ego_name}-{cbv_name}-{args.cbv_recog}-seed{args.seed}'

    eval_dir = base_dir / foler_name / "simulation_results.json"
    if eval_dir.exists():
        entry_status = load_labels_values_as_dict(eval_dir)
        if entry_status is not None and entry_status == "Finished":
            sys.exit(0)

    sys.exit(1)