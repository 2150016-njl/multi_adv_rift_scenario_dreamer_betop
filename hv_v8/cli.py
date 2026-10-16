from __future__ import annotations

import argparse
import sys

from .p0_freeze import main as p0_main
from .p1_native_audit import main as p1_main
from .p2_support import main as p2_main
from .visualize import main as viz_main


def main() -> None:
    # Keep each stage as its own process contract. This dispatcher is only a
    # convenience; the stage modules remain directly executable.
    p = argparse.ArgumentParser(prog="hv_v8")
    sub = p.add_subparsers(dest="stage", required=True)
    sub.add_parser("p0")
    sub.add_parser("p1")
    sub.add_parser("p2")
    sub.add_parser("viz")
    args, rest = p.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    if args.stage == "p0":
        p0_main()
        return
    if args.stage == "p1":
        p1_main()
    elif args.stage == "p2":
        p2_main()
    else:
        viz_main()


if __name__ == "__main__":
    main()
