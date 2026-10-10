"""Build V4C records with target-only RIFT route-interaction supervision.

The implementation deliberately delegates to the V4B builder.  V4C changes
neither the Scenario Dreamer graph condition nor V4B's target selection and
split behavior; it merely enables the opt-in target metadata consumed by the
separate V4C trainer.
"""

import sys

import build_high_value_scene_dataset_v4b as v4b


def main():
    if "--include_route_interaction_supervision" not in sys.argv:
        sys.argv.append("--include_route_interaction_supervision")
    if "--require_route_progress_supervision" not in sys.argv:
        sys.argv.append("--require_route_progress_supervision")
    v4b.main()


if __name__ == "__main__":
    main()
