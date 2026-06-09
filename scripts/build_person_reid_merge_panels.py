"""Build visual panels for one fine-tuned Person ReID association variant."""

import argparse
from pathlib import Path
from typing import Any, Dict

from deep_oc_sort_3d.reid_visual_decision.merge_event_loader import load_and_write_merge_events
from deep_oc_sort_3d.reid_visual_decision.merge_event_selector import select_events_for_review
from deep_oc_sort_3d.reid_visual_decision.merge_panel_builder import build_merge_panels
from deep_oc_sort_3d.reid_visual_decision.visual_decision_config import (
    load_visual_decision_config,
    max_events_for_variant,
    output_root_from_config,
    prepare_visual_output,
)
from deep_oc_sort_3d.reid_visual_decision.visual_decision_io import write_csv_dicts


def main() -> None:
    args = parse_args()
    config = load_visual_decision_config(args.config)
    if args.dataset_root is not None:
        config.setdefault("paths", {})["dataset_root"] = str(args.dataset_root)
    if args.source_root is not None:
        config.setdefault("paths", {})["finetuned_association_root"] = str(args.source_root)
    if args.output_root is not None:
        config.setdefault("person_reid_visual_decision", {})["output_root"] = str(args.output_root)
    output_root = prepare_visual_output(config, overwrite=False)
    loaded = load_and_write_merge_events(config, variants=[args.variant])
    events = [row for row in loaded["events"] if str(row.get("variant")) == args.variant]
    max_events = args.max_events if args.max_events is not None else max_events_for_variant(config, args.variant)
    selected = select_events_for_review(events, max_events=max_events, threshold=float(config.get("heuristics", {}).get("reid_similarity_threshold", 0.80)))
    rows = build_merge_panels(selected, config, output_root, args.variant, progress=args.progress)
    write_csv_dicts(rows, output_root / "merge_audit" / ("%s_visual_review_events.csv" % args.variant))
    print("variant: %s" % args.variant)
    print("panels: %d" % len(rows))
    print("output_root: %s" % output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build visual panels for one ReID merge variant.")
    parser.add_argument("--config", type=Path, default=Path("deep_oc_sort_3d/configs/person_reid_visual_decision.yaml"))
    parser.add_argument("--variant", default="combined_safe_080")
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    main()

