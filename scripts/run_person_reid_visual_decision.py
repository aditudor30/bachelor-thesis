"""Run Step 18D visual decision for fine-tuned Person ReID merges."""

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from deep_oc_sort_3d.reid_visual_decision.manual_review_writer import write_manual_review_sheet, write_review_instructions
from deep_oc_sort_3d.reid_visual_decision.merge_event_loader import load_and_write_merge_events
from deep_oc_sort_3d.reid_visual_decision.merge_event_selector import select_events_for_review
from deep_oc_sort_3d.reid_visual_decision.merge_panel_builder import build_merge_panels
from deep_oc_sort_3d.reid_visual_decision.visual_decision_config import (
    dataset_root_from_config,
    load_visual_decision_config,
    max_events_for_variant,
    output_root_from_config,
    prepare_visual_output,
    save_visual_config,
    variants_from_config,
)
from deep_oc_sort_3d.reid_visual_decision.visual_decision_figures import create_visual_decision_figures
from deep_oc_sort_3d.reid_visual_decision.visual_decision_io import read_json, write_csv_dicts, write_json
from deep_oc_sort_3d.reid_visual_decision.visual_decision_metrics import decide_final_variant, summarize_visual_rows
from deep_oc_sort_3d.reid_visual_decision.visual_decision_report import write_visual_decision_report


def main() -> None:
    args = parse_args()
    config = load_visual_decision_config(args.config)
    if args.dataset_root is not None:
        config.setdefault("paths", {})["dataset_root"] = str(args.dataset_root)
    if args.output_root is not None:
        config.setdefault("person_reid_visual_decision", {})["output_root"] = str(args.output_root)
    if args.source_root is not None:
        config.setdefault("paths", {})["finetuned_association_root"] = str(args.source_root)
    output_root = prepare_visual_output(config, overwrite=args.overwrite)
    save_visual_config(config, args.config, output_root)
    variants = args.variants if args.variants else variants_from_config(config)
    loaded = load_and_write_merge_events(config, variants=variants)
    events = loaded["events"]
    rows_by_variant: Dict[str, List[Dict[str, Any]]] = {}
    for variant in variants:
        variant_events = [row for row in events if str(row.get("variant")) == variant]
        selected = select_events_for_review(
            variant_events,
            max_events=max_events_for_variant(config, variant),
            threshold=float(config.get("heuristics", {}).get("reid_similarity_threshold", 0.80)),
        )
        if not args.no_panels:
            reviewed = build_merge_panels(selected, config, output_root, variant, progress=args.progress)
        else:
            reviewed = selected
            for row in reviewed:
                row.setdefault("auto_label", "ambiguous")
                row.setdefault("risk_score", "")
                row.setdefault("risk_reasons", "panels_disabled")
        rows_by_variant[variant] = reviewed
        write_csv_dicts(reviewed, output_root / "merge_audit" / ("%s_visual_review_events.csv" % variant))
        write_manual_review_sheet(reviewed, output_root, variant)
    all_review_rows = []
    for rows in rows_by_variant.values():
        all_review_rows.extend(rows)
    write_review_instructions(output_root)
    summary = summarize_visual_rows(all_review_rows)
    selected_variant = read_json(Path(str(config.get("paths", {}).get("finetuned_association_root", ""))) / "comparison" / "selected_variant.json") or {}
    final_decision = decide_final_variant(summary, selected_variant)
    figure_summary = create_visual_decision_figures(all_review_rows, output_root)
    review_paths = [output_root / "manual_review" / ("review_sheet_%s.csv" % variant) for variant in variants]
    report = write_visual_decision_report(output_root, summary, final_decision, review_paths, figure_summary)
    write_json({"summary": summary, "final_decision": final_decision, "report": report}, output_root / "comparison" / "visual_decision_status.json")
    print("visual_review_events: %d" % len(all_review_rows))
    print("final_verdict: %s" % final_decision.get("final_verdict"))
    print("output_root: %s" % output_root)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build visual review panels for fine-tuned Person ReID merges.")
    parser.add_argument("--config", type=Path, default=Path("deep_oc_sort_3d/configs/person_reid_visual_decision.yaml"))
    parser.add_argument("--dataset-root", type=Path, default=None)
    parser.add_argument("--source-root", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--variants", nargs="+", default=None)
    parser.add_argument("--no-panels", action="store_true", default=False)
    parser.add_argument("--overwrite", action="store_true", default=False)
    parser.add_argument("--progress", dest="progress", action="store_true", default=True)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    return parser.parse_args()


if __name__ == "__main__":
    main()

