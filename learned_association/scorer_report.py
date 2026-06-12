"""Markdown report generation for Step 20B."""

from pathlib import Path
from typing import Any, Dict


def write_scorer_report(
    output_root: Path, evaluation: Dict[str, Any], config: Dict[str, Any]
) -> None:
    """Write the requested scorer report and README."""
    selection = evaluation.get("selection", {})
    verdict = evaluation.get("verdict", {})
    data = evaluation.get("data_summary", {})
    model_lines = []
    for name, result in evaluation.get("model_evaluations", {}).items():
        metrics = result.get("overall_metrics", {})
        strict = result.get("selected_thresholds", {}).get("strict", {})
        hard = result.get("hard_negative_metrics", {})
        model_lines.append(
            "- `{}`: PR-AUC={}, ROC-AUC={}, strict threshold={}, precision={}, recall={}, FPR={}, hard-negative FPR={}".format(
                name,
                _fmt(metrics.get("pr_auc")),
                _fmt(metrics.get("roc_auc")),
                _fmt(strict.get("threshold")),
                _fmt(strict.get("precision")),
                _fmt(strict.get("recall")),
                _fmt(strict.get("false_positive_rate")),
                _fmt(hard.get("hard_negative_false_positive_rate")),
            )
        )
    text = """# Person Association Scorer Report

## Context

Step 20B trains pairwise Person association scorers from the supervised fragment-pair
dataset produced by Step 20A. This experiment does not modify V1/V2 outputs and does not
run association on test.

## Data

- Train pairs: {train_pairs}
- Validation pairs: {val_pairs}
- Train scenes: {train_scenes}
- Validation scenes: {val_scenes}
- Input dimension after preprocessing: {input_dim}

Warning: train coverage is limited to the available V2 fragments, currently
`Warehouse_014`, `Warehouse_015`, and `Warehouse_016`. The selected scorer is therefore
experimental and must be used conservatively.

## Model Comparison

{model_lines}

## Conservative Selection

- Selected model: `{selected_model}`
- Selection reason: `{selection_reason}`
- Recommended thresholds: `{thresholds}`

## Verdict

`{verdict}`

Reasons: {reasons}

## Recommendation for Step 20C

`ready_for_step_20c = {ready}`. Step 20C should begin as a validation-only conservative
ablation. Apply only the selected strict or very-strict threshold, preserve geometry and
conflict gates, and compare against the ReID-only baseline before any test export.
""".format(
        train_pairs=data.get("num_train_pairs", "unknown"),
        val_pairs=data.get("num_val_pairs", "unknown"),
        train_scenes=", ".join(data.get("scene_split", {}).get("train_scenes", [])),
        val_scenes=", ".join(data.get("scene_split", {}).get("val_scenes", [])),
        input_dim=data.get("input_dim", "unknown"),
        model_lines="\n".join(model_lines) or "No models evaluated.",
        selected_model=selection.get("selected_model"),
        selection_reason=selection.get("reason"),
        thresholds=selection.get("selected_thresholds"),
        verdict=verdict.get("verdict"),
        reasons=", ".join(verdict.get("reasons", [])) or "none",
        ready=str(bool(verdict.get("ready_for_step_20c"))).lower(),
    )
    reports_dir = output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "PERSON_ASSOCIATION_SCORER_REPORT.md").write_text(text, encoding="utf-8")
    readme = """# Person Association Scorer V1

This directory contains Step 20B preprocessing, model checkpoints, validation metrics,
conservative threshold selection, figures, and reports. It contains no test association
results and does not replace the existing V1/V2 pipeline.
"""
    (reports_dir / "README_PERSON_ASSOCIATION_SCORER.md").write_text(readme, encoding="utf-8")


def _fmt(value: Any) -> str:
    if value is None:
        return "not_available"
    if isinstance(value, float):
        return "%.6f" % value
    return str(value)
