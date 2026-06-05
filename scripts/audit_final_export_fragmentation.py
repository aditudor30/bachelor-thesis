"""Run only final export fragmentation audit."""

from deep_oc_sort_3d.scripts.run_fragmentation_audit import run_stage_cli


if __name__ == "__main__":
    run_stage_cli("final_export")

