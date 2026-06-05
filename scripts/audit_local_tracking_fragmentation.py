"""Run only local tracking fragmentation audit."""

from deep_oc_sort_3d.scripts.run_fragmentation_audit import run_stage_cli


if __name__ == "__main__":
    run_stage_cli("local_tracking")

