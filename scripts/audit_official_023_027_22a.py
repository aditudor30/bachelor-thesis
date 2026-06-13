"""Audit official test scenes and class mappings for Step 22A."""

import argparse
from pathlib import Path

from deep_oc_sort_3d.official_023_027.class_mapping_audit import audit_class_mapping
from deep_oc_sort_3d.official_023_027.official_config import load_official_config
from deep_oc_sort_3d.official_023_027.test_scene_audit import audit_test_scenes


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit official scenes 023-027 and class mapping")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--progress", dest="progress", action="store_true")
    group.add_argument("--no-progress", dest="progress", action="store_false")
    parser.set_defaults(progress=True)
    args = parser.parse_args()
    config = load_official_config(args.config)
    scene_audit = audit_test_scenes(config)
    mapping_audit = audit_class_mapping(config)
    print("test_scene_audit: %s" % scene_audit.get("status"))
    print("class_mapping_audit: %s" % mapping_audit.get("status"))
    print("missing_scenes: %s" % scene_audit.get("missing_scenes"))
    if scene_audit.get("status") != "ok" or mapping_audit.get("status") != "ok":
        raise RuntimeError("Official 023-027 audit failed; inspect output/official_023_027/audit")


if __name__ == "__main__":
    main()
