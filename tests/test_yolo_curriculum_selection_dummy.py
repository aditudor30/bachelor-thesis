import csv

from deep_oc_sort_3d.detection2d.yolo_curriculum_selection import select_curriculum_frames


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _audit_row(scene, camera, frame, class_name, class_id, difficulty, area_norm):
    return {
        "split": "train",
        "scene_name": scene,
        "scene_id": int(scene.split("_")[-1]),
        "camera_id": camera,
        "frame_id": frame,
        "object_id": frame + class_id + 1,
        "class_name": class_name,
        "class_id": class_id,
        "x1": 10,
        "y1": 10,
        "x2": 60,
        "y2": 60,
        "width_px": 50,
        "height_px": 50,
        "area_px": 2500,
        "aspect_ratio": 1.0,
        "image_width": 100,
        "image_height": 100,
        "width_norm": 0.5,
        "height_norm": 0.5,
        "area_norm": area_norm,
        "center_x_norm": 0.35,
        "center_y_norm": 0.35,
        "difficulty": difficulty,
    }


def test_curriculum_selection_excludes_holdout_and_limits_person_only(tmp_path):
    audit_csv = tmp_path / "audit.csv"
    class_rich_csv = tmp_path / "class_rich.csv"
    _write_csv(
        audit_csv,
        [
            _audit_row("Warehouse_006", "Camera_0011", 0, "PalletTruck", 2, "easy", 0.02),
            _audit_row("Warehouse_000", "Camera_0000", 1, "Person", 0, "easy", 0.02),
            _audit_row("Warehouse_002", "Camera_0002", 2, "Person", 0, "easy", 0.02),
            _audit_row("Warehouse_014", "Camera_0008", 3, "FourierGR1T2", 4, "easy", 0.02),
        ],
    )
    _write_csv(
        class_rich_csv,
        [
            {
                "split": "train",
                "scene_name": "Warehouse_006",
                "camera_id": "Camera_0011",
                "frame_id": 0,
                "recommended_for_easy_export": "true",
                "recommended_for_medium_export": "true",
            }
        ],
    )

    selected = select_curriculum_frames(
        audit_csv=audit_csv,
        class_rich_frames_csv=class_rich_csv,
        curriculum="easy_allclass",
        target_classes=["PalletTruck", "Person", "FourierGR1T2"],
        allowed_difficulties=["easy"],
        class_priority={"PalletTruck": 5.0, "Person": 1.0, "FourierGR1T2": 4.0},
        scene_priority={"PalletTruck": ["Warehouse_006"]},
        camera_priority={"PalletTruck": ["Camera_0011"]},
        max_frames_total=None,
        max_frames_per_class=None,
        max_person_only_frames=1,
        min_area_norm=0.004,
        exclude_scenes=["Warehouse_014"],
    )

    assert all(row["scene_name"] != "Warehouse_014" for row in selected)
    assert len([row for row in selected if row["contains_person_only"]]) == 1
    assert selected[0]["scene_name"] == "Warehouse_006"
    assert selected[0]["source"] == "class_rich"


def test_medium_selection_keeps_medium_rare_class(tmp_path):
    audit_csv = tmp_path / "audit.csv"
    _write_csv(
        audit_csv,
        [
            _audit_row("Warehouse_013", "Camera_0001", 0, "Transporter", 3, "medium", 0.003),
            _audit_row("Warehouse_000", "Camera_0000", 1, "Person", 0, "easy", 0.02),
        ],
    )

    selected = select_curriculum_frames(
        audit_csv=audit_csv,
        class_rich_frames_csv=None,
        curriculum="medium_allclass",
        target_classes=["Transporter", "Person"],
        allowed_difficulties=["easy", "medium"],
        class_priority={"Transporter": 4.0, "Person": 1.0},
        scene_priority={"Transporter": ["Warehouse_013"]},
        camera_priority={"Transporter": ["Camera_0001"]},
        max_frames_total=1,
        max_frames_per_class=None,
        max_person_only_frames=10,
        min_area_norm=0.001,
        exclude_scenes=None,
    )

    assert len(selected) == 1
    assert selected[0]["scene_name"] == "Warehouse_013"
    assert selected[0]["contains_rare_class"] is True
