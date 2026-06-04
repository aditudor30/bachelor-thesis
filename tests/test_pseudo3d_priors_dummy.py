from deep_oc_sort_3d.audit3d.audit3d_io import write_json
from deep_oc_sort_3d.pseudo3d.pseudo3d_priors import load_pseudo3d_priors


def test_pseudo3d_priors_load_and_lookup(tmp_path):
    path = tmp_path / "priors.json"
    write_json(
        {
            "classes": [
                {
                    "class_id": 0,
                    "class_name": "Person",
                    "robust_width": 0.7,
                    "robust_length": 0.8,
                    "robust_height": 1.7,
                    "confidence_level": "high",
                    "selected_prior_source": "step15b",
                }
            ]
        },
        path,
    )

    table = load_pseudo3d_priors(path)

    assert table.get(0).height == 1.7
    assert table.get(99) is None

