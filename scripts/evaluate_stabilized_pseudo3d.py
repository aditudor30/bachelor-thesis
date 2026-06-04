"""Evaluate stabilized pseudo-3D predictions against GT where available."""

from deep_oc_sort_3d.scripts.evaluate_pseudo3d_predictions import build_arg_parser, run


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
