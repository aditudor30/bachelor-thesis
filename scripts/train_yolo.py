"""Manual Ultralytics YOLO training wrapper."""

import argparse
from typing import Any


def train_yolo(args: Any) -> None:
    """Run YOLO training only when this script is invoked manually."""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Ultralytics is not installed. Install it with: pip install ultralytics")
        return

    model = YOLO(args.model)
    train_kwargs = {
        "data": args.data,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "workers": args.workers,
        "project": args.project,
        "name": args.name,
    }
    if args.resume:
        train_kwargs["resume"] = True
    if args.patience is not None:
        train_kwargs["patience"] = args.patience
    model.train(**train_kwargs)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train YOLO with Ultralytics.")
    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--batch", type=int, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--project", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--patience", type=int, default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    train_yolo(args)


if __name__ == "__main__":
    main()

