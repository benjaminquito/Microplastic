from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .data import validate_dataset
from .demo import make_demo_dataset
from .evaluation import evaluate
from .prediction import predict_image
from .training import train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="microscan", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    demo = commands.add_parser("make-demo", help="create deterministic synthetic test data")
    demo.add_argument("--output", type=Path, default=Path("data/demo"))
    demo.add_argument("--seed", type=int, default=20260906)
    demo.add_argument("--force", action="store_true")

    validate = commands.add_parser("validate-data", help="validate dataset structure and leakage")
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--output", type=Path)

    train_command = commands.add_parser("train", help="train the configured model")
    train_command.add_argument("--config", type=Path, required=True)

    evaluate_command = commands.add_parser("evaluate", help="evaluate once on the test split")
    evaluate_command.add_argument("--config", type=Path, required=True)
    evaluate_command.add_argument("--checkpoint", type=Path, required=True)

    predict = commands.add_parser("predict", help="predict one image")
    predict.add_argument("--checkpoint", type=Path, required=True)
    predict.add_argument("--image", type=Path, required=True)
    predict.add_argument("--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu")
    predict.add_argument("--top-k", type=int, default=3)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "make-demo":
        result: object = {"created": str(make_demo_dataset(args.output, args.seed, args.force))}
    elif args.command == "validate-data":
        report = validate_dataset(load_config(args.config).data)
        result = report.to_dict()
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if not report.valid:
            print(json.dumps(result, indent=2, sort_keys=True))
            raise SystemExit(1)
    elif args.command == "train":
        result = train(load_config(args.config))
    elif args.command == "evaluate":
        result = evaluate(load_config(args.config), args.checkpoint)
    else:
        result = predict_image(args.checkpoint, args.image, args.device, args.top_k)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
