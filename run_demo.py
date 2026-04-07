from __future__ import annotations

import argparse
import json
from pathlib import Path

from toy_pgmorl.core import ToyConfig, run_experiment


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the PG-MORL toy experiment headlessly.")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--population-size", type=int, default=7)
    parser.add_argument("--warmup-steps", type=int, default=25)
    parser.add_argument("--task-steps", type=int, default=8)
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=0.035)
    parser.add_argument("--policy-std", type=float, default=0.6)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--weight-resolution", type=int, default=14)
    parser.add_argument("--gradient-mode", choices=["analytic", "reinforce"], default="analytic")
    parser.add_argument("--summary-json", type=Path)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ToyConfig(
        seed=args.seed,
        population_size=args.population_size,
        warmup_steps=args.warmup_steps,
        task_steps=args.task_steps,
        generations=args.generations,
        learning_rate=args.learning_rate,
        policy_std=args.policy_std,
        batch_size=args.batch_size,
        weight_resolution=args.weight_resolution,
        gradient_mode=args.gradient_mode,
    )
    result = run_experiment(config)
    print(result.method_summary.to_string(index=False))

    if args.summary_json:
        payload = {
            "config": config.to_dict(),
            "summary": result.method_summary.to_dict(orient="records"),
        }
        args.summary_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        print(f"\nSaved summary to {args.summary_json}")


if __name__ == "__main__":
    main()
