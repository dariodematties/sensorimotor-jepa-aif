"""Inspect latent quality for a trained sensorimotor JEPA baseline."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sm_jepa_aif.analysis import (
    build_diagnostic_notes,
    evaluate_linear_probe,
    evaluate_representation,
    load_checkpoint_models,
)
from sm_jepa_aif.data import TransitionDataset
from sm_jepa_aif.train import build_transition_dataset, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/first_baseline.pt"),
        help="trained model checkpoint to inspect",
    )
    parser.add_argument("--load-transitions", type=Path, default=None, help="optional path to an existing transitions .npz file")
    parser.add_argument("--episodes", type=int, default=128, help="number of episodes to collect if no dataset is loaded")
    parser.add_argument("--steps", type=int, default=12, help="maximum actions per episode")
    parser.add_argument("--glimpse-size", type=int, default=7, help="odd crop size")
    parser.add_argument("--seed", type=int, default=None, help="random seed")
    parser.add_argument("--synthetic", action="store_true", help="use the synthetic offline dataset")
    parser.add_argument("--download", action="store_true", help="download MNIST if not present")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="dataset directory")
    parser.add_argument(
        "--start-strategy",
        choices=("center", "random"),
        default="random",
        help="initial fixation placement",
    )
    parser.add_argument("--device", type=str, default=None, help="evaluation device, e.g. cpu or cuda")
    parser.add_argument("--batch-size", type=int, default=256, help="evaluation batch size")
    parser.add_argument("--probe-epochs", type=int, default=25, help="linear probe training epochs")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/analysis/latent_diagnostics.json"),
        help="where to save the diagnostics report",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    encoder, predictor, checkpoint = load_checkpoint_models(str(args.checkpoint), device=device)

    dataset_args = argparse.Namespace(
        load_transitions=args.load_transitions,
        episodes=args.episodes,
        steps=args.steps,
        glimpse_size=args.glimpse_size,
        seed=args.seed,
        synthetic=args.synthetic,
        download=args.download,
        data_root=args.data_root,
        start_strategy=args.start_strategy,
        save_transitions=None,
    )
    dataset, dataset_metadata = build_transition_dataset(dataset_args)
    if not isinstance(dataset, TransitionDataset):
        raise SystemExit("Failed to build transition dataset for latent evaluation.")

    representation_metrics = evaluate_representation(
        encoder=encoder,
        predictor=predictor,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
    )
    probe_metrics = evaluate_linear_probe(
        encoder=encoder,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        epochs=args.probe_epochs,
        seed=args.seed,
    )

    report = {
        "checkpoint": str(args.checkpoint),
        "device": str(device),
        "training_dataset_summary": checkpoint.get("dataset_summary"),
        "evaluation_dataset_metadata": dataset_metadata,
        "evaluation_dataset_summary": dataset.summary(),
        "representation_metrics": representation_metrics,
        "linear_probe": probe_metrics,
        "notes": build_diagnostic_notes(dataset_metadata, probe_metrics),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"saved diagnostics to {args.output}")


if __name__ == "__main__":
    main()
