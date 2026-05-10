"""Collect rollout logs and transition datasets from the glimpse environment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import numpy as np

from sm_jepa_aif import MNISTGlimpseEnv, RolloutLogger, TransitionDataset
from sm_jepa_aif.policies import RandomPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=8, help="number of episodes to collect")
    parser.add_argument("--steps", type=int, default=12, help="maximum number of actions per episode")
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
    parser.add_argument(
        "--log-json",
        type=Path,
        default=Path("artifacts/rollouts/rollouts.json"),
        help="output path for the rollout log",
    )
    parser.add_argument(
        "--transitions-npz",
        type=Path,
        default=Path("artifacts/rollouts/transitions.npz"),
        help="output path for the transition arrays",
    )
    return parser.parse_args()


def build_env(args: argparse.Namespace) -> tuple[MNISTGlimpseEnv, str]:
    if args.synthetic:
        return (
            MNISTGlimpseEnv.from_synthetic(
                glimpse_size=args.glimpse_size,
                max_steps=args.steps,
                seed=args.seed,
            ),
            "synthetic",
        )

    try:
        env = MNISTGlimpseEnv.from_torchvision(
            root=args.data_root,
            train=True,
            glimpse_size=args.glimpse_size,
            max_steps=args.steps,
            start_strategy=args.start_strategy,
            seed=args.seed,
            download=args.download,
        )
        return env, "mnist"
    except RuntimeError as exc:
        raise SystemExit(
            "Failed to load MNIST. Either place the dataset under the data root, "
            "re-run with --download, or use --synthetic for an offline smoke test.\n"
            f"Original error: {exc}"
        ) from exc


def main() -> None:
    args = parse_args()
    env, dataset_name = build_env(args)
    policy = RandomPolicy(seed=args.seed)
    logger = RolloutLogger()
    logger.collect_episodes(env=env, policy=policy, num_episodes=args.episodes, max_steps=args.steps)

    dataset = TransitionDataset.from_rollout_logger(logger)
    arrays = dataset.as_arrays()

    args.log_json.parent.mkdir(parents=True, exist_ok=True)
    logger.save_json(args.log_json)

    args.transitions_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.transitions_npz, **arrays)

    print(f"dataset={dataset_name}")
    print(json.dumps(logger.summary(), indent=2))
    print(json.dumps(dataset.summary(), indent=2))
    print(f"saved rollout log to {args.log_json}")
    print(f"saved transitions to {args.transitions_npz}")


if __name__ == "__main__":
    main()
