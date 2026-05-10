"""Run a short rollout in the MNIST active-vision environment."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from sm_jepa_aif.envs import MNISTGlimpseEnv
from sm_jepa_aif.policies import RandomPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=12, help="number of actions to execute")
    parser.add_argument("--glimpse-size", type=int, default=7, help="odd crop size")
    parser.add_argument("--index", type=int, default=None, help="dataset index to visualize")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument("--data-root", type=Path, default=Path("data"), help="dataset directory")
    parser.add_argument("--download", action="store_true", help="download MNIST if not present")
    parser.add_argument(
        "--start-strategy",
        choices=("center", "random"),
        default="center",
        help="initial fixation placement",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/mnist_glimpse_rollout.png"),
        help="where to save the rendered rollout",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="use a synthetic offline dataset instead of MNIST",
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

    observation, info = env.reset(index=args.index)
    print(
        f"dataset={dataset_name} index={info['index']} label={info['label']} "
        f"start_fixation={info['fixation']} glimpse_shape={observation.shape}"
    )

    for _ in range(args.steps):
        action = policy.act()
        result = env.step(action)
        print(
            f"step={result.info['step_count']:02d} action={result.info['last_action_name']:<5} "
            f"fixation={result.info['fixation']}"
        )
        if result.terminated or result.truncated:
            break

    try:
        fig = env.render()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from exc

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches="tight")
    print(f"saved rollout figure to {args.output}")


if __name__ == "__main__":
    main()
