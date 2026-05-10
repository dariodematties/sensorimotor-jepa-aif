"""Training entrypoint for the first sensorimotor JEPA baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

from sm_jepa_aif.data import RolloutLogger, TransitionDataset
from sm_jepa_aif.envs import MNISTGlimpseEnv
from sm_jepa_aif.losses import latent_prediction_loss
from sm_jepa_aif.models import ActionConditionedPredictor, GlimpseEncoder
from sm_jepa_aif.policies import RandomPolicy


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=128, help="number of episodes to collect if no dataset is loaded")
    parser.add_argument("--steps", type=int, default=12, help="maximum actions per episode")
    parser.add_argument("--glimpse-size", type=int, default=7, help="odd crop size")
    parser.add_argument("--latent-dim", type=int, default=32, help="latent dimensionality")
    parser.add_argument("--hidden-dim", type=int, default=128, help="hidden layer width")
    parser.add_argument("--action-embed-dim", type=int, default=8, help="action embedding size")
    parser.add_argument("--batch-size", type=int, default=64, help="training batch size")
    parser.add_argument("--epochs", type=int, default=20, help="number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="optimizer learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="optimizer weight decay")
    parser.add_argument("--train-split", type=float, default=0.9, help="fraction of data used for training")
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
    parser.add_argument("--device", type=str, default=None, help="training device, e.g. cpu or cuda")
    parser.add_argument("--load-transitions", type=Path, default=None, help="optional path to an existing transitions .npz file")
    parser.add_argument("--save-transitions", type=Path, default=None, help="optional output path for collected transitions")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("artifacts/checkpoints/first_baseline.pt"),
        help="where to save the trained model checkpoint",
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


def build_transition_dataset(args: argparse.Namespace) -> tuple[TransitionDataset, dict[str, Any]]:
    if args.load_transitions is not None:
        dataset = TransitionDataset.from_npz(args.load_transitions)
        return dataset, {"dataset_source": str(args.load_transitions), "environment": None}

    env, dataset_name = build_env(args)
    policy = RandomPolicy(seed=args.seed)
    logger = RolloutLogger()
    logger.collect_episodes(env=env, policy=policy, num_episodes=args.episodes, max_steps=args.steps)
    dataset = TransitionDataset.from_rollout_logger(logger)

    if args.save_transitions is not None:
        dataset.save_npz(args.save_transitions)

    metadata = {
        "dataset_source": "collected",
        "environment": dataset_name,
        "rollout_summary": logger.summary(),
    }
    return dataset, metadata


def resolve_device(requested_device: str | None) -> torch.device:
    if requested_device is not None:
        return torch.device(requested_device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_epoch(
    encoder: nn.Module,
    predictor: nn.Module,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    optimizer: Adam,
    device: torch.device,
) -> float:
    encoder.train()
    predictor.train()

    total_loss = 0.0
    total_examples = 0

    for batch in dataloader:
        observation = batch["observation"].to(device=device, dtype=torch.float32)
        action = batch["action"].to(device=device, dtype=torch.long)
        next_observation = batch["next_observation"].to(device=device, dtype=torch.float32)

        latent = encoder(observation)
        with torch.no_grad():
            target_latent = encoder(next_observation)
        predicted_latent = predictor(latent, action)
        loss = latent_prediction_loss(predicted_latent=predicted_latent, target_latent=target_latent)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        batch_size = observation.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


@torch.no_grad()
def evaluate_epoch(
    encoder: nn.Module,
    predictor: nn.Module,
    dataloader: DataLoader[dict[str, torch.Tensor]],
    device: torch.device,
) -> float:
    encoder.eval()
    predictor.eval()

    total_loss = 0.0
    total_examples = 0

    for batch in dataloader:
        observation = batch["observation"].to(device=device, dtype=torch.float32)
        action = batch["action"].to(device=device, dtype=torch.long)
        next_observation = batch["next_observation"].to(device=device, dtype=torch.float32)

        latent = encoder(observation)
        target_latent = encoder(next_observation)
        predicted_latent = predictor(latent, action)
        loss = latent_prediction_loss(predicted_latent=predicted_latent, target_latent=target_latent)

        batch_size = observation.shape[0]
        total_loss += float(loss.item()) * batch_size
        total_examples += batch_size

    return total_loss / max(total_examples, 1)


def run_training(args: argparse.Namespace) -> dict[str, Any]:
    if args.seed is not None:
        torch.manual_seed(args.seed)

    dataset, dataset_metadata = build_transition_dataset(args)
    if len(dataset) == 0:
        raise SystemExit("Transition dataset is empty; increase --episodes or --steps.")

    train_size = int(len(dataset) * args.train_split)
    train_size = min(max(train_size, 1), len(dataset) - 1) if len(dataset) > 1 else len(dataset)
    val_size = len(dataset) - train_size

    generator = torch.Generator().manual_seed(args.seed if args.seed is not None else torch.seed())
    if val_size > 0:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)
    else:
        train_dataset = dataset
        val_dataset = None

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = None if val_dataset is None else DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    device = resolve_device(args.device)
    encoder = GlimpseEncoder(
        glimpse_size=args.glimpse_size,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)
    predictor = ActionConditionedPredictor(
        latent_dim=args.latent_dim,
        num_actions=5,
        action_embed_dim=args.action_embed_dim,
        hidden_dim=args.hidden_dim,
    ).to(device)

    optimizer = Adam(
        list(encoder.parameters()) + list(predictor.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    history: list[dict[str, float]] = []
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(
            encoder=encoder,
            predictor=predictor,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
        )
        val_loss = None if val_loader is None else evaluate_epoch(
            encoder=encoder,
            predictor=predictor,
            dataloader=val_loader,
            device=device,
        )

        metrics = {"epoch": float(epoch), "train_loss": train_loss}
        if val_loss is not None:
            metrics["val_loss"] = val_loss
        history.append(metrics)
        print(json.dumps(metrics))

    args.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "encoder_state_dict": encoder.state_dict(),
        "predictor_state_dict": predictor.state_dict(),
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in vars(args).items()
        },
        "dataset_metadata": dataset_metadata,
        "dataset_summary": dataset.summary(),
        "history": history,
    }
    torch.save(checkpoint, args.checkpoint)

    return {
        "device": str(device),
        "checkpoint": str(args.checkpoint),
        "dataset_metadata": dataset_metadata,
        "dataset_summary": dataset.summary(),
        "history": history,
    }


def main() -> None:
    args = parse_args()
    result = run_training(args)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
