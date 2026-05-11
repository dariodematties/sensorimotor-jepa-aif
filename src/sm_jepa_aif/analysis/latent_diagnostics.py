"""Utilities for inspecting learned latent representations."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset, random_split

from sm_jepa_aif.data import TransitionDataset
from sm_jepa_aif.models import ActionConditionedPredictor, GlimpseEncoder


def load_checkpoint_models(
    checkpoint_path: str,
    device: torch.device,
) -> tuple[GlimpseEncoder, ActionConditionedPredictor, dict[str, Any]]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]

    encoder = GlimpseEncoder(
        glimpse_size=int(config["glimpse_size"]),
        latent_dim=int(config["latent_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    ).to(device)
    predictor = ActionConditionedPredictor(
        latent_dim=int(config["latent_dim"]),
        num_actions=5,
        action_embed_dim=int(config["action_embed_dim"]),
        hidden_dim=int(config["hidden_dim"]),
    ).to(device)

    encoder.load_state_dict(checkpoint["encoder_state_dict"])
    predictor.load_state_dict(checkpoint["predictor_state_dict"])
    encoder.eval()
    predictor.eval()
    return encoder, predictor, checkpoint


@torch.no_grad()
def evaluate_representation(
    encoder: nn.Module,
    predictor: nn.Module,
    dataset: TransitionDataset,
    device: torch.device,
    batch_size: int = 256,
) -> dict[str, Any]:
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    latents: list[torch.Tensor] = []
    next_latents: list[torch.Tensor] = []
    predicted_latents: list[torch.Tensor] = []
    actions: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    action_sensitivities: list[torch.Tensor] = []

    for batch in dataloader:
        observation = batch["observation"].to(device=device, dtype=torch.float32)
        action = batch["action"].to(device=device, dtype=torch.long)
        next_observation = batch["next_observation"].to(device=device, dtype=torch.float32)
        label = batch["label"].to(device=device, dtype=torch.long)

        latent = encoder(observation)
        next_latent = encoder(next_observation)
        predicted_latent = predictor(latent, action)

        all_actions = torch.arange(5, device=device, dtype=torch.long)
        repeated_latent = latent.unsqueeze(1).expand(-1, all_actions.shape[0], -1)
        repeated_actions = all_actions.unsqueeze(0).expand(latent.shape[0], -1)
        all_predictions = predictor(
            repeated_latent.reshape(-1, latent.shape[-1]),
            repeated_actions.reshape(-1),
        ).reshape(latent.shape[0], all_actions.shape[0], -1)
        centered_predictions = all_predictions - all_predictions.mean(dim=1, keepdim=True)
        action_sensitivity = centered_predictions.norm(dim=-1).mean(dim=1)

        latents.append(latent.cpu())
        next_latents.append(next_latent.cpu())
        predicted_latents.append(predicted_latent.cpu())
        actions.append(action.cpu())
        labels.append(label.cpu())
        action_sensitivities.append(action_sensitivity.cpu())

    latent_tensor = torch.cat(latents, dim=0)
    next_latent_tensor = torch.cat(next_latents, dim=0)
    predicted_latent_tensor = torch.cat(predicted_latents, dim=0)
    action_tensor = torch.cat(actions, dim=0)
    label_tensor = torch.cat(labels, dim=0)
    action_sensitivity_tensor = torch.cat(action_sensitivities, dim=0)

    latent_variance_per_dim = latent_tensor.var(dim=0, unbiased=False)
    next_latent_variance_per_dim = next_latent_tensor.var(dim=0, unbiased=False)
    latent_norm = latent_tensor.norm(dim=-1)
    prediction_error = ((predicted_latent_tensor - next_latent_tensor) ** 2).mean(dim=-1)

    action_delta_norms: dict[str, float] = {}
    for action_id in range(5):
        mask = action_tensor == action_id
        if mask.any():
            delta = next_latent_tensor[mask] - latent_tensor[mask]
            action_delta_norms[str(action_id)] = float(delta.norm(dim=-1).mean().item())
        else:
            action_delta_norms[str(action_id)] = 0.0

    valid_label_mask = label_tensor >= 0
    label_counts: dict[str, int] = {}
    if valid_label_mask.any():
        values, counts = torch.unique(label_tensor[valid_label_mask], return_counts=True)
        label_counts = {str(int(v.item())): int(c.item()) for v, c in zip(values, counts, strict=True)}

    return {
        "num_samples": int(latent_tensor.shape[0]),
        "latent_dim": int(latent_tensor.shape[1]),
        "latent_variance_mean": float(latent_variance_per_dim.mean().item()),
        "latent_variance_min": float(latent_variance_per_dim.min().item()),
        "latent_variance_max": float(latent_variance_per_dim.max().item()),
        "next_latent_variance_mean": float(next_latent_variance_per_dim.mean().item()),
        "active_dimensions@1e-3": int((latent_variance_per_dim > 1e-3).sum().item()),
        "latent_norm_mean": float(latent_norm.mean().item()),
        "latent_norm_std": float(latent_norm.std(unbiased=False).item()),
        "prediction_mse_mean": float(prediction_error.mean().item()),
        "prediction_mse_std": float(prediction_error.std(unbiased=False).item()),
        "action_sensitivity_mean": float(action_sensitivity_tensor.mean().item()),
        "action_sensitivity_std": float(action_sensitivity_tensor.std(unbiased=False).item()),
        "empirical_action_delta_norms": action_delta_norms,
        "label_counts": label_counts,
    }


def evaluate_linear_probe(
    encoder: nn.Module,
    dataset: TransitionDataset,
    device: torch.device,
    batch_size: int = 256,
    train_split: float = 0.8,
    epochs: int = 25,
    learning_rate: float = 1e-2,
    seed: int | None = None,
) -> dict[str, Any]:
    arrays = dataset.as_arrays()
    labels = arrays["label"]
    valid_mask = labels >= 0
    if not np.any(valid_mask):
        return {"status": "skipped", "reason": "dataset has no valid labels"}

    observations = torch.from_numpy(arrays["observation"][valid_mask]).to(device=device, dtype=torch.float32)
    labels_tensor = torch.from_numpy(labels[valid_mask]).to(device=device, dtype=torch.long)

    with torch.no_grad():
        latents = encoder(observations)

    num_classes = int(labels_tensor.max().item()) + 1
    probe_dataset = TensorDataset(latents.detach().cpu(), labels_tensor.detach().cpu())
    if len(probe_dataset) < 2:
        return {"status": "skipped", "reason": "not enough labeled samples for probe"}

    train_size = int(len(probe_dataset) * train_split)
    train_size = min(max(train_size, 1), len(probe_dataset) - 1)
    val_size = len(probe_dataset) - train_size
    generator = torch.Generator().manual_seed(0 if seed is None else seed)
    train_dataset, val_dataset = random_split(probe_dataset, [train_size, val_size], generator=generator)

    train_loader = DataLoader(train_dataset, batch_size=min(batch_size, train_size), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=min(batch_size, max(val_size, 1)), shuffle=False)

    probe = nn.Linear(latents.shape[1], num_classes).to(device)
    optimizer = Adam(probe.parameters(), lr=learning_rate)
    loss_fn = nn.CrossEntropyLoss()

    history: list[dict[str, float]] = []
    for epoch in range(1, epochs + 1):
        probe.train()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0

        for latent_batch, label_batch in train_loader:
            latent_batch = latent_batch.to(device)
            label_batch = label_batch.to(device)

            logits = probe(latent_batch)
            loss = loss_fn(logits, label_batch)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            total_loss += float(loss.item()) * latent_batch.shape[0]
            total_correct += int((logits.argmax(dim=-1) == label_batch).sum().item())
            total_examples += latent_batch.shape[0]

        val_metrics = _evaluate_probe(probe, val_loader, device, loss_fn)
        history.append(
            {
                "epoch": float(epoch),
                "train_loss": total_loss / max(total_examples, 1),
                "train_accuracy": total_correct / max(total_examples, 1),
                "val_loss": val_metrics["loss"],
                "val_accuracy": val_metrics["accuracy"],
            }
        )

    return {
        "status": "ok",
        "num_classes": num_classes,
        "num_samples": len(probe_dataset),
        "train_samples": train_size,
        "val_samples": val_size,
        "final": history[-1],
        "history": history,
    }


@torch.no_grad()
def _evaluate_probe(
    probe: nn.Module,
    dataloader: DataLoader[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    loss_fn: nn.Module,
) -> dict[str, float]:
    probe.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for latent_batch, label_batch in dataloader:
        latent_batch = latent_batch.to(device)
        label_batch = label_batch.to(device)
        logits = probe(latent_batch)
        loss = loss_fn(logits, label_batch)

        total_loss += float(loss.item()) * latent_batch.shape[0]
        total_correct += int((logits.argmax(dim=-1) == label_batch).sum().item())
        total_examples += latent_batch.shape[0]

    return {
        "loss": total_loss / max(total_examples, 1),
        "accuracy": total_correct / max(total_examples, 1),
    }


def build_diagnostic_notes(
    dataset_metadata: dict[str, Any],
    probe_metrics: dict[str, Any],
) -> list[str]:
    notes: list[str] = []
    environment = dataset_metadata.get("environment")

    if environment == "synthetic":
        notes.append(
            "The linear probe is running on synthetic fallback labels, so it measures separability of synthetic image identities rather than true MNIST digit semantics."
        )

    if probe_metrics.get("status") != "ok":
        notes.append(f"Linear probe was skipped: {probe_metrics.get('reason', 'unknown reason')}.")

    return notes
