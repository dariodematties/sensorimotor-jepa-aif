"""Visualize a latent diagnostics report as a compact summary figure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("artifacts/analysis/latent_diagnostics.json"),
        help="path to the latent diagnostics JSON report",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/analysis/latent_diagnostics.png"),
        help="path to the output figure",
    )
    parser.add_argument("--title", type=str, default=None, help="optional figure title override")
    return parser.parse_args()


def load_report(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Diagnostics report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def plot_summary(ax: plt.Axes, report: dict) -> None:
    rep = report["representation_metrics"]
    dataset = report["evaluation_dataset_summary"]
    probe = report["linear_probe"]

    lines = [
        f"samples: {rep['num_samples']}",
        f"latent dim: {rep['latent_dim']}",
        f"active dims @1e-3: {rep['active_dimensions@1e-3']}",
        f"latent variance mean: {rep['latent_variance_mean']:.4f}",
        f"prediction MSE mean: {rep['prediction_mse_mean']:.4f}",
        f"action sensitivity mean: {rep['action_sensitivity_mean']:.4f}",
        f"eval transitions: {dataset['num_transitions']}",
    ]

    if probe.get("status") == "ok":
        final = probe["final"]
        lines.append(f"probe val acc: {final['val_accuracy']:.3f}")
        lines.append(f"probe val loss: {final['val_loss']:.3f}")
    else:
        lines.append(f"probe: {probe.get('reason', 'skipped')}")

    ax.axis("off")
    ax.set_title("Summary", loc="left")
    ax.text(
        0.0,
        1.0,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=10,
        transform=ax.transAxes,
    )


def plot_representation_metrics(ax: plt.Axes, report: dict) -> None:
    rep = report["representation_metrics"]
    names = [
        "latent_var_mean",
        "next_latent_var_mean",
        "pred_mse_mean",
        "action_sens_mean",
        "latent_norm_mean",
    ]
    values = [
        rep["latent_variance_mean"],
        rep["next_latent_variance_mean"],
        rep["prediction_mse_mean"],
        rep["action_sensitivity_mean"],
        rep["latent_norm_mean"],
    ]

    colors = ["#1f77b4", "#6baed6", "#ff7f0e", "#2ca02c", "#d62728"]
    ax.bar(names, values, color=colors)
    ax.set_title("Representation Metrics", loc="left")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)


def plot_action_delta_norms(ax: plt.Axes, report: dict) -> None:
    rep = report["representation_metrics"]
    action_map = {"0": "stay", "1": "up", "2": "down", "3": "left", "4": "right"}
    values = rep["empirical_action_delta_norms"]
    labels = [action_map[key] for key in sorted(values.keys(), key=int)]
    heights = [values[key] for key in sorted(values.keys(), key=int)]

    ax.bar(labels, heights, color="#4c78a8")
    ax.set_title("Empirical Latent Change by Action", loc="left")
    ax.set_ylabel("mean ||s(t+1) - s(t)||")
    ax.grid(axis="y", alpha=0.25)


def plot_action_histogram(ax: plt.Axes, report: dict) -> None:
    histogram = report["evaluation_dataset_summary"]["action_histogram"]
    action_map = {"0": "stay", "1": "up", "2": "down", "3": "left", "4": "right"}
    labels = [action_map[key] for key in sorted(histogram.keys(), key=int)]
    counts = [histogram[key] for key in sorted(histogram.keys(), key=int)]

    ax.bar(labels, counts, color="#72b7b2")
    ax.set_title("Evaluation Action Histogram", loc="left")
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.25)


def plot_probe_history(ax: plt.Axes, report: dict) -> None:
    probe = report["linear_probe"]
    ax.set_title("Linear Probe", loc="left")

    if probe.get("status") != "ok":
        ax.axis("off")
        ax.text(0.0, 1.0, probe.get("reason", "probe skipped"), va="top", ha="left", transform=ax.transAxes)
        return

    history = probe["history"]
    epochs = [entry["epoch"] for entry in history]
    train_acc = [entry["train_accuracy"] for entry in history]
    val_acc = [entry["val_accuracy"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]

    ax.plot(epochs, train_acc, label="train acc", color="#2ca02c", linewidth=2)
    ax.plot(epochs, val_acc, label="val acc", color="#98df8a", linewidth=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy")
    ax.grid(alpha=0.25)

    twin = ax.twinx()
    twin.plot(epochs, train_loss, label="train loss", color="#d62728", linestyle="--")
    twin.plot(epochs, val_loss, label="val loss", color="#ff9896", linestyle="--")
    twin.set_ylabel("loss")

    handles_1, labels_1 = ax.get_legend_handles_labels()
    handles_2, labels_2 = twin.get_legend_handles_labels()
    ax.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper right", fontsize=8)


def plot_notes(ax: plt.Axes, report: dict) -> None:
    notes = report.get("notes", [])
    ax.axis("off")
    ax.set_title("Notes", loc="left")
    if not notes:
        notes = ["No extra notes."]

    wrapped_lines: list[str] = []
    for note in notes:
        chunks = _wrap_text(note, width=72)
        wrapped_lines.extend([f"- {chunks[0]}"] + [f"  {chunk}" for chunk in chunks[1:]])

    ax.text(
        0.0,
        1.0,
        "\n".join(wrapped_lines),
        va="top",
        ha="left",
        fontsize=10,
        transform=ax.transAxes,
    )


def _wrap_text(text: str, width: int) -> list[str]:
    words = text.split()
    if not words:
        return [""]

    lines = [words[0]]
    for word in words[1:]:
        candidate = f"{lines[-1]} {word}"
        if len(candidate) <= width:
            lines[-1] = candidate
        else:
            lines.append(word)
    return lines


def main() -> None:
    args = parse_args()
    report = load_report(args.input)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    ax_summary, ax_metrics, ax_deltas, ax_hist, ax_probe, ax_notes = axes.flat

    plot_summary(ax_summary, report)
    plot_representation_metrics(ax_metrics, report)
    plot_action_delta_norms(ax_deltas, report)
    plot_action_histogram(ax_hist, report)
    plot_probe_history(ax_probe, report)
    plot_notes(ax_notes, report)

    default_title = (
        f"Latent Diagnostics: {Path(report['checkpoint']).name} "
        f"on {report['evaluation_dataset_metadata'].get('environment', 'dataset')}"
    )
    fig.suptitle(args.title or default_title, fontsize=15, y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"saved visualization to {args.output}")


if __name__ == "__main__":
    main()
