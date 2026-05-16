"""Compare two latent diagnostics reports and visualize their differences."""

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
    parser.add_argument("--left", type=Path, required=True, help="first diagnostics JSON report")
    parser.add_argument("--right", type=Path, required=True, help="second diagnostics JSON report")
    parser.add_argument("--left-label", type=str, default="Left", help="label for the first report")
    parser.add_argument("--right-label", type=str, default="Right", help="label for the second report")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts/analysis/latent_diagnostics_comparison.png"),
        help="path to the output comparison figure",
    )
    parser.add_argument("--title", type=str, default=None, help="optional figure title override")
    return parser.parse_args()


def load_report(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Diagnostics report not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def collect_metric_table(report: dict) -> dict[str, float]:
    rep = report["representation_metrics"]
    probe = report["linear_probe"]
    metrics = {
        "latent_variance_mean": rep["latent_variance_mean"],
        "latent_variance_min": rep["latent_variance_min"],
        "latent_variance_max": rep["latent_variance_max"],
        "prediction_mse_mean": rep["prediction_mse_mean"],
        "action_sensitivity_mean": rep["action_sensitivity_mean"],
        "latent_norm_mean": rep["latent_norm_mean"],
        "active_dimensions@1e-3": float(rep["active_dimensions@1e-3"]),
    }
    if probe.get("status") == "ok":
        metrics["probe_val_accuracy"] = probe["final"]["val_accuracy"]
        metrics["probe_val_loss"] = probe["final"]["val_loss"]
    return metrics


def plot_summary(ax: plt.Axes, report: dict, label: str) -> None:
    rep = report["representation_metrics"]
    dataset = report["evaluation_dataset_summary"]
    metadata = report["evaluation_dataset_metadata"]
    probe = report["linear_probe"]

    lines = [
        f"label: {label}",
        f"env: {metadata.get('environment')}",
        f"samples: {rep['num_samples']}",
        f"latent dim: {rep['latent_dim']}",
        f"active dims @1e-3: {rep['active_dimensions@1e-3']}",
        f"pred MSE mean: {rep['prediction_mse_mean']:.4f}",
        f"action sens mean: {rep['action_sensitivity_mean']:.4f}",
        f"eval transitions: {dataset['num_transitions']}",
    ]
    if probe.get("status") == "ok":
        lines.append(f"probe val acc: {probe['final']['val_accuracy']:.3f}")
    else:
        lines.append(f"probe: {probe.get('reason', 'skipped')}")

    ax.axis("off")
    ax.set_title(f"{label} Summary", loc="left")
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


def plot_key_metrics(ax: plt.Axes, left_report: dict, right_report: dict, left_label: str, right_label: str) -> None:
    left_metrics = collect_metric_table(left_report)
    right_metrics = collect_metric_table(right_report)

    metric_names = [name for name in left_metrics if name in right_metrics]
    x = np.arange(len(metric_names))
    width = 0.38

    left_values = [left_metrics[name] for name in metric_names]
    right_values = [right_metrics[name] for name in metric_names]

    ax.bar(x - width / 2, left_values, width=width, label=left_label, color="#4c78a8")
    ax.bar(x + width / 2, right_values, width=width, label=right_label, color="#f58518")
    ax.set_title("Key Metrics", loc="left")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_names, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()


def plot_delta_table(ax: plt.Axes, left_report: dict, right_report: dict, left_label: str, right_label: str) -> None:
    left_metrics = collect_metric_table(left_report)
    right_metrics = collect_metric_table(right_report)

    metric_names = [name for name in left_metrics if name in right_metrics]
    lines = [f"delta = {right_label} - {left_label}"]
    for name in metric_names:
        delta = right_metrics[name] - left_metrics[name]
        lines.append(f"{name:>24}: {delta:+.4f}")

    ax.axis("off")
    ax.set_title("Metric Deltas", loc="left")
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


def plot_action_deltas(ax: plt.Axes, left_report: dict, right_report: dict, left_label: str, right_label: str) -> None:
    action_map = {"0": "stay", "1": "up", "2": "down", "3": "left", "4": "right"}
    left_values = left_report["representation_metrics"]["empirical_action_delta_norms"]
    right_values = right_report["representation_metrics"]["empirical_action_delta_norms"]
    keys = sorted(set(left_values) | set(right_values), key=int)
    labels = [action_map[key] for key in keys]
    x = np.arange(len(labels))
    width = 0.38

    left_heights = [left_values.get(key, 0.0) for key in keys]
    right_heights = [right_values.get(key, 0.0) for key in keys]

    ax.bar(x - width / 2, left_heights, width=width, label=left_label, color="#54a24b")
    ax.bar(x + width / 2, right_heights, width=width, label=right_label, color="#e45756")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("mean ||s(t+1) - s(t)||")
    ax.set_title("Empirical Latent Change by Action", loc="left")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()


def plot_probe_comparison(ax: plt.Axes, left_report: dict, right_report: dict, left_label: str, right_label: str) -> None:
    ax.set_title("Linear Probe Validation Accuracy", loc="left")
    left_probe = left_report["linear_probe"]
    right_probe = right_report["linear_probe"]

    if left_probe.get("status") != "ok" or right_probe.get("status") != "ok":
        ax.axis("off")
        ax.text(0.0, 1.0, "One or both probe histories are unavailable.", va="top", ha="left", transform=ax.transAxes)
        return

    left_epochs = [entry["epoch"] for entry in left_probe["history"]]
    right_epochs = [entry["epoch"] for entry in right_probe["history"]]
    left_val = [entry["val_accuracy"] for entry in left_probe["history"]]
    right_val = [entry["val_accuracy"] for entry in right_probe["history"]]

    ax.plot(left_epochs, left_val, label=f"{left_label} val acc", color="#4c78a8", linewidth=2)
    ax.plot(right_epochs, right_val, label=f"{right_label} val acc", color="#f58518", linewidth=2)
    ax.set_xlabel("epoch")
    ax.set_ylabel("validation accuracy")
    ax.grid(alpha=0.25)
    ax.legend()


def format_notes(left_report: dict, right_report: dict, left_label: str, right_label: str) -> str:
    left_notes = left_report.get("notes", [])
    right_notes = right_report.get("notes", [])

    sections = [
        f"{left_label}:",
        *([f"- {line}" for line in left_notes] or ["- no extra notes"]),
        "",
        f"{right_label}:",
        *([f"- {line}" for line in right_notes] or ["- no extra notes"]),
    ]

    wrapped_lines: list[str] = []
    for line in sections:
        if not line:
            wrapped_lines.append("")
            continue
        wrapped_lines.extend(_wrap_prefixed_line(line, width=72))

    return "\n".join(wrapped_lines)


def _wrap_prefixed_line(text: str, width: int) -> list[str]:
    if text.startswith("- "):
        prefix = "- "
        content = text[2:]
    else:
        prefix = ""
        content = text

    words = content.split()
    if not words:
        return [prefix.rstrip()]

    lines = [prefix + words[0]]
    indent = "  " if prefix else ""
    for word in words[1:]:
        candidate = f"{lines[-1]} {word}"
        if len(candidate) <= width:
            lines[-1] = candidate
        else:
            lines.append(f"{indent}{word}")
    return lines


def main() -> None:
    args = parse_args()
    left_report = load_report(args.left)
    right_report = load_report(args.right)

    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    ax_left, ax_right, ax_delta, ax_metrics, ax_actions, ax_probe = axes.flat

    plot_summary(ax_left, left_report, args.left_label)
    plot_summary(ax_right, right_report, args.right_label)
    plot_delta_table(ax_delta, left_report, right_report, args.left_label, args.right_label)
    plot_key_metrics(ax_metrics, left_report, right_report, args.left_label, args.right_label)
    plot_action_deltas(ax_actions, left_report, right_report, args.left_label, args.right_label)
    plot_probe_comparison(ax_probe, left_report, right_report, args.left_label, args.right_label)

    notes_text = format_notes(left_report, right_report, args.left_label, args.right_label)
    ax_probe.text(
        1.02,
        0.98,
        notes_text,
        va="top",
        ha="left",
        fontsize=8.5,
        transform=ax_probe.transAxes,
        bbox={"boxstyle": "round,pad=0.35", "facecolor": "#f7f7f7", "edgecolor": "#cccccc"},
    )

    default_title = f"Latent Diagnostics Comparison: {args.left_label} vs {args.right_label}"
    fig.suptitle(args.title or default_title, fontsize=16, y=0.98)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    print(f"saved comparison to {args.output}")


if __name__ == "__main__":
    main()
