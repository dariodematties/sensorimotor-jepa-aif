"""MNIST active-vision environment with a movable glimpse sensor."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

try:
    from torchvision import datasets
except ImportError:  # pragma: no cover - optional at runtime
    datasets = None

if TYPE_CHECKING:  # pragma: no cover - typing only
    from matplotlib.figure import Figure


@dataclass(frozen=True)
class StepResult:
    """Return type for environment transitions."""

    observation: np.ndarray
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class MNISTGlimpseEnv:
    """Environment exposing local image crops conditioned on fixation actions."""

    ACTION_STAY = 0
    ACTION_UP = 1
    ACTION_DOWN = 2
    ACTION_LEFT = 3
    ACTION_RIGHT = 4

    ACTION_DELTAS: dict[int, tuple[int, int]] = {
        ACTION_STAY: (0, 0),
        ACTION_UP: (-1, 0),
        ACTION_DOWN: (1, 0),
        ACTION_LEFT: (0, -1),
        ACTION_RIGHT: (0, 1),
    }

    ACTION_NAMES: dict[int, str] = {
        ACTION_STAY: "stay",
        ACTION_UP: "up",
        ACTION_DOWN: "down",
        ACTION_LEFT: "left",
        ACTION_RIGHT: "right",
    }

    def __init__(
        self,
        images: np.ndarray,
        labels: np.ndarray | None = None,
        glimpse_size: int = 7,
        max_steps: int = 25,
        start_strategy: str = "center",
        seed: int | None = None,
    ) -> None:
        if images.ndim != 3:
            raise ValueError("images must have shape (N, H, W)")
        if images.shape[1] <= 0 or images.shape[2] <= 0:
            raise ValueError("images must have positive spatial dimensions")
        if glimpse_size <= 0 or glimpse_size % 2 == 0:
            raise ValueError("glimpse_size must be a positive odd integer")
        if max_steps <= 0:
            raise ValueError("max_steps must be positive")
        if start_strategy not in {"center", "random"}:
            raise ValueError("start_strategy must be 'center' or 'random'")

        self.images = images.astype(np.float32, copy=False)
        self.labels = None if labels is None else np.asarray(labels)
        self.glimpse_size = glimpse_size
        self.max_steps = max_steps
        self.start_strategy = start_strategy
        self.rng = np.random.default_rng(seed)

        self.num_images, self.height, self.width = self.images.shape
        self.radius = glimpse_size // 2

        self.current_index = 0
        self.current_label: int | None = None
        self.fixation = (0, 0)
        self.step_count = 0
        self.trajectory: list[tuple[int, int]] = []

    @classmethod
    def from_torchvision(
        cls,
        root: str | Path = "data",
        train: bool = True,
        glimpse_size: int = 7,
        max_steps: int = 25,
        start_strategy: str = "center",
        seed: int | None = None,
        download: bool = False,
    ) -> "MNISTGlimpseEnv":
        if datasets is None:
            raise RuntimeError("torchvision is not installed")

        dataset = datasets.MNIST(root=str(root), train=train, download=download)
        images = dataset.data.numpy().astype(np.float32) / 255.0
        labels = dataset.targets.numpy()
        return cls(
            images=images,
            labels=labels,
            glimpse_size=glimpse_size,
            max_steps=max_steps,
            start_strategy=start_strategy,
            seed=seed,
        )

    @classmethod
    def from_synthetic(
        cls,
        num_images: int = 16,
        image_size: int = 28,
        glimpse_size: int = 7,
        max_steps: int = 25,
        seed: int | None = None,
    ) -> "MNISTGlimpseEnv":
        rng = np.random.default_rng(seed)
        images = np.zeros((num_images, image_size, image_size), dtype=np.float32)
        yy, xx = np.mgrid[0:image_size, 0:image_size]

        for idx in range(num_images):
            center_y = int(rng.integers(8, image_size - 8))
            center_x = int(rng.integers(8, image_size - 8))
            radius = int(rng.integers(4, 8))
            mask = (yy - center_y) ** 2 + (xx - center_x) ** 2 <= radius**2
            images[idx, mask] = 1.0

            if idx % 2 == 0:
                images[idx, max(center_y - radius, 0) : min(center_y + radius + 1, image_size), center_x] = 0.5
            else:
                images[idx, center_y, max(center_x - radius, 0) : min(center_x + radius + 1, image_size)] = 0.5

        return cls(
            images=images,
            labels=np.arange(num_images),
            glimpse_size=glimpse_size,
            max_steps=max_steps,
            start_strategy="center",
            seed=seed,
        )

    def reset(
        self,
        index: int | None = None,
        fixation: tuple[int, int] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        if index is None:
            self.current_index = int(self.rng.integers(0, self.num_images))
        else:
            if not 0 <= index < self.num_images:
                raise IndexError(f"index {index} out of range for dataset of size {self.num_images}")
            self.current_index = index

        if fixation is None:
            self.fixation = self._initial_fixation()
        else:
            self.fixation = self._clip_fixation(*fixation)

        self.step_count = 0
        self.trajectory = [self.fixation]
        self.current_label = None if self.labels is None else int(self.labels[self.current_index])

        observation = self._extract_glimpse(self.fixation)
        info = self._build_info(last_action=None)
        return observation, info

    def step(self, action: int) -> StepResult:
        if action not in self.ACTION_DELTAS:
            raise ValueError(f"unknown action {action}")

        dy, dx = self.ACTION_DELTAS[action]
        y, x = self.fixation
        self.fixation = self._clip_fixation(y + dy, x + dx)
        self.step_count += 1
        self.trajectory.append(self.fixation)

        observation = self._extract_glimpse(self.fixation)
        terminated = self.step_count >= self.max_steps
        info = self._build_info(last_action=action)

        return StepResult(
            observation=observation,
            reward=0.0,
            terminated=terminated,
            truncated=False,
            info=info,
        )

    def render(self) -> Figure:
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError(
                "render() requires matplotlib. Install project dependencies with "
                "'python -m pip install -r requirements.txt' or run the script from the project venv."
            ) from exc

        image = self.images[self.current_index]
        glimpse = self._extract_glimpse(self.fixation)

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        ax_image, ax_glimpse = axes

        ax_image.imshow(image, cmap="gray", vmin=0.0, vmax=1.0)
        ys, xs = zip(*self.trajectory)
        ax_image.plot(xs, ys, color="tab:red", linewidth=1.5, marker="o", markersize=3)

        y, x = self.fixation
        rect = plt.Rectangle(
            (x - self.radius - 0.5, y - self.radius - 0.5),
            self.glimpse_size,
            self.glimpse_size,
            fill=False,
            edgecolor="tab:cyan",
            linewidth=2,
        )
        ax_image.add_patch(rect)
        label_text = "?" if self.current_label is None else str(self.current_label)
        ax_image.set_title(f"Image #{self.current_index} label={label_text}")
        ax_image.set_xlim(-0.5, self.width - 0.5)
        ax_image.set_ylim(self.height - 0.5, -0.5)
        ax_image.set_aspect("equal")

        ax_glimpse.imshow(glimpse, cmap="gray", vmin=0.0, vmax=1.0)
        ax_glimpse.set_title(f"Glimpse at {self.fixation}")

        for axis in axes:
            axis.set_xticks([])
            axis.set_yticks([])

        fig.tight_layout()
        return fig

    def _initial_fixation(self) -> tuple[int, int]:
        if self.start_strategy == "random":
            y = int(self.rng.integers(0, self.height))
            x = int(self.rng.integers(0, self.width))
            return (y, x)
        return (self.height // 2, self.width // 2)

    def _clip_fixation(self, y: int, x: int) -> tuple[int, int]:
        clipped_y = int(np.clip(y, 0, self.height - 1))
        clipped_x = int(np.clip(x, 0, self.width - 1))
        return (clipped_y, clipped_x)

    def _extract_glimpse(self, fixation: tuple[int, int]) -> np.ndarray:
        y, x = fixation
        padded = np.pad(self.images[self.current_index], self.radius, mode="constant")
        padded_y = y + self.radius
        padded_x = x + self.radius
        glimpse = padded[
            padded_y - self.radius : padded_y + self.radius + 1,
            padded_x - self.radius : padded_x + self.radius + 1,
        ]
        return glimpse.astype(np.float32, copy=False)

    def _build_info(self, last_action: int | None) -> dict[str, Any]:
        return {
            "index": self.current_index,
            "label": self.current_label,
            "fixation": self.fixation,
            "step_count": self.step_count,
            "trajectory": list(self.trajectory),
            "last_action": last_action,
            "last_action_name": None if last_action is None else self.ACTION_NAMES[last_action],
        }
