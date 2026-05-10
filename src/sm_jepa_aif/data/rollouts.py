"""Rollout logging utilities for environment-driven data collection."""

from __future__ import annotations

import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from sm_jepa_aif.envs import MNISTGlimpseEnv


@dataclass(frozen=True)
class RolloutStep:
    """One sensorimotor transition collected from the environment."""

    step_index: int
    action: int
    action_name: str
    fixation_before: tuple[int, int]
    fixation_after: tuple[int, int]
    observation_before: np.ndarray
    observation_after: np.ndarray
    reward: float
    terminated: bool
    truncated: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_index": self.step_index,
            "action": self.action,
            "action_name": self.action_name,
            "fixation_before": list(self.fixation_before),
            "fixation_after": list(self.fixation_after),
            "observation_before": self.observation_before.tolist(),
            "observation_after": self.observation_after.tolist(),
            "reward": self.reward,
            "terminated": self.terminated,
            "truncated": self.truncated,
        }


@dataclass(frozen=True)
class RolloutEpisode:
    """A full episode with observations, actions, and summary metrics."""

    image_index: int
    label: int | None
    initial_fixation: tuple[int, int]
    initial_observation: np.ndarray
    steps: list[RolloutStep]
    image_shape: tuple[int, int]

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def actions(self) -> list[int]:
        return [step.action for step in self.steps]

    @property
    def fixation_path(self) -> list[tuple[int, int]]:
        path = [self.initial_fixation]
        path.extend(step.fixation_after for step in self.steps)
        return path

    @property
    def coverage(self) -> float:
        unique_fixations = len(set(self.fixation_path))
        total_pixels = self.image_shape[0] * self.image_shape[1]
        return unique_fixations / total_pixels

    @property
    def revisit_rate(self) -> float:
        path = self.fixation_path
        if len(path) <= 1:
            return 0.0
        revisits = len(path) - len(set(path))
        return revisits / len(path)

    @property
    def action_histogram(self) -> dict[int, int]:
        histogram = {action: 0 for action in MNISTGlimpseEnv.ACTION_NAMES}
        for action in self.actions:
            histogram[action] += 1
        return histogram

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_index": self.image_index,
            "label": self.label,
            "initial_fixation": list(self.initial_fixation),
            "initial_observation": self.initial_observation.tolist(),
            "image_shape": list(self.image_shape),
            "num_steps": self.num_steps,
            "coverage": self.coverage,
            "revisit_rate": self.revisit_rate,
            "action_histogram": {str(key): value for key, value in self.action_histogram.items()},
            "steps": [step.to_dict() for step in self.steps],
        }


class RolloutLogger:
    """Collect and persist environment rollouts for later analysis or training."""

    def __init__(self) -> None:
        self.episodes: list[RolloutEpisode] = []

    def collect_episode(
        self,
        env: MNISTGlimpseEnv,
        policy: Any,
        index: int | None = None,
        fixation: tuple[int, int] | None = None,
        max_steps: int | None = None,
    ) -> RolloutEpisode:
        observation, info = env.reset(index=index, fixation=fixation)
        initial_observation = observation.copy()
        steps: list[RolloutStep] = []
        limit = env.max_steps if max_steps is None else max_steps

        for _ in range(limit):
            action = _select_action(policy=policy, observation=observation, info=info)
            fixation_before = tuple(info["fixation"])
            result = env.step(action)

            steps.append(
                RolloutStep(
                    step_index=result.info["step_count"],
                    action=action,
                    action_name=result.info["last_action_name"],
                    fixation_before=fixation_before,
                    fixation_after=tuple(result.info["fixation"]),
                    observation_before=observation.copy(),
                    observation_after=result.observation.copy(),
                    reward=result.reward,
                    terminated=result.terminated,
                    truncated=result.truncated,
                )
            )

            observation = result.observation
            info = result.info
            if result.terminated or result.truncated:
                break

        episode = RolloutEpisode(
            image_index=int(info["index"]),
            label=info["label"],
            initial_fixation=tuple(env.trajectory[0]),
            initial_observation=initial_observation,
            steps=steps,
            image_shape=(env.height, env.width),
        )
        self.episodes.append(episode)
        return episode

    def collect_episodes(
        self,
        env: MNISTGlimpseEnv,
        policy: Any,
        num_episodes: int,
        max_steps: int | None = None,
    ) -> list[RolloutEpisode]:
        return [
            self.collect_episode(env=env, policy=policy, max_steps=max_steps)
            for _ in range(num_episodes)
        ]

    def summary(self) -> dict[str, Any]:
        if not self.episodes:
            return {
                "num_episodes": 0,
                "num_transitions": 0,
                "mean_steps": 0.0,
                "mean_coverage": 0.0,
                "mean_revisit_rate": 0.0,
                "action_histogram": {name: 0 for name in MNISTGlimpseEnv.ACTION_NAMES.values()},
            }

        total_histogram = {action: 0 for action in MNISTGlimpseEnv.ACTION_NAMES}
        for episode in self.episodes:
            for action, count in episode.action_histogram.items():
                total_histogram[action] += count

        return {
            "num_episodes": len(self.episodes),
            "num_transitions": int(sum(episode.num_steps for episode in self.episodes)),
            "mean_steps": float(np.mean([episode.num_steps for episode in self.episodes])),
            "mean_coverage": float(np.mean([episode.coverage for episode in self.episodes])),
            "mean_revisit_rate": float(np.mean([episode.revisit_rate for episode in self.episodes])),
            "action_histogram": {
                MNISTGlimpseEnv.ACTION_NAMES[action]: count
                for action, count in total_histogram.items()
            },
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "summary": self.summary(),
            "episodes": [episode.to_dict() for episode in self.episodes],
        }

    def save_json(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def save_npz(self, path: str | Path) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            destination,
            summary=np.array(json.dumps(self.summary()), dtype=object),
            episodes=np.array(json.dumps(self.to_dict()["episodes"]), dtype=object),
        )


def _select_action(policy: Any, observation: np.ndarray, info: dict[str, Any]) -> int:
    act = getattr(policy, "act")
    signature = inspect.signature(act)
    params = list(signature.parameters.values())

    if not params:
        return int(act())

    names = {param.name for param in params}
    if {"observation", "info"}.issubset(names):
        return int(act(observation=observation, info=info))
    if len(params) == 2:
        return int(act(observation, info))
    if len(params) == 1:
        return int(act(observation))
    return int(act())
