# Sensorimotor JEPA + Active Inference

This repository starts from the environment side of the project: an active-vision MNIST setup where an agent moves a small glimpse window over a digit image.

The current first milestone is deliberately narrow:

- implement a movable glimpse environment
- visualize fixation trajectories and observations
- collect rollouts and transition data for the first predictor experiments
- leave model and loss code as placeholders until the sensorimotor loop is stable

## Environment

`MNISTGlimpseEnv` exposes:

- observation: a local `glimpse_size x glimpse_size` crop
- hidden world: a `28 x 28` digit image
- action space:
  - `0 = stay`
  - `1 = up`
  - `2 = down`
  - `3 = left`
  - `4 = right`

The environment supports two data sources:

- real MNIST through `torchvision.datasets.MNIST`
- a small synthetic fallback for offline smoke tests

## Quick Start

Create and populate a virtual environment:

```bash
python -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
```

Then run the synthetic offline smoke test:

```bash
python scripts/test_env.py --synthetic
```

That command is now random by default. If you want a reproducible rollout, pass an explicit seed, for example:

```bash
python scripts/test_env.py --synthetic --seed 0
```

If you prefer not to activate the venv, use:

```bash
.venv/bin/python scripts/test_env.py --synthetic
```

If you already have MNIST cached locally, or want to download it, run:

```bash
python scripts/test_env.py --download
```

This writes a rollout visualization to `artifacts/mnist_glimpse_rollout.png`.

## Rollout Logging

You can also collect training-style transition data:

```bash
python scripts/collect_rollouts.py --synthetic --episodes 8
```

That command writes:

- a rollout log with actions, fixations, and glimpses to `artifacts/rollouts/rollouts.json`
- a transition dataset with `(observation, action, next_observation, fixation, next_fixation)` arrays to `artifacts/rollouts/transitions.npz`

## First Training Baseline

The first learner keeps the architecture intentionally small:

- one encoder `q_phi(s_t | o_t)` mapping each `7x7` glimpse to a latent vector
- one action-conditioned predictor `p_theta(s_{t+1} | s_t, a_t)`
- one deterministic latent prediction loss as a first proxy for the later KL objective

Run a synthetic smoke test with:

```bash
python scripts/train_predictor.py --synthetic --episodes 32 --steps 8 --epochs 5 --seed 0
```

This writes a checkpoint to `artifacts/checkpoints/first_baseline.pt`.

## Repository Layout

```text
sensorimotor-jepa-aif/
├── README.md
├── requirements.txt
├── scripts/
│   ├── collect_rollouts.py
│   ├── test_env.py
│   └── train_predictor.py
└── src/
    └── sm_jepa_aif/
        ├── envs/
        │   └── mnist_glimpse_env.py
        ├── losses/
        ├── models/
        ├── policies/
        └── train.py
```
