# Sensorimotor JEPA + Active Inference

This repository starts from the environment side of the project: an active-vision MNIST setup where an agent moves a small glimpse window over a digit image.

The current first milestone is deliberately narrow:

- implement a movable glimpse environment
- visualize fixation trajectories and observations
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

If you prefer not to activate the venv, use:

```bash
.venv/bin/python scripts/test_env.py --synthetic
```

If you already have MNIST cached locally, or want to download it, run:

```bash
python scripts/test_env.py --download
```

This writes a rollout visualization to `artifacts/mnist_glimpse_rollout.png`.

## Repository Layout

```text
sensorimotor-jepa-aif/
├── README.md
├── requirements.txt
├── scripts/
│   └── test_env.py
└── src/
    └── sm_jepa_aif/
        ├── envs/
        │   └── mnist_glimpse_env.py
        ├── losses/
        ├── models/
        ├── policies/
        └── train.py
```
