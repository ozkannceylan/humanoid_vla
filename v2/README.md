# HUMANOID VLA v2 — IK-Based Data Generation Pipeline

This directory contains a standalone v2 data-generation pipeline added alongside the existing project code.

## Scope

- IK-based pick-and-place demonstrations
- Configurable static camera plus wrist camera observations
- Per-episode domain randomization
- Export to legacy LeRobot v2.1 layout for downstream conversion or training workflows

## Layout

- `config/data_gen_config.yaml` — central configuration
- `env/` — scene wrapper and camera definitions
- `ik/` — grasp strategies, perturbations, and trajectory generation
- `randomization/` — episode randomization
- `data/` — recording and LeRobot export
- `scripts/` — generation and visualization entry points
- `tests/` — targeted verification

## Main command

Run [v2/scripts/generate_episodes.py](v2/scripts/generate_episodes.py) with:

- `--config` path to YAML config
- `--num-episodes` total requested episodes
- `--output-dir` output root
- `--seed` base RNG seed
- `--parallel` optional worker count

The generator writes raw episode artifacts first, then exports a LeRobot v2.1 dataset under `v2/output/lerobot_dataset/` by default.
