# Artifact Guide

This guide maps the public `agent-self-organized-criticality` repository to a reviewer-friendly artifact workflow for `World Model Science`. It is meant to make the release easier to inspect in the style of ICML, ICLR, NeurIPS, and similar artifact-review processes.

## What To Inspect First

- `lib/`: Project-specific implementation subtree.
- `experiments/`: Experiment drivers, ablations, and benchmark-specific runners.
- `assets/`: README and paper-facing visual assets.

## Environment Files

- `requirements.txt`: Primary Python dependency list.
- `pyproject.toml`: Package metadata and optional extras when available.
- `.env.example`: Template for local credentials or backend configuration.

## Minimal Verification

Run these checks in a fresh environment before launching expensive jobs:

```bash
python -m compileall -q .
```

If a smoke command is not tracked, use the README Quick Start with the smallest available seed, sample, or task count.

## Reproduction And Analysis Entry Points

No single reproduction runner is tracked. Use the README experiment commands and the implementation map above; keep first runs small before scaling to full grids.

## Figure Assets

- `assets/agent-soc-overview.webp`
- `assets/agent-soc-pipeline.webp`

## Data, Credentials, And Generated Outputs

- API-backed runs should read credentials from environment variables or local `.env` files only; never commit real keys or provider-specific secrets.
- Record provider endpoint, model/deployment name, sampling parameters, and execution date for every API-backed table or figure.
- Treat generated JSONL files, logs, caches, model checkpoints, and benchmark downloads as local artifacts unless explicitly tracked as fixtures.
- For stochastic experiments, record seeds, task counts, dataset splits, and the exact git commit used for the run.

## Reviewer Reporting Checklist

- `git rev-parse HEAD`
- Python version and dependency-install command
- Full command line for every table, figure, or benchmark cell
- Paths to raw outputs and aggregation scripts
- External data, benchmark, or API-backed steps that were intentionally skipped
