# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

An Apple Silicon (MLX) port of [Karpathy's autoresearch](https://github.com/karpathy/autoresearch) — autonomous LLM-driven research loops that train language models on MLX with a fixed 5-minute time budget per experiment. The goal is to minimize `val_bpb` (bits per byte) by iterating on `train.py`.

## Commands

```bash
uv sync                # install dependencies
uv run prepare.py      # one-time data download + tokenizer training (~/.cache/autoresearch/)
uv run train.py        # run one 5-minute training experiment
```

There are no tests, linter, or build steps — this is a research experiment repo.

## Architecture

Only two Python files, both at the top level:

- **`prepare.py`** — READ-ONLY. Contains fixed constants (`MAX_SEQ_LEN=2048`, `TIME_BUDGET=300`, `EVAL_TOKENS`), data downloading/sharding, BPE tokenizer training (via `rustbpe`), the `make_dataloader` (BOS-aligned best-fit packing), and `evaluate_bpb` (the ground-truth metric). Never modify this file during experiments.
- **`train.py`** — The ONLY file the agent edits. Contains the GPT model (with RoPE, sliding window attention, value embeddings, residual lambdas), a custom `AdamW` optimizer with per-parameter-group LR/decay, hyperparameter constants, and the training loop. All hyperparameters are module-level constants (no CLI flags).

Key data flow: `prepare.py` exports `Tokenizer`, `make_dataloader`, `evaluate_bpb`, `MAX_SEQ_LEN`, and `TIME_BUDGET` → `train.py` imports and uses them.

## Experiment Protocol

The full protocol is in `program.md`. Key rules:

- Each experiment: edit `train.py` → `uv run train.py > run.log 2>&1` → check `val_bpb` → keep or revert
- Results logged to `results.tsv` (tab-separated: commit, val_bpb, memory_gb, status, description)
- Experiments run on a dedicated `autoresearch/<tag>` branch
- If val_bpb improves: amend commit to include results.tsv update
- If val_bpb is equal or worse: `git reset --hard` to previous kept commit
- Never use `git add -A` — only stage specific files
- Never install new packages — only use what's in `pyproject.toml`
- Runs take ~7 minutes total (5 min training + compile/eval overhead)

## Model Details

The model in `train.py` is a GPT variant with:
- Configurable depth/width via `DEPTH` and `ASPECT_RATIO` constants (model_dim derived as `DEPTH * ASPECT_RATIO`, rounded up to `HEAD_DIM`)
- Sliding window attention pattern (`WINDOW_PATTERN = "SSSL"` — short/long alternating)
- Value Embeddings (VE) on alternating layers with gated addition
- RMSNorm-style normalization (inline `norm` function)
- Squared ReLU activation in MLP
- Logit soft-capping at 15.0
- Residual lambdas and x0 skip connections
- Training in bfloat16, optimizer state in float32

## Literature Consultation

Agents should actively consult recent ML research (arXiv, ICML, NeurIPS, ICLR) to source experiment ideas. Notes are saved to `literature/` (one file per paper). See `program.md` for the full protocol.
