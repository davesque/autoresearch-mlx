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
- **`train.py`** — The ONLY file the agent edits. Contains the GPT model (with RoPE, Peri-LN, value embeddings, sparse attention gate, residual lambdas), a hybrid Muon/AdamW optimizer, hyperparameter constants, and the training loop. All hyperparameters are module-level constants (no CLI flags).

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
- Full causal attention on all layers (`WINDOW_PATTERN = "LLLL"`)
- Value Embeddings (VE) on all layers with gated addition (gate uses 128 input channels)
- Peri-LN normalization (pre-norm and post-norm on each sub-layer, using `mx.fast.rms_norm`)
- Sparse attention gate (per-head sigmoid gate on 12 input dimensions, from nanoGPT speedrun Record 28)
- Squared ReLU activation in MLP (4x expansion)
- Logit soft-capping at 15.0
- Residual lambdas and x0 skip connections
- Training in bfloat16, optimizer state in float32

## Optimizer Details

The optimizer is a hybrid Muon/AdamW:
- **Muon** (Newton-Schulz matrix sign, 5 iterations) with Nesterov momentum for all 2D weight matrices in blocks. Uses beta1=0.9, LR=0.01, weight decay=0.1.
- **AdamW** for embeddings (wte, value_embeds), output head (lm_head), and scalar parameters (resid_lambdas, x0_lambdas). Per-parameter-group learning rates.
- 5% warmup, 25% linear warmdown to zero.

## Literature Consultation

Agents should actively consult recent ML research (arXiv, ICML, NeurIPS, ICLR) to source experiment ideas. Notes are saved to `literature/` (one file per paper). See `program.md` for the full protocol.

## Strategy Knowledge Base

The `strategy/` directory is a persistent decision-support system updated after every experiment. It contains curated analysis (not raw logs) that informs experiment selection:

- **`strategy/learnings.md`** — Hardware/config-specific insights with confidence levels (e.g., "this model is step-count-limited")
- **`strategy/hypotheses.md`** — Prioritized queue of untested experiment ideas with rationale
- **`strategy/near-misses.md`** — Experiments within noise margin that deserve revisiting after config changes
- **`strategy/interactions.md`** — Known parameter couplings (e.g., batch size ↔ LR scaling)

The experiment loop in `program.md` includes a pre-experiment checklist that requires consulting these files and a post-experiment step that requires updating them. See `program.md` for full details.
