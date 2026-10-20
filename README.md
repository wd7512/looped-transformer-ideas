# looped-transformer-ideas

Ideas, plans, and small reproducible pilots around looped (weight-shared)
transformers: does attention over past loop iterations help depth
extrapolation, and which memory shape works?

## Layout

- `docs/ideas/` — literature summaries and experiment plans.
- `experiments/<name>/` — results, charts, and reproduce notes per experiment.
- `src/looped_transformer_ideas/` — pilot code (numpy-only, runs anywhere).

## Quickstart

```bash
uv sync --dev
uv run pytest
```

## Status

- `docs/ideas/looped-transformer-option-b-summary.md` — idea B literature status (claims from abstracts/summaries, verify before citing).
- `experiments/tinylm-looped-3way/` — M5 hardware proof: base 8x1 vs looped 4x2 vs 2x4 on TinyStories (run in the private training repo).
- `src/looped_transformer_ideas/rule30/` — Rule 30 depth-extrapolation pilot (in progress).
