# Agent Instructions

This repo holds looped-transformer ideas, experiment plans, and small
reproducible pilots. Python via `uv`, `ruff`, `ty`, `pytest`.

## Commands

- Install dependencies: `uv sync --dev`
- Format: `uv run ruff format`
- Lint: `uv run ruff check`
- Type check: `uv run ty check`
- Test: `uv run pytest`

## Rules

- Use `uv` for dependency management and command execution.
- Keep changes small and focused.
- Add or update tests for behavior changes.
- Run `uv run ruff check`, `uv run ty check`, and `uv run pytest` before finishing code changes.
- Do not add runtime dependencies unless they are required by package behavior.
- Prefer clear, boring Python over clever abstractions.
- Use `logging` (stdlib) for all output — never `print()`. Configure via `logging.basicConfig()` in entry points.
- Pilots stay numpy-only (run anywhere). MLX work lives in the private training repo, not here.

## Layout

- `docs/ideas/` — literature summaries and experiment plans (claims marked with verification status).
- `experiments/<name>/` — one folder per experiment: README with result, `charts/`, `REPRODUCE.md`.
- `src/looped_transformer_ideas/` — pilot code. One subpackage per pilot.

## Worktrees

- Create worktrees only under `.worktrees/` (`git worktree add .worktrees/<name> -b <branch> origin/main`).
- Before every commit, `git branch --show-current` must match the intended branch.
