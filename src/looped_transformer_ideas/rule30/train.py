"""Train the shared-block loop on Rule 30 rollout prediction.

Paper-faithful R4 setting: fixed pool of rows, every iteration 1..N
supervised against CA rows 1..N with the same decoder. AdamW + grad clip.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle

import numpy as np

from looped_transformer_ideas.rule30.ca import random_rows, rollout
from looped_transformer_ideas.rule30.model import Config, LoopedCA, init_params

log = logging.getLogger(__name__)

POOL_ROWS = 100_000
POOL_SEED = 20261020
EVAL_ROWS = 4096


def adamw_step(
    params: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    state: dict[str, np.ndarray],
    lr: float,
    wd: float,
    clip: float,
    t: int,
) -> None:
    b1, b2, eps = 0.9, 0.999, 1e-8
    total = float(sum(float(np.sum(g * g)) for g in grads.values()))
    scale = min(1.0, clip / (np.sqrt(total) + 1e-12))
    for k in params:
        g = grads[k] * scale
        m = state["m_" + k] = b1 * state["m_" + k] + (1 - b1) * g
        v = state["v_" + k] = b2 * state["v_" + k] + (1 - b2) * g * g
        mh = m / (1 - b1**t)
        vh = v / (1 - b2**t)
        params[k] -= lr * (wd * params[k] + mh / (np.sqrt(vh) + eps))


def bit_accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((logits > 0).astype(np.int64) == targets))


def eval_depths(
    model: LoopedCA, eval_x0: np.ndarray, max_depth: int, chunk: int = 512
) -> list[float]:
    """Per-depth bit accuracy for depths 1..max_depth, chunked."""
    accs: list[float] = []
    for s in range(0, len(eval_x0), chunk):
        x = eval_x0[s : s + chunk]
        rows = rollout(x, max_depth)
        logits, _, _ = model.forward(x, None, max_depth, train=False)
        hits = (logits > 0).astype(np.int64) == rows[1:]
        accs.append(hits.mean(axis=(1, 2)))
    return [float(v) for v in np.mean(accs, axis=0)]


def state_cosine_sims(model: LoopedCA, x0: np.ndarray, max_depth: int) -> list[float]:
    """Cosine similarity between successive states h_{n-1}, h_n."""
    states = model.forward_states(x0, max_depth)
    sims = []
    for a, b in zip(states[:-1], states[1:]):
        an = a / (np.linalg.norm(a, axis=-1, keepdims=True) + 1e-12)
        bn = b / (np.linalg.norm(b, axis=-1, keepdims=True) + 1e-12)
        sims.append(float(np.mean(np.sum(an * bn, axis=-1))))
    return sims


def run(
    arm: str,
    seed: int,
    steps: int = 5000,
    train_depth: int = 4,
    batch: int = 128,
    lr: float = 1e-3,
    wd: float = 0.15,
    clip: float = 0.9,
    max_eval_depth: int = 32,
    window: int = 2,
    eval_rows: int = EVAL_ROWS,
    log_every: int = 100,
    weights_path: str | None = None,
) -> dict:
    rng = np.random.default_rng(seed)
    cfg = Config(arm=arm, window=window)
    params = init_params(rng, cfg)
    model = LoopedCA(cfg, params)
    state = {"m_" + k: np.zeros_like(v) for k, v in params.items()}
    state.update({"v_" + k: np.zeros_like(v) for k, v in params.items()})

    pool = random_rows(np.random.default_rng(POOL_SEED), POOL_ROWS, cfg.width)
    eval_x0 = random_rows(np.random.default_rng(9000 + seed), eval_rows, cfg.width)

    curve: list[float] = []
    loss = float("nan")
    for t in range(1, steps + 1):
        idx = rng.integers(0, POOL_ROWS, size=batch)
        x0 = pool[idx]
        rows = rollout(x0, train_depth)
        logits, loss, caches = model.forward(x0, rows[1:], train_depth)
        assert caches is not None
        if not np.isfinite(loss):
            return {
                "arm": arm,
                "seed": seed,
                "diverged": True,
                "diverge_step": t,
            }
        sig = 1.0 / (1.0 + np.exp(-logits))
        dlog = (sig - rows[1:]) / rows[1:].size
        grads = model.backward(caches, dlog, train_depth, batch, x0)
        adamw_step(params, grads, state, lr, wd, clip, t)
        if t % log_every == 0:
            curve.append(round(loss, 6))
            log.info("step %d loss %.4f", t, loss)

    out: dict = {
        "arm": arm,
        "seed": seed,
        "steps": steps,
        "train_depth": train_depth,
        "batch": batch,
        "lr": lr,
        "wd": wd,
        "clip": clip,
        "window": window,
        "train_loss": loss,
        "loss_curve": curve,
        "acc": [round(a, 5) for a in eval_depths(model, eval_x0, max_eval_depth)],
    }
    if weights_path is not None:
        with open(weights_path, "wb") as f:
            pickle.dump({k: np.asarray(v) for k, v in params.items()}, f)
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm",
        default="carry-last",
        choices=["carry-last", "window", "full", "decayed", "papercache"],
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=5000)
    ap.add_argument("--train-depth", type=int, default=4)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.15)
    ap.add_argument("--clip", type=float, default=0.9)
    ap.add_argument("--window", type=int, default=2)
    args = ap.parse_args()
    print(
        json.dumps(
            run(
                args.arm,
                args.seed,
                steps=args.steps,
                train_depth=args.train_depth,
                batch=args.batch,
                lr=args.lr,
                wd=args.wd,
                clip=args.clip,
                window=args.window,
            )
        )
    )


if __name__ == "__main__":
    main()
