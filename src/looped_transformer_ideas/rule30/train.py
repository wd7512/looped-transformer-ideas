"""Train the shared-block loop on Rule 30 rollout prediction.

Fresh random rows every step (infinite data). Supervises iterations 1..N
against CA rows 1..N. Prints one JSON line with train loss and held-out
bit accuracy at each eval depth.
"""

from __future__ import annotations

import argparse
import json
import logging

import numpy as np

from looped_transformer_ideas.rule30.ca import random_rows, rollout
from looped_transformer_ideas.rule30.model import Config, LoopedCA, init_params

log = logging.getLogger(__name__)


def adam_step(
    params: dict[str, np.ndarray],
    grads: dict[str, np.ndarray],
    state: dict[str, np.ndarray],
    lr: float,
    t: int,
) -> None:
    b1, b2, eps = 0.9, 0.999, 1e-8
    for k in params:
        g = grads[k]
        m = state["m_" + k] = b1 * state["m_" + k] + (1 - b1) * g
        v = state["v_" + k] = b2 * state["v_" + k] + (1 - b2) * g * g
        mh = m / (1 - b1**t)
        vh = v / (1 - b2**t)
        params[k] -= lr * mh / (np.sqrt(vh) + eps)


def bit_accuracy(logits: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((logits > 0).astype(np.int64) == targets))


def run(
    arm: str,
    seed: int,
    steps: int = 400,
    train_depth: int = 4,
    batch: int = 256,
    lr: float = 1e-2,
    eval_depths: tuple[int, ...] = (4, 6, 9, 12),
    window: int = 2,
) -> dict:
    rng = np.random.default_rng(seed)
    cfg = Config(arm=arm, window=window)
    params = init_params(rng, cfg)
    model = LoopedCA(cfg, params)
    state = {"m_" + k: np.zeros_like(v) for k, v in params.items()}
    state.update({"v_" + k: np.zeros_like(v) for k, v in params.items()})

    eval_rng = np.random.default_rng(10_000 + seed)
    eval_x0 = random_rows(eval_rng, 256, cfg.width)

    for t in range(1, steps + 1):
        x0 = random_rows(rng, batch, cfg.width)
        rows = rollout(x0, train_depth)
        logits, loss, caches = model.forward(x0, rows[1:], train_depth)
        sig = 1.0 / (1.0 + np.exp(-logits))
        dlog = (sig - rows[1:]) / rows[1:].size
        grads = model.backward(caches, dlog, train_depth, batch, x0)
        adam_step(params, grads, state, lr, t)
        if t % 100 == 0:
            log.info("step %d loss %.4f", t, loss)

    out: dict = {"arm": arm, "seed": seed, "train_loss": loss}
    max_d = max(eval_depths)
    eval_rows = rollout(eval_x0, max_d)
    logits, _, _ = model.forward(eval_x0, None, max_d)
    for d in eval_depths:
        out[f"acc_d{d}"] = round(bit_accuracy(logits[d - 1], eval_rows[d]), 4)
    return out


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm", default="carry-last", choices=["carry-last", "full", "window"]
    )
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=400)
    ap.add_argument("--train-depth", type=int, default=4)
    ap.add_argument("--window", type=int, default=2)
    args = ap.parse_args()
    print(
        json.dumps(
            run(args.arm, args.seed, args.steps, args.train_depth, window=args.window)
        )
    )


if __name__ == "__main__":
    main()
