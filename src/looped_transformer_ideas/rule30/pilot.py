"""Overnight runner: gate probe, ladder, then arms in task order.

Idempotent: reads experiments/rule30-pilot/results.jsonl at startup and
skips finished (arm, seed, steps) triples. Each finished run is also
saved to experiments/rule30-pilot/runs/<arm>_s<seed>.json.
Gate: carry-last seed 0 at 5000 steps needs d1>=0.999 and d4>=0.99.
Ladder on failure: 10k, 20k, lr/wd check (lr 3e-4, wd 0.05), then stop.
Stops starting new runs when under 50 minutes remain.
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import subprocess
import time

import numpy as np

from looped_transformer_ideas.rule30.ca import random_rows
from looped_transformer_ideas.rule30.model import Config, LoopedCA
from looped_transformer_ideas.rule30.train import (
    EVAL_ROWS,
    run,
    state_cosine_sims,
)

log = logging.getLogger(__name__)

OUTDIR = "experiments/rule30-pilot"
RESULTS = os.path.join(OUTDIR, "results.jsonl")
RUNS = os.path.join(OUTDIR, "runs")
MIN_TIME_LEFT = 50 * 60

# (arm, seed, steps, extra-kwargs)
SWEEP: list[tuple] = [
    ("carry-last", 1, 5000, {}),
    ("carry-last", 2, 5000, {}),
    ("full", 0, 5000, {}),
    ("full", 1, 5000, {}),
    ("full", 2, 5000, {}),
    ("window", 0, 5000, {"window": 2}),
    ("window", 1, 5000, {"window": 2}),
    ("window", 2, 5000, {"window": 2}),
    ("window", 0, 5000, {"window": 1}),
    ("decayed", 0, 5000, {}),
    ("decayed", 1, 5000, {}),
    ("decayed", 2, 5000, {}),
    ("papercache", 0, 5000, {}),
    ("papercache", 1, 5000, {}),
]


def git_hash() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception:
        return "unknown"


def finished_keys() -> set:
    keys = set()
    try:
        with open(RESULTS, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                keys.add(
                    (r.get("arm"), r.get("seed"), r.get("steps"), r.get("window", 2))
                )
    except FileNotFoundError:
        pass
    return keys


def run_name(arm: str, seed: int, window: int) -> str:
    extra = "_k1" if arm == "window" and window == 1 else ""
    return f"{arm}_s{seed}{extra}"


def save(res: dict) -> None:
    res = dict(res)
    res["git"] = git_hash()
    os.makedirs(RUNS, exist_ok=True)
    name = run_name(
        str(res.get("arm")), int(str(res.get("seed"))), int(str(res.get("window", 2)))
    )
    with open(os.path.join(RUNS, f"{name}.json"), "w", encoding="utf-8") as f:
        json.dump(res, f)
    with open(RESULTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(res) + "\n")


def do_run(arm: str, seed: int, steps: int, kw: dict) -> dict:
    window = kw.get("window", 2)
    name = run_name(arm, seed, window)
    os.makedirs(RUNS, exist_ok=True)
    res = run(
        arm, seed, steps=steps, weights_path=os.path.join(RUNS, name + ".pkl"), **kw
    )
    save(res)
    return res


def cosine_extra() -> None:
    """Baseline seed-0 successive-state cosine sims at depths 1-32."""
    name = run_name("carry-last", 0, 2)
    pkl_path = os.path.join(RUNS, name + ".pkl")
    try:
        with open(pkl_path, "rb") as f:
            params = pickle.load(f)
    except Exception as e:
        log.info("cosine-sim extra skipped: %s", e)
        return
    cfg = Config(arm="carry-last")
    model = LoopedCA(cfg, params)
    eval_x0 = random_rows(np.random.default_rng(9000), EVAL_ROWS, cfg.width)
    sims = state_cosine_sims(model, eval_x0[:512], 32)
    with open(os.path.join(RUNS, name + "_cossim.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "arm": "carry-last",
                "seed": 0,
                "cosine_sims": [round(s, 5) for s in sims],
            },
            f,
        )
    log.info("cosine sims d1..d32: %s", [round(s, 3) for s in sims[:8]])


def gate_ok(res: dict) -> bool:
    if res.get("diverged"):
        return False
    acc = res.get("acc", [])
    return len(acc) >= 4 and acc[0] >= 0.999 and acc[3] >= 0.99


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    t_end = time.time() + 8 * 3600
    done = finished_keys()

    # Step 1: gate probe (or reuse if already finished)
    probe_key = ("carry-last", 0, 5000, 2)
    if probe_key not in done:
        res = do_run("carry-last", 0, 5000, {})
        done.add(probe_key)
        log.info("probe: %s", {k: res.get(k) for k in ("train_loss", "diverged")})
        if not res.get("diverged"):
            log.info("probe acc d1=%.5f d4=%.5f", res["acc"][0], res["acc"][3])
    else:
        log.info("probe already finished, skipping")

    # Re-read probe result for the gate decision
    probe = None
    with open(RESULTS, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if (r.get("arm"), r.get("seed"), r.get("steps")) == ("carry-last", 0, 5000):
                probe = r
    if probe is None or not gate_ok(probe):
        ladder = [(10000, {}), (20000, {}), (5000, {"lr": 3e-4, "wd": 0.05})]
        for steps, kw in ladder:
            key = ("carry-last", 0, steps, 2)
            if key in done:
                continue
            if time.time() > t_end - MIN_TIME_LEFT:
                break
            res = do_run("carry-last", 0, steps, kw)
            res["ladder_kw"] = kw
            save(res)
            done.add(key)
            if gate_ok(res):
                probe = res
                break
        else:
            log.info("gate ladder exhausted without pass")
        if probe is None or not gate_ok(probe):
            log.info("GATE FAILED - stopping, no arms")
            return

    log.info("GATE PASSED")
    cosine_extra()

    for arm, seed, steps, kw in SWEEP:
        window = kw.get("window", 2)
        if (arm, seed, steps, window) in done:
            continue
        if time.time() > t_end - MIN_TIME_LEFT:
            log.info("under 50 min remain - stopping")
            break
        res = do_run(arm, seed, steps, kw)
        done.add((arm, seed, steps, window))
        if res.get("diverged"):
            log.info(
                "DIVERGED %s s%d at step %s - skipping arm rest",
                arm,
                seed,
                res.get("diverge_step"),
            )
            continue
        log.info(
            "done %s s%d loss %.4f d1 %.4f d4 %.4f",
            arm,
            seed,
            res["train_loss"],
            res["acc"][0],
            res["acc"][3],
        )


if __name__ == "__main__":
    main()
