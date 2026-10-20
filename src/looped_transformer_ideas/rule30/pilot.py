"""Stage-0 pilot: carry-last vs full-cache vs sliding-window on Rule 30.

Trains through depth 4, evaluates 1..12. Five seeds per arm, in-range
checkpoint selection is N/A here (fixed step count, reported as-is).
Appends one JSON line per run to experiments/rule30-pilot/results.jsonl.
"""

from __future__ import annotations

import json
import logging

from looped_transformer_ideas.rule30.train import run

log = logging.getLogger(__name__)

ARMS = ["carry-last", "full", "window"]
SEEDS = [0, 1, 2, 3, 4]
EVAL_DEPTHS = (1, 2, 3, 4, 6, 9, 12)


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    with open("experiments/rule30-pilot/results.jsonl", "a", encoding="utf-8") as f:
        for arm in ARMS:
            for seed in SEEDS:
                res = run(
                    arm,
                    seed,
                    steps=2000,
                    train_depth=4,
                    eval_depths=EVAL_DEPTHS,
                )
                f.write(json.dumps(res) + "\n")
                f.flush()
                log.info("done %s", res)


if __name__ == "__main__":
    main()
