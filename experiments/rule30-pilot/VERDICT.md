# Rule 30 verdict: fixed-data sweep (2026-10-21/22)

Data fix: `step()` rolled the flattened batch; now `axis=-1` with a
regression test. All below is paper-faithful, unpadded, 5000 steps,
batch 128, AdamW lr 1e-3 / wd 0.15 / clip 0.9, fixed 100k pool.

## Gate

carry-last s0: d1 1.0 (seam 1.0), d4 0.99994 (seam 0.998). GATE PASSED.
s1/s2 confirm: d1-d4 1.0 on all three seeds.

## Sweep (d1 / d4)

- carry-last (3 seeds): 1.0 / ~1.0 on all three.
- full (3): s1, s2 match at 1.0; s0 degrades to 0.996 / 0.992.
- window K=2 (3): s1 matches; s0 1.0 / 0.995; s2 0.996 / 0.995.
- window K=1 (1): 0.9997 / 0.951 — worst d4 of the sweep.
- decayed (3): s2 matches; s1 0.9996 / 0.997; s0 0.9938 / 0.991.
- papercache (2): s0 1.0 / 0.9999, s1 1.0 / 1.0 — matches carry-last.

Extrapolation: every arm is at chance from d9 on. Best mid-range:
carry-last s2 d6 0.90, window s1 d6 0.91. Nobody reaches 0.9 @ d12.

## Verdict (pre-registered rules)

Saturation clause did not trigger (carry-last collapses past d8, so arms
can be separated on extrapolation — but none do). Worst-seed-beats-best:
no arm beats carry-last anywhere. The result is harm-only, and the harm
is variance, not ordering: every non-baseline arm degrades on a minority
of seeds (full 1/3, K=2 2/3, decayed 2/3) while matching otherwise.
Decay does not prevent the wobble. K=1 hurting most cuts against a simple
"history degrades" story. The paper's full-history degradation reproduces
in direction (full s0) but not as a stable effect. Benefit question moves
to delayed recall or Sudoku, as pre-registered.

## Dynamics finding (pre-registered prediction confirmed)

With the rule learned exactly (d1 1.0 everywhere), successive-state cosine
similarity still climbs to 0.9999 by ~depth 26 (carry-last s2, 512 rows).
True Rule 30 is chaotic — distinct rows should not merge. This is a
genuine contraction beyond the training range, not a training artefact.
runs/carry-last-fixed_s2_cossim.json holds the 1-32 series.
