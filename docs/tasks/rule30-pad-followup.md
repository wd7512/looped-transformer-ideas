# Rule 30 follow-up: circular-pad variant (deviation A)

Written before any pad-variant results. Paper-faithful failure (d1 frozen
at 0.988, all error at seam cells 0/63, interior d1 exactly 1.0) is kept as
a separate reported result; see experiments/rule30-pilot/README.md.

## Working baseline

Documented circular-padding variant: input rows are wrap-padded with one
cell each side (width 64 -> 66), pads refreshed from the real edge states
at every iteration (not once), pad grads accumulated back into the edge
cells (finite-difference checked). History arms store padded states;
identical rule for every arm. Decode and loss on the 64 real cells only.
Label: every pad-variant result carries `"pad": 1`.

Known wart (recorded, not fixed): pad cells act as extra keys, so the two
edge cells appear twice in each softmax. Interior was already exact, so
expected harmless; revisit only if seam cells misbehave under padding.

## Standing eval

Every run reports per-position accuracy at d1 and d4 (`acc_pos`): overall,
seam (cells 0 and 63), interior (1-62). Seam is judged separately.

## Probe then gate

1. N=1 probe (train depth 1, pad 1). Passes if every position, seam
   included, reaches at least 0.999.
2. Only then: full gate rerun at N=4 (d1 >= 0.999, d4 >= 0.99),
   everything else unchanged.

## Pre-registered arm rules (before seeing any arm results)

- Saturation. If the fixed carry-last holds at least 0.99 through depth
  32, Rule 30 cannot separate the arms; nothing can beat the baseline.
  The arm verdict then becomes "no worse than carry-last" (harm only):
  does full history degrade as the paper reports, and does decay
  prevent it? The benefit question moves to delayed recall or Sudoku.
- Similarity prediction. If the rule is learned exactly, successive-state
  cosine similarity should stop collapsing to 1.0 by ~depth 20, because
  the true dynamics are chaotic. If it still collapses while d1 sits at
  1.0, that is a genuine contraction beyond the training range and
  becomes the real dynamics finding.
