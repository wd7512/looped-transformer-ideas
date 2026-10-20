# Overnight task: Rule 30 memory-arm experiment (idea B)

Written 20 October 2026. Budget: about 8 hours of wall-clock on one laptop. Output is judged at dawn.

## 0. Purpose and priorities

Question: when each pass of a shared block can look back at earlier passes, which memory shape lets the model keep iterating correctly beyond its training depth?

Priorities, in order:
1. A **valid baseline.** A carry-last (no-memory) model that reproduces the published behaviour. Nothing else is interpretable without it.
2. A **correct comparison** of memory arms against that baseline, using the pre-registered rules in section 5.
3. **Honest reporting** of what finished, what did not, and every deviation from the paper-faithful setup.

Do not trade a valid result for more arms. A clean negative on the gate is a good outcome.

## 1. Fixed configuration (paper-faithful)

Reproduction target is the Rule 30 experiment in the LSTM-UT preprint (arXiv:2609.19521), R4 setting. Identical for every arm:

- Rule 30 on a **periodic ring**, 64 cells, initial rows Bernoulli(0.5).
- Model: width/dim 64, 4 heads, FFN 256, **rotary positions (RoPE)**, bidirectional attention, one shared block applied repeatedly. No iteration index anywhere. Input enters only at the start (`h_0 = embed(x0)`).
- Training depth N = 4. Loss: mean BCE over iterations 1-4 (every iteration decoded with the same decoder, scored against the exact CA row at that depth).
- AdamW, lr 1e-3, weight decay 0.15, grad clip 0.9, batch 128, 5000 steps.
- **Fixed pool of 100k training rows** (not fresh rows) for gate and arm runs.
- Evaluation: **4096 fixed rows** per seed (256 is too few to resolve 0.999), per-depth bit accuracy at **every depth 1-32** with the same decoder.
- Seeds: 0, 1, 2 for the sweep. Same data pool and config across arms; only the memory differs.

Paper reference numbers (carry-last, trained to depth 4, 3 seeds): about 0.9997 at depth 4, 0.9974 at depth 6, 0.93 (SD 0.13) at depth 9. Its cached model: about 0.94, 0.61, 0.50 at depths 4, 6, 9.

Known note: at depth 32 on a 64-cell ring the light cone wraps by one cell. Record it in the write-up; do not try to fix it.

## 2. Arms

| Arm | Memory definition | Seeds |
|---|---|---|
| carry-last | No memory: `h_(n+1) = F(h_n)` | 3 |
| K=1 control | Same history-attention module, window of 1 (current state only). Parameter control, adds weights but no information | 1, extend if time |
| K=2 window | History attention over the 2 most recent states | 3 |
| full history | History attention over all past states, per position over time (my variant) | 3 |
| decayed | History attention with a learned recency bias: logits get `+ age * log(beta)`, beta in (0,1) learned, init 0.55, weights renormalised (a true weighted average). If the implementation differs, document how | 3 if time |
| paper-style cache | Queries from current positions attend over all positions at all past repeats (matches the paper's cache) | 2 |

Window K=4 and wider are dropped: with training depth 4 they train identically to full history.

## 3. Run order (about 46 min per run at batch 128, measured)

**Step 0 (optional, timebox 45 minutes).** Measured throughput is roughly 20 GFLOP/s by my rough count, which looks like CPU numpy rather than the GPU. If an MLX (or MPS) port of the same model is easy, time 50 steps. Switch **only if** (a) it is at least 5x faster and (b) parity holds: same init and same batch give the same loss to 1e-4 and gradients within 1e-3 relative. If not, keep numpy and continue. If the backward pass is hand-written, add a finite-difference gradient check for each new memory arm before training.

**Step 1: convergence probe.** carry-last, seed 0, 5000 steps.
- Gate passes if **d1 >= 0.999 and d4 >= 0.99**.
- If it fails, climb the ladder, stopping at the first pass: 10k steps, then 20k steps, then an lr / weight-decay check, then **stop and write up the negative result**. Do not run arms on a failed baseline.

**Step 2 (only if the gate passed).** In this order:
1. carry-last seeds 1 and 2 (needed for the worst-vs-best comparison).
2. full history, seeds 0-2.
3. K=2, seeds 0-2.
4. K=1 control, seed 0.
5. decayed, seeds 0-2.
6. paper-style cache, seeds 0-1.
Expect about 10 runs tonight; stop starting new runs when less than 50 minutes remain.

Do not run the TinyStories rerun tonight, and do not run anything concurrently that slows these runs.

## 4. Logging and reproducibility

- Save per run: config, seed, git hash, loss curve, and the full per-depth accuracy array for depths 1-32 (JSON or CSV), written when the run ends so partial results survive.
- Never delete or silently drop a failed or odd run; list it.
- Commit code and results as you go. Keep deviations from section 1 in a `DEVIATIONS` list.
- Cheap extra, baseline only: cosine similarity between successive states at depths 1-32 (seed 0). Skip all other dynamics diagnostics tonight.

## 5. Pre-registered rules (do not change after seeing results)

- **Extrapolates:** per-seed accuracy >= 0.9 at depth 12 and >= 0.75 at depth 24, **and** the arm's **worst seed** beats carry-last's **best seed** at depths 12 and 24. Judged only against a baseline that passed the gate.
- Three seeds make every verdict **provisional**; top up to 8 or more seeds for carry-last and the leading arm afterwards, not tonight.
- Report the paper-style cache separately from my full-history variant. Only the paper-style cache tests "the cache degrades as published".
- The 100M language-model green-light is **not decided tonight**. Report the evidence; the decision is made after reading it.

## 6. Dawn deliverables

1. Gate verdict: probe numbers (d1, d4) and which rung of the ladder was reached.
2. Table of per-seed accuracy at depths 1, 4, 6, 9, 12, 24, 32 for every finished run.
3. Worst-vs-best table against carry-last, and the extrapolation verdict for each finished arm.
4. A depth-accuracy chart (depths 1-32, arms overlaid, seed spread shown).
5. List of runs finished, not finished, and an ETA for the rest.
6. The deviations list, including: history attention here is per position over time while the paper's spans all positions (only the paper-style cache matches it).

## 7. Stop conditions

- Gate fails after the full ladder: stop, write the negative.
- Any arm's training diverges (NaN loss): record it, skip to the next arm, report it.
- Machine instability or thermal throttling that doubles step time: finish the current run, report, stop.
