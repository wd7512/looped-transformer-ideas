# Looped Transformers: Idea, Literature Status and Option B Experiment Plan

*Summary as of 20 October 2026. Claims come from abstracts, summaries and partial reads of the papers; none of the code has been run. Verify before citing.*

## 1. The idea

A shared middle layer `F_θ` applied N times turns depth into a dynamical system: `h_0 = E(x)`, `h_(n+1) = F_θ(h_n)`, `y = D(h_N)`. Behaviour as N → ∞ depends on the dynamics of `F_θ`. Three variants:

- **A, attention within the middle layer:** `h_(n+1) = h_n + Attn(h_n) + MLP(h_n)`.
- **B, attention over previous iterations:** `h_(n+1) = F_θ(h_n, Attn(h_n, {h_0..h_n}))`. Non-Markovian; memory and compute grow with N.
- **C, input injection:** `h_(n+1) = F_θ(h_n, E(x))`. For fixed x this is an input-dependent autonomous system.

Five limit behaviours considered: fixed point, periodic cycle, bounded non-periodic dynamics, multiple attractors, divergence.

**Mapping to the literature.** A is the standard looped transformer (token-axis attention in the shared block). B is cross-iteration (history) attention. C is input injection (recall, DEQ, Huginn-style). They are orthogonal axes, not alternatives.

## 2. Is it scoped out? Status by option

| Option | Status | Evidence |
|---|---|---|
| A | Not a separate idea | It is the standard layer; only attention variants differ |
| B | Heavily explored, mostly positive | See below |
| C | Explored, contested | See below |

**B in ordinary transformers.** Attention Residuals (Kimi, arXiv:2603.15031) replaces fixed residual accumulation with attention over earlier layer outputs, with a block variant to bound memory. Related: Mixture-of-Depths Attention (2603.15619), DenseFormer, hyper-connections, DeepCrossAttention, Dual Attention Residuals (2607.18730), which reports gains on 0.1B–1B dense and a 7B MoE. These are authors' own claims.

**B inside loops.** Mixed. CoTFormer, Think-at-Hard (2511.08577), RecurTrace (2609.03379), InfiLoop (2610.11570) and LOOM (2610.01153) add history; LSTM-UT (2609.19521) finds unbounded caches hurt depth extrapolation while bounded gated memory helps.

**C.** ResFormer / Value Residual Learning (2410.17897) is the non-looped analogue. In loops, LOOM re-injects the input embedding each loop; other strong looped models use no injection. Parcae (2604.12946) links stability to the spectral norm of injection parameters.

**Does looping itself help?** Only conditionally. A recent survey (Azizi et al., Preprints.org, 15 Oct 2026, not peer-reviewed) reports a dense recurrence is worth roughly r^0.46 of a fresh block at matched compute; it helps when parameters are the limiting resource or with sparse experts. A looped-MoE scaling study reports about 3× active-parameter efficiency from sparsity and about 2× total-parameter efficiency from recurrence on reasoning.

## 3. What looks open (from my searches, not guaranteed)

1. Depth attention combined with weight sharing at language-model scale, with bounded memory and a stability analysis.
2. Limit behaviour beyond fixed points. Fixed points are crowded (FPRM 2606.18206, SFPRM 2610.25646, STARS 2605.26733, Attractor Models 2605.12466, Parcae); cycles, chaos and multiple attractors are barely mapped.
3. How history terms change contraction and spectral conditions.

**Recommendation:** B inside loops with bounded memory plus a dynamics analysis is the strongest bet, but expect competition. Do not pitch A or C alone as novel.

## 4. Option B experiment plan

**Question.** The literature disagrees about which memory shape works, so memory type is the single independent variable. Fix backbone, injection, data and compute.

**Arms.** Carry-last; full cache; sliding window (W = 2, 4, 8); AttnRes-style block summaries with a shared query; InfiLoop-style decayed running summary; LSTM-UT-style gated memory.

### Stage 0: reproduce baselines first

**Rule 30 (LSTM-UT).** Tiny shared GPT-2-style block, width 64, 4 heads, 64-cell rows, bidirectional attention.
- Trained through depth 4: current-state-only ≈ 0.9997 / 0.9974 / 0.93 (SD 0.13) at depths 4 / 6 / 9; cache ≈ 0.94 / 0.61 / 0.50.
- Trained through depth 6: current-state-only ≈ 1.000 / 0.9999 / 0.981 at depths 6 / 9 / 12; cache ≈ 0.9997 / 0.894 / 0.503.
- Caveats: small single-source preprint, no code link seen, high seed variance, and its checkpoint selection used unseen-depth validation (leakage). Select on in-range depths only.

**Sudoku-Extreme / Maze-Hard (TRM codebase, InfiLoop).**
- TRM retrained: 73.6% (reported 74.7%); FPRM 94.2%; InfiLoop 97.9%.
- Memory ablations: uniform weighting over history 38.4% / 40.2%; fixed forward decay 74.0% / 77.9%; learned decay 82.1% / 89.7%; reverse decay 2.0%.
- Carry-last TRM lost 19.7% of solved puzzles after 36 updates; InfiLoop lost none.

**Language-model loop (~100M parameters, Parcae backbone).** Use Parcae's stable parameterisation, otherwise memory effects are confounded with instability. Parcae validation perplexity: 13.59 vs 14.23 at 100M (16 loops); 10.09 vs 10.76 at 350M (8 loops).

**Depth attention without looping (Attention Residuals ablation).** Reproduce the ordering, not the numbers: baseline 1.766, DenseFormer 1.767, full AttnRes 1.737, block 1.746, sliding window 1.764. The original is a 436M-active-parameter MoE on about 88B tokens, too heavy to repeat.

**Open tension to resolve.** Kimi's window barely helps; InfiLoop finds decay essential and uniform weighting poor. Different regimes (16 distinct layers vs hundreds of shared iterations); reconciling them is the scientific question.

### Stage 1: the new experiment

- **Axes:** memory type × training-depth distribution (fixed or sampled), evaluated from 0.5× to at least 32× training depth.
- **Tasks:** Rule 30, then Sudoku/Maze, then the 100M language model with two or three arms. Skip ARC at first (protocol-sensitive).
- **Measures:** accuracy vs depth; solved-then-lost rate; retention after mid-run perturbation; cosine similarity of successive states (collapse signature); attention entropy over lag; fixed-point residual and Lyapunov estimates; cache-reset vs clean-state interventions.

**Hypotheses with direct literature support**
1. Uniform or full history degrades beyond the training depth.
2. Windowing alone is not the fix.
3. History helps in range, so a crossover depth should exist.
4. Decayed or gated memory keeps dynamics bounded.

**New contribution:** label each arm's limit class and test whether it predicts extrapolation.

### Controls and gates

- Report parameters, FLOPs and cache size per arm.
- At least five seeds.
- In-range checkpoint selection only.
- Gate: if a baseline misses tolerance (about 2 points on TRM), fix it before continuing. A failed Rule 30 reproduction is itself a finding.

## 5. Open decision

Compute budget (one GPU for weeks vs larger) determines whether the 100M language-model stage is realistic.

## 6. Reading queue (titles seen, not read)

- Learning to Compute Is Not Learning to Retain or Reuse in Looped Transformers
- Looped Reasoning Finishes Earlier Than You Think
- Monitoring latent reasoning in looped language models
- Think Shallow, Solve Deep: Controlling Recurrent Dynamics for Reliable Test-Time Depth
- Simply Stabilizing the Loop via Fully Looped Transformer
- Stability and Generalization in Looped Transformers
