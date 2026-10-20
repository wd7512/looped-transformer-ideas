# Deviations from the paper-faithful setup

Recorded during the overnight run. The task's fixed configuration
(overnight_task.md section 1) holds except where noted here.

1. Step 0 outcome: MLX GPU port verified exact against numpy in float64
   (loss diff 4e-9, all grads rel err < 1e-3). In float32 the same port
   differs by ~1.5e-3 absolute on the loss, exceeding the 1e-4 parity bar,
   so numpy was retained per the pre-registered rule. MLX f32 measured
   0.021 s/step vs numpy 0.55 s/step (26x) at batch 128, dim 64. The port
   is proven and available for the seed top-up; the parity bar as written
   cannot pass in float32 for this model.
2. Norm: RMSNorm pre-norm throughout. The paper says "GPT-2-style block";
   GPT-2 uses LayerNorm. Chose not to churn the grad-checked norm.
3. FFN activation is ReLU. Paper-era GPT-2 uses GELU; the difference is
   minor for this task and ReLU keeps the backward simple.
4. My full-history / window / decayed arms attend per position over time
   only. The paper's cache spans all positions at all past repeats. Only
   the paper-style cache arm matches it; the others test a narrower
   variant and are reported as such.
5. History attention here is single-head with full-dim projections; the
   paper-style cache arm is multi-head (4 heads), matching the backbone.
6. Eval rows are 4096 fixed rows per seed (same count as paper). Training
   pool is 100k fixed rows shared across arms and seeds; batch sampling
   order differs per run (seeded by arm/seed).
7. At depth 32 on the 64-cell ring the light cone wraps by one cell.
   Recorded, not fixed, per the task.
