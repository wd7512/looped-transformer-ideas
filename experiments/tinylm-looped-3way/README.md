# TinyLM looped 3-way — hardware proof (MacBook Air M5, 24GB)

Source repo: overnight-llm (private work log). This folder is the portable proof.

Question: same effective depth (8 block applications), fewer distinct weights — what happens to loss and speed on one laptop GPU?

## Result
- Base 8x1, 49.9M: valid ppl 3.77. 21h45 to 12k steps. Infer ~124 tok/s.
- Looped 4x2, 33.2M: valid ppl 3.96 (+0.19). ~10h20. Infer ~234 tok/s.
- Looped 2x4, 24.8M: valid ppl 4.21 (+0.44). ~10h. Infer ~236 tok/s.

Same TinyStories data, same seed to step 1000, same eval. Train throughput ~8k tok/s looped vs ~7k base; inference ~2x (fewer weights to move). Valid ranking stable from step 2000: more distinct blocks wins. Full curves in charts/.

## Hardware
MacBook Air M5, 10-core, 24GB unified, MLX GPU (Metal, Device gpu/0), Python 3.11, no CUDA, no torch. Checkpoints: 191MB / 127MB / 95MB.

## Caveat
Looped runs resumed at step 1000 (pilot to full, weights-only, fresh optimizer). Causes a train bump / valid flat spot 1000-1500. Base ran straight through. Costs ~500 steps, ranking unaffected. Fix resume to carry optimizer state before citing closely.
