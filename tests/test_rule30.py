from __future__ import annotations

import numpy as np

from looped_transformer_ideas.rule30.ca import random_rows, rollout, step
from looped_transformer_ideas.rule30.model import (
    Config,
    LoopedCA,
    init_params,
    pad_wrap,
)
from looped_transformer_ideas.rule30.train import bit_accuracy, run


def test_rule30_single_one() -> None:
    row = np.zeros(11, dtype=np.int64)
    row[5] = 1
    out = step(row)
    assert out.tolist() == [0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0]


def test_rollout_shape_and_first_row() -> None:
    rng = np.random.default_rng(0)
    x0 = random_rows(rng, 3, 16)
    rows = rollout(x0, 4)
    assert rows.shape == (5, 3, 16)
    assert bool((rows[0] == x0).all())


def test_step_wraps_within_each_row() -> None:
    # Regression: np.roll without axis flattens (B, W) and leaks cells
    # across rows, corrupting the seam targets. Roll must be per-row.
    rng = np.random.default_rng(5)
    x0 = random_rows(rng, 4, 8)
    rows = rollout(x0, 1)
    for b in range(4):
        assert bool((rows[1, b] == step(x0[b])).all())
    # seam target of row b depends only on row b
    for b in range(4):
        lf, c, r = x0[b, 7], x0[b, 0], x0[b, 1]
        assert rows[1, b, 0] == (lf ^ (c | r))
        lf, c, r = x0[b, 6], x0[b, 7], x0[b, 0]
        assert rows[1, b, 7] == (lf ^ (c | r))


def test_forward_shapes() -> None:
    rng = np.random.default_rng(1)
    cfg = Config(width=16, dim=8, heads=2, ffn=16, arm="window", window=2)
    model = LoopedCA(cfg, init_params(rng, cfg))
    x0 = random_rows(rng, 4, 16)
    rows = rollout(x0, 3)
    logits, loss, caches = model.forward(x0, rows[1:], 3)
    assert caches is not None
    assert logits.shape == (3, 4, 16)
    assert loss > 0 and np.isfinite(loss)
    assert len(caches) == 3


def _numeric_grads(cfg: Config, seed: int) -> None:
    rng = np.random.default_rng(seed)
    params = {k: v.astype(np.float64) for k, v in init_params(rng, cfg).items()}
    model = LoopedCA(cfg, params)
    xrng = np.random.default_rng(seed + 1)
    x0 = pad_wrap(random_rows(xrng, 2, cfg.width), cfg.pad)
    rows = rollout(x0[:, cfg.pad : cfg.pad + cfg.width], 2).astype(np.float64)

    logits, loss, caches = model.forward(x0, rows[1:], 2)
    assert caches is not None
    sig = 1.0 / (1.0 + np.exp(-logits))
    dlog = (sig - rows[1:]) / rows[1:].size
    grads = model.backward(caches, dlog, 2, 2, x0)

    eps = 1e-5
    for k, v in params.items():
        num = np.zeros_like(v)
        it = np.nditer(v, flags=["multi_index"])
        while not it.finished:
            ix = it.multi_index
            old = v[ix]
            v[ix] = old + eps
            _, lp, _ = model.forward(x0, rows[1:], 2)
            v[ix] = old - eps
            _, lm, _ = model.forward(x0, rows[1:], 2)
            v[ix] = old
            num[ix] = (lp - lm) / (2 * eps)
            it.iternext()
        denom = np.maximum(1e-8, np.abs(num) + np.abs(grads[k]))
        rel = float(np.max(np.abs(num - grads[k]) / denom))
        assert rel < 1e-4, (k, rel)


def test_backward_carry_last() -> None:
    _numeric_grads(Config(width=7, dim=4, heads=1, ffn=8, arm="carry-last"), 0)


def test_backward_window() -> None:
    _numeric_grads(Config(width=7, dim=4, heads=1, ffn=8, arm="window", window=2), 3)


def test_backward_full() -> None:
    _numeric_grads(Config(width=7, dim=4, heads=1, ffn=8, arm="full"), 5)


def test_backward_decayed() -> None:
    _numeric_grads(Config(width=7, dim=4, heads=1, ffn=8, arm="decayed"), 9)


def test_backward_papercache() -> None:
    _numeric_grads(Config(width=7, dim=4, heads=1, ffn=8, arm="papercache"), 11)


def test_backward_padded_carry_last() -> None:
    _numeric_grads(Config(width=7, dim=4, heads=1, ffn=8, arm="carry-last", pad=1), 21)


def test_backward_padded_full() -> None:
    _numeric_grads(Config(width=7, dim=4, heads=1, ffn=8, arm="full", pad=1), 22)


def test_pad_refresh_and_decode_slice() -> None:
    rng = np.random.default_rng(23)
    cfg = Config(width=7, dim=4, heads=1, ffn=8, arm="carry-last", pad=1)
    model = LoopedCA(cfg, init_params(rng, cfg))
    x0 = pad_wrap(random_rows(rng, 2, 7), 1)
    assert x0.shape == (2, 9)
    xu = x0[:, 1:8]
    rows = rollout(xu, 2)
    logits, loss, caches = model.forward(x0, rows[1:], 2)
    assert logits.shape == (2, 2, 7)  # decoded on real cells only
    assert np.isfinite(loss)
    assert caches is not None
    states = model.forward_states(x0, 2)
    for h in states[1:]:
        assert bool((h[:, 0, :] == h[:, 7, :]).all())  # left pad = cell 6
        assert bool((h[:, 8, :] == h[:, 1, :]).all())  # right pad = cell 0
    # full-width logits equal the sliced real-cell columns
    lg_full = states[2] @ model.p["wd"] + model.p["bd"]
    assert bool((logits[1] == lg_full[:, 1:8]).all())


def test_rope_position_zero_is_identity() -> None:
    from looped_transformer_ideas.rule30.model import (
        _rope_transform,
        rope_tables,
    )

    rng = np.random.default_rng(2)
    q = rng.normal(size=(2, 1, 7, 4))
    rope = rope_tables(7, 4)
    qr = _rope_transform(q, rope.cos, rope.sin)
    assert np.allclose(qr[:, :, 0, :], q[:, :, 0, :])


def test_rope_preserves_norm() -> None:
    from looped_transformer_ideas.rule30.model import (
        _rope_transform,
        rope_tables,
    )

    rng = np.random.default_rng(3)
    q = rng.normal(size=(2, 1, 7, 4))
    rope = rope_tables(7, 4)
    qr = _rope_transform(q, rope.cos, rope.sin)
    assert np.allclose(np.linalg.norm(qr, axis=-1), np.linalg.norm(q, axis=-1))


def test_window_one_control_shapes() -> None:
    rng = np.random.default_rng(4)
    cfg = Config(width=16, dim=8, heads=2, ffn=16, arm="window", window=1)
    model = LoopedCA(cfg, init_params(rng, cfg))
    x0 = random_rows(rng, 4, 16)
    logits, loss, caches = model.forward(x0, None, 3, train=False)
    assert logits.shape == (3, 4, 16)
    assert np.isnan(loss)
    assert caches is None


def test_smoke_training_learns() -> None:
    res = run("carry-last", seed=0, steps=100, train_depth=1, batch=64)
    assert res["train_loss"] < 0.66  # chance is 0.6931
    assert bit_accuracy(np.zeros((2, 3)), np.zeros((2, 3))) == 1.0
