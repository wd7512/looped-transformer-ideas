"""Shared-block loop with memory arms, numpy forward + manual backward.

Task: initial CA row x0 embedded as h_0; iteration n decodes to CA row n.
Arms differ only in the history term inside the shared block:
  carry-last: h = Block(h), no history.
  full:       per-position attention over all past states.
  window:     same attention over the last K states.

Shapes: B batch, W ring width, d model dim, H heads, dh head dim.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-6


@dataclass
class Config:
    width: int = 40
    dim: int = 32
    heads: int = 2
    ffn: int = 64
    arm: str = "carry-last"  # carry-last | full | window
    window: int = 2


def init_params(rng: np.random.Generator, cfg: Config) -> dict[str, np.ndarray]:
    d = cfg.dim

    def xavier(n_in: int, n_out: int) -> np.ndarray:
        lim = float(np.sqrt(6.0 / (n_in + n_out)))
        return rng.uniform(-lim, lim, size=(n_in, n_out))

    return {
        "E": rng.normal(0, 0.1, size=(2, d)),
        "P": rng.normal(0, 0.1, size=(cfg.width, d)),
        "qkv": xavier(d, 3 * d),
        "bo": xavier(d, d),
        "m1": xavier(d, cfg.ffn),
        "m2": xavier(cfg.ffn, d),
        "g1": np.ones(d),
        "g2": np.ones(d),
        "g3": np.ones(d),
        "hq": xavier(d, d),
        "hk": xavier(d, d),
        "hv": xavier(d, d),
        "ho": xavier(d, d),
        "wd": xavier(d, 1).reshape(d),
        "bd": np.zeros(()),
    }


def _rms_forward(x: np.ndarray, g: np.ndarray) -> tuple[np.ndarray, tuple]:
    r = np.sqrt(np.mean(x * x, axis=-1, keepdims=True) + EPS)
    return (x / r) * g, (x, r, g)


def _rms_backward(dy: np.ndarray, cache: tuple) -> tuple[np.ndarray, np.ndarray]:
    x, r, g = cache
    s = np.mean(dy * g * x, axis=-1, keepdims=True)
    dx = (g / r) * (dy - x * s / (r * r))
    dg = np.sum(dy * x / r, axis=tuple(range(dy.ndim - 1)))
    return dx, dg


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / np.sum(e, axis=axis, keepdims=True)


def _softmax_backward(ds: np.ndarray, s: np.ndarray, axis: int = -1) -> np.ndarray:
    return s * (ds - np.sum(ds * s, axis=axis, keepdims=True))


def _split_heads(x: np.ndarray, h: int) -> np.ndarray:
    b, w, d = x.shape
    return x.reshape(b, w, h, d // h).transpose(0, 2, 1, 3)


def _merge_heads(x: np.ndarray) -> np.ndarray:
    b, h, w, dh = x.shape
    return x.transpose(0, 2, 1, 3).reshape(b, w, h * dh)


def _bce_with_logits(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    return np.maximum(logits, 0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))


class LoopedCA:
    def __init__(self, cfg: Config, params: dict[str, np.ndarray]) -> None:
        self.cfg = cfg
        self.p = params

    # -- forward ------------------------------------------------------
    def forward(
        self,
        x0: np.ndarray,
        targets: np.ndarray | None,
        n_iters: int,
    ) -> tuple[np.ndarray, float, list[dict]]:
        p = self.p
        h = p["E"][x0] + p["P"][None, :, :]
        states = [h]
        logits, caches = [], []
        for _ in range(1, n_iters + 1):
            h, cache = self._block(h, states)
            states.append(h)
            logits.append(h @ p["wd"] + p["bd"])
            caches.append(cache)
        logits_arr = np.stack(logits, axis=0)
        if targets is None:
            return logits_arr, float("nan"), caches
        return logits_arr, float(np.mean(_bce_with_logits(logits_arr, targets))), caches

    def _block(
        self, h: np.ndarray, states: list[np.ndarray]
    ) -> tuple[np.ndarray, dict]:
        cfg, p = self.cfg, self.p
        d, dh = cfg.dim, cfg.dim // cfg.heads
        cache: dict = {"h_in": h, "dh": dh}
        r1, c1 = _rms_forward(h, p["g1"])
        qkv = r1 @ p["qkv"]
        q = _split_heads(qkv[..., :d], cfg.heads)
        k = _split_heads(qkv[..., d : 2 * d], cfg.heads)
        v = _split_heads(qkv[..., 2 * d :], cfg.heads)
        s = _softmax(q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh))
        h1 = h + _merge_heads(s @ v) @ p["bo"]
        r2, c2 = _rms_forward(h1, p["g2"])
        f = r2 @ p["m1"]
        a = np.maximum(f, 0)
        h2 = h1 + a @ p["m2"]
        if cfg.arm == "carry-last":
            h3 = h2
        else:
            ctx = states if cfg.arm == "full" else states[-cfg.window :]
            C = np.stack(ctx, axis=0)
            r3, c3 = _rms_forward(h2, p["g3"])
            qh = r3 @ p["hq"]
            kh = C @ p["hk"]
            vh = C @ p["hv"]
            sh = _softmax(np.sum(qh[None] * kh, axis=-1) / np.sqrt(d), axis=0)
            g = np.sum(sh[..., None] * vh, axis=0)
            h3 = h2 + g @ p["ho"]
            cache.update(C=C, c3=c3, r3=r3, qh=qh, kh=kh, vh=vh, sh=sh, g=g)
        cache.update(
            c1=c1,
            r1=r1,
            q=q,
            k=k,
            s=s,
            v=v,
            h1=h1,
            c2=c2,
            r2=r2,
            f=f,
            a=a,
            h2=h2,
            h3=h3,
        )
        return h3, cache

    # -- backward -----------------------------------------------------
    def backward(
        self,
        caches: list[dict],
        dlog: np.ndarray,
        n_iters: int,
        batch: int,
        x0: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """dlog: (N, B, W) decoder-logit grads (already loss-averaged)."""
        cfg = self.cfg
        grads = {k: np.zeros_like(v) for k, v in self.p.items()}
        d = cfg.dim
        n_states = n_iters + 1
        gs = [np.zeros((batch, cfg.width, d)) for _ in range(n_states)]
        for n in range(n_iters, 0, -1):
            dh3 = dlog[n - 1][..., None] * self.p["wd"]  # (B,W,d)
            grads["wd"] += np.sum(
                dlog[n - 1][..., None] * caches[n - 1]["h3"], axis=(0, 1)
            )
            grads["bd"] += np.sum(dlog[n - 1])
            dh3 = dh3 + gs[n]
            dh_in, ctx_grads = self._block_backward(dh3, caches[n - 1], grads)
            gs[n - 1] += dh_in
            if ctx_grads is not None:
                t = ctx_grads.shape[0]
                if cfg.arm == "full":
                    idx = list(range(n - t, n))
                else:
                    idx = list(range(n - t, n))
                for j, i in enumerate(idx):
                    gs[i] += ctx_grads[j]
        # embedding + pos grads from state h_0
        dh0 = gs[0]
        np.add.at(grads["E"], x0, dh0)
        grads["P"] += np.sum(dh0, axis=0)
        return grads

    def _block_backward(
        self, dh3: np.ndarray, c: dict, grads: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray | None]:
        cfg, p = self.cfg, self.p
        d = cfg.dim
        ctx_grads = None
        if cfg.arm == "carry-last":
            dh2 = dh3
        else:
            C = c["C"]
            dgo = dh3
            dg = dgo @ p["ho"].T
            grads["ho"] += np.einsum("bwe,bwf->ef", c["g"], dgo)
            dsh = np.einsum("bwe,tbwe->tbw", dg, c["vh"])
            dvh = c["sh"][..., None] * dg[None]
            ds = _softmax_backward(dsh, c["sh"], axis=0) / np.sqrt(d)
            dqh = np.einsum("tbw,tbwe->bwe", ds, c["kh"])
            dkh = ds[..., None] * c["qh"][None]
            dr3 = dqh @ p["hq"].T
            grads["hq"] += np.einsum("bwd,bwe->de", c["r3"], dqh)
            grads["hk"] += np.einsum("tbwd,tbwe->de", C, dkh)
            grads["hv"] += np.einsum("tbwd,tbwe->de", C, dvh)
            ctx_grads = dkh @ p["hk"].T + dvh @ p["hv"].T
            dr3n, dg3 = _rms_backward(dr3, c["c3"])
            grads["g3"] += dg3
            dh2 = dh3 + dr3n
        da = dh2 @ p["m2"].T
        grads["m2"] += np.einsum("bwf,bwd->fd", c["a"], dh2)
        df = da * (c["f"] > 0)
        grads["m1"] += np.einsum("bwd,bwf->df", c["r2"], df)
        dr2 = df @ p["m1"].T
        dr2n, dg2 = _rms_backward(dr2, c["c2"])
        grads["g2"] += dg2
        dh1 = dh2 + dr2n
        dov_m = dh1 @ p["bo"].T
        grads["bo"] += np.einsum("bwd,bwe->de", _merge_heads(c["s"] @ c["v"]), dh1)
        dov = _split_heads(dov_m, cfg.heads)
        ds_attn = np.einsum("bhwe,bhve->bhwv", dov, c["v"])
        dv = np.einsum("bhwv,bhwe->bhve", c["s"], dov)
        dscores = _softmax_backward(ds_attn, c["s"]) / np.sqrt(c["dh"])
        dq = np.einsum("bhwv,bhve->bhwe", dscores, c["k"])
        dk = np.einsum("bhwv,bhwe->bhve", dscores, c["q"])
        dqkv = np.concatenate([_merge_heads(x) for x in (dq, dk, dv)], axis=-1)
        dr1 = dqkv @ p["qkv"].T
        grads["qkv"] += np.einsum("bwd,bwe->de", c["r1"], dqkv)
        dr1n, dg1 = _rms_backward(dr1, c["c1"])
        grads["g1"] += dg1
        return dh1 + dr1n, ctx_grads
