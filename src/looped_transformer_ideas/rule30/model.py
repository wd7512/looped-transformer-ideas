"""Shared-block loop with memory arms, numpy forward + manual backward.

Task: initial CA row x0 embedded as h_0; iteration n decodes to CA row n.
Arms differ only in the history term inside the shared block:
  carry-last: h = Block(h), no history.
  window/full: per-position attention over the last K / all past states.
  decayed:    same, plus a learned recency bias age*log(beta) on logits.
  papercache: paper-style cache: queries from current positions attend over
              all positions at all past repeats (multi-head).

Shapes: B batch, W ring width, d model dim, H heads, dh head dim.
Positions are rotary (RoPE) on the token axis; no iteration index anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-6
THETA = 10000.0


@dataclass
class Config:
    width: int = 64
    dim: int = 64
    heads: int = 4
    ffn: int = 256
    arm: str = "carry-last"  # carry-last | window | full | decayed | papercache
    window: int = 2


@dataclass
class RopeCache:
    cos: np.ndarray
    sin: np.ndarray


def init_params(rng: np.random.Generator, cfg: Config) -> dict[str, np.ndarray]:
    d = cfg.dim

    def xavier(n_in: int, n_out: int) -> np.ndarray:
        lim = float(np.sqrt(6.0 / (n_in + n_out)))
        return rng.uniform(-lim, lim, size=(n_in, n_out))

    p = {
        "E": rng.normal(0, 0.1, size=(2, d)),
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
    if cfg.arm == "decayed":
        # beta = sigmoid(lb); init beta = 0.55
        p["lb"] = np.full((), float(np.log(0.55 / 0.45)))
    if cfg.arm == "papercache":
        p["g4"] = np.ones(d)
        p["pc_q"] = xavier(d, d)
        p["pc_k"] = xavier(d, d)
        p["pc_v"] = xavier(d, d)
        p["pc_o"] = xavier(d, d)
    return p


def rope_tables(width: int, dh: int) -> RopeCache:
    i = np.arange(dh // 2)
    freq = THETA ** (-2.0 * i / dh)
    ang = np.arange(width)[:, None] * freq[None, :]
    return RopeCache(cos=np.cos(ang), sin=np.sin(ang))


def _rope_transform(x: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    """Rotate pairs of last-dim features. x: (..., W, dh)."""
    xe = x[..., 0::2]
    xo = x[..., 1::2]
    c = cos[..., : xe.shape[-1]]
    s = sin[..., : xe.shape[-1]]
    y = np.empty_like(x)
    y[..., 0::2] = c * xe - s * xo
    y[..., 1::2] = s * xe + c * xo
    return y


def _rope_backward(dy: np.ndarray, cos: np.ndarray, sin: np.ndarray) -> np.ndarray:
    dye = dy[..., 0::2]
    dyo = dy[..., 1::2]
    c = cos[..., : dye.shape[-1]]
    s = sin[..., : dye.shape[-1]]
    dx = np.empty_like(dy)
    dx[..., 0::2] = c * dye + s * dyo
    dx[..., 1::2] = -s * dye + c * dyo
    return dx


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
        dh = cfg.dim // cfg.heads
        assert dh % 2 == 0, "RoPE needs even head dim"
        self.rope = rope_tables(cfg.width, dh)

    # -- forward ------------------------------------------------------
    def forward(
        self,
        x0: np.ndarray,
        targets: np.ndarray | None,
        n_iters: int,
        train: bool = True,
    ) -> tuple[np.ndarray, float, list[dict] | None]:
        p = self.p
        h = p["E"][x0]
        states = [h]
        logits: list[np.ndarray] = []
        cache_list: list[dict] = []
        for _ in range(1, n_iters + 1):
            h, cache = self._block(h, states)
            states.append(h)
            logits.append(h @ p["wd"] + p["bd"])
            if train:
                cache_list.append(cache)
        logits_arr = np.stack(logits, axis=0)
        if targets is None:
            return logits_arr, float("nan"), None
        loss = float(np.mean(_bce_with_logits(logits_arr, targets)))
        return logits_arr, loss, cache_list

    def forward_states(self, x0: np.ndarray, n_iters: int) -> list[np.ndarray]:
        """States h_0..h_N without caches (eval/diagnostics)."""
        h = self.p["E"][x0]
        states = [h]
        for _ in range(1, n_iters + 1):
            h, _ = self._block(h, states)
            states.append(h)
        return states

    def _block(
        self, h: np.ndarray, states: list[np.ndarray]
    ) -> tuple[np.ndarray, dict]:
        cfg, p = self.cfg, self.p
        d, dh = cfg.dim, cfg.dim // cfg.heads
        cache: dict = {"h_in": h, "dh": dh}
        r1, c1 = _rms_forward(h, p["g1"])
        qkv = r1 @ p["qkv"]
        q = _rope_transform(
            _split_heads(qkv[..., :d], cfg.heads), self.rope.cos, self.rope.sin
        )
        k = _rope_transform(
            _split_heads(qkv[..., d : 2 * d], cfg.heads),
            self.rope.cos,
            self.rope.sin,
        )
        v = _split_heads(qkv[..., 2 * d :], cfg.heads)
        s = _softmax(q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh))
        h1 = h + _merge_heads(s @ v) @ p["bo"]
        r2, c2 = _rms_forward(h1, p["g2"])
        f = r2 @ p["m1"]
        a = np.maximum(f, 0)
        h2 = h1 + a @ p["m2"]
        if cfg.arm == "carry-last":
            h3 = h2
        elif cfg.arm in ("window", "full", "decayed"):
            ctx = states if cfg.arm != "window" else states[-cfg.window :]
            C = np.stack(ctx, axis=0)
            r3, c3 = _rms_forward(h2, p["g3"])
            qh = r3 @ p["hq"]
            kh = C @ p["hk"]
            vh = C @ p["hv"]
            logits_h = np.sum(qh[None] * kh, axis=-1) / np.sqrt(d)
            if cfg.arm == "decayed":
                beta = float(1.0 / (1.0 + np.exp(-p["lb"])))
                logbeta = float(-np.logaddexp(0.0, -float(p["lb"])))
                t = C.shape[0]
                age = (t - 1 - np.arange(t))[:, None, None]
                logits_h = logits_h + age * logbeta
                cache["age"] = age
                cache["beta"] = beta
            sh = _softmax(logits_h, axis=0)
            g = np.sum(sh[..., None] * vh, axis=0)
            h3 = h2 + g @ p["ho"]
            cache.update(C=C, c3=c3, r3=r3, qh=qh, kh=kh, vh=vh, sh=sh, g=g)
        elif cfg.arm == "papercache":
            C = np.stack(states, axis=0)  # (T,B,W,d)
            t = C.shape[0]
            b = C.shape[1]
            r4, c4 = _rms_forward(h2, p["g4"])
            qh = _split_heads(r4 @ p["pc_q"], cfg.heads)  # (B,H,W,dh)
            Cb = C.transpose(1, 0, 2, 3).reshape(b, -1, d)  # (B,T*W,d)
            kh = _split_heads(Cb @ p["pc_k"], cfg.heads)
            vh = _split_heads(Cb @ p["pc_v"], cfg.heads)
            s4 = _softmax(qh @ kh.transpose(0, 1, 3, 2) / np.sqrt(dh))
            g4 = _merge_heads(s4 @ vh)
            h3 = h2 + g4 @ p["pc_o"]
            cache.update(C=C, c4=c4, r4=r4, qh=qh, kh=kh, vh=vh, s4=s4, g4=g4, t=t)
        else:
            raise ValueError(f"unknown arm {cfg.arm}")
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
        d = cfg.dim
        n_states = n_iters + 1
        gs = [np.zeros((batch, cfg.width, d)) for _ in range(n_states)]
        grads = {k: np.zeros_like(v) for k, v in self.p.items()}
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
                for j, i in enumerate(range(n - t, n)):
                    gs[i] += ctx_grads[j]
        np.add.at(grads["E"], x0, gs[0])
        return grads

    def _block_backward(
        self, dh3: np.ndarray, c: dict, grads: dict[str, np.ndarray]
    ) -> tuple[np.ndarray, np.ndarray | None]:
        cfg, p = self.cfg, self.p
        d = cfg.dim
        ctx_grads = None
        if cfg.arm == "carry-last":
            dh2 = dh3
        elif cfg.arm in ("window", "full", "decayed"):
            C = c["C"]
            dgo = dh3
            dg = dgo @ p["ho"].T
            grads["ho"] += np.einsum("bwd,bwe->de", c["g"], dgo)
            dsh = np.einsum("bwd,tbwd->tbw", dg, c["vh"])
            dvh = c["sh"][..., None] * dg[None]
            dS = _softmax_backward(dsh, c["sh"], axis=0)
            ds = dS / np.sqrt(d)
            if cfg.arm == "decayed":
                # z = raw/sqrt(d) + age*logbeta: bias path has no 1/sqrt(d)
                grads["lb"] += float(np.sum(dS * c["age"])) * (1.0 - c["beta"])
            dqh = np.einsum("tbw,tbwd->bwd", ds, c["kh"])
            dkh = ds[..., None] * c["qh"][None]
            dr3 = dqh @ p["hq"].T
            grads["hq"] += np.einsum("bwd,bwe->de", c["r3"], dqh)
            grads["hk"] += np.einsum("tbwd,tbwf->df", C, dkh)
            grads["hv"] += np.einsum("tbwd,tbwf->df", C, dvh)
            ctx_grads = dkh @ p["hk"].T + dvh @ p["hv"].T
            dr3n, dg3 = _rms_backward(dr3, c["c3"])
            grads["g3"] += dg3
            dh2 = dh3 + dr3n
        elif cfg.arm == "papercache":
            C, t = c["C"], c["t"]
            b, w = dh3.shape[0], dh3.shape[1]
            grads["pc_o"] += np.einsum("bwd,bwe->de", c["g4"], dh3)
            dov = _split_heads(dh3 @ p["pc_o"].T, cfg.heads)
            ds4 = np.einsum("bhwe,bhve->bhwv", dov, c["vh"])
            dv = np.einsum("bhwv,bhwe->bhve", c["s4"], dov)
            dsc = _softmax_backward(ds4, c["s4"]) / np.sqrt(c["dh"])
            dq = np.einsum("bhwv,bhve->bhwe", dsc, c["kh"])
            dk = np.einsum("bhwv,bhwe->bhve", dsc, c["qh"])
            dq_m = _merge_heads(dq)
            dr4 = dq_m @ p["pc_q"].T
            grads["pc_q"] += np.einsum("bwd,bwe->de", c["r4"], dq_m)
            dkf = _merge_heads(dk).reshape(b, t, w, d).transpose(1, 0, 2, 3)
            dvf = _merge_heads(dv).reshape(b, t, w, d).transpose(1, 0, 2, 3)
            grads["pc_k"] += np.einsum("tbwd,tbwe->de", C, dkf)
            grads["pc_v"] += np.einsum("tbwd,tbwe->de", C, dvf)
            ctx_grads = dkf @ p["pc_k"].T + dvf @ p["pc_v"].T
            dr4n, dg4 = _rms_backward(dr4, c["c4"])
            grads["g4"] += dg4
            dh2 = dh3 + dr4n
        else:
            raise ValueError(f"unknown arm {cfg.arm}")
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
        dv_tok = np.einsum("bhwv,bhwe->bhve", c["s"], dov)
        dscores = _softmax_backward(ds_attn, c["s"]) / np.sqrt(c["dh"])
        dq_raw = np.einsum("bhwv,bhve->bhwe", dscores, c["k"])
        dk_raw = np.einsum("bhwv,bhwe->bhve", dscores, c["q"])
        dq = _rope_backward(dq_raw, self.rope.cos, self.rope.sin)
        dk = _rope_backward(dk_raw, self.rope.cos, self.rope.sin)
        dqkv = np.concatenate([_merge_heads(x) for x in (dq, dk, dv_tok)], axis=-1)
        dr1 = dqkv @ p["qkv"].T
        grads["qkv"] += np.einsum("bwd,bwe->de", c["r1"], dqkv)
        dr1n, dg1 = _rms_backward(dr1, c["c1"])
        grads["g1"] += dg1
        return dh1 + dr1n, ctx_grads
