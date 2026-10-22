"""WAP (Zhang ve ark., Pattern Recognition 71, 2017) — PyTorch portu.

Orijinal Theano kodu ``VGG/nmt.py`` (JianshuZhang/WAP) birebir izlenmiştir:

* **Watcher (kodlayıcı):** VGG-benzeri FCN; 4 blok × 4 konvolüsyon (3×3, ``same``), kanal
  ``[32, 64, 64, 128]``, her konvolüsyon sonrası BatchNorm(eps=1e-4) + ReLU; 4. bloğun ilk üç
  katmanında dropout 0.2; her blok sonunda alt/sağ ``-1`` dolgu + 2×2 max-pool (⇒ ``ceil(H/2)``).
  Çıktı ``(B, 128, H/16, W/16)``; maske aynı oranda alt-örneklenir.
* **Parser (çözücü):** koşullu GRU (Bahdanau tarzı ``gru_cond``): GRU1(emb, h) → h1; kapsama
  (coverage) tabanlı dikkat: ``e = v^T tanh(W_c ctx + W_h h1 + W_f conv(sum alpha_past))``;
  GRU2(ctx_t, h1) → h. Kapsama evrişimi 5×5, 128 kanal; dikkat boyutu 128; GRU 256; gömme 256.
* **Okuma:** ``W_h h + W_e emb + W_c ctx`` → maxout(2) → dropout 0.2 → softmax(111).
* ``<eol>`` kimliği 0; ilk adımda gömme sıfırdır; başlangıç durumu ``tanh(W · maskeli_ortalama(ctx))``.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
import torch.nn.functional as F  # noqa: N812


class Encoder(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        blocks: tuple[int, ...] = (32, 64, 64, 128),
        layers: tuple[int, ...] = (4, 4, 4, 4),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.drop_names: set[str] = {"b3_l0", "b3_l1", "b3_l2"}
        self.names: list[str] = []
        self.layers = layers
        ch = in_ch
        for b, out in enumerate(blocks):
            for level in range(layers[b]):
                self.convs.append(nn.Conv2d(ch, out, 3, padding=1, bias=False))
                self.bns.append(nn.BatchNorm2d(out, eps=1e-4))
                self.names.append(f"b{b}_l{level}")
                ch = out
        self.out_dim = ch
        self.dropout = dropout

    def forward(self, x: Tensor, mask: Tensor) -> tuple[Tensor, Tensor]:
        i = 0
        for b in range(len(self.layers)):
            for _ in range(self.layers[b]):
                x = F.relu(self.bns[i](self.convs[i](x)))
                if self.names[i] in self.drop_names:
                    x = F.dropout(x, self.dropout, self.training)
                i += 1
            x = F.pad(x, (0, 1, 0, 1), value=-1.0)
            x = F.max_pool2d(x, 2, 2)
            mask = mask[:, 0::2, 0::2]
        return x, mask


class CoverageAttentionGRU(nn.Module):
    """Theano ``gru_cond`` + kapsama dikkati; tek adım."""

    def __init__(self, dim_word: int, dim: int, dimctx: int, dimatt: int, dim_cov: int, k_cov: int):
        super().__init__()
        self.dim = dim
        # GRU1 (girdi: gömme)
        self.W = nn.Linear(dim_word, 2 * dim)  # r,u
        self.Wx = nn.Linear(dim_word, dim)
        self.U = nn.Linear(dim, 2 * dim, bias=False)
        self.Ux = nn.Linear(dim, dim, bias=False)
        # dikkat
        self.Wc_att = nn.Conv2d(dimctx, dimatt, 1)
        self.W_comb_att = nn.Linear(dim, dimatt, bias=False)
        self.conv_Q = nn.Conv2d(1, dim_cov, k_cov, padding=k_cov // 2, bias=False)
        self.conv_Uf = nn.Linear(dim_cov, dimatt)
        self.U_att = nn.Linear(dimatt, 1)
        # GRU2 (girdi: bağlam)
        self.Wc = nn.Linear(dimctx, 2 * dim, bias=False)
        self.Wcx = nn.Linear(dimctx, dim, bias=False)
        self.U_nl = nn.Linear(dim, 2 * dim)
        self.Ux_nl = nn.Linear(dim, dim)

    def project_context(self, ctx: Tensor) -> Tensor:
        return self.Wc_att(ctx)  # (B, A, H, W)

    def step(
        self, emb: Tensor, h: Tensor, ctx: Tensor, pctx: Tensor, mask: Tensor, alpha_past: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        d = self.dim
        pre = torch.sigmoid(self.W(emb) + self.U(h))
        r1, u1 = pre[:, :d], pre[:, d:]
        h1 = torch.tanh(self.Wx(emb) + r1 * self.Ux(h))
        h1 = u1 * h + (1.0 - u1) * h1

        cover = self.conv_Q(alpha_past[:, None])  # (B, C, H, W)
        cover = self.conv_Uf(cover.permute(0, 2, 3, 1))  # (B, H, W, A)
        pstate = self.W_comb_att(h1)[:, None, None, :]
        e = self.U_att(torch.tanh(pctx.permute(0, 2, 3, 1) + pstate + cover)).squeeze(-1)
        e = e - e.amax(dim=(1, 2), keepdim=True)
        alpha = torch.exp(e) * mask
        alpha = alpha / alpha.sum(dim=(1, 2), keepdim=True).clamp_min(1e-12)
        alpha_past = alpha_past + alpha
        ctx_t = (ctx * alpha[:, None]).sum(dim=(2, 3))  # (B, dimctx)

        pre2 = torch.sigmoid(self.U_nl(h1) + self.Wc(ctx_t))
        r2, u2 = pre2[:, :d], pre2[:, d:]
        h2 = torch.tanh(r2 * self.Ux_nl(h1) + self.Wcx(ctx_t))
        h2 = u2 * h1 + (1.0 - u2) * h2
        return h2, ctx_t, alpha, alpha_past


class WAP(nn.Module):
    def __init__(
        self,
        vocab: int = 111,
        dim_word: int = 256,
        dim_dec: int = 256,
        dim_att: int = 128,
        dim_cov: int = 128,
        k_cov: int = 5,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.encoder = Encoder(dropout=dropout)
        ctxdim = self.encoder.out_dim
        self.emb = nn.Embedding(vocab, dim_word)
        self.ff_state = nn.Linear(ctxdim, dim_dec)
        self.dec = CoverageAttentionGRU(dim_word, dim_dec, ctxdim, dim_att, dim_cov, k_cov)
        self.ff_logit_gru = nn.Linear(dim_dec, dim_word)
        self.ff_logit_prev = nn.Linear(dim_word, dim_word)
        self.ff_logit_ctx = nn.Linear(ctxdim, dim_word)
        self.ff_logit = nn.Linear(dim_word // 2, vocab)
        self.dropout = dropout
        self.vocab = vocab

    def encode(self, x: Tensor, x_mask: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        ctx, mask = self.encoder(x, x_mask)
        ctx_mean = (ctx * mask[:, None]).sum(dim=(2, 3)) / mask.sum(dim=(1, 2))[:, None]
        h0 = torch.tanh(self.ff_state(ctx_mean))
        pctx = self.dec.project_context(ctx)
        return ctx, mask, pctx, h0

    def readout(self, h: Tensor, emb: Tensor, ctx_t: Tensor) -> Tensor:
        logit = self.ff_logit_gru(h) + self.ff_logit_prev(emb) + self.ff_logit_ctx(ctx_t)
        logit = logit.view(logit.shape[0], -1, 2).amax(-1)
        logit = F.dropout(logit, self.dropout, self.training)
        return self.ff_logit(logit)

    def forward(self, x: Tensor, x_mask: Tensor, y: Tensor, y_mask: Tensor) -> Tensor:
        """Öğretmen zorlamalı maliyet: örnek başına toplam CE'nin yığın ortalaması.

        ``y`` (B, T) hedef (son geçerli konum ``<eol>``=0), ``y_mask`` (B, T).
        """
        ctx, mask, pctx, h = self.encode(x, x_mask)
        n_b, n_t = y.shape
        emb_all = self.emb(y)
        emb_prev = torch.cat([emb_all.new_zeros(n_b, 1, emb_all.shape[-1]), emb_all[:, :-1]], 1)
        alpha_past = ctx.new_zeros(n_b, ctx.shape[2], ctx.shape[3])
        logits = []
        for t in range(n_t):
            emb = emb_prev[:, t]
            h_new, ctx_t, _alpha, alpha_past = self.dec.step(emb, h, ctx, pctx, mask, alpha_past)
            m = y_mask[:, t, None]
            h = m * h_new + (1.0 - m) * h
            logits.append(self.readout(h, emb, ctx_t))
        logits_t = torch.stack(logits, 1)  # (B, T, V)
        nll = F.cross_entropy(logits_t.reshape(-1, self.vocab), y.reshape(-1), reduction="none")
        nll = (nll.view(n_b, n_t) * y_mask).sum(1)
        return nll.mean()

    @torch.no_grad()
    def greedy(self, x: Tensor, x_mask: Tensor, maxlen: int = 200) -> list[list[int]]:
        ctx, mask, pctx, h = self.encode(x, x_mask)
        n_b = x.shape[0]
        alpha_past = ctx.new_zeros(n_b, ctx.shape[2], ctx.shape[3])
        emb = ctx.new_zeros(n_b, self.emb.embedding_dim)
        out = torch.zeros(n_b, maxlen, dtype=torch.long, device=x.device)
        done = torch.zeros(n_b, dtype=torch.bool, device=x.device)
        for t in range(maxlen):
            h, ctx_t, _a, alpha_past = self.dec.step(emb, h, ctx, pctx, mask, alpha_past)
            w = self.readout(h, emb, ctx_t).argmax(-1)
            out[:, t] = w
            done |= w == 0
            if bool(done.all()):
                break
            emb = self.emb(w)
        res = []
        for b in range(n_b):
            seq = out[b].tolist()
            res.append(seq[: seq.index(0)] if 0 in seq else seq)
        return res

    @torch.no_grad()
    def beam(self, x: Tensor, x_mask: Tensor, k: int = 10, maxlen: int = 200) -> list[int]:
        """Tek örnek için ışın araması (orijinal ``gen_sample`` karşılığı)."""
        ctx, mask, pctx, h = self.encode(x, x_mask)
        h_c, w_c = ctx.shape[2], ctx.shape[3]
        alpha_past = ctx.new_zeros(1, h_c, w_c)
        emb = ctx.new_zeros(1, self.emb.embedding_dim)
        hyps: list[list[int]] = [[]]
        scores = ctx.new_zeros(1)
        finished: list[tuple[float, list[int]]] = []
        live = 1
        for _ in range(maxlen):
            ctx_r = ctx.expand(live, -1, -1, -1)
            pctx_r = pctx.expand(live, -1, -1, -1)
            mask_r = mask.expand(live, -1, -1)
            h, ctx_t, _a, alpha_past = self.dec.step(emb, h, ctx_r, pctx_r, mask_r, alpha_past)
            logp = F.log_softmax(self.readout(h, emb, ctx_t), -1)
            cand = (scores[:, None] - logp).flatten()
            n_keep = k - len(finished)
            costs, idx = cand.topk(n_keep, largest=False)
            ti = idx // self.vocab
            wi = idx % self.vocab
            new_hyps, keep, new_scores = [], [], []
            for c, t_, w_ in zip(costs.tolist(), ti.tolist(), wi.tolist(), strict=True):
                seq = hyps[t_] + [w_]
                if w_ == 0:
                    finished.append((c, seq[:-1]))
                else:
                    new_hyps.append(seq)
                    keep.append(t_)
                    new_scores.append(c)
            live = len(new_hyps)
            if live == 0 or len(finished) >= k:
                break
            keep_t = torch.tensor(keep, device=x.device)
            hyps = new_hyps
            scores = torch.tensor(new_scores, device=x.device)
            h = h[keep_t]
            alpha_past = alpha_past[keep_t]
            emb = self.emb(torch.tensor([s[-1] for s in hyps], device=x.device))
        if live:
            for s, hyp in zip(scores.tolist(), hyps, strict=True):
                finished.append((s, hyp))
        finished.sort(key=lambda p: p[0] / max(len(p[1]), 1))
        return finished[0][1]
