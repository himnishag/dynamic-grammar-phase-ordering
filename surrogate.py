"""Differentiable surrogate: (IR features, pass sequence) -> predicted log(size_after / size_before).

The sequence enters as a (soft) one-hot tensor [B,L,V], so gradients flow back to the
relaxed pass choices.  An ensemble provides mean + epistemic std (risk-aware search).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class Surrogate(nn.Module):
    def __init__(self, D, V, L, H=96, layers=2, heads=4, drop=0.1):
        super().__init__()
        self.cfg = dict(D=D, V=V, L=L, H=H, layers=layers, heads=heads, drop=drop)
        self.feat = nn.Sequential(nn.Linear(D, H), nn.GELU(), nn.Linear(H, H))
        self.tok = nn.Linear(V, H, bias=False)
        self.pos = nn.Parameter(torch.randn(L, H) * 0.02)
        self.film = nn.Linear(H, 2 * H)
        layer = nn.TransformerEncoderLayer(H, heads, 2 * H, drop, batch_first=True,
                                           norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.Linear(2 * H, H), nn.GELU(), nn.Linear(H, 1))

    def forward(self, f, x):  # f: normalised features [B,D]; x: [B,L,V]
        c = self.feat(f)
        g, b = self.film(c).chunk(2, -1)
        h = (self.tok(x) + self.pos) * (1 + g[:, None]) + b[:, None]
        h = self.enc(h)
        return self.head(torch.cat([h.mean(1), c], -1)).squeeze(-1)


class Ensemble(nn.Module):
    def __init__(self, models, mean, std, tokens, L):
        super().__init__()
        self.models = nn.ModuleList(models)
        self.register_buffer("mu", torch.as_tensor(mean, dtype=torch.float32))
        self.register_buffer("sd", torch.as_tensor(std, dtype=torch.float32))
        self.tokens, self.L = list(tokens), L

    def prep(self, f_raw):
        return (f_raw - self.mu) / self.sd

    def forward(self, f_raw, x):
        f = self.prep(f_raw)
        p = torch.stack([m(f, x) for m in self.models], 0)
        return p.mean(0), p.std(0, unbiased=False)


def save_ensemble(path, ens: Ensemble, extra=None):
    torch.save({"cfg": ens.models[0].cfg, "n": len(ens.models), "state": ens.state_dict(),
                "mu": ens.mu.cpu(), "sd": ens.sd.cpu(), "tokens": ens.tokens, "L": ens.L,
                "extra": extra or {}}, path)


def load_ensemble(path, device="cpu"):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    models = [Surrogate(**ck["cfg"]) for _ in range(ck["n"])]
    ens = Ensemble(models, ck["mu"], ck["sd"], ck["tokens"], ck["L"])
    ens.load_state_dict(ck["state"])
    ens.to(device).eval()
    for p in ens.parameters():
        p.requires_grad_(False)
    return ens, ck["extra"]
