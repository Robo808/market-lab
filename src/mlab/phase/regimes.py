"""Regimes: a Gaussian hidden Markov model over the standardised feature cube, the "weather systems" of
phase space (Hamilton 1989; Ang & Timmermann 2012 survey regime models in finance).

Point-in-time rules. The scaler and the HMM are fitted on a training window only. Regime probabilities
come from the forward filter, P(state_t | x_1..x_t), which never looks past t. The Viterbi path and the
smoothed (forward-backward) probabilities condition on the whole sample, so they are used inside the fit
only and never exposed as a signal.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

REGIME_FEATURES = ["ret_21", "ret_63", "rv_21", "rv_ratio", "er_21", "dist_sma50", "dist_sma200", "adx_14",
                   "pe_63", "ac1_63", "htf_dist_sma10"]
LOG_FEATURES = {"rv_21"}        # positive and skewed: modelled in logs
# categorical slots 1-8 of the desk's validated palette (light mode); colour follows the regime id
PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#7a5cd6", "#6b6b6b"]


def _design(cube: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    x = cube[cols].copy()
    for c in LOG_FEATURES & set(cols):
        x[c] = np.log(x[c].clip(lower=1e-6))
    return x


def _kmeans(x: np.ndarray, k: int, rng: np.random.Generator, iters: int = 30) -> np.ndarray:
    c = x[[rng.integers(len(x))]]
    for _ in range(1, k):                                   # k-means++ seeding
        d = ((x[:, None, :] - c[None]) ** 2).sum(-1).min(1)
        c = np.vstack([c, x[rng.choice(len(x), p=d / d.sum())]])
    for _ in range(iters):
        lab = ((x[:, None, :] - c[None]) ** 2).sum(-1).argmin(1)
        c = np.vstack([x[lab == j].mean(0) if (lab == j).any() else c[j] for j in range(k)])
    return c


def _logpdf(x: np.ndarray, mu: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """log N(x | mu_k, cov_k) for x of shape (..., D); returns (..., K)."""
    out = []
    d = x.shape[-1]
    for m, s in zip(mu, cov):
        ch = np.linalg.cholesky(s)
        z = np.linalg.solve(ch, (x - m).reshape(-1, d).T).T.reshape(x.shape)
        out.append(-0.5 * (z ** 2).sum(-1) - np.log(np.diag(ch)).sum() - 0.5 * d * np.log(2 * np.pi))
    return np.stack(out, -1)


@dataclass
class GaussianHMM:
    """Full-covariance Gaussian HMM fitted by Baum-Welch over many sequences in parallel (one per symbol)."""
    k: int = 4
    n_iter: int = 60
    tol: float = 1e-3
    n_init: int = 2
    reg: float = 1e-3
    seed: int = 0
    pi: np.ndarray = field(default=None, repr=False)
    A: np.ndarray = field(default=None, repr=False)
    mu: np.ndarray = field(default=None, repr=False)
    cov: np.ndarray = field(default=None, repr=False)
    loglik: float = float("nan")

    @staticmethod
    def _pad(seqs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        T = max(len(s) for s in seqs)
        X = np.zeros((len(seqs), T, seqs[0].shape[1]))
        M = np.zeros((len(seqs), T), bool)
        for i, s in enumerate(seqs):
            X[i, :len(s)], M[i, :len(s)] = s, True
        return X, M

    def _emissions(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lb = _logpdf(X, self.mu, self.cov)
        shift = lb.max(-1, keepdims=True)
        return np.exp(lb - shift), shift[..., 0]

    def _forward(self, B: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        S, T, K = B.shape
        alpha, c = np.zeros((S, T, K)), np.ones((S, T))
        a = self.pi[None] * B[:, 0]
        c[:, 0] = a.sum(1)
        alpha[:, 0] = a / c[:, 0:1]
        for t in range(1, T):
            a = (alpha[:, t - 1] @ self.A) * B[:, t]
            ct = a.sum(1)
            m = M[:, t]
            c[m, t] = ct[m]
            alpha[:, t] = np.where(m[:, None], a / np.where(ct > 0, ct, 1)[:, None], alpha[:, t - 1])
        return alpha, c

    def _em(self, X, M, rng) -> float:
        S, T, D = X.shape
        flat = X[M]
        self.mu = _kmeans(flat, self.k, rng)
        lab = ((flat[:, None, :] - self.mu[None]) ** 2).sum(-1).argmin(1)
        base = np.cov(flat.T) + self.reg * np.eye(D)
        self.cov = np.stack([np.cov(flat[lab == j].T) + self.reg * np.eye(D) if (lab == j).sum() > D else base
                             for j in range(self.k)])
        self.pi = np.full(self.k, 1 / self.k)
        self.A = np.full((self.k, self.k), 0.05 / max(self.k - 1, 1)) + np.eye(self.k) * (0.95 - 0.05 / max(self.k - 1, 1))
        prev = -np.inf
        for _ in range(self.n_iter):
            B, shift = self._emissions(X)
            alpha, c = self._forward(B, M)
            ll = float((np.log(c) * M).sum() + (shift * M).sum())
            beta = np.ones((S, T, self.k))
            xi = np.zeros((self.k, self.k))
            for t in range(T - 2, -1, -1):
                both = M[:, t] & M[:, t + 1]
                nb = (B[:, t + 1] * beta[:, t + 1]) / c[:, t + 1:t + 2]
                beta[:, t] = np.where(both[:, None], nb @ self.A.T, 1.0)
                xi += np.einsum("si,ij,sj->ij", alpha[both, t], self.A, nb[both])
            g = alpha * beta
            g /= g.sum(-1, keepdims=True)
            g *= M[..., None]
            self.pi = g[:, 0].sum(0) / S
            self.A = xi / xi.sum(1, keepdims=True)
            w = g[M]
            nk = w.sum(0) + 1e-12
            self.mu = (w.T @ flat) / nk[:, None]
            self.cov = np.stack([((flat - self.mu[j]) * w[:, j:j + 1]).T @ (flat - self.mu[j]) / nk[j]
                                 + self.reg * np.eye(D) for j in range(self.k)])
            if ll - prev < self.tol * abs(ll):
                break
            prev = ll
        return ll

    def fit(self, seqs: list[np.ndarray]) -> GaussianHMM:
        X, M = self._pad(seqs)
        best = None
        for i in range(self.n_init):
            rng = np.random.default_rng(self.seed + i)
            ll = self._em(X, M, rng)
            if best is None or ll > best[0]:
                best = (ll, self.pi.copy(), self.A.copy(), self.mu.copy(), self.cov.copy())
        self.loglik, self.pi, self.A, self.mu, self.cov = best
        return self

    def filter(self, seq: np.ndarray) -> np.ndarray:
        """Forward-filtered state probabilities, P(s_t | x_1..x_t). Point-in-time by construction."""
        B, _ = self._emissions(seq[None])
        alpha, _ = self._forward(B, np.ones(B.shape[:2], bool))
        return alpha[0]


@dataclass
class RegimeModel:
    """Scaler + HMM fitted on a training cube; `probs` labels any cube with filtered probabilities."""
    k: int = 4
    features: list[str] = field(default_factory=lambda: list(REGIME_FEATURES))
    seed: int = 0
    clip: float = 5.0
    center: pd.Series = None
    scale: pd.Series = None
    hmm: GaussianHMM = None
    names: list[str] = None
    profile: pd.DataFrame = None

    def _z(self, cube: pd.DataFrame) -> pd.DataFrame:
        return ((_design(cube, self.features) - self.center) / self.scale).clip(-self.clip, self.clip)

    def fit(self, cube: pd.DataFrame) -> RegimeModel:
        x = _design(cube, self.features).dropna()
        self.center, self.scale = x.mean(), x.std().replace(0, 1.0)
        z = self._z(cube).dropna()
        seqs = [g.to_numpy() for _, g in z.groupby(level="symbol") if len(g) > 50]
        self.hmm = GaussianHMM(k=self.k, seed=self.seed).fit(seqs)
        lab = self.probs(cube)["regime"]
        self._describe(cube, lab)
        return self

    def probs(self, cube: pd.DataFrame) -> pd.DataFrame:
        z = self._z(cube)
        out = []
        for sym, g in z.groupby(level="symbol"):
            g = g.dropna()
            if g.empty:
                continue
            p = self.hmm.filter(g.to_numpy())
            out.append(pd.DataFrame(p, index=g.index, columns=[f"p{j}" for j in range(self.k)]))
        res = pd.concat(out).sort_index()
        res["regime"] = res.to_numpy().argmax(1)
        return res

    def _describe(self, cube: pd.DataFrame, lab: pd.Series) -> None:
        cols = ["ret_63", "rv_21", "er_21", "dist_sma200", "pe_63", "ac1_63"]
        d = cube.loc[lab.index, cols].assign(regime=lab.to_numpy())
        prof = d.groupby("regime").mean()
        prof["rv_ann"] = prof.pop("rv_21") * np.sqrt(252)
        prof["share"] = d["regime"].value_counts(normalize=True).sort_index()
        A = self.hmm.A
        prof["stay_prob"] = np.diag(A)
        prof["avg_spell_bars"] = 1 / (1 - np.clip(np.diag(A), 0, 0.9999))
        vol_rank = prof["rv_ann"].rank(method="first").astype(int)
        words = {1: ["steady"], 2: ["calm", "volatile"], 3: ["calm", "active", "volatile"],
                 4: ["calm", "quiet", "active", "volatile"]}.get(len(prof))
        sd = cube["ret_63"].std()
        names = []
        for j, row in prof.iterrows():
            r = vol_rank[j]
            v = words[r - 1] if words else ("calm" if r == 1 else "volatile" if r == len(prof) else f"vol-rank {r}")
            t = "uptrend" if row["ret_63"] > 0.35 * sd else "downtrend" if row["ret_63"] < -0.35 * sd else "range"
            names.append(f"{v} {t}")
        seen: dict[str, int] = {}
        for i, n in enumerate(names):
            seen[n] = seen.get(n, 0) + 1
            if seen[n] > 1:
                names[i] = f"{n} {seen[n]}"
        prof.insert(0, "name", names)
        self.names, self.profile = names, prof

    def to_json(self) -> str:
        return json.dumps({"k": self.k, "features": self.features, "names": self.names,
                           "transition": np.round(self.hmm.A, 4).tolist(), "loglik": self.hmm.loglik})


def phase_plot(cube: pd.DataFrame, regimes: pd.DataFrame, names: list[str], path, every: int = 3,
               title: str = "Phase space") -> str:
    """Interactive 3D view: trend (close vs SMA50 in ATRs) x annualised realised vol x permutation entropy,
    one trajectory per symbol, points coloured by filtered regime. Pick a symbol from the menu."""
    import plotly.graph_objects as go
    d = cube[["dist_sma50", "rv_21", "pe_63"]].join(regimes["regime"], how="inner").dropna()
    syms = list(d.index.get_level_values("symbol").unique())
    fig = go.Figure()
    for s in syms:
        g = d.xs(s, level="symbol").iloc[::every]
        t = g.index.strftime("%Y-%m-%d")
        fig.add_trace(go.Scatter3d(
            x=g["dist_sma50"], y=g["rv_21"] * np.sqrt(252) * 100, z=g["pe_63"], mode="lines+markers", name=s,
            showlegend=False,
            visible=s == syms[0],
            line={"color": "rgba(120,120,120,0.35)", "width": 2},
            marker={"size": 3, "color": [PALETTE[int(r) % len(PALETTE)] for r in g["regime"]]},
            text=[f"{s} {ti}<br>{names[int(r)]}" for ti, r in zip(t, g["regime"])],
            hovertemplate="%{text}<br>trend %{x:.2f} ATR<br>vol %{y:.1f}%<br>entropy %{z:.3f}<extra></extra>"))
    for j, n in enumerate(names):   # legend entries for the regimes
        fig.add_trace(go.Scatter3d(x=[None], y=[None], z=[None], mode="markers", name=n, showlegend=True,
                                   marker={"size": 6, "color": PALETTE[j % len(PALETTE)]}))
    nreg = len(names)
    buttons = [{"label": s, "method": "update",
                "args": [{"visible": [k == i for k in range(len(syms))] + [True] * nreg}]} for i, s in enumerate(syms)]
    buttons.insert(0, {"label": "all", "method": "update", "args": [{"visible": [True] * (len(syms) + nreg)}]})
    fig.update_layout(title={"text": title, "x": 0.1}, updatemenus=[{"buttons": buttons, "x": 0, "y": 1.08, "xanchor": "left"}],
                      scene={"xaxis_title": "trend: (close - SMA50) / ATR", "yaxis_title": "realised vol, % ann",
                             "zaxis_title": "permutation entropy"},
                      legend={"itemsizing": "constant"}, margin={"l": 0, "r": 0, "t": 60, "b": 0})
    fig.write_html(str(path), include_plotlyjs="https://cdn.jsdelivr.net/npm/plotly.js-dist-min@2.35.2/plotly.min.js")
    return str(path)
