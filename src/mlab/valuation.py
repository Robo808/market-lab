"""Valuation-based long-horizon expected returns (CAPE and a "fair-value" CAPE), plus lump-sum vs staged entry.

Methods follow published work rather than new inventions:
- Campbell and Shiller (1998), Shiller CAPE: log(CAPE) against the next 10 years' real total return.
- Davis, Aliaga-Diaz, Ahluwalia and Tolani (2018, JPM), "Improving U.S. stock return forecasts: a 'fair-value'
  CAPE approach": the earnings yield is explained by real bond yields, inflation and volatility first, then the
  return forecast uses the gap between CAPE and that fair value. This module is a reduced-form OLS version of the
  two steps (the paper uses a VAR), so its numbers are ours, not the paper's.
- Shtekhman, Tasopoulos and Wimmer (2012, Vanguard), "Dollar-cost averaging just means taking risk later": lump
  sum vs a staged entry with the uninvested balance in T-bills.

Data: Shiller monthly file (shillerdata.com, Yale mirror as fallback) for price, dividends, CPI, GS10 and CAPE; the
latest CAPE print from multpl.com (the Shiller file lags); FRED for T-bills (TB3MS) and the 10y TIPS yield (DFII10).
Raw downloads are cached under $MLAB_DATA_DIR/data/valuation/ and never committed.

Splits follow the frozen-dataset convention on the OUTCOME window: a 10-year forward return belongs to the split
in which its window ends. Train = windows ending by 2014-12, validation = ending 2015-2019, test = ending 2020+,
which stays locked here (no function evaluates it).
"""
from __future__ import annotations

import io
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import DATA_DIR

SHILLER_URLS = (
    "https://img1.wsimg.com/blobby/go/e5e77e0b-59d1-44d9-ab25-4763ac982e53/downloads/ie_data.xls",
    "http://www.econ.yale.edu/~shiller/data/ie_data.xls",
)
MULTPL_CAPE_URL = "https://www.multpl.com/shiller-pe/table/by-month"
VAL_DIR = DATA_DIR / "data" / "valuation"
TRAIN_END, VAL_END = "2014-12", "2019-12"


# ---- loaders (network, cached raw files) ------------------------------------------
def _fetch(url: str, path: Path, max_age_days: float = 7, user_agent: str | None = None) -> Path:
    if path.exists() and (time.time() - path.stat().st_mtime) < max_age_days * 86400:
        return path
    from .net import session
    s = session(user_agent) if user_agent else session()
    r = s.get(url, timeout=60)
    r.raise_for_status()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(r.content)
    return path


def parse_shiller(raw: pd.DataFrame) -> pd.DataFrame:
    """Shiller 'Data' sheet (header row 8) -> monthly frame: price, div, earn, cpi, gs10, cape, real_tr."""
    d = raw.copy()
    d = d[pd.to_numeric(d["Date"], errors="coerce").notna()]
    date = d["Date"].astype(float)
    yr = np.floor(date).astype(int)
    mo = np.rint((date - yr) * 100).astype(int)
    mo = np.where(mo == 0, 10, mo)  # Excel drops the trailing zero of October (2023.1 == 2023.10)
    idx = pd.to_datetime(pd.DataFrame({"year": yr, "month": mo, "day": 1}))
    num = lambda c: pd.to_numeric(d[c], errors="coerce").to_numpy()
    out = pd.DataFrame({"price": num("P"), "div": num("D"), "earn": num("E"), "cpi": num("CPI"),
                        "gs10": num("Rate GS10"), "cape": num("CAPE"), "real_tr": num("Price.1")}, index=idx)
    out.index.name = "date"
    return out[~out.index.duplicated(keep="last")].sort_index()


def load_shiller(refresh: bool = False) -> tuple[pd.DataFrame, str]:
    last = None
    for url in SHILLER_URLS:
        try:
            p = _fetch(url, VAL_DIR / "ie_data.xls", max_age_days=0 if refresh else 30)
            raw = pd.read_excel(p, "Data", header=7)
            return parse_shiller(raw), url
        except Exception as e:  # try the mirror
            last = e
    raise RuntimeError(f"Shiller data unavailable: {last}")


def parse_multpl(html: str) -> pd.Series:
    t = pd.read_html(io.StringIO(html))[0]
    s = pd.Series(pd.to_numeric(t.iloc[:, 1].astype(str).str.extract(r"([\d.]+)")[0], errors="coerce").to_numpy(),
                  index=pd.to_datetime(t.iloc[:, 0]), name="cape").dropna().sort_index()
    return s


def load_cape_latest(refresh: bool = False) -> tuple[pd.Series, str]:
    p = _fetch(MULTPL_CAPE_URL, VAL_DIR / "multpl_shiller_pe.html", max_age_days=0 if refresh else 1,
               user_agent="Mozilla/5.0")
    return parse_multpl(p.read_text()), MULTPL_CAPE_URL


# ---- maths (pure, offline-testable) ------------------------------------------------
def forward_annualised(level: pd.Series, months: int) -> pd.Series:
    """Annualised return from month t to t+months on a total-return level series."""
    return (level.shift(-months) / level) ** (12 / months) - 1


def ols(x, y) -> tuple[np.ndarray, float, float]:
    """OLS with intercept: (coefficients, R^2, residual std)."""
    x = np.asarray(x, float)
    X = np.c_[np.ones(len(x)), x.reshape(len(x), -1)]
    y = np.asarray(y, float)
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ b
    return b, float(1 - r.var() / y.var()), float(r.std())


def split_by_outcome(df: pd.DataFrame, months: int = 120) -> dict[str, pd.DataFrame]:
    """Assign each start month to train/val/test by the month its forward window ENDS."""
    ends = df.index + pd.DateOffset(months=months)
    return {"train": df[ends <= pd.Timestamp(TRAIN_END) + pd.offsets.MonthEnd(0)],
            "val": df[(ends > pd.Timestamp(TRAIN_END) + pd.offsets.MonthEnd(0)) & (ends <= pd.Timestamp(VAL_END) + pd.offsets.MonthEnd(0))]}


def features(sh: pd.DataFrame, months: int = 120) -> pd.DataFrame:
    f = pd.DataFrame(index=sh.index)
    f["cape"] = sh["cape"]
    f["fwd"] = forward_annualised(sh["real_tr"], months)
    f["fwd12"] = forward_annualised(sh["real_tr"], 12)
    f["infl"] = sh["cpi"].pct_change(12)
    # real yield proxy before TIPS existed: nominal 10y minus trailing 10y average inflation
    f["ry"] = sh["gs10"] / 100 - ((sh["cpi"] / sh["cpi"].shift(120)) ** 0.1 - 1)
    f["vol"] = sh["real_tr"].pct_change().rolling(12).std() * np.sqrt(12)
    f["ey"] = 1 / f["cape"]
    return f


@dataclass
class CapeModels:
    simple: np.ndarray       # fwd = a + b*log(cape)
    fv: np.ndarray           # ey = c0 + c1*ry + c2*infl + c3*vol
    step2: np.ndarray        # fwd = d0 + d1*log(cape/fv_cape) + d2*(1/fv_cape)
    r2: dict
    n_train: int
    hist_mean: float

    def fair_cape(self, ry, infl, vol):
        return 1 / (self.fv[0] + self.fv[1] * np.asarray(ry) + self.fv[2] * np.asarray(infl) + self.fv[3] * np.asarray(vol))

    def predict_simple(self, cape):
        return self.simple[0] + self.simple[1] * np.log(cape)

    def predict_fv(self, cape, ry, infl, vol):
        fc = self.fair_cape(ry, infl, vol)
        return self.step2[0] + self.step2[1] * np.log(np.asarray(cape) / fc) + self.step2[2] / fc


def fit_models(f: pd.DataFrame, start: str = "1891") -> CapeModels:
    tr = split_by_outcome(f.loc[start:])["train"].dropna(subset=["cape", "fwd", "ry", "infl", "vol"])
    b1, r1, _ = ols(np.log(tr["cape"]), tr["fwd"])
    X = tr[["ry", "infl", "vol"]].to_numpy()
    bf, rf, _ = ols(X, tr["ey"])
    fc = 1 / (bf[0] + X @ bf[1:])
    b2, r2, _ = ols(np.c_[np.log(tr["cape"].to_numpy() / fc), 1 / fc], tr["fwd"])
    return CapeModels(b1, bf, b2, {"simple": r1, "fair_value_step1": rf, "fair_value_step2": r2}, len(tr),
                      float(tr["fwd"].mean()))


def validate(m: CapeModels, f: pd.DataFrame) -> dict:
    va = split_by_outcome(f)["val"].dropna(subset=["cape", "fwd", "ry", "infl", "vol"])
    rmse = lambda p: float(np.sqrt(np.mean((va["fwd"].to_numpy() - p) ** 2)))
    return {"n_val": len(va), "rmse_hist_mean": rmse(np.full(len(va), m.hist_mean)),
            "rmse_simple": rmse(m.predict_simple(va["cape"].to_numpy())),
            "rmse_fair_value": rmse(m.predict_fv(va["cape"], va["ry"], va["infl"], va["vol"])),
            "realised_mean": float(va["fwd"].mean())}


def bucket_table(f: pd.DataFrame, bins=(0, 10, 15, 20, 25, 30, 35, 100)) -> pd.DataFrame:
    """Realised forward real return by starting CAPE. Overlapping monthly windows: count is NOT independent obs."""
    a = f.dropna(subset=["cape", "fwd"])
    g = a.groupby(pd.cut(a["cape"], list(bins)), observed=False)["fwd"]
    t = pd.DataFrame({"months": g.size(), "min": g.min(), "median": g.median(), "max": g.max(),
                      "indep_decades": g.size() / 120})
    return t


def lump_vs_staged(level: pd.Series, cash_annual: pd.Series, tranches: int = 5, horizon: int = 12) -> pd.DataFrame:
    """Wealth after `horizon` months of 1 unit invested at once vs in `tranches` equal monthly buys (first buy at t),
    uninvested balance earning `cash_annual` (decimal, annual) monthly. Returns one row per start month."""
    lv = level.dropna()
    c = cash_annual.reindex(lv.index).ffill().to_numpy() / 12
    p = lv.to_numpy()
    rows = []
    for i in range(len(p) - horizon):
        if np.isnan(c[i]):
            continue
        cash, units = 1.0, 0.0
        for k in range(tranches):
            if k:
                cash *= 1 + c[i + k - 1]
            buy = 1.0 / tranches
            units += buy / p[i + k]
            cash -= buy
        cash *= np.prod(1 + c[i + tranches - 1:i + horizon])
        staged = units * p[i + horizon] + cash
        rows.append((lv.index[i], p[i + horizon] / p[i], staged))
    out = pd.DataFrame(rows, columns=["start", "lump", "staged"]).set_index("start")
    out["gap"] = out["lump"] - out["staged"]
    return out


def lvs_summary(t: pd.DataFrame, cape: pd.Series, bins=(0, 15, 20, 25, 30, 100)) -> pd.DataFrame:
    t = t.join(cape.rename("cape"), how="inner").dropna()
    g = t.groupby(pd.cut(t["cape"], list(bins)), observed=False)
    return pd.DataFrame({"months": g.size(), "lump_wins": g["gap"].apply(lambda s: (s > 0).mean()),
                         "mean_gap": g["gap"].mean(), "worst_gap": g["gap"].min(), "best_gap": g["gap"].max()})


# ---- CLI ---------------------------------------------------------------------------
def cmd_valuation(a):
    from .cli import show
    from .providers.macro import fred
    sh, sh_src = load_shiller(a.refresh)
    f = features(sh)
    m = fit_models(f)
    v = validate(m, f)
    cape_s, cape_src = load_cape_latest(a.refresh)
    cape_now, cape_dt = float(cape_s.iloc[-1]), cape_s.index[-1].date()
    monthly = cape_s[cape_s.index.day == 1]
    ry = fred("DFII10").dropna()
    vol, vol_src = a.vol, "--vol"
    if vol is None:
        try:
            from .data import get_prices
            px = get_prices("^GSPC", interval="1mo", period="2y")["Close"]
            vol, vol_src = float(px.pct_change().tail(12).std() * np.sqrt(12)), "Yahoo ^GSPC 12 monthly returns"
        except Exception:
            vol, vol_src = float(f["vol"].dropna().iloc[-1]), "Shiller file (stale)"
    infl = a.infl
    now = {"cape": cape_now, "cape_date": str(cape_dt), "cape_pct_since_1871": float((monthly < cape_now).mean()),
           "tips10_real_yield": float(ry.iloc[-1]) / 100, "tips_date": str(ry.index[-1].date()),
           "infl_assumed": infl, "vol_used": vol, "vol_source": vol_src,
           "fair_value_cape": float(m.fair_cape(ry.iloc[-1] / 100, infl, vol)),
           "fcst_10y_real_simple": float(m.predict_simple(cape_now)),
           "fcst_10y_real_fair_value": float(m.predict_fv(cape_now, ry.iloc[-1] / 100, infl, vol))}
    if a.action == "dca":
        tb = fred("TB3MS") / 100
        nominal = sh["real_tr"] * sh["cpi"] / sh["cpi"].dropna().iloc[-1]
        t = lump_vs_staged(nominal, tb, a.tranches, a.horizon)
        show(lvs_summary(t, sh["cape"]), a.json, f"Lump sum vs {a.tranches} monthly tranches, {a.horizon}m wealth gap "
             f"(US TR, nominal, cash in T-bills) | {sh_src}, FRED TB3MS")
        return
    show(pd.Series(now), a.json, f"Now | CAPE {cape_src} {cape_dt}; FRED DFII10 {ry.index[-1].date()}")
    show(pd.Series({**{f"r2_{k}": x for k, x in m.r2.items()}, "n_train_months": m.n_train,
                    "hist_mean_train": m.hist_mean, **v}), a.json,
         f"Models fit on 10y windows ending <= {TRAIN_END}; validation = ending 2015-19; test (2020+) locked "
         f"| {sh_src}, data to {sh['real_tr'].dropna().index[-1].date()}")
    show(bucket_table(f), a.json, "Realised 10y real total return by starting CAPE (overlapping windows)")
    tr = f.dropna(subset=["cape", "fwd12"]).loc["1891":]
    tr = tr[tr.index + pd.DateOffset(months=12) <= pd.Timestamp(TRAIN_END)]
    show(pd.Series({"r2_cape_vs_next_12m": ols(np.log(tr["cape"]), tr["fwd12"])[1], "r2_cape_vs_next_10y": m.r2["simple"]}),
         a.json, "Valuation is a decade signal, not a timing signal")


def register(add):
    q = add("valuation", cmd_valuation, "CAPE and fair-value CAPE 10y expected returns | valuation dca: lump sum vs staged entry")
    q.add_argument("action", nargs="?", default="show", choices=["show", "dca"])
    q.add_argument("--infl", type=float, default=0.03, help="inflation assumption for the fair-value CAPE (decimal)")
    q.add_argument("--vol", type=float, help="annualised equity vol for the fair-value CAPE (default: last 12m in Shiller data)")
    q.add_argument("--tranches", type=int, default=5); q.add_argument("--horizon", type=int, default=12)
    q.add_argument("--refresh", action="store_true")
