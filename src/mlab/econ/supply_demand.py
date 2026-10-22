"""Supply and demand categories against industry returns (`mlab econ supply-demand`, `mlab econ sd-study`).

The map lives in data/supply_demand.yaml. Each category pairs a supply series (Federal Reserve G.17 industrial
production, plus capacity utilisation where published) with a consumer-demand series (Census monthly retail sales
by kind of business) and the industry that earns from it (Ken French 49 industry portfolios).

The study asks two separate questions, because they have different answers:
  explain   do 12-month industry excess returns move with same-window demand and supply growth? (not tradeable)
  predict   does what is published by the end of month t (data for t-1) forecast the excess return over the
            next 1, 3, 6 or 12 months? (tradeable)
Every statistic is set against null stubs: the same feature circularly shifted by random offsets, so the null
keeps each series' own persistence. A bar that the shifted features clear as often as the real one is noise.
"""
from __future__ import annotations

import io
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from ..config import DATA_DIR
from ..net import session

HERE = Path(__file__).parent / "data"
G17 = "https://www.federalreserve.gov/releases/g17/Current/ipdisk/{}.txt"
MRTS = "https://www.census.gov/retail/mrts/www/mrtssales92-present.xlsx"
FRENCH = "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/{}_CSV.zip"
SPLITS = {"train": ("1993-01-01", "2014-12-31"), "validation": ("2015-01-01", "2019-12-31"),
          "test": ("2020-01-01", "2100-01-01")}
HORIZONS = (1, 3, 6, 12)


def load_map(path: Path | None = None) -> dict:
    with open(path or HERE / "supply_demand.yaml") as f:
        return yaml.safe_load(f)


def map_frame(m: dict | None = None) -> pd.DataFrame:
    m = m or load_map()
    rows = [{"category": k, "side": c["side"], "supply": c.get("supply", ""), "util": c.get("util", ""),
             "demand": "+".join(c.get("demand", [])), "industry": c["industry"], "etf": c.get("etf", "")}
            for k, c in m["categories"].items()]
    return pd.DataFrame(rows).set_index("category")


# ---- parsers (pure: text or frames in, Series out) -------------------------------------------------------------
def parse_g17(text: str, codes: set[str] | None = None) -> pd.DataFrame:
    """G.17 ip_sa.txt / utl_sa.txt: rows `"CODE"  YEAR  v1 .. v12`, blank cells where not yet published."""
    rows = {}
    for ln in text.splitlines():
        mt = re.match(r'^"([A-Z0-9.]+)"\s+(\d{4})\s+(.*)$', ln)
        if not mt or (codes and mt.group(1) not in codes):
            continue
        code, year = mt.group(1), int(mt.group(2))
        for i, v in enumerate(mt.group(3).split()[:12]):
            rows.setdefault(code, {})[pd.Timestamp(year, i + 1, 1)] = float(v)
    return pd.DataFrame({c: pd.Series(v) for c, v in rows.items()}).sort_index()


def _mrts_key(code, label) -> str:
    code = "" if pd.isna(code) else str(code).strip()
    return code if code and code[0].isdigit() else str(label).strip()


def parse_mrts_sheet(sheet: pd.DataFrame) -> pd.DataFrame:
    """One year sheet of mrtssales92-present.xlsx (header=None) -> seasonally adjusted sales, one column per code."""
    lab = sheet[1].map(lambda v: "" if pd.isna(v) else str(v).strip())
    month = re.compile(r"^([A-Z][a-z]{2})\.? (\d{4})(\(p\))?$")  # "Jan. 2026", "May 2026", "Aug. 2026(p)"
    hdr = next(i for i in sheet.index if month.match(str(sheet.iat[i, 2])))
    months = {j: pd.to_datetime(" ".join(mt.group(1, 2)), format="%b %Y")
              for j in range(2, sheet.shape[1]) if (mt := month.match(str(sheet.iat[hdr, j]).strip()))}
    start = next(i for i in sheet.index if lab[i].startswith("ADJUSTED"))
    out = {}
    for i in sheet.index[sheet.index > start]:
        if not lab[i]:
            if out:
                break  # the adjusted block ends at the first blank row
            continue
        key = _mrts_key(sheet.iat[i, 0], lab[i])
        out[key] = {d: pd.to_numeric(sheet.iat[i, j], errors="coerce") for j, d in months.items()}
    return pd.DataFrame(out).sort_index()


def parse_mrts(book: dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = [parse_mrts_sheet(s) for name, s in book.items() if name.strip().isdigit()]
    df = pd.concat(frames).sort_index()
    return df[~df.index.duplicated(keep="last")]


def parse_french_monthly(text: str) -> pd.DataFrame:
    """First monthly block of a Ken French CSV (value-weighted for industry files), percent -> decimal."""
    lines = text.splitlines()
    head = next(i for i, ln in enumerate(lines) if ln.startswith(","))
    cols = [c.strip() for c in lines[head].split(",")[1:]]
    rows = []
    for ln in lines[head + 1:]:
        parts = [p.strip() for p in ln.split(",")]
        if not (parts[0].isdigit() and len(parts[0]) == 6):
            break
        rows.append([pd.Timestamp(int(parts[0][:4]), int(parts[0][4:]), 1)] + [float(x) for x in parts[1:len(cols) + 1]])
    df = pd.DataFrame(rows, columns=["date"] + cols).set_index("date")
    return df.where(df > -99) / 100


# ---- fetch with a small cache in the data dir (public-domain / research-licensed sources only) ----------------
def _cache() -> Path:
    p = Path(DATA_DIR) / "data" / "econ_sd"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _get(url: str, binary=False):
    r = session().get(url, timeout=120)
    r.raise_for_status()
    return r.content if binary else r.text


@dataclass
class Raw:
    ip: pd.DataFrame
    util: pd.DataFrame
    sales: pd.DataFrame
    ind: pd.DataFrame
    mkt: pd.Series
    asof: str


def fetch(m: dict | None = None, refresh: bool = False, log=print) -> Raw:
    m = m or load_map()
    cats = m["categories"].values()
    codes = {c["supply"] for c in cats if c.get("supply")} | {c["util"] for c in cats if c.get("util")}
    path = _cache() / "raw.parquet"
    if path.exists() and not refresh:
        df = pd.read_parquet(path)
        log(f"[sd] cache {path}")
        return _split(df)
    log("[sd] G.17 industrial production + utilisation (federalreserve.gov)")
    ip = parse_g17(_get(G17.format("ip_sa")), codes)
    util = parse_g17(_get(G17.format("utl_sa")), codes)
    log("[sd] Census MRTS retail sales by kind of business")
    sales = parse_mrts(pd.read_excel(io.BytesIO(_get(MRTS, binary=True)), sheet_name=None, header=None))
    log("[sd] Ken French 49 industries + market")
    ind = parse_french_monthly(_unzip(_get(FRENCH.format("49_Industry_Portfolios"), binary=True)))
    ff = parse_french_monthly(_unzip(_get(FRENCH.format("F-F_Research_Data_Factors"), binary=True)))
    mkt = (ff["Mkt-RF"] + ff["RF"]).rename("mkt")
    asof = f"{pd.Timestamp.now(tz='UTC'):%Y-%m-%d %H:%M} UTC"
    df = pd.concat([ip.add_prefix("ip:"), util.add_prefix("util:"), sales.add_prefix("rs:"),
                    ind.add_prefix("ind:"), mkt.rename("ind:mkt")], axis=1)
    df.attrs["asof"] = asof
    df.to_parquet(path)
    return _split(df, asof)


def _unzip(b: bytes) -> str:
    with zipfile.ZipFile(io.BytesIO(b)) as z:
        return z.read(z.namelist()[0]).decode("latin-1")


def _split(df: pd.DataFrame, asof: str = "") -> Raw:
    pick = lambda p: df[[c for c in df if c.startswith(p)]].rename(columns=lambda c: c[len(p):])  # noqa: E731
    return Raw(pick("ip:"), pick("util:"), pick("rs:"), pick("ind:").drop(columns="mkt"), df["ind:mkt"],
               asof or str(df.attrs.get("asof", "")))


# ---- features ---------------------------------------------------------------------------------------------------
def yoy(s: pd.Series) -> pd.Series:
    return 100 * np.log(s / s.shift(12))


def features(raw: Raw, m: dict | None = None) -> dict[str, pd.DataFrame]:
    """Per category: monthly features dated by the data month (not yet lagged for publication)."""
    m = m or load_map()
    out = {}
    for k, c in m["categories"].items():
        f = {}
        if c.get("supply"):
            s = yoy(raw.ip[c["supply"]])
            f["s_yoy"], f["s_acc"] = s, s - s.shift(3)
        if c.get("util") and c["util"] in raw.util:
            u = raw.util[c["util"]]
            f["util_chg"] = u - u.shift(12)
        if c.get("demand"):
            d = yoy(raw.sales[[f"{x}" for x in c["demand"]]].sum(axis=1, min_count=len(c["demand"])))
            f["d_yoy"], f["d_acc"] = d, d - d.shift(3)
        if "s_yoy" in f and "d_yoy" in f:
            f["gap"] = f["d_yoy"] - f["s_yoy"]
            f["gap_acc"] = f["gap"] - f["gap"].shift(3)
        out[k] = pd.DataFrame(f)
    return out


def excess_returns(raw: Raw, m: dict | None = None) -> pd.DataFrame:
    m = m or load_map()
    cols = {}
    for k, c in m["categories"].items():
        r = raw.mkt if c["industry"] == "mkt" else raw.ind[c["industry"]] - raw.mkt
        cols[k] = r
    return pd.DataFrame(cols)


def fwd_sum(r: pd.Series, h: int) -> pd.Series:
    """Sum of log-ish monthly returns over months t+1..t+h, dated t."""
    return r[::-1].rolling(h, min_periods=h).sum()[::-1].shift(-1)


# ---- statistics ---------------------------------------------------------------------------------------------------
def nw_t(y: np.ndarray, x: np.ndarray, lags: int) -> tuple[float, float]:
    """OLS slope of y on standardised x and its Newey-West t (Bartlett kernel)."""
    x = (x - x.mean()) / x.std()
    X = np.column_stack([np.ones_like(x), x])
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    e = y - X @ beta
    n = len(y)
    S = (X * e[:, None]).T @ (X * e[:, None])
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = (X[L:] * e[L:, None]).T @ (X[:-L] * e[:-L, None])
        S += w * (G + G.T)
    XtX = np.linalg.inv(X.T @ X)
    V = XtX @ S @ XtX
    return float(beta[1]), float(beta[1] / np.sqrt(V[1, 1])) if V[1, 1] > 0 else float("nan")


def shift_null(y: np.ndarray, x: np.ndarray, lags: int, n: int, rng: np.random.Generator, min_shift=24) -> np.ndarray:
    """|t| of the same regression with x circularly shifted by random offsets (keeps x's own persistence)."""
    N = len(x)
    if N < 2 * min_shift + 1:
        return np.array([])
    offs = rng.integers(min_shift, N - min_shift, size=n)
    return np.array([abs(nw_t(y, np.roll(x, o), lags)[1]) for o in offs])


def study(raw: Raw, split: str = "train", m: dict | None = None, n_null: int = 300, seed: int = 7,
          allow_test: bool = False) -> dict[str, pd.DataFrame]:
    if split == "test" and not allow_test:
        raise PermissionError("test split is locked: pass allow_test=True (and log the look)")
    m = m or load_map()
    lo, hi = SPLITS[split]
    lag = int(m.get("meta", {}).get("publication_lag_months", 1))
    feats, ret = features(raw, m), excess_returns(raw, m)
    rng = np.random.default_rng(seed)
    pred, expl = [], []
    for k, F in feats.items():
        r = ret[k]
        for f in F:
            x_pub = F[f].shift(lag)  # known at the end of month t
            for h in HORIZONS:
                y = fwd_sum(r, h)
                d = pd.concat([x_pub, y], axis=1, keys=["x", "y"]).loc[lo:hi].dropna()
                if len(d) < 48:
                    continue
                b, t = nw_t(d.y.values, d.x.values, max(h, 3))
                null = shift_null(d.y.values, d.x.values, max(h, 3), n_null, rng)
                pred.append({"category": k, "feature": f, "h": h, "n": len(d), "beta_pp": 100 * b, "t": t,
                             "p_null": float(((null >= abs(t)).sum() + 1) / (len(null) + 1)) if len(null) else np.nan,
                             "null_t3": float((null >= 3).mean()) if len(null) else np.nan})
            # explain: same-window 12m excess return vs same-window yoy
            if f in ("s_yoy", "d_yoy", "gap"):
                y12 = r.rolling(12, min_periods=12).sum()
                d = pd.concat([F[f], y12], axis=1, keys=["x", "y"]).loc[lo:hi].dropna()
                if len(d) >= 48:
                    b, t = nw_t(d.y.values, d.x.values, 12)
                    # and the reverse lead: does the past 12m return foretell the next 12m of growth?
                    fut = F[f].shift(-12)
                    e = pd.concat([y12, fut], axis=1, keys=["r", "g"]).loc[lo:hi].dropna()
                    bl, tl = nw_t(e.g.values, e.r.values, 12) if len(e) >= 48 else (np.nan, np.nan)
                    expl.append({"category": k, "feature": f, "n": len(d), "corr": float(d.x.corr(d.y)),
                                 "beta_pp": 100 * b, "t": t, "price_leads_corr": float(e.r.corr(e.g)),
                                 "price_leads_t": tl})
    P = pd.DataFrame(pred)
    if len(P):
        P["q_bh"] = bh(P["p_null"].fillna(1).values)
    return {"predict": P, "explain": pd.DataFrame(expl), "xs": cross_section(feats, ret, lo, hi, lag, rng)}


def bh(p: np.ndarray) -> np.ndarray:
    n = len(p)
    o = np.argsort(p)
    q = p[o] * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(n)
    out[o] = np.minimum(q, 1)
    return out


def cross_section(feats, ret, lo, hi, lag, rng, n_null: int = 1000) -> pd.DataFrame:
    """Each month rank the categories that have the feature, long the top third, short the bottom third, hold
    one month. Null stubs: random rankings each month (coin flips with the same book shape)."""
    rows = []
    for f in ("gap", "gap_acc", "d_yoy", "d_acc", "s_yoy", "s_acc", "util_chg"):
        X = pd.DataFrame({k: F[f] for k, F in feats.items() if f in F and k != "economy"}).shift(lag)
        R = ret[[c for c in X.columns]].shift(-1)
        X, R = X.loc[lo:hi], R.loc[lo:hi]
        if X.shape[1] < 6:
            continue
        spreads, nulls = [], []
        for dt in X.index:
            x, r = X.loc[dt], R.loc[dt]
            ok = x.notna() & r.notna()
            if ok.sum() < 6:
                continue
            x, r = x[ok], r[ok].values
            k = max(1, len(x) // 3)
            order = np.argsort(x.values)
            spreads.append(r[order[-k:]].mean() - r[order[:k]].mean())
            perm = np.argsort(rng.random((n_null, len(r))), axis=1)
            nulls.append(r[perm[:, -k:]].mean(axis=1) - r[perm[:, :k]].mean(axis=1))
        if len(spreads) < 36:
            continue
        s = np.array(spreads)
        t = s.mean() / (s.std(ddof=1) / np.sqrt(len(s)))
        N = np.array(nulls)  # months x draws
        nt = N.mean(axis=0) / (N.std(axis=0, ddof=1) / np.sqrt(N.shape[0]))
        rows.append({"feature": f, "months": len(s), "n_categories": X.shape[1], "spread_pp_m": 100 * s.mean(),
                     "t": t, "hit": float((s > 0).mean()), "p_null": float(((np.abs(nt) >= abs(t)).sum() + 1) / (len(nt) + 1)),
                     "null_t3": float((np.abs(nt) >= 3).mean())})
    return pd.DataFrame(rows)


def latest(raw: Raw, m: dict | None = None) -> pd.DataFrame:
    """Where each category stands on its latest published month."""
    rows = []
    for k, F in features(raw, m).items():
        row = {"category": k}
        for c in F:
            v = F[c].dropna()
            if len(v):
                row[c] = round(float(v.iloc[-1]), 1)
                side = "s" if c.startswith(("s_", "util")) else "d"
                row[f"{side}_month"] = f"{v.index[-1]:%Y-%m}"
        rows.append(row)
    cols = ["s_month", "s_yoy", "s_acc", "util_chg", "d_month", "d_yoy", "d_acc", "gap", "gap_acc"]
    return pd.DataFrame(rows).set_index("category").reindex(columns=cols)
