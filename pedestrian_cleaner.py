"""
Cleaning and imputation of pedestrian count time-series.

Implements the pipeline from Shen, Wijayaratna & Saberi, "Time-Series Approaches
for Cleaning and Imputing Pedestrian Count Data" (Transportation Research Record):
robust STL decomposition, Isolation Forest anomaly detection, a day-level
whole-day reconstruction rule, and imputation by linear interpolation (isolated
hours) or SARIMA (whole days and deployment-edge gaps).

CLI:   python pedestrian_cleaner.py sample_input.csv -o cleaned.csv --report report.csv
API:   from pedestrian_cleaner import clean_dataframe, CleanConfig
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from statsmodels.tsa.seasonal import STL, seasonal_decompose
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.ensemble import IsolationForest


@dataclass
class CleanConfig:
    period: int = 24                 # seasonal period (24 = daily cycle on hourly data)
    weekdays_only: bool = True       # drop weekends before processing
    contamination: float = 0.05      # Isolation Forest outlier fraction (paper: 0.02-0.17 per site)
    day_low: float = 0.40            # rebuild a day whose total < day_low * median day
    day_high: float = 2.5            # rebuild a day whose total > day_high * median day
    hourly_backstop: int = 8         # or rebuild a day carrying >= this many hourly anomalies
    sarima_order: Optional[tuple] = None   # fixed ((p,d,q),(P,D,Q)); None -> auto per site
    use_pmdarima: bool = False       # if True and pmdarima is installed, use auto_arima
    enforce_stationarity: bool = True
    enforce_invertibility: bool = True
    clip_factor: float = 1.5         # clip reconstructions to [0, clip_factor * observed max]
    max_iter: int = 200
    random_state: int = 0
    order_candidates: Sequence[tuple] = field(default_factory=lambda: [
        ((1, 0, 0), (0, 1, 1)), ((0, 1, 1), (0, 1, 1)), ((1, 1, 0), (0, 1, 1)),
        ((2, 0, 0), (0, 1, 1)), ((1, 0, 1), (0, 1, 1)), ((0, 0, 0), (0, 1, 1)),
    ])


@dataclass
class SiteResult:
    name: str
    raw: pd.Series            # weekday hourly series with gaps (NaN)
    cleaned: pd.Series        # fully imputed series
    anomalies: pd.Series      # boolean mask of flagged hours
    rebuilt_days: list        # dates rebuilt whole ('YYYY-MM-DD')
    order: tuple              # SARIMA order used
    n_linear: int             # hours imputed by linear interpolation
    n_sarima: int             # hours imputed by SARIMA

    def volume_change(self) -> float:
        """Percent change in mean daily volume, raw vs cleaned."""
        def daily_mean(s):
            s = s.dropna()
            return s.groupby(s.index.normalize()).sum().mean()
        b, a = daily_mean(self.raw), daily_mean(self.cleaned)
        return np.nan if not b else (a - b) / b * 100.0


def _weekday_hourly(series: pd.Series, weekdays_only: bool) -> pd.Series:
    """Aggregate to an hourly grid and (optionally) keep weekdays only."""
    s = series.dropna().sort_index().resample("1h").sum(min_count=1)
    grid = pd.date_range(s.index.min().normalize(),
                         s.index.max().normalize() + pd.Timedelta("23h"), freq="h")
    if weekdays_only:
        grid = grid[grid.dayofweek < 5]
    return s.reindex(grid)


def select_sarima_order(values, period, candidates, max_iter, use_pmdarima=False) -> tuple:
    """Lowest-AIC order from a small stable grid (or pmdarima.auto_arima if requested)."""
    x = pd.Series(values).interpolate("linear", limit_direction="both").values
    if use_pmdarima:
        try:
            import pmdarima as pm
            m = pm.auto_arima(x, seasonal=True, m=period, suppress_warnings=True,
                              error_action="ignore", stepwise=True, max_order=6)
            o, so = m.order, m.seasonal_order
            return ((o[0], o[1], o[2]), (so[0], so[1], so[2]))
        except Exception:
            pass
    best, best_aic = candidates[0], np.inf
    for order in candidates:
        try:
            res = SARIMAX(x, order=order[0], seasonal_order=order[1] + (period,),
                          enforce_stationarity=True, enforce_invertibility=True
                          ).fit(disp=False, maxiter=max_iter)
            if np.isfinite(res.aic) and res.aic < best_aic:
                best, best_aic = order, res.aic
        except Exception:
            continue
    return best


def clean_series(series: pd.Series, config: CleanConfig, name: str = "site") -> SiteResult:
    """Clean and impute one site's hourly count series."""
    raw = _weekday_hourly(series, config.weekdays_only)
    idx = raw.index
    observed = raw.notna().values
    if observed.sum() < 2 * config.period:
        cleaned = raw.interpolate("linear", limit_direction="both").clip(lower=0)
        return SiteResult(name, raw, cleaned, pd.Series(False, index=idx),
                          [], config.sarima_order or (), 0, 0)

    # positional series so Fri -> Mon is contiguous (weekend never modelled)
    pos = pd.Series(raw.values, index=np.arange(len(raw)))
    filled = pos.interpolate("linear", limit_direction="both")

    # robust decomposition -> residuals
    try:
        resid = STL(filled, period=config.period, robust=True).fit().resid.values
    except Exception:
        resid = seasonal_decompose(filled, model="additive", period=config.period,
                                   extrapolate_trend="freq").resid.values

    # Isolation Forest on residuals (observed hours only)
    r = resid[observed].reshape(-1, 1)
    iso = IsolationForest(contamination=config.contamination, n_estimators=150,
                          max_samples=min(len(r), 256), random_state=config.random_state)
    flags = iso.fit_predict(r) == -1
    anom = pd.Series(False, index=idx)
    anom.values[np.where(observed)[0]] = flags

    # day-level classification: which whole days to rebuild
    day_key = pd.Series(idx.normalize(), index=idx)
    obs_per_day = pd.Series(observed, index=idx).groupby(idx.normalize()).sum()
    totals = raw.groupby(idx.normalize()).sum()
    full_totals = totals[obs_per_day >= config.period]
    per_day_anom = anom.groupby(idx.normalize()).sum()
    med = full_totals.median() if len(full_totals) else np.nan
    rebuild_days = set()
    if np.isfinite(med) and med > 0:
        for d in full_totals.index:
            if (full_totals[d] < config.day_low * med
                    or full_totals[d] > config.day_high * med
                    or per_day_anom.get(d, 0) >= config.hourly_backstop):
                rebuild_days.add(d)

    is_rebuild = day_key.isin(rebuild_days).values
    single_mask = anom.values & ~is_rebuild
    edge_or_rebuild = (~observed) | is_rebuild

    order = config.sarima_order or select_sarima_order(
        raw.values, config.period, config.order_candidates, config.max_iter,
        config.use_pmdarima)

    proc = pos.copy()
    proc[anom.values | is_rebuild] = np.nan
    proc[single_mask] = proc.interpolate("linear")[single_mask]   # isolated -> linear

    need = edge_or_rebuild & proc.isna().values
    n_sarima = int(need.sum())
    if need.any():
        first_obs = int(np.argmax(observed))
        lead = np.zeros(len(proc), bool)
        lead[:first_obs] = proc.isna().values[:first_obs]

        def sarima_predict(v):
            res = SARIMAX(v, order=order[0], seasonal_order=order[1] + (config.period,),
                          enforce_stationarity=config.enforce_stationarity,
                          enforce_invertibility=config.enforce_invertibility
                          ).fit(disp=False, maxiter=config.max_iter)
            return np.asarray(res.get_prediction(start=0, end=len(v) - 1).predicted_mean)

        try:
            fwd = sarima_predict(proc.values)                 # Kalman smoother (interior gaps)
            proc.values[need & ~lead] = fwd[need & ~lead]
            if lead.any():                                    # leading gap via time reversal
                tmp = proc.values.copy(); tmp[lead] = np.nan
                proc.values[lead] = sarima_predict(tmp[::-1])[::-1][lead]
        except Exception:
            proc = proc.interpolate("linear", limit_direction="both")

    proc = proc.interpolate("linear", limit_direction="both")
    cap = config.clip_factor * np.nanmax(raw.values)
    proc = proc.clip(lower=0, upper=cap)
    cleaned = pd.Series(proc.values, index=idx)

    # mark hours on rebuilt days that deviate from the reconstruction (for plotting)
    anom_disp = anom.copy()
    if rebuild_days:
        dev = np.abs(raw.values - cleaned.values) > np.maximum(0.35 * cleaned.values, 5)
        anom_disp.values[is_rebuild & observed & dev] = True

    return SiteResult(name, raw, cleaned, anom_disp,
                      sorted(d.strftime("%Y-%m-%d") for d in rebuild_days),
                      order, int(single_mask.sum()), n_sarima)


def clean_dataframe(df, time_col=None, value_cols=None, config=None, site_params=None):
    """Clean every count column in a wide dataframe (datetime + one column per site).

    site_params: optional {column: {"contamination": .., "sarima_order": ..}} to
    override the config per site (e.g. to reproduce the paper's Table 1 settings).
    """
    from dataclasses import replace
    config = config or CleanConfig()
    site_params = site_params or {}
    df = df.copy()
    if time_col is None:
        for c in df.columns:
            if np.issubdtype(df[c].dtype, np.datetime64):
                time_col = c; break
        time_col = time_col or df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()
    if value_cols is None:
        value_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]

    cleaned_cols, rows = {}, []
    for col in value_cols:
        cfg = replace(config, **site_params[col]) if col in site_params else config
        res = clean_series(df[col], cfg, name=col)
        cleaned_cols[col] = res.cleaned
        rows.append({
            "site": col,
            "sarima_order": f"{res.order[0]}{res.order[1]}[{config.period}]" if res.order else "-",
            "n_anomalies": int(res.anomalies.sum()),
            "n_linear_imputed": res.n_linear,
            "n_sarima_imputed": res.n_sarima,
            "rebuilt_days": ", ".join(res.rebuilt_days),
            "volume_change_pct": round(res.volume_change(), 1),
        })
    cleaned = pd.DataFrame(cleaned_cols)
    cleaned.index.name = time_col
    return cleaned, pd.DataFrame(rows)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Clean and impute pedestrian count time-series.")
    p.add_argument("input", help="Input CSV: one datetime column + one column per site.")
    p.add_argument("-o", "--output", default="cleaned.csv", help="Output CSV path.")
    p.add_argument("--time-col", default=None, help="Name of the datetime column.")
    p.add_argument("--report", default=None, help="Optional per-site summary CSV.")
    p.add_argument("--contamination", type=float, default=0.05)
    p.add_argument("--period", type=int, default=24)
    p.add_argument("--keep-weekends", action="store_true", help="Process all days, not just weekdays.")
    p.add_argument("--order", default=None, help="Fixed SARIMA order 'p,d,q,P,D,Q' (else auto).")
    args = p.parse_args(argv)

    order = None
    if args.order:
        v = [int(x) for x in args.order.split(",")]
        order = ((v[0], v[1], v[2]), (v[3], v[4], v[5]))
    cfg = CleanConfig(period=args.period, weekdays_only=not args.keep_weekends,
                      contamination=args.contamination, sarima_order=order)

    df = pd.read_csv(args.input)
    cleaned, report = clean_dataframe(df, time_col=args.time_col, config=cfg)
    cleaned.to_csv(args.output)
    print(f"Wrote {args.output}  ({cleaned.shape[1]} sites, {cleaned.shape[0]} timestamps)")
    print(report.to_string(index=False))
    if args.report:
        report.to_csv(args.report, index=False)
        print(f"Wrote {args.report}")


if __name__ == "__main__":
    main()
