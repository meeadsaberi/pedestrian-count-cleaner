"""
pedestrian_cleaner
==================

Automated cleaning and imputation of pedestrian count time-series, implementing the
methodology in:

    Shen, Y., Wijayaratna, K., & Saberi, M. "Time-Series Approaches for Cleaning and
    Imputing Pedestrian Count Data: Implications for Urban Street Classification in
    Sydney." Transportation Research Record (2025).

Pipeline (per site / per sensor column)
---------------------------------------
1.  Aggregate to an hourly grid and (optionally) keep weekdays only. Consecutive
    weekdays are treated as a continuous 24-hour-periodic sequence, so the weekend
    gap is never modelled (weekly effects are disregarded, as in the paper).
2.  Robust seasonal-trend decomposition (STL with robust weighting) so that
    anomalies do not contaminate the estimated daily profile.
3.  Anomaly detection with an Isolation Forest on the decomposition residuals.
4.  Day classification:
      * "whole-day" anomaly  -> the day's total volume is a strong outlier vs the
        site's typical day (systematically depressed/inflated), or the day carries
        an extreme number of hourly anomalies. These days are discarded entirely.
      * "isolated" anomaly    -> a few scattered hourly outliers on an otherwise
        normal day.
5.  Imputation:
      * isolated hourly anomalies        -> linear interpolation
      * whole-day anomalies + gaps       -> SARIMA (seasonal ARIMA, period 24)
      * partial first/last deployment day-> SARIMA; the leading (pre-install)
        morning is reconstructed by time-reversing the series so the backcast
        becomes a stable forward forecast.
    Stationarity and invertibility are enforced during SARIMA estimation.

Usage
-----
Command line::

    python pedestrian_cleaner.py counts.csv -o counts_cleaned.csv --plots plots/

Python API::

    import pandas as pd
    from pedestrian_cleaner import clean_dataframe, CleanConfig

    df = pd.read_csv("counts.csv", parse_dates=["Time"])
    cleaned, report = clean_dataframe(df, time_col="Time", config=CleanConfig())

Input format
------------
A CSV with one datetime column and one or more numeric columns, each holding the
hourly (or sub-hourly) pedestrian count for one sensor/site. Sub-hourly data is
aggregated to hourly automatically. Missing timestamps are allowed.

License: MIT.
"""
from __future__ import annotations

import argparse
import warnings
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

try:  # STL is the preferred (robust) decomposer
    from statsmodels.tsa.seasonal import STL, seasonal_decompose
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from sklearn.ensemble import IsolationForest
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "pedestrian_cleaner requires numpy, pandas, scikit-learn and statsmodels. "
        "Install them with:  pip install numpy pandas scikit-learn statsmodels"
    ) from exc


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
@dataclass
class CleanConfig:
    """Tuneable parameters for the cleaning pipeline.

    Attributes
    ----------
    period:
        Seasonal period in hours (24 for a daily cycle on hourly data).
    weekdays_only:
        If True, drop Saturdays and Sundays before processing (recommended when
        weekend coverage is sparse, as in the source study).
    contamination:
        Expected proportion of anomalous hours for the Isolation Forest
        (statsmodels/sklearn's `contamination`). The paper hand-tuned this per
        site within 0.02-0.17; 0.05 is a sensible default.
    day_low, day_high:
        A fully-observed day is rebuilt whole when its total volume is below
        `day_low * median` or above `day_high * median` of the site's daily
        totals (systematic day-level corruption).
    hourly_backstop:
        A fully-observed day is also rebuilt whole if it carries at least this
        many hourly anomalies (concentrated corruption).
    sarima_order:
        ((p, d, q), (P, D, Q)) to use for every site. If None, an order is
        selected automatically per site (see `select_sarima_order`).
    enforce_stationarity, enforce_invertibility:
        Passed to SARIMAX. Enforcing both (paper default) keeps reconstructions
        stable when several whole days are rebuilt.
    clip_factor:
        Reconstructed values are clipped to [0, clip_factor * observed_max].
    max_iter:
        Maximum optimiser iterations for SARIMA fitting.
    random_state:
        Seed for the Isolation Forest.
    """

    period: int = 24
    weekdays_only: bool = True
    contamination: float = 0.05
    day_low: float = 0.40
    day_high: float = 2.5
    hourly_backstop: int = 8
    sarima_order: Optional[tuple] = None
    enforce_stationarity: bool = True
    enforce_invertibility: bool = True
    clip_factor: float = 1.5
    max_iter: int = 200
    random_state: int = 0
    # candidate orders searched when sarima_order is None
    order_candidates: Sequence[tuple] = field(
        default_factory=lambda: [
            ((1, 0, 0), (0, 1, 1)),
            ((0, 1, 1), (0, 1, 1)),
            ((1, 1, 0), (0, 1, 1)),
            ((2, 0, 0), (0, 1, 1)),
            ((1, 0, 1), (0, 1, 1)),
            ((0, 0, 0), (0, 1, 1)),
        ]
    )


@dataclass
class SiteResult:
    """Cleaning result for one site."""

    name: str
    raw: pd.Series            # weekday hourly series with gaps (NaN)
    cleaned: pd.Series        # fully imputed series
    anomalies: pd.Series      # boolean mask of flagged hours (on raw index)
    rebuilt_days: list        # list of dates rebuilt whole (as 'YYYY-MM-DD')
    order: tuple              # SARIMA order used
    n_linear: int             # hours imputed by linear interpolation
    n_sarima: int             # hours imputed by SARIMA

    def volume_change(self) -> float:
        """Percent change in mean daily volume, raw vs cleaned (weekdays)."""
        def daily_mean(s):
            s = s.dropna()
            return s.groupby(s.index.normalize()).sum().mean()
        b, a = daily_mean(self.raw), daily_mean(self.cleaned)
        return np.nan if not b else (a - b) / b * 100.0


# --------------------------------------------------------------------------- #
# Core steps
# --------------------------------------------------------------------------- #
def _weekday_hourly(series: pd.Series, weekdays_only: bool) -> pd.Series:
    """Aggregate to hourly and reindex onto a regular (weekday) hourly grid."""
    s = series.dropna().sort_index()
    s = s.resample("1h").sum(min_count=1)                 # hourly aggregation
    grid = pd.date_range(s.index.min().normalize(),
                         s.index.max().normalize() + pd.Timedelta("23h"), freq="h")
    if weekdays_only:
        grid = grid[grid.dayofweek < 5]
    return s.reindex(grid)


def select_sarima_order(values: np.ndarray, period: int,
                        candidates: Sequence[tuple], max_iter: int) -> tuple:
    """Pick the ((p,d,q),(P,D,Q)) candidate with the lowest AIC.

    Uses pmdarima.auto_arima when available (more thorough); otherwise a small
    AIC grid search over `candidates`. Falls back to ((1,0,0),(0,1,1))."""
    x = pd.Series(values).interpolate("linear", limit_direction="both").values
    try:  # optional, better search
        import pmdarima as pm
        model = pm.auto_arima(x, seasonal=True, m=period, suppress_warnings=True,
                              error_action="ignore", stepwise=True, max_order=6)
        o, so = model.order, model.seasonal_order
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
    """Clean and impute a single site's pedestrian count series."""
    raw = _weekday_hourly(series, config.weekdays_only)
    idx = raw.index
    observed = raw.notna().values
    if observed.sum() < 2 * config.period:
        # too little data to model; return as-is with linear fill
        cleaned = raw.interpolate("linear", limit_direction="both").clip(lower=0)
        return SiteResult(name, raw, cleaned, pd.Series(False, index=idx),
                          [], config.sarima_order or (), 0, 0)

    # positional series so Fri->Mon is contiguous (no weekend modelled)
    pos = pd.Series(raw.values, index=np.arange(len(raw)))
    filled = pos.interpolate("linear", limit_direction="both")

    # 1. robust decomposition -> residuals
    try:
        resid = STL(filled, period=config.period, robust=True).fit().resid.values
    except Exception:
        resid = seasonal_decompose(filled, model="additive", period=config.period,
                                   extrapolate_trend="freq").resid.values

    # 2. Isolation Forest on residuals (observed hours only)
    r = resid[observed].reshape(-1, 1)
    iso = IsolationForest(contamination=config.contamination, n_estimators=150,
                          max_samples=min(len(r), 256), random_state=config.random_state)
    flags = iso.fit_predict(r) == -1
    anom = pd.Series(False, index=idx)
    anom.values[np.where(observed)[0]] = flags

    # 3. day-level classification
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

    # 4. impute
    order = config.sarima_order or select_sarima_order(
        raw.values, config.period, config.order_candidates, config.max_iter)

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
            fwd = sarima_predict(proc.values)
            proc.values[need & ~lead] = fwd[need & ~lead]
            if lead.any():                       # backcast leading morning via reversal
                tmp = proc.values.copy(); tmp[lead] = np.nan
                proc.values[lead] = sarima_predict(tmp[::-1])[::-1][lead]
        except Exception:
            proc = proc.interpolate("linear", limit_direction="both")

    proc = proc.interpolate("linear", limit_direction="both")
    cap = config.clip_factor * np.nanmax(raw.values)
    proc = proc.clip(lower=0, upper=cap)

    cleaned = pd.Series(proc.values, index=idx)
    # for display: also mark hours on rebuilt days that deviate from the rebuild
    anom_disp = anom.copy()
    if rebuild_days:
        dev = np.abs(raw.values - cleaned.values) > np.maximum(0.35 * cleaned.values, 5)
        anom_disp.values[is_rebuild & observed & dev] = True

    return SiteResult(
        name=name, raw=raw, cleaned=cleaned, anomalies=anom_disp,
        rebuilt_days=sorted(d.strftime("%Y-%m-%d") for d in rebuild_days),
        order=order, n_linear=int(single_mask.sum()), n_sarima=n_sarima,
    )


def clean_dataframe(df: pd.DataFrame, time_col: Optional[str] = None,
                    value_cols: Optional[Sequence[str]] = None,
                    config: Optional[CleanConfig] = None):
    """Clean every site column in a wide dataframe.

    Returns
    -------
    cleaned : pd.DataFrame   (time index + one cleaned column per site)
    report  : pd.DataFrame   (per-site summary: order, anomalies, volume change)
    """
    config = config or CleanConfig()
    df = df.copy()
    if time_col is None:                      # first datetime-like column
        for c in df.columns:
            if np.issubdtype(df[c].dtype, np.datetime64):
                time_col = c; break
        if time_col is None:
            time_col = df.columns[0]
    df[time_col] = pd.to_datetime(df[time_col])
    df = df.set_index(time_col).sort_index()
    if value_cols is None:
        value_cols = [c for c in df.columns if np.issubdtype(df[c].dtype, np.number)]

    cleaned_cols, rows = {}, []
    for col in value_cols:
        res = clean_series(df[col], config, name=col)
        cleaned_cols[col] = res.cleaned
        rows.append({
            "site": col, "sarima_order": f"{res.order[0]}{res.order[1]}[{config.period}]"
            if res.order else "-",
            "n_anomalies": int(res.anomalies.sum()),
            "n_linear_imputed": res.n_linear, "n_sarima_imputed": res.n_sarima,
            "rebuilt_days": ", ".join(res.rebuilt_days),
            "volume_change_pct": round(res.volume_change(), 1),
        })
    cleaned = pd.DataFrame(cleaned_cols)
    cleaned.index.name = time_col
    return cleaned, pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Optional plotting
# --------------------------------------------------------------------------- #
def plot_site(result: SiteResult, out_path: str) -> None:
    """Save a raw-vs-cleaned overlay (green raw, orange cleaned, red anomalies)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates

    idx = result.raw.index
    cal = pd.date_range(idx.min().normalize(),
                        idx.max().normalize() + pd.Timedelta("23h"), freq="h")
    raw_c = result.raw.reindex(cal)
    clean_c = result.cleaned.reindex(cal)
    an = result.anomalies[result.anomalies]

    fig, ax = plt.subplots(figsize=(13, 3.2))
    ax.plot(clean_c.index, clean_c.values, color="orange", lw=2, label="Cleaned", zorder=2)
    ax.plot(raw_c.index, raw_c.values, color="green", lw=0.9, label="Raw", zorder=3)
    ax.scatter(an.index, result.raw.reindex(an.index), color="red", s=14,
               zorder=4, label="Anomaly")
    for d in pd.date_range(cal.min().normalize(), cal.max().normalize(), freq="D"):
        if d.dayofweek >= 5:
            ax.axvspan(d, d + pd.Timedelta("1D"), color="lightgrey", alpha=0.5)
    title = f"{result.name}  SARIMA{result.order[0]}{result.order[1]}"
    if result.rebuilt_days:
        title += f"  | whole-day rebuild: {', '.join(result.rebuilt_days)}"
    ax.set_title(title, loc="left", fontsize=9, fontweight="bold")
    ax.set_ylabel("Hourly count"); ax.margins(x=0); ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout(); fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Command-line interface
# --------------------------------------------------------------------------- #
def main(argv=None) -> None:
    p = argparse.ArgumentParser(
        description="Clean and impute pedestrian count time-series (Shen, "
                    "Wijayaratna & Saberi, TRR 2025).")
    p.add_argument("input", help="Input CSV: one datetime column + one column per site.")
    p.add_argument("-o", "--output", default="cleaned.csv", help="Output CSV path.")
    p.add_argument("--time-col", default=None, help="Name of the datetime column.")
    p.add_argument("--report", default=None, help="Optional CSV path for the per-site report.")
    p.add_argument("--plots", default=None, help="Optional directory for per-site overlay PNGs.")
    p.add_argument("--contamination", type=float, default=0.05)
    p.add_argument("--period", type=int, default=24)
    p.add_argument("--keep-weekends", action="store_true",
                   help="Process all days instead of weekdays only.")
    p.add_argument("--order", default=None,
                   help="Fixed SARIMA order 'p,d,q,P,D,Q' for all sites (else auto).")
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
    print(f"Wrote cleaned series -> {args.output}  ({cleaned.shape[1]} sites, "
          f"{cleaned.shape[0]} timestamps)")
    print(report.to_string(index=False))
    if args.report:
        report.to_csv(args.report, index=False)
        print(f"Wrote report -> {args.report}")
    if args.plots:
        import os
        os.makedirs(args.plots, exist_ok=True)
        for col in cleaned.columns:
            res = clean_series(df.set_index(pd.to_datetime(
                df[args.time_col or df.columns[0]]))[col], cfg, name=col)
            safe = "".join(c if c.isalnum() else "_" for c in str(col))
            plot_site(res, os.path.join(args.plots, f"{safe}.png"))
        print(f"Wrote per-site plots -> {args.plots}/")


if __name__ == "__main__":
    main()
