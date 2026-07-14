# Methodology

This document describes the pedestrian count cleaning and imputation method implemented
in `pedestrian_cleaner.py`, following Shen, Wijayaratna & Saberi (Transportation Research
Record, 2025), with two refinements noted at the end.

The pipeline runs independently on each site (sensor). The goal is to convert a raw
hourly count series — which contains sensor noise, malfunctions, occlusions, tampering,
and deployment-edge gaps — into a clean, gap-free series suitable for analysis.

---

## 0. Preprocessing

**Hourly aggregation.** Sub-hourly readings (e.g. 15-minute intervals) are summed into
hourly counts.

**Weekday-only sequence.** By default, Saturdays and Sundays are removed. Temporary
deployments often span only a week or two, giving too few weekend days to model reliably,
and the study focuses on weekday activity. The remaining weekday hours are placed on a
single ordered sequence, so consecutive weekdays are treated as continuous. This means the
seasonal model sees a pure 24-hour daily cycle and never has to represent the weekend gap
(weekly seasonality is deliberately disregarded, matching the seasonal period of 24).

The series is indexed positionally (`0, 1, 2, …`) for all modelling so that Friday 23:00
is adjacent to Monday 00:00.

## 1. Seasonal-trend decomposition

Pedestrian counts have a strong, repeating daily profile (morning/evening commute peaks,
near-zero overnight). We separate this structure from irregular variation:

```
observed(t) = trend(t) + seasonal(t) + residual(t)
```

We use **STL with robust weighting** (Seasonal-Trend decomposition using Loess). Robust
weighting down-weights outlying observations when estimating the trend and seasonal
components. This is important: with an ordinary (non-robust) decomposition, a stretch of
corrupted data pulls the estimated daily profile toward itself, which both (a) hides the
corruption in the residuals and (b) makes the *normal* daily peaks look anomalous. Robust
STL keeps the seasonal profile clean, so genuine anomalies stand out and normal peaks do
not. (If STL is unavailable, the code falls back to a classical additive decomposition.)

The **residual** series — what remains after removing the trend and daily profile — is the
input to anomaly detection.

## 2. Anomaly detection (Isolation Forest)

An **Isolation Forest** is fit to the residuals of the observed hours. The algorithm
isolates points with few random partition splits; anomalous residuals (unusually large
positive or negative deviations from the expected daily profile) are isolated quickly and
flagged. The `contamination` parameter sets the expected proportion of anomalies and is
the main sensitivity knob (0.02–0.17 in the source study; default 0.05 here).

The output is a boolean mask of anomalous hours.

## 3. Day-level classification

Not all anomalies should be treated the same way. We distinguish two situations:

- **Isolated anomalies** — a few scattered bad hours on an otherwise normal day (e.g. a
  transient spike). These are patched locally.
- **Whole-day anomalies** — a day that is *systematically* wrong (a sensor that
  under-counted all day, or a day-long malfunction). Patching individual hours cannot fix
  such a day; it must be rebuilt.

A fully-observed weekday is classified as **whole-day** if either:

1. its **total volume is a strong outlier** versus the site's typical day —
   `total < 0.40 × median` or `total > 2.5 × median` of the site's daily totals; or
2. it carries an **extreme count of hourly anomalies** (`≥ hourly_backstop`, default 8).

Rule (1) is the key refinement (see below): it catches days that are uniformly depressed
or inflated even when no single hour is extreme, and — crucially — it does *not* flag a
normal-height day that merely carries a couple of peak spikes. Partial days (the first/last
day of a deployment) are never rebuilt whole; their missing portion is reconstructed as a
gap (§4).

## 4. Imputation

Every value that is missing, flagged as an isolated anomaly, or belongs to a whole-day
rebuild is removed and then reconstructed:

**Linear interpolation** for isolated anomalies. A single bad hour between two good
neighbours is well approximated by a straight line — cheap and accurate for short gaps.

**SARIMA** for whole-day rebuilds and for deployment-edge gaps. A Seasonal ARIMA model,
`SARIMA(p,d,q)(P,D,Q)[24]`, captures both short-term autocorrelation and the daily seasonal
structure, so it can synthesise a plausible full day (or a missing morning) consistent with
the surrounding data. **Stationarity and invertibility are enforced** during estimation;
this keeps predictions stable even when several whole days are reconstructed at once
(without it, the model can produce explosive/oscillating output).

The SARIMA **order** is either supplied by the user (one order for all sites) or selected
automatically per site — via `pmdarima.auto_arima` if installed, otherwise by a small
AIC-based search over common orders (default fallback `(1,0,0)(0,1,1)[24]`).

**Leading-edge (first-day) reconstruction.** Sensors are typically installed mid-morning,
so the first day is missing its early hours *at the very start of the series*. SARIMA
cannot backcast across seasonal differencing (there is no prior season to difference
against), and naive backcasting returns zeros. We instead **time-reverse the series**, so
the leading gap becomes a gap at the *end* — a stable forward forecast — reconstruct it,
and reverse back. This recovers a realistic morning profile instead of zeros.

**Post-processing.** Reconstructed counts are clipped to be non-negative and to not exceed
`clip_factor × observed maximum` (default 1.5×), guarding against occasional numerical
overshoot.

## 5. Validation (masking experiment)

To assess imputation accuracy, hold-out masking is used: known-good values are hidden,
reconstructed, and compared to truth via Absolute Percentage Error, `|actual − imputed| /
actual`. Three scenarios of increasing difficulty are evaluated — a single hidden hour
(linear), a masked 7am–9pm daytime block (SARIMA), and a masked full day (SARIMA). Error
grows with gap length, as expected, and percentage error is only meaningful above a small
count floor (near-zero overnight hours are excluded).

---

## Refinements relative to the published method

The published pipeline used classical additive decomposition and defined a "whole-day"
case purely by the count of hourly anomalies (`≥ 5` per day). This implementation makes two
changes that improve robustness without departing from the paper's intent:

1. **Robust STL decomposition** instead of classical decomposition, so anomalies do not
   contaminate the estimated daily profile.
2. **Day-level whole-day trigger** based on the day's total volume (a systematic-corruption
   test), instead of counting hourly flags. This correctly separates a genuinely corrupted
   day from a normal day that happens to carry a few spikes — a case where the hourly-count
   rule misclassifies (rebuilding a good day, or missing a moderately depressed one).

Both changes are exposed as parameters, so the original behaviour can be approximated by
adjusting the configuration.

---

## Parameter summary

| Parameter | Default | Role |
|---|---|---|
| `period` | 24 | Seasonal period (hours). |
| `weekdays_only` | True | Drop weekends; treat weekdays as continuous. |
| `contamination` | 0.05 | Isolation-Forest anomaly fraction (main knob, 0.02–0.17). |
| `day_low` / `day_high` | 0.40 / 2.5 | Day-total bounds for whole-day rebuild. |
| `hourly_backstop` | 8 | Hourly-anomaly count that also triggers whole-day rebuild. |
| `sarima_order` | None (auto) | Fixed SARIMA order, or automatic per-site selection. |
| `enforce_stationarity` / `enforce_invertibility` | True | Stable SARIMA reconstructions. |
| `clip_factor` | 1.5 | Upper clip on reconstructed values. |
| `random_state` | 0 | Seed for reproducible anomaly detection. |
