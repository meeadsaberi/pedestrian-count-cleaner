# Pedestrian Time-Series Count Data Cleaner

Automated cleaning and imputation of pedestrian count time-series.

This tool reproduces the methodology in:

> Shen, Y., Wijayaratna, K., & Saberi, M. (2025). *Time-Series Approaches for Cleaning
> and Imputing Pedestrian Count Data: Implications for Urban Street Classification in
> Sydney.* Transportation Research Record.

Pedestrian counters (passive infrared, video, etc.) routinely produce erroneous data
from occlusion, tampering, malfunction, and installation/removal mid-day. This package
takes raw count series and returns a cleaned, gap-free series by combining seasonal-trend
decomposition, Isolation-Forest anomaly detection, and SARIMA/linear imputation. See
[`METHODOLOGY.md`](METHODOLOGY.md) for the full method.

---

## Installation

```bash
pip install -r requirements.txt
```

Requires Python 3.9+, `numpy`, `pandas`, `scikit-learn`, `statsmodels` (and `matplotlib`
for plots). `pmdarima` is optional and, if installed, is used for more thorough automatic
SARIMA order selection.

## Input format

A CSV with **one datetime column** and **one numeric column per sensor/site**:

| Time                | Belmore Rd | Concord Rd West | Middle St A |
|---------------------|-----------:|----------------:|------------:|
| 2024-03-15 00:00:00 | 0          | ...             | ...         |
| 2024-03-15 01:00:00 | 2          | ...             | ...         |

Sub-hourly data (e.g. 15-minute) is aggregated to hourly automatically. Missing
timestamps and gaps are allowed. A comma inside a column name will break CSV parsing —
quote such headers.

## Quick start

Command line:

```bash
python pedestrian_cleaner.py counts.csv -o counts_cleaned.csv --report report.csv --plots plots/
```

Python API:

```python
import pandas as pd
from pedestrian_cleaner import clean_dataframe, CleanConfig

df = pd.read_csv("counts.csv", parse_dates=["Time"])
cleaned, report = clean_dataframe(df, time_col="Time", config=CleanConfig())
cleaned.to_csv("counts_cleaned.csv")
print(report)
```

`report` summarises, per site: the SARIMA order used, number of anomalies, hours imputed
by linear interpolation vs SARIMA, which days were rebuilt whole, and the resulting
percentage change in mean daily volume.

## Key options (`CleanConfig`)

| Option | Default | Meaning |
|---|---|---|
| `period` | `24` | Seasonal period (hours). 24 = daily cycle on hourly data. |
| `weekdays_only` | `True` | Drop weekends before processing (recommended for sparse weekend coverage). |
| `contamination` | `0.05` | Expected fraction of anomalous hours (Isolation Forest). Tune within `0.02–0.17`. |
| `day_low`, `day_high` | `0.40`, `2.5` | A day is rebuilt whole if its total is `< day_low×median` or `> day_high×median`. |
| `hourly_backstop` | `8` | ...or if it carries at least this many hourly anomalies. |
| `sarima_order` | `None` | Fix `((p,d,q),(P,D,Q))` for all sites, or leave `None` to auto-select per site. |
| `enforce_stationarity` / `enforce_invertibility` | `True` | Keep SARIMA reconstructions stable. |
| `clip_factor` | `1.5` | Reconstructed values clipped to `[0, clip_factor × observed max]`. |

### CLI flags

```
python pedestrian_cleaner.py INPUT.csv
    -o, --output        output CSV (default cleaned.csv)
    --time-col          name of the datetime column (auto-detected if omitted)
    --report            write the per-site report CSV
    --plots             directory for raw-vs-cleaned overlay PNGs
    --contamination     Isolation Forest contamination (default 0.05)
    --period            seasonal period in hours (default 24)
    --keep-weekends     process all days instead of weekdays only
    --order p,d,q,P,D,Q fix one SARIMA order for all sites (else auto)
```

## Output

- **Cleaned CSV** — the datetime index plus one fully-imputed column per site.
- **Report CSV** — per-site diagnostics.
- **Plots** (optional) — for each site, an overlay of raw (green), cleaned (orange),
  detected anomalies (red), with weekends shaded.

## Notes & tips

- **Contamination** is the main tuning knob. Higher values flag more hourly outliers.
  In the source study it was set per site within `0.02–0.17`; start at `0.05` and adjust
  by inspecting the overlay plots.
- **Short deployments** (a few days) and **very low-volume sites** (a few pedestrians per
  hour) are intrinsically hard: SARIMA has little history to learn from and percentage
  errors explode near zero. Treat their output with caution.
- **Reproducibility.** Isolation-Forest results depend on `random_state`; set it for
  deterministic runs.

## License

MIT — see [`LICENSE`](LICENSE).
