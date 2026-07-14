# Pedestrian Time-Series Count Data Cleaner

Cleaning and imputation of pedestrian count time-series, implementing the method in:

> Shen, Y., Wijayaratna, K., & Saberi, M. *Time-Series Approaches for Cleaning and
> Imputing Pedestrian Count Data: Implications for Urban Street Classification in
> Sydney.* Transportation Research Record.

Give it raw hourly counts and it returns a cleaned, gap-free series per site.

## Method

For each site the pipeline:

1. Aggregates to an hourly grid and keeps weekdays only (consecutive weekdays are
   treated as continuous, so the weekend gap is never modelled).
2. Runs a robust STL decomposition so anomalies do not distort the daily profile.
3. Flags anomalous hours with an Isolation Forest on the residuals.
4. Classifies each day: a *whole-day* anomaly (daily total a strong outlier versus
   the site's median day) is discarded and rebuilt; *isolated* hourly anomalies are
   repaired in place.
5. Imputes: isolated hours by linear interpolation; whole days and deployment-edge
   gaps by SARIMA (Kalman smoother for interior gaps; forward forecast for the
   trailing edge; a time-reversed forecast for the leading edge).

## Install

```
pip install -r requirements.txt
```

Requires numpy, pandas, scikit-learn and statsmodels. `pmdarima` is optional
(enables a more thorough order search via `use_pmdarima=True`).

## Usage

Command line:

```
python pedestrian_cleaner.py data/sample_input.csv -o cleaned.csv --report report.csv
```

Python:

```python
import pandas as pd
from pedestrian_cleaner import clean_dataframe, CleanConfig

df = pd.read_csv("data/sample_input.csv")
cleaned, report = clean_dataframe(df, time_col="timestamp", config=CleanConfig())
cleaned.to_csv("cleaned.csv")
```

## Input / output format

**Input** (`data/sample_input.csv`): a datetime column plus one numeric column per
site holding the hourly count. Missing timestamps and gaps are allowed.

```
timestamp,Belmore Rd,Concord Rd West,Elizabeth St B,Middle St A
2024-03-15 00:00:00,2.0,,,
2024-03-15 01:00:00,0.0,,,
```

**Output** (`data/sample_output.csv`): the same columns on a regular weekday hourly
grid, fully imputed. The `report` also lists, per site, the SARIMA order used, the
number of anomalies, hours imputed by linear vs SARIMA, days rebuilt, and the
percentage change in mean daily volume.

## Reproducing the paper

`example_usage.py` cleans the four sample sites using the per-site contamination and
SARIMA order from the paper (`data/paper_params.csv`, i.e. Table 1) and reproduces
the Table 3 volume changes:

```
python example_usage.py
```

| Site | Volume change | Hours imputed (linear + SARIMA) |
|---|---|---|
| Belmore Rd | +40.6% | 3 + 63 |
| Concord Rd West | +12.2% | 8 + 24 |
| Elizabeth St B | +22.5% | 6 + 25 |
| Middle St A | +23.6% | 3 + 26 |

## Parameters

Key `CleanConfig` fields: `contamination` (Isolation Forest outlier fraction,
0.02–0.17 per site in the paper), `day_low` / `day_high` (whole-day rebuild
thresholds, 0.40 / 2.5), and `sarima_order` (fixed order, or `None` for automatic
selection). Very short or sparse deployments are sensitive to these settings;
supplying a per-site `contamination` and `sarima_order` (as in `example_usage.py`)
gives the most reliable reconstruction.

## License

MIT — see [LICENSE](LICENSE).
