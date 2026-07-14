"""Minimal example of the pedestrian_cleaner API.

Run:  python example_usage.py
"""
import pandas as pd

from pedestrian_cleaner import CleanConfig, clean_dataframe, clean_series, plot_site

# 1. Load a wide CSV: one datetime column + one column per sensor/site.
df = pd.read_csv("example_data.csv", parse_dates=["Time"])

# 2. Clean every site with default settings.
config = CleanConfig(
    period=24,            # daily cycle on hourly data
    weekdays_only=True,   # drop weekends (sparse coverage)
    contamination=0.05,   # expected fraction of anomalous hours (tune 0.02-0.17)
    # sarima_order=((1, 0, 0), (0, 1, 1)),  # fix an order, or leave None to auto-select
)
cleaned, report = clean_dataframe(df, time_col="Time", config=config)

cleaned.to_csv("example_cleaned.csv")
print(report.to_string(index=False))

# 3. Inspect / plot a single site.
result = clean_series(df.set_index("Time")["Belmore Rd"], config, name="Belmore Rd")
print(f"\nBelmore Rd: {result.n_linear} linear + {result.n_sarima} SARIMA hours imputed; "
      f"whole-day rebuilds: {result.rebuilt_days}; volume change {result.volume_change():.1f}%")
plot_site(result, "belmore_overlay.png")
