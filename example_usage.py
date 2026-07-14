"""
Reproduce the paper's cleaning for the four sample sites.

Reads the raw hourly counts, applies the per-site contamination and SARIMA order
from the paper (data/paper_params.csv, i.e. Table 1), writes the cleaned series to
data/sample_output.csv, and prints the per-site summary (matches Table 3).

For NEW data you do not need paper_params.csv; the default configuration selects a
SARIMA order automatically per site:

    from pedestrian_cleaner import clean_dataframe
    cleaned, report = clean_dataframe(pd.read_csv("my_counts.csv"))
"""
import pandas as pd
from pedestrian_cleaner import clean_dataframe, CleanConfig

df = pd.read_csv("data/sample_input.csv")

params = pd.read_csv("data/paper_params.csv")
site_params = {
    r["site"]: {
        "contamination": r["contamination"],
        "sarima_order": ((int(r.p), int(r.d), int(r.q)), (int(r.P), int(r.D), int(r.Q))),
    }
    for _, r in params.iterrows()
}

cleaned, report = clean_dataframe(df, time_col="timestamp",
                                  config=CleanConfig(), site_params=site_params)
cleaned.to_csv("data/sample_output.csv")
report.to_csv("data/sample_report.csv", index=False)
print(report.to_string(index=False))
