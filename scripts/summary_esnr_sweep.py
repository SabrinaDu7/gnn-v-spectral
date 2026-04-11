import pandas as pd

df = pd.read_csv("results/esnr_synthetic_sweep.csv")

summary = (
    df.groupby(["family", "noise_type", "noise_frac"])
      .agg(
          mean_esnr=("esnr", "mean"),
          std_esnr=("esnr", "std"),
          mean_outlier_mass=("outlier_mass", "mean"),
          mean_n_outliers=("n_outliers", "mean"),
          mean_sv5=("sv5", "mean"),
      )
      .reset_index()
)

print(summary)