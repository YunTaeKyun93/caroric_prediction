import pandas as pd

def remove_outliers(df : pd.DataFrame):
  return df[df["Exercise_Duration"] >= 3].copy()
