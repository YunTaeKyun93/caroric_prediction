import pandas as pd
from src.preprocessing.clean import remove_outliers
from src.features.build_features import build_features


df_raw =  pd.read_csv("data/raw/train.csv")
print("Raw Shape", df_raw.shape)


df_clean = remove_outliers(df_raw)
print("After", df_clean.shape)

df_fe = build_features(df_clean)

output_path = "data/processed/train_clean_fe.csv"
df_fe.to_csv(output_path, index=False)

print(f"⭕️ Saved to {output_path}")

