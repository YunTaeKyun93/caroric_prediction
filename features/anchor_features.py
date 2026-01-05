import pandas as pd



def make_features(X : pd.DataFrame) -> pd.DataFrame:
  Xf = X.copy()
  return Xf