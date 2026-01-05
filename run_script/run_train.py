import pandas as pd
from src.modeling.pipeline import make_preporcessor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
df = pd.read_csv("data/processed/train_clean_fe.csv")


y = df["Calories_Burned"]
X = df.drop(columns=["Calories_Burned"])




preprocessor  = make_preporcessor()

pipe = Pipeline(
  steps=[
    ("preprocessor", preprocessor),
    ("model",LinearRegression )
  ]
)


pipe.fit(X,y)
