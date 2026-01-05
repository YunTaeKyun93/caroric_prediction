import pandas as pd
from src.modeling.pipeline import make_preporcessor
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import root_mean_squared_error
# from sklearn.ensemble import 


df = pd.read_csv("data/processed/train_clean_fe.csv")

y = df["Calories_Burned"]
X = df.drop(columns=["Calories_Burned", "ID"])

preprocessor  = make_preporcessor()

X_train, X_valid , y_train, y_valid = train_test_split(
  X,y,
  test_size=0.2,
  random_state=42
)

pipe_lr = Pipeline(
  steps=[
    ("preprocessor", preprocessor),
    ("model",LinearRegression() )
  ]
)

pipe_lr = Pipeline(
  steps=[
    ("preprocessor", preprocessor),
    ("model",LinearRegression() )
  ]
)



pipe_lr.fit(X_train, y_train)
pred_lr = pipe_lr.predict(X_valid)

rmse_lr = root_mean_squared_error(
y_valid, pred_lr
)

print(f"Base ProtoType Linear Regression {rmse_lr:.3f}" )



