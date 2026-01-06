import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.ensemble import GradientBoostingRegressor

from src.modeling.pipeline import make_preprocessor


train_df = pd.read_csv("data/processed/train_clean_fe.csv")
test_df  = pd.read_csv("data/processed/test_clean_fe.csv")


train_df = train_df[train_df["Exercise_Duration"] >= 3].copy()


X_train = train_df.drop(columns=["Calories_Burned", "ID"])
y_raw   = train_df["Calories_Burned"]
y_log   = np.log1p(y_raw)

X_test = test_df.drop(columns=["ID"])


model = GradientBoostingRegressor(
    n_estimators=1517,
    max_depth=3,
    learning_rate=0.049247820455736474,
    subsample=0.7054530162494841,
    min_samples_leaf=10,
    loss="huber",
    random_state=42
)

pipe = Pipeline(
    steps=[
        ("preprocess", make_preprocessor()),
        ("model", model)
    ]
)


pipe.fit(X_train, y_log)


y_pred_log = pipe.predict(X_test)
y_pred = np.expm1(y_pred_log)

y_pred = np.clip(y_pred, 0, None)


submission = pd.DataFrame({
    "ID": test_df["ID"],
    "Calories_Burned": y_pred
})

submission.to_csv("submission_anchor_gb.csv", index=False)

print("submission_anchor_gb.csv 생성 완료")
