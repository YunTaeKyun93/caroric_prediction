from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def make_preporcessor():
  num_features= [
    "Exercise_Duration",
    "Body_Temperature_C",
    "BPM",
    "Weight_kg",
    "BMI",
    "Age",
    "Height_m"
  ]
  cat_features =[
    "Gender",
    "Weight_Status"
  ]
  
  numeric_transformer = StandardScaler()
  categorical_transformer = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
  )

  preprocessor = ColumnTransformer(
    transformers=[
      ("num", numeric_transformer, num_features),
      ("cat", categorical_transformer, cat_features),
    ]
  )
  return preprocessor






