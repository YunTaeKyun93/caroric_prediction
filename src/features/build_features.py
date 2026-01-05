import pandas as pd



def add_height_features(df : pd.DataFrame) :
  df_fe = df.copy()  

  df_fe["Height_in"] = (
      df_fe["Height(Feet)"] * 12 +
      df_fe["Height(Remainder_Inches)"]
  )

  df_fe["Height_m"] = df_fe["Height_in"] * 0.0254
  return df_fe


def add_bmi(df: pd.DataFrame):
  df = df.copy()
  df["Weight_kg"] = df["Weight(lb)"]  *0.453592
  df["BMI"] = df["Weight_kg"]  / (df["Height_m"] *2)
  return df 


def convert_temp(df:pd.DataFrame):
  df = df.copy()
  df["Body_Temperature_C"] = (
  (  df["Body_Temperature(F)"]-32) * 5 / 9
  )

  return df 

def drop_raw_units (df: pd.DataFrame)->pd.DataFrame:
  df = df.copy()
  
  drop_cols =[
    "Height(Feet)",
    "Height(Remainder_Inches)",
    "Weight(lb)",
    "Body_Temperature(F)",
    "Height_in"
  ]

  return df.drop(columns=drop_cols)

def build_features (df):
  df = add_height_features(df)
  df = add_bmi(df)
  df = convert_temp(df)
  df = drop_raw_units(df)

  return df

# anchor feature
