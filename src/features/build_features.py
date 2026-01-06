import pandas as pd
import numpy as np

def load_data(path):
  
    return pd.read_csv(path)

def clean_data(df: pd.DataFrame, is_train: bool = True) -> pd.DataFrame:

    df = df.copy()
    
    if is_train and "Calories_Burned" in df.columns:
        mask = (df["Exercise_Duration"] >= 2) & (df["Calories_Burned"] > 2)
        df = df[mask]
        
    return df

def generate_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()


    height_inch = df["Height(Feet)"] * 12 + df["Height(Remainder_Inches)"]
    df["Height_m"] = height_inch * 0.0254
    df["Weight_kg"] = df["Weight(lb)"] * 0.453592
    df["Body_Temp_C"] = (df["Body_Temperature(F)"] - 32) * 5.0/9.0
    
    df["Gender_Num"] = df["Gender"].map({"M": 1, "F": 0})
    df["BMI"] = df["Weight_kg"] / (df["Height_m"] ** 2)
  
    df["Duration_x_BPM"] = df["Exercise_Duration"] * df["BPM"]
    df["Duration_x_Weight"] = df["Exercise_Duration"] * df["Weight_kg"] 
    df["Duration_x_Age"] = df["Exercise_Duration"] * df["Age"]
    df["Duration_x_Gender"] = df["Exercise_Duration"] * df["Gender_Num"]

    df["Total_Factor"] = df["Exercise_Duration"] * df["BPM"] * df["Weight_kg"]
    df["Intensity"] = df["BPM"] / df["Exercise_Duration"].replace(0, 1)

    
    drop_cols = [
        "Height(Feet)", "Height(Remainder_Inches)", 
        "Weight(lb)", "Body_Temperature(F)", "Gender", "ID"
    ]
    df = df.drop(columns=drop_cols, errors='ignore')

    return df