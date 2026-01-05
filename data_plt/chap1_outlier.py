import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import pandas as pd
from IPython.display import display
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier 
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
import os 
plt.rcParams['font.family'] = 'AppleGothic'
plt.rcParams['axes.unicode_minus'] = False
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
df = pd.read_csv('data/train.csv')

num_cols = [
  "Exercise_Duration",
  "Body_Temperature(F)",
  "BPM",
  "Calories_Burned"
]

plt.figure(figsize=(14,8))
for i, col in enumerate(num_cols, 1):
  plt.subplot(2,2,i)
  sns.boxplot(x=df[col])
  plt.title(col)


plt.tight_layout()
plt.show()