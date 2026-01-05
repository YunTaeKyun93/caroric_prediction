import pandas as pd





train = pd.read_csv('./data/train.csv')
test  = pd.read_csv('./data/test.csv')
sub   = pd.read_csv('./data/sample_submission.csv')




print(train.shape)
print(test.shape)
print(sub.shape)
print(train.dtypes)

print('===================================================')


print(train.columns)
print(test.columns)
print(sub.columns)

target_col = "Calories_Burned"

def summary_feature_info(src_df):
  df_info = pd.DataFrame()
  df_info["feature_name"] = train.columns
  df_info["dtype"] = train.dtypes

  df_info["고유값수"] =train.nunique().values
  df_info["결측치"] =train.isna().sum()

  df_info["샘플값_0"] = train.sample(1).values[0]
  df_info["샘플값_1"] = train.sample(1).values[0]
  df_info["샘플값_2"] = train.sample(1).values[0]
  return df_info

summary_df = summary_feature_info(train)




print(summary_df)


