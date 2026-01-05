import pandas as pd

def remove_outliers(df : pd.DataFrame):
  make_noise =(
    (df["Exercise_Duration"]>3)&
    (df["Calories_Burned"]>5)
  )
  return df[~make_noise].copy()


""" 보수적
1. 운동시간 3미만 운동칼로리 5미만  (데이터 239개 정도 날아갓지만 대량 은 아님)
2. 고강도 / 칼로리소모 많은 데이터는 일단 남겨둠
3. 이상치는 여기서 일단 완료 
"""
# plan B
"""중도적
1.  운동시간 2미만 운동칼로리 2미만
"""

"""완화적 접근
1.  운동시간 1미만 운동칼로리 1미만
"""



