# run_script/finalize_submission.py
import os
import pandas as pd
import numpy as np

# 경로 설정
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# 파일 경로
input_path = os.path.join(project_root, "outputs", "submission_stacking.csv")
output_path = os.path.join(project_root, "outputs", "submission_final_clipped.csv")

def main():
    print("🧹 Final Polish for Submission...")
    
    if not os.path.exists(input_path):
        print(f"❌ Error: {input_path} 파일이 없습니다. train_stacking.py를 먼저 실행하세요.")
        return

    # 불러오기
    df = pd.read_csv(input_path)
    original_mean = df['Calories_Burned'].mean()
    
    # 1. 음수값 확인 및 보정
    negatives = df[df['Calories_Burned'] < 0]
    print(f"🔍 Found {len(negatives)} negative predictions.")
    
    if len(negatives) > 0:
        print(negatives.head())
        print("🛠️  Clipping negative values to 0...")
        # 음수면 0으로, 아니면 그대로
        df['Calories_Burned'] = df['Calories_Burned'].apply(lambda x: max(0, x))
    else:
        print("✅ No negative values found.")

    # 2. 너무 작은 값 보정 (선택 사항)
    # 훈련 데이터에서 최소 칼로리가 1이었으므로, 1보다 작은 값을 1로 맞추는 것도 방법입니다.
    # df['Calories_Burned'] = df['Calories_Burned'].apply(lambda x: max(1.0, x))

    print(f"📊 Mean Value Change: {original_mean:.4f} -> {df['Calories_Burned'].mean():4f}")

    # 저장
    df.to_csv(output_path, index=False)
    print("="*50)
    print(f"🚀 FINAL SUBMISSION READY: {output_path}")
    print("="*50)

if __name__ == "__main__":
    main()