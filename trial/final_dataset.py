import pandas as pd

# CSV 불러오기
tess = pd.read_csv("datasets/tess_cleaned.csv")
k2 = pd.read_csv("datasets/k2_cleaned.csv")
kepler = pd.read_csv("datasets/kepler_cleaned.csv")

print("원본 데이터 개수:")
print(f"TESS: {len(tess)}")
print(f"K2: {len(k2)}")
print(f"Kepler: {len(kepler)}")

# 합치기
combined = pd.concat([tess, k2, kepler], axis=0, ignore_index=True)
combined = combined.drop(columns=["confidence"])

print(f"\n합친 데이터 개수: {len(combined)}")

# CSV로 저장
combined.to_csv("datasets/final_exoplanets.csv", index=False)
print("합친 데이터셋 저장 완료: datasets/final_exoplanets.csv")


# 결측치 확인
# 각 컬럼별 결측치 개수 확인
print(combined.isnull().sum())

# 결측치 비율 확인
print((combined.isnull().mean() * 100).round(2))
