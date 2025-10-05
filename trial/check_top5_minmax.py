import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('datasets/exoplanets.csv', low_memory=False)

y_full = df['koi_disposition']
X_full = df.drop(['koi_disposition', 'kepoi_name'], axis=1, errors='ignore')

# 수치형 컬럼만
numeric_cols = X_full.select_dtypes(include=[np.number]).columns
X_full = X_full[numeric_cols]

# 피처 엔지니어링 (핵심 5개에 포함된 것만)
X_full['planet_star_ratio'] = X_full['koi_prad'] / (X_full['koi_srad'] + 1e-10)
X_full['planet_density_proxy'] = X_full['koi_prad']**3 / (X_full['koi_sma']**2 + 1e-10)

X_full = X_full.replace([np.inf, -np.inf], np.nan)

# 핵심 5개 컬럼
top5_features = ['koi_prad', 'dec', 'koi_smet', 'planet_star_ratio', 'planet_density_proxy']

print("=" * 100)
print("📊 핵심 5개 컬럼 통계")
print("=" * 100)

print(f"\n{'컬럼명':<30} {'최솟값':<20} {'최댓값':<20} {'중앙값':<20} {'평균':<20}")
print("-" * 100)

for feat in top5_features:
    min_val = X_full[feat].min()
    max_val = X_full[feat].max()
    median_val = X_full[feat].median()
    mean_val = X_full[feat].mean()
    
    print(f"{feat:<30} {min_val:<20.6f} {max_val:<20.6f} {median_val:<20.6f} {mean_val:<20.6f}")

print("\n" + "=" * 100)
print("📋 상세 통계")
print("=" * 100)

for feat in top5_features:
    print(f"\n🔹 {feat}")
    print(f"   최솟값:  {X_full[feat].min():.6f}")
    print(f"   최댓값:  {X_full[feat].max():.6f}")
    print(f"   범위:    {X_full[feat].max() - X_full[feat].min():.6f}")
    print(f"   중앙값:  {X_full[feat].median():.6f}")
    print(f"   평균:    {X_full[feat].mean():.6f}")
    print(f"   표준편차: {X_full[feat].std():.6f}")
    
    # 백분위수
    p25 = X_full[feat].quantile(0.25)
    p75 = X_full[feat].quantile(0.75)
    print(f"   25% 지점: {p25:.6f}")
    print(f"   75% 지점: {p75:.6f}")
    print(f"   IQR:      {p75 - p25:.6f}")
