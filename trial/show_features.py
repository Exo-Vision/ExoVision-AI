import pandas as pd
import numpy as np

# 데이터 로드
df = pd.read_csv('datasets/exoplanets.csv', low_memory=False)
X = df.drop(['koi_disposition', 'kepoi_name'], axis=1, errors='ignore')

# 수치형 컬럼만
numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()

print("=" * 80)
print("🔹 기본 수치형 컬럼 (19개)")
print("=" * 80)
for i, col in enumerate(numeric_cols, 1):
    print(f"{i:2d}. {col}")

print("\n" + "=" * 80)
print("🔸 피처 엔지니어링으로 추가된 컬럼 (10개)")
print("=" * 80)

engineered_features = [
    ("planet_star_ratio", "koi_prad / koi_srad", "행성/별 크기 비율"),
    ("orbital_energy", "1 / koi_sma", "궤도 에너지"),
    ("transit_signal", "koi_depth × koi_duration", "통과 신호 강도"),
    ("stellar_density", "koi_smass / koi_srad³", "별 밀도"),
    ("planet_density_proxy", "koi_prad³ / koi_sma²", "행성 밀도 근사"),
    ("log_period", "log(1 + koi_period)", "로그 주기"),
    ("log_depth", "log(1 + koi_depth)", "로그 깊이"),
    ("log_insol", "log(1 + koi_insol)", "로그 복사 에너지"),
    ("orbit_stability", "koi_eccen × koi_impact", "궤도 안정성"),
    ("transit_snr", "koi_depth / koi_duration", "통과 신호 대 잡음비")
]

for i, (name, formula, desc) in enumerate(engineered_features, 1):
    print(f"{i:2d}. {name:25s} = {formula:30s} ({desc})")

print("\n" + "=" * 80)
print("📊 최종 입력 피처")
print("=" * 80)
print(f"총 피처 수: {len(numeric_cols) + len(engineered_features)}개")
print(f"  • 기본 피처: {len(numeric_cols)}개")
print(f"  • 엔지니어링 피처: {len(engineered_features)}개")

print("\n" + "=" * 80)
print("🎯 모델 입력값 형식")
print("=" * 80)
print("입력 데이터: 29차원 벡터")
print("형식: numpy array 또는 pandas DataFrame")
print("스케일링: StandardScaler로 정규화")
print("\n예시:")
print("  input_shape = (n_samples, 29)")
print("  input_data = scaler.transform(X)")
