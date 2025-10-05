"""
Feature Importance 분석 - 핵심 5개 컬럼 추출
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings('ignore')

print("=" * 100)
print("🔍 Feature Importance 분석 - 핵심 컬럼 추출")
print("=" * 100)

# 데이터 로드
df = pd.read_csv('datasets/exoplanets.csv', low_memory=False)
print(f"\n원본 데이터: {df.shape[0]:,} 샘플")

y_full = df['koi_disposition']
X_full = df.drop(['koi_disposition', 'kepoi_name'], axis=1, errors='ignore')

# 수치형 컬럼만
numeric_cols = X_full.select_dtypes(include=[np.number]).columns
X_full = X_full[numeric_cols]

# 피처 엔지니어링
X_full['planet_star_ratio'] = X_full['koi_prad'] / (X_full['koi_srad'] + 1e-10)
X_full['orbital_energy'] = 1.0 / (X_full['koi_sma'] + 1e-10)
X_full['transit_signal'] = X_full['koi_depth'] * X_full['koi_duration']
X_full['stellar_density'] = X_full['koi_smass'] / (X_full['koi_srad']**3 + 1e-10)
X_full['planet_density_proxy'] = X_full['koi_prad']**3 / (X_full['koi_sma']**2 + 1e-10)
X_full['log_period'] = np.log1p(X_full['koi_period'])
X_full['log_depth'] = np.log1p(X_full['koi_depth'])
X_full['log_insol'] = np.log1p(X_full['koi_insol'])
X_full['orbit_stability'] = X_full['koi_eccen'] * X_full['koi_impact']
X_full['transit_snr'] = X_full['koi_depth'] / (X_full['koi_duration'] + 1e-10)

X_full = X_full.replace([np.inf, -np.inf], np.nan)
X_full = X_full.fillna(X_full.median())

# 모델 1: CONFIRMED vs FALSE POSITIVE로 importance 분석
y_binary = y_full[y_full.isin(['CONFIRMED', 'FALSE POSITIVE'])]
X_binary = X_full.loc[y_binary.index]
y_binary_encoded = (y_binary == 'CONFIRMED').astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X_binary, y_binary_encoded, test_size=0.2, random_state=42, stratify=y_binary_encoded
)

print(f"\n학습 데이터: {len(X_train):,} 샘플")

# Feature Importance 계산
feature_names = X_train.columns.tolist()
importance_dict = {}

print("\n" + "=" * 100)
print("📊 모델별 Feature Importance 계산")
print("=" * 100)

# CatBoost
print("\n[1/3] CatBoost 학습 중...")
cat = CatBoostClassifier(
    iterations=500,
    learning_rate=0.02,
    depth=4,
    l2_leaf_reg=10.0,
    random_state=42,
    verbose=False
)
cat.fit(X_train, y_train)
importance_dict['CatBoost'] = cat.feature_importances_

# XGBoost
print("[2/3] XGBoost 학습 중...")
xgb = XGBClassifier(
    n_estimators=500,
    learning_rate=0.02,
    max_depth=4,
    reg_lambda=10.0,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42,
    eval_metric='logloss'
)
xgb.fit(X_train, y_train)
importance_dict['XGBoost'] = xgb.feature_importances_

# LightGBM
print("[3/3] LightGBM 학습 중...")
lgb = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.02,
    max_depth=4,
    reg_lambda=10.0,
    subsample=0.7,
    colsample_bytree=0.7,
    random_state=42,
    verbose=-1
)
lgb.fit(X_train, y_train)
importance_dict['LightGBM'] = lgb.feature_importances_

# 평균 importance 계산
print("\n" + "=" * 100)
print("🎯 통합 Feature Importance (3개 모델 평균)")
print("=" * 100)

# 정규화된 importance 평균
normalized_importance = {}
for model_name, importances in importance_dict.items():
    # 정규화 (합이 1이 되도록)
    normalized = importances / importances.sum()
    normalized_importance[model_name] = normalized

# 평균 계산
avg_importance = np.mean([normalized_importance[m] for m in normalized_importance.keys()], axis=0)

# DataFrame으로 정리
importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': avg_importance,
    'CatBoost': normalized_importance['CatBoost'],
    'XGBoost': normalized_importance['XGBoost'],
    'LightGBM': normalized_importance['LightGBM']
})

importance_df = importance_df.sort_values('Importance', ascending=False)

print("\n상위 10개 피처:")
print("-" * 100)
print(f"{'순위':<5} {'피처명':<30} {'평균 중요도':<12} {'CatBoost':<12} {'XGBoost':<12} {'LightGBM':<12}")
print("-" * 100)
for idx, row in importance_df.head(10).iterrows():
    print(f"{importance_df.index.get_loc(idx)+1:<5} {row['Feature']:<30} "
          f"{row['Importance']:.6f}    {row['CatBoost']:.6f}    "
          f"{row['XGBoost']:.6f}    {row['LightGBM']:.6f}")

# 핵심 5개 컬럼
top5_features = importance_df.head(5)['Feature'].tolist()

print("\n" + "=" * 100)
print("⭐ 핵심 5개 컬럼")
print("=" * 100)
for i, feat in enumerate(top5_features, 1):
    imp = importance_df[importance_df['Feature'] == feat]['Importance'].values[0]
    print(f"{i}. {feat:<30} (중요도: {imp:.4f})")

# 나머지 컬럼들의 기본값 계산
print("\n" + "=" * 100)
print("📋 나머지 컬럼들의 기본값 (중앙값/평균)")
print("=" * 100)

other_features = [f for f in feature_names if f not in top5_features]

# 통계 정보 출력
print(f"\n{'컬럼명':<30} {'중앙값':<15} {'평균값':<15} {'표준편차':<15} {'추천값':<15}")
print("-" * 100)

default_values = {}
for feat in other_features:
    median_val = X_full[feat].median()
    mean_val = X_full[feat].mean()
    std_val = X_full[feat].std()
    
    # 추천값: 중앙값 선택 (더 안정적)
    recommended = median_val
    default_values[feat] = recommended
    
    print(f"{feat:<30} {median_val:<15.6f} {mean_val:<15.6f} {std_val:<15.6f} {recommended:<15.6f}")

# 결과 저장
print("\n" + "=" * 100)
print("💾 결과 저장")
print("=" * 100)

# 1. Feature Importance CSV
importance_df.to_csv('feature_importance_ranking.csv', index=False)
print("✅ Feature Importance: feature_importance_ranking.csv")

# 2. 기본값 설정 파일
config_data = {
    'top5_features': top5_features,
    'default_values': default_values
}

import json
with open('model_config_top5.json', 'w', encoding='utf-8') as f:
    json.dump(config_data, f, indent=2, ensure_ascii=False)
print("✅ 설정 파일: model_config_top5.json")

# 3. 사용 예제 코드 생성
example_code = f"""# ========================================
# 핵심 5개 컬럼만으로 예측하기
# ========================================
import pandas as pd
import numpy as np
import json
import joblib

# 1. 설정 로드
with open('model_config_top5.json', 'r', encoding='utf-8') as f:
    config = json.load(f)

top5_features = config['top5_features']
default_values = config['default_values']

print("핵심 5개 컬럼:")
for i, feat in enumerate(top5_features, 1):
    print(f"  {{i}}. {{feat}}")

# 2. 모델 로드 (저장된 모델 경로 지정)
model1 = joblib.load('saved_models/model1_binary_catboost_YYYYMMDD_HHMMSS.pkl')
scaler1 = joblib.load('saved_models/scaler1_YYYYMMDD_HHMMSS.pkl')
model2 = joblib.load('saved_models/model2_candidate_voting_YYYYMMDD_HHMMSS.pkl')
scaler2 = joblib.load('saved_models/scaler2_YYYYMMDD_HHMMSS.pkl')

# 3. 입력 데이터 (핵심 5개 컬럼만 제공)
input_data_top5 = {{
{chr(10).join(f"    '{feat}': 0.0,  # 실제 값 입력" for feat in top5_features)}
}}

# 4. 전체 29개 피처로 확장
def expand_to_full_features(top5_data, default_values):
    full_data = default_values.copy()
    full_data.update(top5_data)
    return full_data

full_input = expand_to_full_features(input_data_top5, default_values)

# 5. DataFrame으로 변환
all_features = top5_features + list(default_values.keys())
X_input = pd.DataFrame([full_input])[all_features]

# 6. 예측
X_scaled = scaler1.transform(X_input)
pred_proba = model1.predict_proba(X_scaled)[0]
confidence = max(pred_proba)

if confidence >= 0.96:
    prediction = "CONFIRMED" if pred_proba[1] > 0.5 else "FALSE POSITIVE"
else:
    X_scaled2 = scaler2.transform(X_input)
    is_candidate = model2.predict(X_scaled2)[0]
    if is_candidate:
        prediction = "CANDIDATE"
    else:
        prediction = "CONFIRMED" if pred_proba[1] > 0.5 else "FALSE POSITIVE"

print(f"\\n예측 결과: {{prediction}}")
print(f"확신도: {{confidence:.4f}}")
"""

with open('predict_with_top5.py', 'w', encoding='utf-8') as f:
    f.write(example_code)
print("✅ 사용 예제: predict_with_top5.py")

print("\n" + "=" * 100)
print("✅ 분석 완료!")
print("=" * 100)
print(f"\n핵심 5개 컬럼: {', '.join(top5_features)}")
print(f"나머지 24개 컬럼: 기본값 설정됨")
