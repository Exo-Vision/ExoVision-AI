# ========================================
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
    print(f"  {i}. {feat}")

# 2. 모델 로드 (저장된 모델 경로 지정)
model1 = joblib.load('saved_models/model1_binary_catboost_YYYYMMDD_HHMMSS.pkl')
scaler1 = joblib.load('saved_models/scaler1_YYYYMMDD_HHMMSS.pkl')
model2 = joblib.load('saved_models/model2_candidate_voting_YYYYMMDD_HHMMSS.pkl')
scaler2 = joblib.load('saved_models/scaler2_YYYYMMDD_HHMMSS.pkl')

# 3. 입력 데이터 (핵심 5개 컬럼만 제공)
input_data_top5 = {
    'koi_prad': 0.0,  # 실제 값 입력
    'dec': 0.0,  # 실제 값 입력
    'koi_smet': 0.0,  # 실제 값 입력
    'planet_star_ratio': 0.0,  # 실제 값 입력
    'planet_density_proxy': 0.0,  # 실제 값 입력
}

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

print(f"\n예측 결과: {prediction}")
print(f"확신도: {confidence:.4f}")
