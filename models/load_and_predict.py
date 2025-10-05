import joblib
import pandas as pd

# 모델 및 스케일러 로드
model1 = joblib.load("model1_binary_catboost_20251005_195122.pkl")
scaler1 = joblib.load("scaler1_20251005_195122.pkl")
model2 = joblib.load("model2_candidate_voting_20251005_195122.pkl")
scaler2 = joblib.load("scaler2_20251005_195122.pkl")
config = joblib.load("config_20251005_195122.pkl")

print("모델 로드 완료!")
print(f"모델 1: {config['model1_name']} - {config['model1_accuracy']*100:.2f}%")
print(f"모델 2: {config['model2_name']} - {config['model2_accuracy']*100:.2f}%")
print(f"최종 정확도: {config['final_accuracy']*100:.2f}%")
print(f"최적 임계값: {config['best_threshold']:.2f}")


# 새로운 데이터 예측
def predict_exoplanet(X_new):
    """
    새로운 데이터에 대해 3-클래스 예측
    X_new: pandas DataFrame (피처 엔지니어링 완료된 데이터)
    """
    # 스케일링
    X_scaled_1 = scaler1.transform(X_new)
    X_scaled_2 = scaler2.transform(X_new)

    # 모델 1 예측
    proba1 = model1.predict_proba(X_scaled_1)
    pred1 = model1.predict(X_scaled_1)

    # 모델 2 예측
    pred2 = model2.predict(X_scaled_2)

    # 최종 예측
    final_pred = []
    for i in range(len(X_new)):
        if proba1[i].max() >= config["best_threshold"]:
            # 고확신도 → 모델 1 결과 사용
            final_pred.append("CONFIRMED" if pred1[i] == 1 else "FALSE POSITIVE")
        else:
            # 저확신도 → 모델 2로 CANDIDATE 판별
            if pred2[i] == 1:
                final_pred.append("CANDIDATE")
            else:
                final_pred.append("CONFIRMED" if pred1[i] == 1 else "FALSE POSITIVE")

    return final_pred


# 사용 예시:
# 입력값:
# float 대신 실제 값으로 변경하면 됨.
input = {
    "ra": float,
    "dec": float,
    "koi_period": float,
    "koi_eccen": float,
    "koi_longp": float,
    "koi_incl": float,
    "koi_impact": float,
    "koi_sma": float,
    "koi_duration": float,
    "koi_depth": float,
    "koi_prad": float,
    "koi_insol": float,
    "koi_teq": float,
    "koi_srad": float,
    "koi_smass": float,
    "koi_sage": float,
    "koi_steff": float,
    "koi_slogg": float,
    "koi_smet": float,
    "data_source": "TESS",
    "koi_smass_calculated": True,
    "koi_sma_calculated": True,
    "koi_incl_calculated": True,
}
input_pd = pd.DataFrame([input])
predictions = predict_exoplanet(input_pd)
