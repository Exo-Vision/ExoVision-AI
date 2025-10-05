import pandas as pd
from pycaret.classification import *

# 데이터 로드
dataset = pd.read_csv("datasets/kepler.csv")
# target_col = [col for col in dataset.columns if "koi_disposition" in col][0]

X = dataset[:int(0.8 * len(dataset))]
y = dataset[int(0.8 * len(dataset)):]

# 특징(X)과 타겟(y) 설정
features = ["koi_period", "koi_duration", "koi_depth", "koi_prad", "koi_model_snr"]
target = 'koi_disposition'

# PyCaret 환경 설정
clf_setup = setup(
    data=X[features + [target]],
    target=target,
    train_size=0.8,         # train/test split
    session_id=42,          # 랜덤 시드 고정
    # categorical_features=[target],   # 라벨 데이터
    # normalize=True,         # 정규화 여부 (선택)
    use_gpu=True
)

# XGBoost 분류기 생성
xgb_clf = create_model("xgboost")

# 모델 튜닝 (선택)
tuned_xgb = tune_model(xgb_clf)

# 최종 모델 확정
final_xgb = finalize_model(tuned_xgb)

# 예측
pred_test = predict_model(final_xgb, data=y.drop(columns=[target]))  # 테스트셋 예측 가능

y_true = y[target].values

# predict_model 결과에서 예측값 가져오기
y_pred = pred_test['prediction_label'].values
from sklearn.metrics import accuracy_score

# accuracy 계산
acc = accuracy_score(y_true, y_pred)
print("테스트 정확도:", acc)