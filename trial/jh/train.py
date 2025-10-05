import optuna
import pandas as pd
from lightgbm import LGBMClassifier
from optuna.samplers import TPESampler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# 데이터 로드
datasets = pd.read_csv("C:/workspace/nasa/datasets/preprocessed_all.csv")

# mission 컬럼 원-핫 인코딩
datasets_encoded = pd.get_dummies(datasets, columns=["mission"], prefix="mission")

print(f"원-핫 인코딩 후 shape: {datasets_encoded.shape}")
print(
    f"생성된 mission 컬럼: {[col for col in datasets_encoded.columns if 'mission' in col]}"
)

# 레이블 인코딩
label_encoder = LabelEncoder()
datasets_encoded["label"] = label_encoder.fit_transform(datasets_encoded["label"])
print(
    f"\n레이블 매핑: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}"
)

# 학습/테스트 데이터 분할
x_train, x_test, y_train, y_test = train_test_split(
    datasets_encoded.drop(columns=["label"]),
    datasets_encoded["label"],
    test_size=0.05,
    random_state=42,
    stratify=datasets_encoded["label"],
)

print(f"\n학습 데이터 크기: {x_train.shape}")
print(f"테스트 데이터 크기: {x_test.shape}")


# XGBoost Optuna 목적 함수 정의
def objective_xgb(trial):
    # 하이퍼파라미터 탐색 공간 정의
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 0.5),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 1.0),
        "random_state": 42,
        "eval_metric": "mlogloss",
        "verbosity": 0,
        "n_jobs": -1,  # 모든 CPU 코어 사용
    }

    # 모델 생성 및 교차 검증
    model = XGBClassifier(**params)
    scores = cross_val_score(
        model, x_train, y_train, cv=5, scoring="accuracy", n_jobs=-1
    )

    return scores.mean()


# LightGBM Optuna 목적 함수 정의
def objective_lgb(trial):
    # 하이퍼파라미터 탐색 공간 정의
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0, 1.0),
        "num_leaves": trial.suggest_int("num_leaves", 20, 100),
        "random_state": 42,
        "verbosity": -1,
        "n_jobs": -1,
    }

    # 모델 생성 및 교차 검증
    model = LGBMClassifier(**params)
    scores = cross_val_score(
        model, x_train, y_train, cv=5, scoring="accuracy", n_jobs=-1
    )

    return scores.mean()


# XGBoost 최적화
print("\n=== XGBoost 하이퍼파라미터 튜닝 시작 ===")
study_xgb = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study_xgb.optimize(objective_xgb, n_trials=50, show_progress_bar=True, n_jobs=-1)

print("\n=== XGBoost 최적 하이퍼파라미터 ===")
print(f"최고 정확도: {study_xgb.best_value:.4f}")
print(f"최적 파라미터:")
for key, value in study_xgb.best_params.items():
    print(f"  {key}: {value}")


# LightGBM 최적화
print("\n=== LightGBM 하이퍼파라미터 튜닝 시작 ===")
study_lgb = optuna.create_study(direction="maximize", sampler=TPESampler(seed=42))
study_lgb.optimize(objective_lgb, n_trials=50, show_progress_bar=True, n_jobs=-1)

print("\n=== LightGBM 최적 하이퍼파라미터 ===")
print(f"최고 정확도: {study_lgb.best_value:.4f}")
print(f"최적 파라미터:")
for key, value in study_lgb.best_params.items():
    print(f"  {key}: {value}")


# 두 모델 비교 및 최종 모델 선택
print("\n" + "=" * 60)
print("모델 비교 결과")
print("=" * 60)
print(f"XGBoost 교차 검증 정확도: {study_xgb.best_value:.4f}")
print(f"LightGBM 교차 검증 정확도: {study_lgb.best_value:.4f}")

if study_xgb.best_value > study_lgb.best_value:
    print("\n최종 선택: XGBoost")
    best_params = study_xgb.best_params.copy()
    best_params.update({"random_state": 42, "eval_metric": "mlogloss", "n_jobs": -1})
    final_model = XGBClassifier(**best_params)
    model_name = "XGBoost"
else:
    print("\n최종 선택: LightGBM")
    best_params = study_lgb.best_params.copy()
    best_params.update({"random_state": 42, "verbosity": -1, "n_jobs": -1})
    final_model = LGBMClassifier(**best_params)
    model_name = "LightGBM"

# 최종 모델 학습
print(f"\n=== {model_name} 최종 모델 학습 ===")
final_model.fit(x_train, y_train)

# 예측
y_pred = final_model.predict(x_test)

# 성능 평가
print(f"\n=== {model_name} 최적화된 모델 성능 평가 ===")
print(f"정확도: {accuracy_score(y_test, y_pred):.4f}")
print("\n혼동 행렬:")
print(confusion_matrix(y_test, y_pred))
print("\n분류 리포트:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))
