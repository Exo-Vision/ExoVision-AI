import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# [1] 데이터 로드
print("=" * 80)
print("🚀 고성능 앙상블 모델 - 외계행성 분류")
print("=" * 80)

data = pd.read_csv("datasets/preprocessed_all.csv")
print(f"\n데이터 크기: {data.shape}")
print(f"타겟 분포:\n{data['label'].value_counts()}")

# [2] Feature Engineering
def create_features(df):
    """고급 특성 생성"""
    df = df.copy()
    
    # 기존 특성들
    if 'pl_rade' in df.columns and 'pl_orbper' in df.columns:
        df['rade_orbper_ratio'] = df['pl_rade'] / (df['pl_orbper'] + 1e-8)
    
    if 'pl_trandep' in df.columns and 'pl_trandurh' in df.columns:
        df['trandep_durh_product'] = df['pl_trandep'] * df['pl_trandurh']
    
    if 'st_teff' in df.columns and 'st_dist' in df.columns:
        df['teff_dist_ratio'] = df['st_teff'] / (df['st_dist'] + 1e-8)
    
    if 'pl_insol' in df.columns and 'snr' in df.columns:
        df['insol_snr_ratio'] = df['pl_insol'] / (df['snr'] + 1e-8)
    
    # 새로운 특성들
    if 'pl_rade' in df.columns:
        df['rade_squared'] = df['pl_rade'] ** 2
        df['rade_log'] = np.log1p(df['pl_rade'])
    
    if 'pl_orbper' in df.columns:
        df['orbper_log'] = np.log1p(df['pl_orbper'])
    
    if 'snr' in df.columns:
        df['snr_log'] = np.log1p(df['snr'])
        df['snr_squared'] = df['snr'] ** 2
    
    if 'st_teff' in df.columns:
        df['teff_normalized'] = df['st_teff'] / 6000  # 태양 온도 기준
    
    # Confidence 특성 조합
    confidence_cols = [col for col in df.columns if 'confidence' in col]
    if len(confidence_cols) > 0:
        df['avg_confidence'] = df[confidence_cols].mean(axis=1)
        df['min_confidence'] = df[confidence_cols].min(axis=1)
        df['max_confidence'] = df[confidence_cols].max(axis=1)
    
    # Limit 특성 조합
    lim_cols = [col for col in df.columns if 'lim' in col]
    if len(lim_cols) > 0:
        df['sum_limits'] = df[lim_cols].sum(axis=1)
    
    return df

# [3] 데이터 준비
target = "label"
X = data.drop(target, axis=1)
y = data[target]

# Feature Engineering 적용
X = create_features(X)

# 범주형 처리
X = pd.get_dummies(X, columns=['mission'], drop_first=False)

# 타겟 인코딩
le = LabelEncoder()
y = le.fit_transform(y)

print(f"\n특성 엔지니어링 후 특성 수: {X.shape[1]}")

# Train/Test 분할 (Stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"훈련 세트 크기: {X_train.shape}")
print(f"테스트 세트 크기: {X_test.shape}")

# [4] 결측값 처리 - KNN Imputer (더 정교한 방법)
print("\n결측값 처리 중...")
imputer = KNNImputer(n_neighbors=5, weights='distance')
X_train = imputer.fit_transform(X_train)
X_test = imputer.transform(X_test)

# [5] 스케일링 - Robust Scaler (이상치에 강함)
print("스케일링 중...")
scaler = RobustScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# [6] 고성능 모델 정의
print("\n모델 학습 중...")

# XGBoost - 최적화된 하이퍼파라미터
xgb = XGBClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=8,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    tree_method="hist",
    random_state=42,
    eval_metric='mlogloss',
    early_stopping_rounds=50
)

# LightGBM - 최적화
lgb = LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=8,
    num_leaves=64,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=30,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=42,
    verbose=-1
)

# CatBoost - 최적화
cb = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.03,
    depth=8,
    l2_leaf_reg=5,
    bootstrap_type='Bernoulli',
    subsample=0.8,
    random_strength=0.1,
    verbose=False,
    random_state=42,
    early_stopping_rounds=50
)

# RandomForest - 최적화
rf = RandomForestClassifier(
    n_estimators=800,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    oob_score=True,
    random_state=42,
    n_jobs=-1
)

# ExtraTrees - 추가 다양성
et = ExtraTreesClassifier(
    n_estimators=800,
    max_depth=15,
    min_samples_split=5,
    min_samples_leaf=2,
    max_features='sqrt',
    bootstrap=True,
    random_state=42,
    n_jobs=-1
)

# GradientBoosting
gb = GradientBoostingClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=7,
    min_samples_split=5,
    min_samples_leaf=2,
    subsample=0.8,
    max_features='sqrt',
    random_state=42
)

# [7] 개별 모델 학습 및 평가
models = {
    'XGBoost': xgb,
    'LightGBM': lgb,
    'CatBoost': cb,
    'RandomForest': rf,
    'ExtraTrees': et,
    'GradientBoosting': gb
}

predictions = {}
accuracies = {}

print("\n" + "=" * 80)
print("개별 모델 성능")
print("=" * 80)

for name, model in models.items():
    print(f"\n학습 중: {name}...")
    
    # Early stopping을 위한 eval_set (XGBoost, CatBoost)
    if name in ['XGBoost', 'CatBoost']:
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
    else:
        model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    predictions[name] = y_pred
    accuracies[name] = acc
    
    print(f"{name} 정확도: {acc:.4f}")

# [8] Soft Voting Ensemble (확률 기반)
print("\n" + "=" * 80)
print("앙상블 예측 (Soft Voting)")
print("=" * 80)

# 각 모델의 확률 예측
proba_predictions = {}
for name, model in models.items():
    proba_predictions[name] = model.predict_proba(X_test)

# 가중 평균 (성능 기반 가중치)
total_acc = sum(accuracies.values())
weights = {name: acc / total_acc for name, acc in accuracies.items()}

print("\n모델 가중치:")
for name, weight in weights.items():
    print(f"  {name}: {weight:.4f}")

# Weighted soft voting
weighted_proba = np.zeros_like(proba_predictions['XGBoost'])
for name, proba in proba_predictions.items():
    weighted_proba += weights[name] * proba

y_pred_ensemble = np.argmax(weighted_proba, axis=1)
acc_ensemble = accuracy_score(y_test, y_pred_ensemble)

print(f"\n{'=' * 80}")
print(f"🎯 최종 앙상블 정확도: {acc_ensemble:.4f}")
print(f"{'=' * 80}")

# [9] 상세 분석
print("\n분류 리포트:")
print(classification_report(y_test, y_pred_ensemble, 
                          target_names=le.classes_))

print("\n혼동 행렬:")
cm = confusion_matrix(y_test, y_pred_ensemble)
cm_df = pd.DataFrame(cm, 
                     index=le.classes_, 
                     columns=le.classes_)
print(cm_df)

# [10] 성능 요약
print("\n" + "=" * 80)
print("📊 최종 성능 요약")
print("=" * 80)
print(f"베이스라인 (최빈 클래스): {max(np.bincount(y_test)) / len(y_test):.4f}")
print(f"최고 개별 모델: {max(accuracies.values()):.4f} ({max(accuracies, key=accuracies.get)})")
print(f"앙상블 모델: {acc_ensemble:.4f}")
print(f"향상도: {(acc_ensemble - max(accuracies.values())) * 100:.2f}%p")
print("=" * 80)
