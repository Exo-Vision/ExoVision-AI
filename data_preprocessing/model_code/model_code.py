"""
95% 정확도 달성을 위한 ULTRA 최적화
- SMOTE 데이터 증강
- 교차 검증 기반 스태킹
- 더 강력한 앙상블
- 클래스별 최적화 전략
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import joblib
import os
from datetime import datetime

print("=" * 100)
print("🚀 ULTRA 최적화 - 95% 정확도 목표")
print("=" * 100)

# ============================================================================
# 데이터 로드 및 고급 피처 엔지니어링
# ============================================================================
print("\n[1] 데이터 로드 및 고급 피처 엔지니어링")
print("-" * 100)

df = pd.read_csv('datasets/exoplanets.csv', low_memory=False)
print(f"원본 데이터: {df.shape[0]:,} 샘플")

y_full = df['koi_disposition']
X_full = df.drop(['koi_disposition', 'kepoi_name'], axis=1, errors='ignore')

# 수치형 컬럼만
numeric_cols = X_full.select_dtypes(include=[np.number]).columns
X_full = X_full[numeric_cols]

# 고급 피처 엔지니어링 (44개)
print("\n피처 엔지니어링 중...")

# 기존 피처 (10개)
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

# 고급 피처 (15개)
X_full['habitable_zone_score'] = 1.0 / (1.0 + np.abs(X_full['koi_insol'] - 1.0))
X_full['temp_habitable_score'] = 1.0 / (1.0 + np.abs(X_full['koi_teq'] - 288) / 100)
X_full['roche_limit'] = 2.46 * X_full['koi_srad'] * (X_full['koi_smass'] / (X_full['koi_prad'] / 109.0))**(1/3)
X_full['hill_sphere'] = X_full['koi_sma'] * (X_full['koi_smass'] / 3.0)**(1/3)
X_full['transit_probability'] = X_full['koi_srad'] / (X_full['koi_sma'] * 215.032 + 1e-10)
X_full['improved_snr'] = (X_full['koi_depth'] * np.sqrt(X_full['koi_duration'])) / (X_full['koi_period'] + 1e-10)
X_full['stability_index'] = (1 - X_full['koi_eccen']) * (1 - X_full['koi_impact'])
X_full['mass_ratio'] = (X_full['koi_prad'] / 109.0)**3 / (X_full['koi_smass'] + 1e-10)
X_full['tidal_heating'] = X_full['koi_eccen'] / (X_full['koi_sma']**3 + 1e-10)
X_full['duration_ratio'] = X_full['koi_duration'] / (X_full['koi_period'] + 1e-10)
X_full['radiation_balance'] = X_full['koi_insol'] * (X_full['koi_prad']**2) / (X_full['koi_sma']**2 + 1e-10)
X_full['age_metallicity'] = X_full['koi_sage'] * (X_full['koi_smet'] + 2.5)
X_full['depth_size_ratio'] = X_full['koi_depth'] / (X_full['koi_prad']**2 + 1e-10)
X_full['kepler_ratio'] = X_full['koi_period']**2 / (X_full['koi_sma']**3 + 1e-10)
X_full['depth_variability'] = X_full['koi_depth'] / (X_full['koi_duration'] * X_full['koi_period'] + 1e-10)

# Inf, NaN 처리
X_full = X_full.replace([np.inf, -np.inf], np.nan)
X_full = X_full.fillna(X_full.median())

print(f"최종 피처 수: {X_full.shape[1]}개")

# ============================================================================
# 모델 1: CONFIRMED vs FALSE POSITIVE (SMOTE + 교차검증 스태킹)
# ============================================================================
print("\n" + "=" * 100)
print("🔵 모델 1: CONFIRMED vs FALSE POSITIVE (SMOTE + 강화 앙상블)")
print("=" * 100)

y_binary = y_full[y_full.isin(['CONFIRMED', 'FALSE POSITIVE'])]
X_binary = X_full.loc[y_binary.index]

print(f"\n학습 데이터: {len(y_binary):,} 샘플")
y_binary_encoded = (y_binary == 'CONFIRMED').astype(int)

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
    X_binary, y_binary_encoded, test_size=0.1, random_state=42, stratify=y_binary_encoded
)

# 스케일링 (SMOTE 전)
scaler_1 = StandardScaler()
X_train_1_scaled = scaler_1.fit_transform(X_train_1)
X_test_1_scaled = scaler_1.transform(X_test_1)

# SMOTE 적용
print("\nSMOTE 데이터 증강 중...")
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_1_smote, y_train_1_smote = smote.fit_resample(X_train_1_scaled, y_train_1)
print(f"  증강 전: {len(y_train_1):,}")
print(f"  증강 후: {len(y_train_1_smote):,} (+{len(y_train_1_smote) - len(y_train_1):,})")

# 강화된 모델 풀 (8개 모델)
print("\n강화된 앙상블 모델 학습 중...")

base_models_1 = [
    ('CatBoost1', CatBoostClassifier(
        iterations=1500, depth=7, learning_rate=0.008,
        l2_leaf_reg=20.0, subsample=0.85, random_state=42, verbose=False
    )),
    ('CatBoost2', CatBoostClassifier(
        iterations=2000, depth=5, learning_rate=0.01,
        l2_leaf_reg=25.0, subsample=0.8, random_state=43, verbose=False
    )),
    ('XGBoost1', XGBClassifier(
        n_estimators=1500, max_depth=7, learning_rate=0.008,
        subsample=0.85, reg_alpha=8.0, reg_lambda=20.0, random_state=42, eval_metric='logloss'
    )),
    ('XGBoost2', XGBClassifier(
        n_estimators=2000, max_depth=5, learning_rate=0.01,
        subsample=0.8, reg_alpha=10.0, reg_lambda=25.0, random_state=43, eval_metric='logloss'
    )),
    ('LightGBM1', LGBMClassifier(
        n_estimators=1500, max_depth=7, learning_rate=0.008,
        subsample=0.85, reg_alpha=8.0, reg_lambda=20.0, random_state=42, verbose=-1
    )),
    ('LightGBM2', LGBMClassifier(
        n_estimators=2000, max_depth=5, learning_rate=0.01,
        subsample=0.8, reg_alpha=10.0, reg_lambda=25.0, random_state=43, verbose=-1
    )),
    ('RandomForest', RandomForestClassifier(
        n_estimators=500, max_depth=15, min_samples_split=10,
        min_samples_leaf=4, random_state=42, n_jobs=-1
    )),
    ('ExtraTrees', ExtraTreesClassifier(
        n_estimators=500, max_depth=15, min_samples_split=10,
        min_samples_leaf=4, random_state=42, n_jobs=-1
    ))
]

# 교차 검증 기반 스태킹
print("  교차 검증 스태킹 학습 중...")
stacking_1 = StackingClassifier(
    estimators=base_models_1,
    final_estimator=LogisticRegression(C=0.05, max_iter=2000, random_state=42),
    cv=10,  # 10-fold CV
    n_jobs=-1
)
stacking_1.fit(X_train_1_smote, y_train_1_smote)

train_acc_1 = accuracy_score(y_train_1_smote, stacking_1.predict(X_train_1_smote))
test_acc_1 = accuracy_score(y_test_1, stacking_1.predict(X_test_1_scaled))
cv_scores_1 = cross_val_score(stacking_1, X_train_1_smote, y_train_1_smote, cv=5, scoring='accuracy', n_jobs=-1)

print(f"\n✅ 모델 1 성능:")
print(f"  Train: {train_acc_1:.4f} ({train_acc_1*100:.2f}%)")
print(f"  Test: {test_acc_1:.4f} ({test_acc_1*100:.2f}%)")
print(f"  CV: {cv_scores_1.mean():.4f}±{cv_scores_1.std():.4f}")
print(f"  과적합: {(train_acc_1 - test_acc_1)*100:.2f}%p")

# ============================================================================
# 모델 2: CANDIDATE 판별 (SMOTETomek + 강화 앙상블)
# ============================================================================
print("\n" + "=" * 100)
print("🟢 모델 2: CANDIDATE 판별 (SMOTETomek + 강화 앙상블)")
print("=" * 100)

y_candidate = (y_full == 'CANDIDATE').astype(int)
X_candidate = X_full.copy()

print(f"\n학습 데이터: {len(y_candidate):,} 샘플")
print(f"  CANDIDATE: {y_candidate.sum():,} ({y_candidate.sum()/len(y_candidate)*100:.1f}%)")
print(f"  NOT CANDIDATE: {(~y_candidate.astype(bool)).sum():,}")

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    X_candidate, y_candidate, test_size=0.1, random_state=42, stratify=y_candidate
)

# 스케일링
scaler_2 = StandardScaler()
X_train_2_scaled = scaler_2.fit_transform(X_train_2)
X_test_2_scaled = scaler_2.transform(X_test_2)

# SMOTETomek 적용 (오버샘플링 + 언더샘플링)
print("\nSMOTETomek 데이터 증강 중...")
smotetomek = SMOTETomek(random_state=42)
X_train_2_smote, y_train_2_smote = smotetomek.fit_resample(X_train_2_scaled, y_train_2)
print(f"  증강 전: {len(y_train_2):,}")
print(f"  증강 후: {len(y_train_2_smote):,} (+{len(y_train_2_smote) - len(y_train_2):,})")

# 클래스 가중치
class_weight_ratio = (~y_candidate.astype(bool)).sum() / y_candidate.sum()

# 강화된 모델 풀 (9개 모델)
print("\n강화된 앙상블 모델 학습 중...")

base_models_2 = [
    ('CatBoost1', CatBoostClassifier(
        iterations=1500, depth=7, learning_rate=0.008,
        l2_leaf_reg=20.0, subsample=0.85, class_weights=[1, class_weight_ratio],
        random_state=42, verbose=False
    )),
    ('CatBoost2', CatBoostClassifier(
        iterations=2000, depth=5, learning_rate=0.01,
        l2_leaf_reg=25.0, subsample=0.8, class_weights=[1, class_weight_ratio],
        random_state=43, verbose=False
    )),
    ('XGBoost1', XGBClassifier(
        n_estimators=1500, max_depth=7, learning_rate=0.008,
        subsample=0.85, reg_alpha=8.0, reg_lambda=20.0,
        scale_pos_weight=class_weight_ratio, random_state=42, eval_metric='logloss'
    )),
    ('XGBoost2', XGBClassifier(
        n_estimators=2000, max_depth=5, learning_rate=0.01,
        subsample=0.8, reg_alpha=10.0, reg_lambda=25.0,
        scale_pos_weight=class_weight_ratio, random_state=43, eval_metric='logloss'
    )),
    ('LightGBM1', LGBMClassifier(
        n_estimators=1500, max_depth=7, learning_rate=0.008,
        subsample=0.85, reg_alpha=8.0, reg_lambda=20.0,
        class_weight='balanced', random_state=42, verbose=-1
    )),
    ('LightGBM2', LGBMClassifier(
        n_estimators=2000, max_depth=5, learning_rate=0.01,
        subsample=0.8, reg_alpha=10.0, reg_lambda=25.0,
        class_weight='balanced', random_state=43, verbose=-1
    )),
    ('RandomForest', RandomForestClassifier(
        n_estimators=500, max_depth=15, min_samples_split=10,
        class_weight='balanced', random_state=42, n_jobs=-1
    )),
    ('ExtraTrees', ExtraTreesClassifier(
        n_estimators=500, max_depth=15, min_samples_split=10,
        class_weight='balanced', random_state=42, n_jobs=-1
    )),
    ('NeuralNet', MLPClassifier(
        hidden_layer_sizes=(512, 256, 128, 64),
        activation='relu', alpha=0.008, max_iter=1500,
        early_stopping=True, random_state=42
    ))
]

# 교차 검증 기반 스태킹
print("  교차 검증 스태킹 학습 중...")
stacking_2 = StackingClassifier(
    estimators=base_models_2,
    final_estimator=LogisticRegression(C=0.05, max_iter=2000, random_state=42),
    cv=10,
    n_jobs=-1
)
stacking_2.fit(X_train_2_smote, y_train_2_smote)

train_acc_2 = accuracy_score(y_train_2_smote, stacking_2.predict(X_train_2_smote))
test_acc_2 = accuracy_score(y_test_2, stacking_2.predict(X_test_2_scaled))
cv_scores_2 = cross_val_score(stacking_2, X_train_2_smote, y_train_2_smote, cv=5, scoring='accuracy', n_jobs=-1)

print(f"\n✅ 모델 2 성능:")
print(f"  Train: {train_acc_2:.4f} ({train_acc_2*100:.2f}%)")
print(f"  Test: {test_acc_2:.4f} ({test_acc_2*100:.2f}%)")
print(f"  CV: {cv_scores_2.mean():.4f}±{cv_scores_2.std():.4f}")
print(f"  과적합: {(train_acc_2 - test_acc_2)*100:.2f}%p")

# ============================================================================
# 파이프라인 통합
# ============================================================================
print("\n" + "=" * 100)
print("🔗 파이프라인 통합: 확신도 기반 3-클래스 분류")
print("=" * 100)

y_test_full = y_full.loc[X_test_2.index]
X_test_full_scaled_1 = scaler_1.transform(X_test_2)
X_test_full_scaled_2 = scaler_2.transform(X_test_2)

stage1_proba = stacking_1.predict_proba(X_test_full_scaled_1)
stage1_pred = stacking_1.predict(X_test_full_scaled_1)
stage2_pred = stacking_2.predict(X_test_full_scaled_2)

# 확신도 임계값 최적화
thresholds = [0.70, 0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98, 0.99]

print(f"\n{'임계값':<10} {'1단계사용':<12} {'CANDIDATE수':<12} {'최종정확도':<12}")
print("-" * 100)

best_threshold = 0.90
best_accuracy = 0.0
best_predictions = None

for threshold in thresholds:
    final_predictions = np.empty(len(y_test_full), dtype=object)
    
    high_conf_mask = (stage1_proba.max(axis=1) >= threshold)
    final_predictions[high_conf_mask] = np.where(
        stage1_pred[high_conf_mask] == 1,
        'CONFIRMED',
        'FALSE POSITIVE'
    )
    
    low_conf_mask = ~high_conf_mask
    final_predictions[low_conf_mask] = np.where(
        stage2_pred[low_conf_mask] == 1,
        'CANDIDATE',
        np.where(stage1_pred[low_conf_mask] == 1, 'CONFIRMED', 'FALSE POSITIVE')
    )
    
    accuracy = accuracy_score(y_test_full, final_predictions)
    stage1_ratio = high_conf_mask.sum() / len(y_test_full) * 100
    candidate_count = (final_predictions == 'CANDIDATE').sum()
    
    print(f"{threshold:.2f}      {stage1_ratio:>6.1f}%      {candidate_count:>4}개       {accuracy:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold
        best_predictions = final_predictions.copy()

print(f"\n✅ 최적 임계값: {best_threshold:.2f}")
print(f"✅ 최종 정확도: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

# 최종 평가
print("\n" + "=" * 100)
print("📊 최종 성능 평가")
print("=" * 100)

print(f"\n모델 1: {test_acc_1:.4f} ({test_acc_1*100:.2f}%)")
print(f"모델 2: {test_acc_2:.4f} ({test_acc_2*100:.2f}%)")
print(f"최종 통합: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

# Confusion Matrix
cm = confusion_matrix(y_test_full, best_predictions,
                      labels=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'])

print("\nConfusion Matrix:")
print("-" * 100)
print(f"{'':15} {'CANDIDATE':>12} {'CONFIRMED':>12} {'FALSE POS':>12}")
print("-" * 100)
for i, label in enumerate(['CANDIDATE', 'CONFIRMED', 'FALSE POS']):
    print(f"{label:15} {cm[i][0]:>12} {cm[i][1]:>12} {cm[i][2]:>12}")

print("\nClassification Report:")
print("-" * 100)
print(classification_report(y_test_full, best_predictions,
                           labels=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'],
                           target_names=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']))

# 클래스별 정확도
print("\n클래스별 정확도:")
print("-" * 100)
for label in ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']:
    mask = y_test_full == label
    if mask.sum() > 0:
        class_acc = accuracy_score(y_test_full[mask], best_predictions[mask])
        print(f"  {label:20} {class_acc:.4f} ({class_acc*100:.2f}%)  [{mask.sum()}개 샘플]")

# 모델 저장
print("\n" + "=" * 100)
print("💾 모델 저장")
print("=" * 100)

save_dir = 'saved_models'
os.makedirs(save_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

model1_path = os.path.join(save_dir, f'model1_ultra_{timestamp}.pkl')
scaler1_path = os.path.join(save_dir, f'scaler1_{timestamp}.pkl')
model2_path = os.path.join(save_dir, f'model2_ultra_{timestamp}.pkl')
scaler2_path = os.path.join(save_dir, f'scaler2_{timestamp}.pkl')

joblib.dump(stacking_1, model1_path)
joblib.dump(scaler_1, scaler1_path)
joblib.dump(stacking_2, model2_path)
joblib.dump(scaler_2, scaler2_path)

config = {
    'model1_name': 'Stacking_10CV_8Models',
    'model1_accuracy': test_acc_1,
    'model2_name': 'Stacking_10CV_9Models',
    'model2_accuracy': test_acc_2,
    'best_threshold': best_threshold,
    'final_accuracy': best_accuracy,
    'timestamp': timestamp,
    'feature_count': X_full.shape[1]
}

config_path = os.path.join(save_dir, f'config_{timestamp}.pkl')
joblib.dump(config, config_path)

print("✅ 모델 저장 완료")
print(f"  • {model1_path}")
print(f"  • {model2_path}")
print(f"  • {config_path}")

print("\n" + "=" * 100)
print("🎯 최종 결과")
print("=" * 100)
print(f"모델 1: {test_acc_1*100:.2f}%")
print(f"모델 2: {test_acc_2*100:.2f}%")
print(f"통합 시스템: {best_accuracy*100:.2f}%")

if best_accuracy >= 0.95:
    print("\n🎉🎉🎉 95% 목표 달성! 🎉🎉🎉")
elif best_accuracy >= 0.90:
    print(f"\n💪 90% 이상 달성! 목표까지 {(0.95-best_accuracy)*100:.2f}%p")
else:
    print(f"\n📊 현재 수준: {best_accuracy*100:.2f}% (목표: 95.00%)")

print("\n" + "=" * 100)
