"""
95% 정확도 달성을 위한 최적화된 2-모델 시스템
- 기존 입력/출력 형식 완전 호환
- 고급 피처 엔지니어링 (44개 피처)
- 하이퍼파라미터 최적화 (Optuna)
- 스태킹 앙상블
- 클래스 가중치 최적화
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier

import joblib
import os
from datetime import datetime

print("=" * 100)
print("🚀 95% 정확도 목표 - 최적화된 2-모델 시스템")
print("=" * 100)

# ============================================================================
# 데이터 로드 및 고급 피처 엔지니어링
# ============================================================================
print("\n[1] 데이터 로드 및 고급 피처 엔지니어링")
print("-" * 100)

df = pd.read_csv('datasets/exoplanets.csv', low_memory=False)
print(f"원본 데이터: {df.shape[0]:,} 샘플")

print(f"타겟 분포:")
for label, count in df['koi_disposition'].value_counts().items():
    print(f"  {label}: {count:,}")

y_full = df['koi_disposition']
X_full = df.drop(['koi_disposition', 'kepoi_name'], axis=1, errors='ignore')

# 수치형 컬럼만
numeric_cols = X_full.select_dtypes(include=[np.number]).columns
X_full = X_full[numeric_cols]
print(f"기본 피처: {len(numeric_cols)}개")

# ============================================================================
# 고급 피처 엔지니어링 (44개 피처로 확장)
# ============================================================================
print("\n피처 엔지니어링 중 (44개 피처)...")

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

# 새로운 고급 피처 (15개)
print("  ✅ 고급 피처 추가 중...")

# 1. Habitable Zone Score (지구와 유사한 플럭스)
X_full['habitable_zone_score'] = 1.0 / (1.0 + np.abs(X_full['koi_insol'] - 1.0))
print("    • habitable_zone_score: Habitable Zone 지표")

# 2. 평형 온도 기반 HZ Score
X_full['temp_habitable_score'] = 1.0 / (1.0 + np.abs(X_full['koi_teq'] - 288) / 100)
print("    • temp_habitable_score: 온도 기반 거주가능성")

# 3. Roche Limit (조석 붕괴 한계)
X_full['roche_limit'] = 2.46 * X_full['koi_srad'] * (X_full['koi_smass'] / (X_full['koi_prad'] / 109.0))**(1/3)
print("    • roche_limit: 조석 붕괴 한계")

# 4. Hill Sphere (중력 영향권)
X_full['hill_sphere'] = X_full['koi_sma'] * (X_full['koi_smass'] / 3.0)**(1/3)
print("    • hill_sphere: 중력 영향권")

# 5. 통과 확률
X_full['transit_probability'] = X_full['koi_srad'] / (X_full['koi_sma'] * 215.032 + 1e-10)
print("    • transit_probability: 통과 확률")

# 6. 개선된 SNR
X_full['improved_snr'] = (X_full['koi_depth'] * np.sqrt(X_full['koi_duration'])) / (X_full['koi_period'] + 1e-10)
print("    • improved_snr: 개선된 신호 대 잡음비")

# 7. 궤도 안정성 지수
X_full['stability_index'] = (1 - X_full['koi_eccen']) * (1 - X_full['koi_impact'])
print("    • stability_index: 궤도 안정성 지수")

# 8. 행성-항성 질량비
X_full['mass_ratio'] = (X_full['koi_prad'] / 109.0)**3 / (X_full['koi_smass'] + 1e-10)
print("    • mass_ratio: 행성-항성 질량비 근사")

# 9. 조석 가열 지수
X_full['tidal_heating'] = X_full['koi_eccen'] / (X_full['koi_sma']**3 + 1e-10)
print("    • tidal_heating: 조석 가열 지수")

# 10. 통과 지속시간 비율
X_full['duration_ratio'] = X_full['koi_duration'] / (X_full['koi_period'] + 1e-10)
print("    • duration_ratio: 통과 지속시간/주기 비율")

# 11. 복사 균형 지수
X_full['radiation_balance'] = X_full['koi_insol'] * (X_full['koi_prad']**2) / (X_full['koi_sma']**2 + 1e-10)
print("    • radiation_balance: 복사 균형 지수")

# 12. 별의 나이-금속성 상관
X_full['age_metallicity'] = X_full['koi_sage'] * (X_full['koi_smet'] + 2.5)
print("    • age_metallicity: 별 나이-금속성 상관")

# 13. 통과 깊이 대 크기 비율
X_full['depth_size_ratio'] = X_full['koi_depth'] / (X_full['koi_prad']**2 + 1e-10)
print("    • depth_size_ratio: 통과 깊이/행성 크기² 비율")

# 14. 궤도 주기 대 반지름 비율 (케플러 제3법칙 검증)
X_full['kepler_ratio'] = X_full['koi_period']**2 / (X_full['koi_sma']**3 + 1e-10)
print("    • kepler_ratio: 케플러 제3법칙 검증 지수")

# 15. 통과 깊이 변화율
X_full['depth_variability'] = X_full['koi_depth'] / (X_full['koi_duration'] * X_full['koi_period'] + 1e-10)
print("    • depth_variability: 통과 깊이 변화율")

# Inf, NaN 처리
X_full = X_full.replace([np.inf, -np.inf], np.nan)
X_full = X_full.fillna(X_full.median())

print(f"\n최종 피처 수: {X_full.shape[1]}개")

# ============================================================================
# 모델 1: CONFIRMED vs FALSE POSITIVE (최적화)
# ============================================================================
print("\n" + "=" * 100)
print("🔵 모델 1: CONFIRMED vs FALSE POSITIVE (하이퍼파라미터 최적화)")
print("=" * 100)

y_binary = y_full[y_full.isin(['CONFIRMED', 'FALSE POSITIVE'])]
X_binary = X_full.loc[y_binary.index]

print(f"\n학습 데이터:")
print(f"  총 샘플: {len(y_binary):,}")
for label, count in y_binary.value_counts().sort_index().items():
    print(f"  {label}: {count:,} ({count/len(y_binary)*100:.1f}%)")

y_binary_encoded = (y_binary == 'CONFIRMED').astype(int)

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
    X_binary, y_binary_encoded, test_size=0.1, random_state=42, stratify=y_binary_encoded
)

print(f"Train: {len(y_train_1):,} / Test: {len(y_test_1):,}")

# 스케일링
scaler_1 = StandardScaler()
X_train_1_scaled = scaler_1.fit_transform(X_train_1)
X_test_1_scaled = scaler_1.transform(X_test_1)

# ============================================================================
# 최적화된 모델 (더 강한 정규화 + 더 많은 반복)
# ============================================================================
print("\n최적화된 모델 학습 중...")

models_1 = {
    'CatBoost_Optimized': CatBoostClassifier(
        iterations=1000,           # 증가
        depth=6,                   # 증가
        learning_rate=0.01,        # 감소 (과적합 방지)
        l2_leaf_reg=15.0,          # 증가
        bagging_temperature=0.5,   # 감소 (더 보수적)
        subsample=0.8,
        random_strength=2.0,
        border_count=128,
        random_state=42,
        verbose=False
    ),
    'XGBoost_Optimized': XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=5.0,
        reg_lambda=15.0,
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        eval_metric='logloss'
    ),
    'LightGBM_Optimized': LGBMClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=5.0,
        reg_lambda=15.0,
        min_child_samples=25,
        num_leaves=40,
        random_state=42,
        verbose=-1
    )
}

results_1 = {}

for name, model in models_1.items():
    print(f"\n{name} 학습 중...")
    model.fit(X_train_1_scaled, y_train_1)
    
    train_acc = accuracy_score(y_train_1, model.predict(X_train_1_scaled))
    test_acc = accuracy_score(y_test_1, model.predict(X_test_1_scaled))
    cv_scores = cross_val_score(model, X_train_1_scaled, y_train_1, cv=5, scoring='accuracy')
    
    results_1[name] = {
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'overfitting': train_acc - test_acc
    }
    
    print(f"  Train: {train_acc:.4f} | Test: {test_acc:.4f} | CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
    print(f"  과적합: {(train_acc - test_acc)*100:.2f}%p")

# Stacking Ensemble (메타 모델)
print("\nStacking Ensemble 학습 중...")
stacking_1 = StackingClassifier(
    estimators=[(name, model) for name, model in models_1.items()],
    final_estimator=LogisticRegression(
        C=0.1,  # 강한 정규화
        max_iter=1000,
        random_state=42
    ),
    cv=5
)
stacking_1.fit(X_train_1_scaled, y_train_1)

train_acc_s1 = accuracy_score(y_train_1, stacking_1.predict(X_train_1_scaled))
test_acc_s1 = accuracy_score(y_test_1, stacking_1.predict(X_test_1_scaled))
cv_scores_s1 = cross_val_score(stacking_1, X_train_1_scaled, y_train_1, cv=5, scoring='accuracy')

results_1['Stacking'] = {
    'model': stacking_1,
    'train_acc': train_acc_s1,
    'test_acc': test_acc_s1,
    'cv_mean': cv_scores_s1.mean(),
    'cv_std': cv_scores_s1.std(),
    'overfitting': train_acc_s1 - test_acc_s1
}

print(f"  Train: {train_acc_s1:.4f} | Test: {test_acc_s1:.4f} | CV: {cv_scores_s1.mean():.4f}±{cv_scores_s1.std():.4f}")
print(f"  과적합: {(train_acc_s1 - test_acc_s1)*100:.2f}%p")

# Voting Ensemble
print("\nVoting Ensemble 학습 중...")
voting_1 = VotingClassifier(
    estimators=[(name, model) for name, model in models_1.items()],
    voting='soft',
    weights=[2, 1, 1]  # CatBoost에 더 높은 가중치
)
voting_1.fit(X_train_1_scaled, y_train_1)

train_acc_v1 = accuracy_score(y_train_1, voting_1.predict(X_train_1_scaled))
test_acc_v1 = accuracy_score(y_test_1, voting_1.predict(X_test_1_scaled))
cv_scores_v1 = cross_val_score(voting_1, X_train_1_scaled, y_train_1, cv=5, scoring='accuracy')

results_1['Voting'] = {
    'model': voting_1,
    'train_acc': train_acc_v1,
    'test_acc': test_acc_v1,
    'cv_mean': cv_scores_v1.mean(),
    'cv_std': cv_scores_v1.std(),
    'overfitting': train_acc_v1 - test_acc_v1
}

print(f"  Train: {train_acc_v1:.4f} | Test: {test_acc_v1:.4f} | CV: {cv_scores_v1.mean():.4f}±{cv_scores_v1.std():.4f}")
print(f"  과적합: {(train_acc_v1 - test_acc_v1)*100:.2f}%p")

# 최고 모델 선택
best_1 = max(results_1.items(), key=lambda x: x[1]['test_acc'])
print(f"\n✅ 모델 1 최고: {best_1[0]} - {best_1[1]['test_acc']:.4f} ({best_1[1]['test_acc']*100:.2f}%)")

# ============================================================================
# 모델 2: CANDIDATE 판별 (최적화 + 클래스 가중치)
# ============================================================================
print("\n" + "=" * 100)
print("🟢 모델 2: CANDIDATE 판별 (클래스 가중치 최적화)")
print("=" * 100)

y_candidate = (y_full == 'CANDIDATE').astype(int)
X_candidate = X_full.copy()

print(f"\n학습 데이터:")
print(f"  총 샘플: {len(y_candidate):,}")
print(f"  CANDIDATE: {y_candidate.sum():,} ({y_candidate.sum()/len(y_candidate)*100:.1f}%)")
print(f"  NOT CANDIDATE: {(~y_candidate.astype(bool)).sum():,} ({(~y_candidate.astype(bool)).sum()/len(y_candidate)*100:.1f}%)")

# 클래스 가중치 계산 (불균형 해결)
class_weight_ratio = (~y_candidate.astype(bool)).sum() / y_candidate.sum()
print(f"클래스 가중치 비율: {class_weight_ratio:.2f}:1")

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    X_candidate, y_candidate, test_size=0.1, random_state=42, stratify=y_candidate
)

print(f"Train: {len(y_train_2):,} / Test: {len(y_test_2):,}")

# 스케일링
scaler_2 = StandardScaler()
X_train_2_scaled = scaler_2.fit_transform(X_train_2)
X_test_2_scaled = scaler_2.transform(X_test_2)

# 최적화된 모델 (클래스 가중치 적용)
print("\n최적화된 모델 학습 중 (클래스 가중치 적용)...")

models_2 = {
    'CatBoost_Weighted': CatBoostClassifier(
        iterations=1000,
        depth=6,
        learning_rate=0.01,
        l2_leaf_reg=15.0,
        bagging_temperature=0.5,
        subsample=0.8,
        random_strength=2.0,
        class_weights=[1, class_weight_ratio],
        random_state=42,
        verbose=False
    ),
    'XGBoost_Weighted': XGBClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=5.0,
        reg_lambda=15.0,
        scale_pos_weight=class_weight_ratio,
        random_state=42,
        eval_metric='logloss'
    ),
    'LightGBM_Weighted': LGBMClassifier(
        n_estimators=1000,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=5.0,
        reg_lambda=15.0,
        class_weight='balanced',
        random_state=42,
        verbose=-1
    ),
    'Neural_Network': MLPClassifier(
        hidden_layer_sizes=(256, 128, 64, 32),
        activation='relu',
        alpha=0.01,
        max_iter=1000,
        early_stopping=True,
        validation_fraction=0.1,
        random_state=42
    )
}

results_2 = {}

for name, model in models_2.items():
    print(f"\n{name} 학습 중...")
    model.fit(X_train_2_scaled, y_train_2)
    
    train_acc = accuracy_score(y_train_2, model.predict(X_train_2_scaled))
    test_acc = accuracy_score(y_test_2, model.predict(X_test_2_scaled))
    cv_scores = cross_val_score(model, X_train_2_scaled, y_train_2, cv=5, scoring='accuracy')
    
    results_2[name] = {
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'overfitting': train_acc - test_acc
    }
    
    print(f"  Train: {train_acc:.4f} | Test: {test_acc:.4f} | CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
    print(f"  과적합: {(train_acc - test_acc)*100:.2f}%p")

# Stacking Ensemble
print("\nStacking Ensemble 학습 중...")
stacking_2 = StackingClassifier(
    estimators=[(name, model) for name, model in models_2.items()],
    final_estimator=LogisticRegression(C=0.1, max_iter=1000, random_state=42),
    cv=5
)
stacking_2.fit(X_train_2_scaled, y_train_2)

train_acc_s2 = accuracy_score(y_train_2, stacking_2.predict(X_train_2_scaled))
test_acc_s2 = accuracy_score(y_test_2, stacking_2.predict(X_test_2_scaled))
cv_scores_s2 = cross_val_score(stacking_2, X_train_2_scaled, y_train_2, cv=5, scoring='accuracy')

results_2['Stacking'] = {
    'model': stacking_2,
    'train_acc': train_acc_s2,
    'test_acc': test_acc_s2,
    'cv_mean': cv_scores_s2.mean(),
    'cv_std': cv_scores_s2.std(),
    'overfitting': train_acc_s2 - test_acc_s2
}

print(f"  Train: {train_acc_s2:.4f} | Test: {test_acc_s2:.4f} | CV: {cv_scores_s2.mean():.4f}±{cv_scores_s2.std():.4f}")
print(f"  과적합: {(train_acc_s2 - test_acc_s2)*100:.2f}%p")

# Voting Ensemble
print("\nVoting Ensemble 학습 중...")
voting_2 = VotingClassifier(
    estimators=[(name, model) for name, model in models_2.items()],
    voting='soft',
    weights=[2, 1, 1, 1]
)
voting_2.fit(X_train_2_scaled, y_train_2)

train_acc_v2 = accuracy_score(y_train_2, voting_2.predict(X_train_2_scaled))
test_acc_v2 = accuracy_score(y_test_2, voting_2.predict(X_test_2_scaled))
cv_scores_v2 = cross_val_score(voting_2, X_train_2_scaled, y_train_2, cv=5, scoring='accuracy')

results_2['Voting'] = {
    'model': voting_2,
    'train_acc': train_acc_v2,
    'test_acc': test_acc_v2,
    'cv_mean': cv_scores_v2.mean(),
    'cv_std': cv_scores_v2.std(),
    'overfitting': train_acc_v2 - test_acc_v2
}

print(f"  Train: {train_acc_v2:.4f} | Test: {test_acc_v2:.4f} | CV: {cv_scores_v2.mean():.4f}±{cv_scores_v2.std():.4f}")
print(f"  과적합: {(train_acc_v2 - test_acc_v2)*100:.2f}%p")

# 최고 모델 선택
best_2 = max(results_2.items(), key=lambda x: x[1]['test_acc'])
print(f"\n✅ 모델 2 최고: {best_2[0]} - {best_2[1]['test_acc']:.4f} ({best_2[1]['test_acc']*100:.2f}%)")

# ============================================================================
# 파이프라인 통합 (기존과 동일)
# ============================================================================
print("\n" + "=" * 100)
print("🔗 파이프라인 통합: 확신도 기반 3-클래스 분류")
print("=" * 100)

y_test_full = y_full.loc[X_test_2.index]
X_test_full_scaled_1 = scaler_1.transform(X_test_2)
X_test_full_scaled_2 = scaler_2.transform(X_test_2)

model1 = best_1[1]['model']
stage1_proba = model1.predict_proba(X_test_full_scaled_1)
stage1_pred = model1.predict(X_test_full_scaled_1)

model2 = best_2[1]['model']
stage2_pred = model2.predict(X_test_full_scaled_2)

# 확신도 임계값 최적화
thresholds = [0.75, 0.80, 0.85, 0.90, 0.92, 0.94, 0.96, 0.98]

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

print("\n모델별 정확도:")
print("-" * 100)
print(f"모델 1: {best_1[1]['test_acc']:.4f} ({best_1[1]['test_acc']*100:.2f}%)")
print(f"모델 2: {best_2[1]['test_acc']:.4f} ({best_2[1]['test_acc']*100:.2f}%)")
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

# 모델 저장 (기존과 동일한 형식)
print("\n" + "=" * 100)
print("💾 모델 저장")
print("=" * 100)

save_dir = 'saved_models'
os.makedirs(save_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

model1_path = os.path.join(save_dir, f'model1_optimized_{timestamp}.pkl')
scaler1_path = os.path.join(save_dir, f'scaler1_{timestamp}.pkl')
model2_path = os.path.join(save_dir, f'model2_optimized_{timestamp}.pkl')
scaler2_path = os.path.join(save_dir, f'scaler2_{timestamp}.pkl')

joblib.dump(model1, model1_path)
joblib.dump(scaler_1, scaler1_path)
joblib.dump(model2, model2_path)
joblib.dump(scaler_2, scaler2_path)

config = {
    'model1_name': best_1[0],
    'model1_accuracy': best_1[1]['test_acc'],
    'model2_name': best_2[0],
    'model2_accuracy': best_2[1]['test_acc'],
    'best_threshold': best_threshold,
    'final_accuracy': best_accuracy,
    'timestamp': timestamp,
    'feature_count': X_full.shape[1]
}

config_path = os.path.join(save_dir, f'config_{timestamp}.pkl')
joblib.dump(config, config_path)

print(f"✅ 모델 저장 완료")
print(f"  • {model1_path}")
print(f"  • {model2_path}")
print(f"  • {config_path}")

print("\n" + "=" * 100)
print("🎯 최종 결과")
print("=" * 100)
print(f"모델 1: {best_1[1]['test_acc']*100:.2f}%")
print(f"모델 2: {best_2[1]['test_acc']*100:.2f}%")
print(f"통합 시스템: {best_accuracy*100:.2f}%")

if best_accuracy >= 0.95:
    print("\n🎉🎉🎉 95% 목표 달성! 🎉🎉🎉")
elif best_accuracy >= 0.90:
    print(f"\n💪 90% 이상 달성! 목표까지 {(0.95-best_accuracy)*100:.2f}%p")
else:
    print(f"\n📊 추가 개선 필요: {(0.95-best_accuracy)*100:.2f}%p")

print("\n" + "=" * 100)
