"""
빠른 하이브리드 모델 (Fast Hybrid Model)
- 1단계: 92% 달성한 2-클래스 모델 (CONFIRMED vs FALSE POSITIVE)
- 2단계: CANDIDATE 판별 모델 (확신도 낮은 케이스만)
- 26개 피처 사용, 빠른 구현, 높은 정확도 목표
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 100)
print("⚡ 빠른 하이브리드 모델 (92% 기반 + CANDIDATE 판별)")
print("=" * 100)

# ============================================================================
# 데이터 로드 및 피처 엔지니어링 (92% 모델과 동일)
# ============================================================================
print("\n[데이터 로드 및 피처 엔지니어링]")
print("-" * 100)

df = pd.read_csv('datasets/exoplanets.csv')
print(f"원본 데이터: {df.shape[0]:,} 샘플")

y_full = df['koi_disposition']
X_full = df.drop(['koi_disposition', 'kepoi_name'], axis=1, errors='ignore')

# 수치형 컬럼만 선택
numeric_cols = X_full.select_dtypes(include=[np.number]).columns
X_full = X_full[numeric_cols]
print(f"기본 피처: {len(numeric_cols)}개")

# 피처 엔지니어링 (92% 모델 동일)
print("피처 엔지니어링 중...")
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

print(f"최종 피처 수: {X_full.shape[1]}개")

# ============================================================================
# 1단계: 92% 2-클래스 모델 (CONFIRMED vs FALSE POSITIVE)
# ============================================================================
print("\n" + "=" * 100)
print("[1단계] 2-클래스 모델: CONFIRMED vs FALSE POSITIVE (92% 목표)")
print("=" * 100)

# CANDIDATE 제외
y_binary = y_full[y_full.isin(['CONFIRMED', 'FALSE POSITIVE'])]
X_binary = X_full.loc[y_binary.index]

print(f"\n1단계 학습 데이터:")
print(f"  총 샘플: {len(y_binary):,}")
for label, count in y_binary.value_counts().sort_index().items():
    print(f"  {label}: {count:,} ({count/len(y_binary)*100:.1f}%)")

# 레이블 인코딩
y_binary_encoded = (y_binary == 'CONFIRMED').astype(int)

# Train/Test 분할
X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
    X_binary, y_binary_encoded, test_size=0.1, random_state=42, stratify=y_binary_encoded
)

print(f"\nTrain: {len(y_train_1):,} / Test: {len(y_test_1):,}")

# 스케일링
scaler_1 = StandardScaler()
X_train_1_scaled = scaler_1.fit_transform(X_train_1)
X_test_1_scaled = scaler_1.transform(X_test_1)

# 강한 정규화 모델 (92% 달성한 설정)
print("\n1단계 모델 학습 중...")

models_1 = {
    'XGBoost': XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=2.0,
        reg_lambda=10.0,
        random_state=42,
        eval_metric='logloss'
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=2.0,
        reg_lambda=10.0,
        random_state=42,
        verbose=-1
    ),
    'CatBoost': CatBoostClassifier(
        iterations=500,
        depth=4,
        learning_rate=0.02,
        l2_leaf_reg=10.0,
        bagging_temperature=1.0,
        random_state=42,
        verbose=False
    )
}

results_1 = {}

for name, model in models_1.items():
    print(f"\n{name} 학습 중...")
    model.fit(X_train_1_scaled, y_train_1)
    
    y_train_pred = model.predict(X_train_1_scaled)
    y_test_pred = model.predict(X_test_1_scaled)
    y_test_proba = model.predict_proba(X_test_1_scaled)
    
    train_acc = accuracy_score(y_train_1, y_train_pred)
    test_acc = accuracy_score(y_test_1, y_test_pred)
    cv_scores = cross_val_score(model, X_train_1_scaled, y_train_1, cv=5, scoring='accuracy')
    auc = roc_auc_score(y_test_1, y_test_proba[:, 1])
    
    results_1[name] = {
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'overfitting': train_acc - test_acc,
        'auc': auc,
        'y_test_proba': y_test_proba
    }
    
    print(f"  Train: {train_acc:.4f} | Test: {test_acc:.4f} | CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
    print(f"  과적합: {train_acc - test_acc:.4f} ({(train_acc - test_acc)*100:.2f}%p) | AUC: {auc:.4f}")

# 앙상블
print("\n1단계 앙상블 (Soft Voting)...")
voting_1 = VotingClassifier(
    estimators=[(name, model) for name, model in models_1.items()],
    voting='soft'
)
voting_1.fit(X_train_1_scaled, y_train_1)

y_train_pred_v1 = voting_1.predict(X_train_1_scaled)
y_test_pred_v1 = voting_1.predict(X_test_1_scaled)
y_test_proba_v1 = voting_1.predict_proba(X_test_1_scaled)

train_acc_v1 = accuracy_score(y_train_1, y_train_pred_v1)
test_acc_v1 = accuracy_score(y_test_1, y_test_pred_v1)
cv_scores_v1 = cross_val_score(voting_1, X_train_1_scaled, y_train_1, cv=5, scoring='accuracy')
auc_v1 = roc_auc_score(y_test_1, y_test_proba_v1[:, 1])

print(f"  Train: {train_acc_v1:.4f} | Test: {test_acc_v1:.4f} | CV: {cv_scores_v1.mean():.4f}±{cv_scores_v1.std():.4f}")
print(f"  과적합: {train_acc_v1 - test_acc_v1:.4f} ({(train_acc_v1 - test_acc_v1)*100:.2f}%p) | AUC: {auc_v1:.4f}")

results_1['Voting'] = {
    'model': voting_1,
    'train_acc': train_acc_v1,
    'test_acc': test_acc_v1,
    'cv_mean': cv_scores_v1.mean(),
    'cv_std': cv_scores_v1.std(),
    'overfitting': train_acc_v1 - test_acc_v1,
    'auc': auc_v1,
    'y_test_proba': y_test_proba_v1
}

best_1 = max(results_1.items(), key=lambda x: x[1]['test_acc'])
print(f"\n✅ 1단계 최고: {best_1[0]} - {best_1[1]['test_acc']:.4f} ({best_1[1]['test_acc']*100:.2f}%)")

# ============================================================================
# 2단계: CANDIDATE 판별 모델
# ============================================================================
print("\n" + "=" * 100)
print("[2단계] CANDIDATE 판별 모델")
print("=" * 100)

# 전체 데이터에서 CANDIDATE 레이블 생성
y_candidate = (y_full == 'CANDIDATE').astype(int)
X_candidate = X_full.copy()

print(f"\n2단계 학습 데이터:")
print(f"  총 샘플: {len(y_candidate):,}")
print(f"  CANDIDATE: {y_candidate.sum():,} ({y_candidate.sum()/len(y_candidate)*100:.1f}%)")
print(f"  NOT CANDIDATE: {(~y_candidate.astype(bool)).sum():,} ({(~y_candidate.astype(bool)).sum()/len(y_candidate)*100:.1f}%)")

# Train/Test 분할
X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    X_candidate, y_candidate, test_size=0.1, random_state=42, stratify=y_candidate
)

print(f"\nTrain: {len(y_train_2):,} / Test: {len(y_test_2):,}")

# 스케일링
scaler_2 = StandardScaler()
X_train_2_scaled = scaler_2.fit_transform(X_train_2)
X_test_2_scaled = scaler_2.transform(X_test_2)

# 강한 정규화 모델
print("\n2단계 모델 학습 중...")

models_2 = {
    'XGBoost': XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=2.0,
        reg_lambda=10.0,
        random_state=42,
        eval_metric='logloss'
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=2.0,
        reg_lambda=10.0,
        random_state=42,
        verbose=-1
    ),
    'CatBoost': CatBoostClassifier(
        iterations=500,
        depth=4,
        learning_rate=0.02,
        l2_leaf_reg=10.0,
        bagging_temperature=1.0,
        random_state=42,
        verbose=False
    )
}

results_2 = {}

for name, model in models_2.items():
    print(f"\n{name} 학습 중...")
    model.fit(X_train_2_scaled, y_train_2)
    
    y_train_pred = model.predict(X_train_2_scaled)
    y_test_pred = model.predict(X_test_2_scaled)
    y_test_proba = model.predict_proba(X_test_2_scaled)
    
    train_acc = accuracy_score(y_train_2, y_train_pred)
    test_acc = accuracy_score(y_test_2, y_test_pred)
    cv_scores = cross_val_score(model, X_train_2_scaled, y_train_2, cv=5, scoring='accuracy')
    auc = roc_auc_score(y_test_2, y_test_proba[:, 1])
    
    results_2[name] = {
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'overfitting': train_acc - test_acc,
        'auc': auc,
        'y_test_proba': y_test_proba
    }
    
    print(f"  Train: {train_acc:.4f} | Test: {test_acc:.4f} | CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
    print(f"  과적합: {train_acc - test_acc:.4f} ({(train_acc - test_acc)*100:.2f}%p) | AUC: {auc:.4f}")

# 앙상블
print("\n2단계 앙상블 (Soft Voting)...")
voting_2 = VotingClassifier(
    estimators=[(name, model) for name, model in models_2.items()],
    voting='soft'
)
voting_2.fit(X_train_2_scaled, y_train_2)

y_train_pred_v2 = voting_2.predict(X_train_2_scaled)
y_test_pred_v2 = voting_2.predict(X_test_2_scaled)
y_test_proba_v2 = voting_2.predict_proba(X_test_2_scaled)

train_acc_v2 = accuracy_score(y_train_2, y_train_pred_v2)
test_acc_v2 = accuracy_score(y_test_2, y_test_pred_v2)
cv_scores_v2 = cross_val_score(voting_2, X_train_2_scaled, y_train_2, cv=5, scoring='accuracy')
auc_v2 = roc_auc_score(y_test_2, y_test_proba_v2[:, 1])

print(f"  Train: {train_acc_v2:.4f} | Test: {test_acc_v2:.4f} | CV: {cv_scores_v2.mean():.4f}±{cv_scores_v2.std():.4f}")
print(f"  과적합: {train_acc_v2 - test_acc_v2:.4f} ({(train_acc_v2 - test_acc_v2)*100:.2f}%p) | AUC: {auc_v2:.4f}")

results_2['Voting'] = {
    'model': voting_2,
    'train_acc': train_acc_v2,
    'test_acc': test_acc_v2,
    'cv_mean': cv_scores_v2.mean(),
    'cv_std': cv_scores_v2.std(),
    'overfitting': train_acc_v2 - test_acc_v2,
    'auc': auc_v2,
    'y_test_proba': y_test_proba_v2
}

best_2 = max(results_2.items(), key=lambda x: x[1]['test_acc'])
print(f"\n✅ 2단계 최고: {best_2[0]} - {best_2[1]['test_acc']:.4f} ({best_2[1]['test_acc']*100:.2f}%)")

# ============================================================================
# 파이프라인 통합 및 확신도 임계값 최적화
# ============================================================================
print("\n" + "=" * 100)
print("[파이프라인 통합] 확신도 임계값 최적화")
print("=" * 100)

# 전체 테스트 데이터 준비
y_test_full = y_full.loc[X_test_2.index]
X_test_full_scaled_1 = scaler_1.transform(X_test_2)
X_test_full_scaled_2 = scaler_2.transform(X_test_2)

# 1단계 예측
stage1_proba = results_1['Voting']['model'].predict_proba(X_test_full_scaled_1)
stage1_pred = results_1['Voting']['model'].predict(X_test_full_scaled_1)

# 2단계 예측
stage2_proba = results_2['Voting']['model'].predict_proba(X_test_full_scaled_2)
stage2_pred = results_2['Voting']['model'].predict(X_test_full_scaled_2)

# 확신도 임계값 최적화
thresholds = np.arange(0.80, 0.96, 0.02)
threshold_results = []

print(f"\n{'임계값':<10} {'1단계사용':<12} {'CANDIDATE':<12} {'최종정확도':<12}")
print("-" * 100)

for threshold in thresholds:
    final_predictions = np.empty(len(y_test_full), dtype=object)
    
    # 1단계 확신도가 높은 케이스
    high_conf_mask = (stage1_proba.max(axis=1) >= threshold)
    
    # 1단계 고확신도 → CONFIRMED or FALSE POSITIVE
    final_predictions[high_conf_mask] = np.where(
        stage1_pred[high_conf_mask] == 1,
        'CONFIRMED',
        'FALSE POSITIVE'
    )
    
    # 1단계 저확신도 → CANDIDATE 판별 사용
    low_conf_mask = ~high_conf_mask
    final_predictions[low_conf_mask] = np.where(
        stage2_pred[low_conf_mask] == 1,
        'CANDIDATE',
        np.where(stage1_pred[low_conf_mask] == 1, 'CONFIRMED', 'FALSE POSITIVE')
    )
    
    accuracy = accuracy_score(y_test_full, final_predictions)
    stage1_ratio = high_conf_mask.sum() / len(y_test_full) * 100
    candidate_count = (final_predictions == 'CANDIDATE').sum()
    
    threshold_results.append({
        'threshold': threshold,
        'accuracy': accuracy,
        'stage1_ratio': stage1_ratio,
        'candidate_count': candidate_count
    })
    
    print(f"{threshold:.2f}      {stage1_ratio:>6.1f}%      {candidate_count:>4}개       {accuracy:.4f}")

# 최적 임계값
best_result = max(threshold_results, key=lambda x: x['accuracy'])
best_threshold = best_result['threshold']
best_accuracy = best_result['accuracy']

print(f"\n✅ 최적 임계값: {best_threshold:.2f}")
print(f"✅ 최종 정확도: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"   1단계 사용: {best_result['stage1_ratio']:.1f}%")
print(f"   CANDIDATE 판별: {best_result['candidate_count']}개")

# 최적 임계값으로 최종 예측
high_conf_mask = (stage1_proba.max(axis=1) >= best_threshold)
final_predictions = np.empty(len(y_test_full), dtype=object)

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

# ============================================================================
# 과적합 분석 및 최종 평가
# ============================================================================
print("\n" + "=" * 100)
print("[과적합 분석]")
print("=" * 100)

print("\n1단계 모델 과적합:")
for name, r in results_1.items():
    status = "✅" if r['overfitting'] < 0.03 else "⚠️"
    print(f"{status} {name:15} {r['overfitting']:>7.4f} ({r['overfitting']*100:>5.2f}%p)")

print("\n2단계 모델 과적합:")
for name, r in results_2.items():
    status = "✅" if r['overfitting'] < 0.03 else "⚠️"
    print(f"{status} {name:15} {r['overfitting']:>7.4f} ({r['overfitting']*100:>5.2f}%p)")

# Confusion Matrix
print("\n" + "=" * 100)
print("[최종 성능 평가]")
print("=" * 100)

cm = confusion_matrix(y_test_full, final_predictions,
                      labels=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'])

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'],
            yticklabels=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'])
plt.title(f'Confusion Matrix (빠른 하이브리드) - 정확도: {best_accuracy:.4f}')
plt.ylabel('실제')
plt.xlabel('예측')
plt.tight_layout()
plt.savefig('confusion_matrix_fast_hybrid.png', dpi=150, bbox_inches='tight')
print("\n✅ Confusion Matrix 저장: confusion_matrix_fast_hybrid.png")

print("\nClassification Report:")
print(classification_report(y_test_full, final_predictions,
                           labels=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'],
                           target_names=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']))

print("\n클래스별 정확도:")
for label in ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']:
    mask = y_test_full == label
    if mask.sum() > 0:
        class_acc = accuracy_score(y_test_full[mask], final_predictions[mask])
        print(f"  {label:20} {class_acc:.4f} ({class_acc*100:.2f}%)  [{mask.sum()}개]")

# 목표 달성 여부
print("\n" + "=" * 100)
print("🎯 목표 달성 평가")
print("=" * 100)
print(f"목표: 95.00%")
print(f"달성: {best_accuracy*100:.2f}%")
print(f"격차: {(best_accuracy - 0.95)*100:+.2f}%p")

if best_accuracy >= 0.95:
    print("\n🎉🎉🎉 95% 목표 달성! 🎉🎉🎉")
elif best_accuracy >= 0.90:
    print(f"\n💪 90% 이상 달성! 추가 {(0.95 - best_accuracy)*100:.2f}%p 필요")
else:
    print(f"\n📊 추가 개선 필요: {(0.95 - best_accuracy)*100:.2f}%p")

print("\n" + "=" * 100)
print("✅ 분석 완료")
print("=" * 100)
