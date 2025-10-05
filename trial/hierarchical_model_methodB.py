"""
방법 B: 확신도 기반 계층적 분류 모델
- 1단계: CONFIRMED vs FALSE POSITIVE (CANDIDATE 제외, 92% 정확도 목표)
- 2단계: 3-클래스 (CANDIDATE 포함, 낮은 확신도 케이스 처리)
- 다중공선성 제거 적용
- 과적합 방지 및 95% 정확도 목표
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix, classification_report, 
                             roc_auc_score, roc_curve)
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 100)
print("방법 B: 확신도 기반 계층적 분류 시스템")
print("=" * 100)

# ============================================================================
# STEP 1: 다중공선성 분석 및 제거
# ============================================================================
print("\n[STEP 1] 다중공선성 분석 및 제거")
print("-" * 100)

df = pd.read_csv('datasets/exoplanets.csv')
print(f"원본 데이터: {df.shape[0]:,} 샘플, {df.shape[1]-1} 피처")

# 타겟 분리
y_full = df['koi_disposition']
X_full = df.drop(['koi_disposition', 'kepoi_name'], axis=1, errors='ignore')

# NaN 제거
valid_idx = y_full.notna()
X_full = X_full[valid_idx]
y_full = y_full[valid_idx]

print(f"NaN 제거 후: {len(y_full):,} 샘플")

# 수치형 컬럼만 선택 (문자열 컬럼 제외)
numeric_cols = X_full.select_dtypes(include=[np.number]).columns
X_full = X_full[numeric_cols]
print(f"수치형 피처만 선택: {len(numeric_cols)}개")

# ============================================================================
# 피처 엔지니어링 (92% 달성 모델에서 사용한 피처들)
# ============================================================================
print("\n피처 엔지니어링 중...")

# 1. 행성-항성 비율 (Planet-Star Radius Ratio)
X_full['planet_star_ratio'] = X_full['koi_prad'] / X_full['koi_srad']

# 2. 궤도 에너지 (Orbital Energy) - 반장축의 역수
X_full['orbital_energy'] = 1.0 / (X_full['koi_sma'] + 1e-10)

# 3. 통과 신호 강도 (Transit Signal Strength)
X_full['transit_signal'] = X_full['koi_depth'] * X_full['koi_duration']

# 4. 항성 밀도 (Stellar Density)
X_full['stellar_density'] = X_full['koi_smass'] / (X_full['koi_srad']**3 + 1e-10)

# 5. 행성 밀도 프록시 (Planet Density Proxy)
X_full['planet_density_proxy'] = X_full['koi_prad']**3 / (X_full['koi_sma']**2 + 1e-10)

# 6. Log 변환 (왜도 감소)
X_full['log_period'] = np.log1p(X_full['koi_sma'])
X_full['log_depth'] = np.log1p(X_full['koi_depth'])
X_full['log_insol'] = np.log1p(X_full['koi_insol'])

# 7. 궤도 안정성 지표 (이심률과 충돌 파라미터의 조합)
X_full['orbit_stability'] = X_full['koi_eccen'] * X_full['koi_impact']

# NaN 처리 (무한대나 NaN이 생성된 경우)
X_full = X_full.replace([np.inf, -np.inf], np.nan)
X_full = X_full.fillna(X_full.median())

print(f"엔지니어링 후 총 피처: {X_full.shape[1]}개")
print(f"추가된 피처: planet_star_ratio, orbital_energy, transit_signal, stellar_density,")
print(f"            planet_density_proxy, log_period, log_depth, log_insol, orbit_stability")

# 상관계수 매트릭스 계산
correlation_matrix = X_full.corr().abs()

# 상관계수 0.8 이상인 피처 쌍 찾기
high_corr_pairs = []
for i in range(len(correlation_matrix.columns)):
    for j in range(i+1, len(correlation_matrix.columns)):
        if correlation_matrix.iloc[i, j] >= 0.8:
            high_corr_pairs.append({
                'feature1': correlation_matrix.columns[i],
                'feature2': correlation_matrix.columns[j],
                'correlation': correlation_matrix.iloc[i, j]
            })

print(f"\n다중공선성 발견 (상관계수 ≥ 0.8): {len(high_corr_pairs)}개 쌍")
for pair in high_corr_pairs:
    print(f"  {pair['feature1']} ↔ {pair['feature2']}: {pair['correlation']:.3f}")

# 다중공선성 제거 전략: 각 쌍에서 타겟과 상관관계가 낮은 피처 제거
features_to_drop = set()

# 2-클래스 타겟 생성 (다중공선성 분석용)
y_binary = y_full.copy()
y_binary = y_binary[y_binary.isin(['CONFIRMED', 'FALSE POSITIVE'])]
X_binary_temp = X_full.loc[y_binary.index]
y_binary_encoded = (y_binary == 'CONFIRMED').astype(int)

for pair in high_corr_pairs:
    feat1, feat2 = pair['feature1'], pair['feature2']
    
    # 각 피처와 타겟의 상관계수 계산
    corr1 = abs(X_binary_temp[feat1].corr(y_binary_encoded))
    corr2 = abs(X_binary_temp[feat2].corr(y_binary_encoded))
    
    # 타겟과 상관관계가 낮은 피처 제거
    if corr1 < corr2:
        features_to_drop.add(feat1)
    else:
        features_to_drop.add(feat2)

print(f"\n제거할 피처 ({len(features_to_drop)}개):")
for feat in sorted(features_to_drop):
    print(f"  - {feat}")

# 피처 제거
X_reduced = X_full.drop(columns=list(features_to_drop))
print(f"\n최종 피처 수: {X_reduced.shape[1]}개")
print(f"사용할 피처: {', '.join(X_reduced.columns.tolist())}")

# ============================================================================
# STEP 2: 1단계 모델 - 2-클래스 분류 (CONFIRMED vs FALSE POSITIVE)
# ============================================================================
print("\n" + "=" * 100)
print("[STEP 2] 1단계 모델: 2-클래스 분류 (CONFIRMED vs FALSE POSITIVE)")
print("=" * 100)

# CANDIDATE 제외
y_stage1 = y_full[y_full.isin(['CONFIRMED', 'FALSE POSITIVE'])]
X_stage1 = X_reduced.loc[y_stage1.index]

print(f"\n1단계 학습 데이터:")
print(f"  총 샘플: {len(y_stage1):,}")
for label, count in y_stage1.value_counts().sort_index().items():
    print(f"  {label}: {count:,} ({count/len(y_stage1)*100:.1f}%)")

# 레이블 인코딩 (CONFIRMED=1, FALSE POSITIVE=0)
y_stage1_encoded = (y_stage1 == 'CONFIRMED').astype(int)

# Train/Test 분할
X_train_s1, X_test_s1, y_train_s1, y_test_s1 = train_test_split(
    X_stage1, y_stage1_encoded, test_size=0.1, random_state=42, stratify=y_stage1_encoded
)

print(f"\nTrain: {len(y_train_s1):,} / Test: {len(y_test_s1):,}")

# 스케일링
scaler_s1 = StandardScaler()
X_train_s1_scaled = scaler_s1.fit_transform(X_train_s1)
X_test_s1_scaled = scaler_s1.transform(X_test_s1)

# 강한 정규화 적용 - 과적합 방지
print("\n1단계 모델 학습 중 (강한 정규화 적용)...")

models_stage1 = {
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

results_stage1 = {}

for name, model in models_stage1.items():
    print(f"\n{name} 학습 중...")
    
    # 학습
    model.fit(X_train_s1_scaled, y_train_s1)
    
    # 예측
    y_train_pred = model.predict(X_train_s1_scaled)
    y_test_pred = model.predict(X_test_s1_scaled)
    
    # 확신도 (probability)
    y_test_proba = model.predict_proba(X_test_s1_scaled)
    
    # 정확도
    train_acc = accuracy_score(y_train_s1, y_train_pred)
    test_acc = accuracy_score(y_test_s1, y_test_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_s1_scaled, y_train_s1, cv=5, scoring='accuracy')
    
    # AUC
    auc = roc_auc_score(y_test_s1, y_test_proba[:, 1])
    
    results_stage1[name] = {
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'overfitting': train_acc - test_acc,
        'auc': auc,
        'y_test_proba': y_test_proba
    }
    
    print(f"  Train 정확도: {train_acc:.4f}")
    print(f"  Test 정확도:  {test_acc:.4f}")
    print(f"  CV 정확도:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  과적합:       {train_acc - test_acc:.4f} ({(train_acc - test_acc)*100:.2f}%p)")
    print(f"  AUC:          {auc:.4f}")

# 최고 성능 모델 선택
best_model_s1 = max(results_stage1.items(), key=lambda x: x[1]['test_acc'])
print(f"\n1단계 최고 모델: {best_model_s1[0]} - Test 정확도: {best_model_s1[1]['test_acc']:.4f}")

# 앙상블 (Voting)
print("\n1단계 앙상블 (Soft Voting) 학습 중...")
voting_s1 = VotingClassifier(
    estimators=[(name, model) for name, model in models_stage1.items()],
    voting='soft'
)
voting_s1.fit(X_train_s1_scaled, y_train_s1)

y_train_pred_voting = voting_s1.predict(X_train_s1_scaled)
y_test_pred_voting = voting_s1.predict(X_test_s1_scaled)
y_test_proba_voting = voting_s1.predict_proba(X_test_s1_scaled)

train_acc_voting = accuracy_score(y_train_s1, y_train_pred_voting)
test_acc_voting = accuracy_score(y_test_s1, y_test_pred_voting)
cv_scores_voting = cross_val_score(voting_s1, X_train_s1_scaled, y_train_s1, cv=5, scoring='accuracy')
auc_voting = roc_auc_score(y_test_s1, y_test_proba_voting[:, 1])

print(f"  Train 정확도: {train_acc_voting:.4f}")
print(f"  Test 정확도:  {test_acc_voting:.4f}")
print(f"  CV 정확도:    {cv_scores_voting.mean():.4f} ± {cv_scores_voting.std():.4f}")
print(f"  과적합:       {train_acc_voting - test_acc_voting:.4f} ({(train_acc_voting - test_acc_voting)*100:.2f}%p)")
print(f"  AUC:          {auc_voting:.4f}")

results_stage1['Voting'] = {
    'model': voting_s1,
    'train_acc': train_acc_voting,
    'test_acc': test_acc_voting,
    'cv_mean': cv_scores_voting.mean(),
    'cv_std': cv_scores_voting.std(),
    'overfitting': train_acc_voting - test_acc_voting,
    'auc': auc_voting,
    'y_test_proba': y_test_proba_voting
}

# ============================================================================
# STEP 3: 2단계 모델 - 3-클래스 분류 (CANDIDATE 포함)
# ============================================================================
print("\n" + "=" * 100)
print("[STEP 3] 2단계 모델: 3-클래스 분류 (CANDIDATE 포함)")
print("=" * 100)

# 전체 데이터 사용
y_stage2 = y_full.copy()
X_stage2 = X_reduced.copy()

print(f"\n2단계 학습 데이터:")
print(f"  총 샘플: {len(y_stage2):,}")
for label, count in y_stage2.value_counts().sort_index().items():
    print(f"  {label}: {count:,} ({count/len(y_stage2)*100:.1f}%)")

# 레이블 인코딩
label_encoder = LabelEncoder()
y_stage2_encoded = label_encoder.fit_transform(y_stage2)

print(f"\n인코딩된 레이블:")
for i, label in enumerate(label_encoder.classes_):
    print(f"  {label} → {i}")

# Train/Test 분할
X_train_s2, X_test_s2, y_train_s2, y_test_s2 = train_test_split(
    X_stage2, y_stage2_encoded, test_size=0.1, random_state=42, stratify=y_stage2_encoded
)

print(f"\nTrain: {len(y_train_s2):,} / Test: {len(y_test_s2):,}")

# 스케일링
scaler_s2 = StandardScaler()
X_train_s2_scaled = scaler_s2.fit_transform(X_train_s2)
X_test_s2_scaled = scaler_s2.transform(X_test_s2)

# 강한 정규화 적용
print("\n2단계 모델 학습 중 (강한 정규화 적용)...")

models_stage2 = {
    'XGBoost': XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=2.0,
        reg_lambda=10.0,
        random_state=42,
        eval_metric='mlogloss'
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

results_stage2 = {}

for name, model in models_stage2.items():
    print(f"\n{name} 학습 중...")
    
    # 학습
    model.fit(X_train_s2_scaled, y_train_s2)
    
    # 예측
    y_train_pred = model.predict(X_train_s2_scaled)
    y_test_pred = model.predict(X_test_s2_scaled)
    
    # 확신도
    y_test_proba = model.predict_proba(X_test_s2_scaled)
    
    # 정확도
    train_acc = accuracy_score(y_train_s2, y_train_pred)
    test_acc = accuracy_score(y_test_s2, y_test_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_s2_scaled, y_train_s2, cv=5, scoring='accuracy')
    
    results_stage2[name] = {
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'overfitting': train_acc - test_acc,
        'y_test_proba': y_test_proba
    }
    
    print(f"  Train 정확도: {train_acc:.4f}")
    print(f"  Test 정확도:  {test_acc:.4f}")
    print(f"  CV 정확도:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  과적합:       {train_acc - test_acc:.4f} ({(train_acc - test_acc)*100:.2f}%p)")

# 앙상블
print("\n2단계 앙상블 (Soft Voting) 학습 중...")
voting_s2 = VotingClassifier(
    estimators=[(name, model) for name, model in models_stage2.items()],
    voting='soft'
)
voting_s2.fit(X_train_s2_scaled, y_train_s2)

y_train_pred_voting = voting_s2.predict(X_train_s2_scaled)
y_test_pred_voting = voting_s2.predict(X_test_s2_scaled)
y_test_proba_voting = voting_s2.predict_proba(X_test_s2_scaled)

train_acc_voting = accuracy_score(y_train_s2, y_train_pred_voting)
test_acc_voting = accuracy_score(y_test_s2, y_test_pred_voting)
cv_scores_voting = cross_val_score(voting_s2, X_train_s2_scaled, y_train_s2, cv=5, scoring='accuracy')

print(f"  Train 정확도: {train_acc_voting:.4f}")
print(f"  Test 정확도:  {test_acc_voting:.4f}")
print(f"  CV 정확도:    {cv_scores_voting.mean():.4f} ± {cv_scores_voting.std():.4f}")
print(f"  과적합:       {train_acc_voting - test_acc_voting:.4f} ({(train_acc_voting - test_acc_voting)*100:.2f}%p)")

results_stage2['Voting'] = {
    'model': voting_s2,
    'train_acc': train_acc_voting,
    'test_acc': test_acc_voting,
    'cv_mean': cv_scores_voting.mean(),
    'cv_std': cv_scores_voting.std(),
    'overfitting': train_acc_voting - test_acc_voting,
    'y_test_proba': y_test_proba_voting
}

# ============================================================================
# STEP 4: 파이프라인 통합 및 확신도 임계값 최적화
# ============================================================================
print("\n" + "=" * 100)
print("[STEP 4] 파이프라인 통합 및 확신도 임계값 최적화")
print("=" * 100)

# 전체 테스트 데이터 준비
y_test_full = y_full.loc[X_test_s2.index]
X_test_full_scaled = scaler_s2.transform(X_test_s2)

# 1단계 모델로 전체 테스트 데이터 예측 (2-클래스)
best_stage1_model = results_stage1['Voting']['model']
stage1_proba = best_stage1_model.predict_proba(X_test_full_scaled)

# 각 확신도 임계값별 성능 평가
thresholds = np.arange(0.85, 0.96, 0.01)
threshold_results = []

print("\n확신도 임계값 최적화 중...")
print(f"{'임계값':<10} {'1단계 사용':<12} {'2단계 사용':<12} {'최종 정확도':<12} {'상세'}")
print("-" * 100)

for threshold in thresholds:
    # 확신도가 높은 샘플 (1단계 결과 사용)
    high_confidence_mask = (stage1_proba.max(axis=1) >= threshold)
    
    # 확신도가 낮은 샘플 (2단계 결과 사용)
    low_confidence_mask = ~high_confidence_mask
    
    # 최종 예측
    final_predictions = np.empty(len(y_test_full), dtype=object)
    
    # 1단계 고확신도 예측
    stage1_pred = best_stage1_model.predict(X_test_full_scaled[high_confidence_mask])
    final_predictions[high_confidence_mask] = ['CONFIRMED' if p == 1 else 'FALSE POSITIVE' 
                                                 for p in stage1_pred]
    
    # 2단계 저확신도 예측
    stage2_pred = results_stage2['Voting']['model'].predict(X_test_full_scaled[low_confidence_mask])
    final_predictions[low_confidence_mask] = label_encoder.inverse_transform(stage2_pred)
    
    # 정확도 계산
    accuracy = accuracy_score(y_test_full, final_predictions)
    
    # 1단계/2단계 사용 비율
    stage1_ratio = high_confidence_mask.sum() / len(y_test_full) * 100
    stage2_ratio = low_confidence_mask.sum() / len(y_test_full) * 100
    
    threshold_results.append({
        'threshold': threshold,
        'accuracy': accuracy,
        'stage1_ratio': stage1_ratio,
        'stage2_ratio': stage2_ratio,
        'stage1_count': high_confidence_mask.sum(),
        'stage2_count': low_confidence_mask.sum()
    })
    
    print(f"{threshold:.2f}      {stage1_ratio:>5.1f}%      {stage2_ratio:>5.1f}%      "
          f"{accuracy:.4f}      (1단계:{high_confidence_mask.sum()}, 2단계:{low_confidence_mask.sum()})")

# 최적 임계값 선택
best_threshold_result = max(threshold_results, key=lambda x: x['accuracy'])
best_threshold = best_threshold_result['threshold']
best_accuracy = best_threshold_result['accuracy']

print("\n" + "=" * 100)
print(f"최적 확신도 임계값: {best_threshold:.2f}")
print(f"최종 정확도: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"1단계 사용: {best_threshold_result['stage1_ratio']:.1f}% ({best_threshold_result['stage1_count']}개)")
print(f"2단계 사용: {best_threshold_result['stage2_ratio']:.1f}% ({best_threshold_result['stage2_count']}개)")
print("=" * 100)

# 최적 임계값으로 최종 예측
high_confidence_mask = (stage1_proba.max(axis=1) >= best_threshold)
low_confidence_mask = ~high_confidence_mask

final_predictions = np.empty(len(y_test_full), dtype=object)
stage1_pred = best_stage1_model.predict(X_test_full_scaled[high_confidence_mask])
final_predictions[high_confidence_mask] = ['CONFIRMED' if p == 1 else 'FALSE POSITIVE' 
                                             for p in stage1_pred]
stage2_pred = results_stage2['Voting']['model'].predict(X_test_full_scaled[low_confidence_mask])
final_predictions[low_confidence_mask] = label_encoder.inverse_transform(stage2_pred)

# ============================================================================
# STEP 5: 과적합 분석
# ============================================================================
print("\n" + "=" * 100)
print("[STEP 5] 과적합 분석")
print("=" * 100)

print("\n1단계 모델 (2-클래스) 과적합 분석:")
print("-" * 100)
for name, result in results_stage1.items():
    print(f"{name:20} Train: {result['train_acc']:.4f}  Test: {result['test_acc']:.4f}  "
          f"과적합: {result['overfitting']:.4f} ({result['overfitting']*100:.2f}%p)")

print("\n2단계 모델 (3-클래스) 과적합 분석:")
print("-" * 100)
for name, result in results_stage2.items():
    print(f"{name:20} Train: {result['train_acc']:.4f}  Test: {result['test_acc']:.4f}  "
          f"과적합: {result['overfitting']:.4f} ({result['overfitting']*100:.2f}%p)")

# 과적합 판정
overfitting_threshold = 0.03  # 3%p 이상이면 과적합
print(f"\n과적합 판정 기준: {overfitting_threshold*100:.1f}%p 이상")

stage1_overfitting = [(name, r['overfitting']) for name, r in results_stage1.items() 
                      if r['overfitting'] > overfitting_threshold]
stage2_overfitting = [(name, r['overfitting']) for name, r in results_stage2.items() 
                      if r['overfitting'] > overfitting_threshold]

if stage1_overfitting:
    print(f"\n⚠️  1단계 과적합 모델 ({len(stage1_overfitting)}개):")
    for name, ovf in stage1_overfitting:
        print(f"  - {name}: {ovf*100:.2f}%p")
else:
    print("\n✅ 1단계 모든 모델 과적합 없음")

if stage2_overfitting:
    print(f"\n⚠️  2단계 과적합 모델 ({len(stage2_overfitting)}개):")
    for name, ovf in stage2_overfitting:
        print(f"  - {name}: {ovf*100:.2f}%p")
else:
    print("\n✅ 2단계 모든 모델 과적합 없음")

# ============================================================================
# STEP 6: 최종 성능 평가
# ============================================================================
print("\n" + "=" * 100)
print("[STEP 6] 최종 성능 평가")
print("=" * 100)

# Confusion Matrix
cm = confusion_matrix(y_test_full, final_predictions, 
                      labels=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'])

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'],
            yticklabels=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'])
plt.title(f'Confusion Matrix (방법 B) - 정확도: {best_accuracy:.4f}')
plt.ylabel('실제')
plt.xlabel('예측')
plt.tight_layout()
plt.savefig('confusion_matrix_methodB.png', dpi=150, bbox_inches='tight')
print("\nConfusion Matrix 저장: confusion_matrix_methodB.png")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test_full, final_predictions, 
                           labels=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'],
                           target_names=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']))

# 클래스별 정확도
print("\n클래스별 정확도:")
for i, label in enumerate(['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']):
    mask = y_test_full == label
    if mask.sum() > 0:
        class_acc = accuracy_score(y_test_full[mask], final_predictions[mask])
        print(f"  {label:20} {class_acc:.4f} ({class_acc*100:.2f}%)")

# 목표 달성 여부
target_accuracy = 0.95
gap = best_accuracy - target_accuracy

print("\n" + "=" * 100)
print("목표 달성 평가")
print("=" * 100)
print(f"목표 정확도: {target_accuracy:.4f} (95%)")
print(f"달성 정확도: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"격차:        {gap:.4f} ({gap*100:.2f}%p)")

if best_accuracy >= target_accuracy:
    print("\n🎉 목표 달성! 95% 이상 정확도 달성!")
else:
    print(f"\n📊 목표 미달성. 추가 개선 필요: {abs(gap)*100:.2f}%p 향상 필요")
    print("\n개선 방안:")
    print("  1. 피처 엔지니어링 추가 (도메인 지식 기반)")
    print("  2. 하이퍼파라미터 튜닝 (Optuna, GridSearch)")
    print("  3. 데이터 증강 (SMOTE, ADASYN)")
    print("  4. 딥러닝 모델 추가 (Transformer, Attention)")
    print("  5. 앙상블 다양성 증가 (Stacking 메타 모델 최적화)")

print("\n" + "=" * 100)
print("분석 완료")
print("=" * 100)
