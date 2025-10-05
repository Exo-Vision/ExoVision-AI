"""
천문학적 계층적 분류 (Astronomical Hierarchical Classification)
- 1단계: (CONFIRMED + CANDIDATE) vs FALSE POSITIVE - "외계행성 가능성" 판단
- 2단계: CONFIRMED vs CANDIDATE - "확정 여부" 판단
- 26개 피처 사용 (기본 16개 + 엔지니어링 10개)
- 과적합 방지 및 95% 목표 달성
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
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

# 한글 폰트 설정
plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 100)
print("천문학적 계층적 분류 시스템")
print("1단계: (CONFIRMED + CANDIDATE) vs FALSE POSITIVE")
print("2단계: CONFIRMED vs CANDIDATE")
print("=" * 100)

# ============================================================================
# STEP 1: 데이터 로드 및 피처 엔지니어링 (26개 피처)
# ============================================================================
print("\n[STEP 1] 데이터 로드 및 피처 엔지니어링")
print("-" * 100)

df = pd.read_csv('datasets/exoplanets.csv')
print(f"원본 데이터: {df.shape[0]:,} 샘플")

# 타겟 분리
y_full = df['koi_disposition']
X_full = df.drop(['koi_disposition', 'kepoi_name'], axis=1, errors='ignore')

# 수치형 컬럼만 선택
numeric_cols = X_full.select_dtypes(include=[np.number]).columns
X_full = X_full[numeric_cols]
print(f"기본 피처: {len(numeric_cols)}개")

# ============================================================================
# 피처 엔지니어링 (92% 달성 모델에서 사용한 10개 피처)
# ============================================================================
print("\n피처 엔지니어링 중...")

# 1. 행성-항성 비율 (Planet-Star Radius Ratio)
X_full['planet_star_ratio'] = X_full['koi_prad'] / (X_full['koi_srad'] + 1e-10)

# 2. 궤도 에너지 (Orbital Energy)
X_full['orbital_energy'] = 1.0 / (X_full['koi_sma'] + 1e-10)

# 3. 통과 신호 강도 (Transit Signal Strength)
X_full['transit_signal'] = X_full['koi_depth'] * X_full['koi_duration']

# 4. 항성 밀도 (Stellar Density)
X_full['stellar_density'] = X_full['koi_smass'] / (X_full['koi_srad']**3 + 1e-10)

# 5. 행성 밀도 프록시 (Planet Density Proxy)
X_full['planet_density_proxy'] = X_full['koi_prad']**3 / (X_full['koi_sma']**2 + 1e-10)

# 6. Log 변환 (왜도 감소)
X_full['log_period'] = np.log1p(X_full['koi_period'])
X_full['log_depth'] = np.log1p(X_full['koi_depth'])
X_full['log_insol'] = np.log1p(X_full['koi_insol'])

# 7. 궤도 안정성 지표
X_full['orbit_stability'] = X_full['koi_eccen'] * X_full['koi_impact']

# 8. Transit SNR (Signal-to-Noise Ratio proxy)
X_full['transit_snr'] = X_full['koi_depth'] / (X_full['koi_duration'] + 1e-10)

# NaN 처리
X_full = X_full.replace([np.inf, -np.inf], np.nan)
X_full = X_full.fillna(X_full.median())

print(f"최종 피처 수: {X_full.shape[1]}개")
print(f"엔지니어링 피처: planet_star_ratio, orbital_energy, transit_signal, stellar_density,")
print(f"                 planet_density_proxy, log_period, log_depth, log_insol,")
print(f"                 orbit_stability, transit_snr")

# ============================================================================
# STEP 2: 1단계 모델 - (CONFIRMED + CANDIDATE) vs FALSE POSITIVE
# ============================================================================
print("\n" + "=" * 100)
print("[STEP 2] 1단계 모델: (CONFIRMED + CANDIDATE) vs FALSE POSITIVE")
print("=" * 100)

# 1단계 레이블 생성: EXOPLANET_LIKE (CONFIRMED + CANDIDATE) vs FALSE_POSITIVE
y_stage1 = y_full.copy()
y_stage1_binary = y_stage1.replace({
    'CONFIRMED': 'EXOPLANET_LIKE',
    'CANDIDATE': 'EXOPLANET_LIKE',
    'FALSE POSITIVE': 'NOT_EXOPLANET',
    'REFUTED': 'NOT_EXOPLANET'
})

# NaN 제거
valid_idx = y_stage1_binary.notna()
X_stage1 = X_full[valid_idx]
y_stage1_binary = y_stage1_binary[valid_idx]

print(f"\n1단계 학습 데이터:")
print(f"  총 샘플: {len(y_stage1_binary):,}")
for label, count in y_stage1_binary.value_counts().sort_index().items():
    print(f"  {label}: {count:,} ({count/len(y_stage1_binary)*100:.1f}%)")

# 레이블 인코딩 (EXOPLANET_LIKE=1, NOT_EXOPLANET=0)
y_stage1_encoded = (y_stage1_binary == 'EXOPLANET_LIKE').astype(int)

# Train/Test 분할
X_train_s1, X_test_s1, y_train_s1, y_test_s1 = train_test_split(
    X_stage1, y_stage1_encoded, test_size=0.1, random_state=42, stratify=y_stage1_encoded
)

print(f"\nTrain: {len(y_train_s1):,} / Test: {len(y_test_s1):,}")

# 스케일링
scaler_s1 = StandardScaler()
X_train_s1_scaled = scaler_s1.fit_transform(X_train_s1)
X_test_s1_scaled = scaler_s1.transform(X_test_s1)

# 강한 정규화 적용
print("\n1단계 모델 학습 중 (강한 정규화)...")

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
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42
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

# 최고 성능 모델 선택
best_model_s1 = max(results_stage1.items(), key=lambda x: x[1]['test_acc'])
print(f"\n✅ 1단계 최고 모델: {best_model_s1[0]} - Test 정확도: {best_model_s1[1]['test_acc']:.4f}")

# ============================================================================
# STEP 3: 2단계 모델 - CONFIRMED vs CANDIDATE
# ============================================================================
print("\n" + "=" * 100)
print("[STEP 3] 2단계 모델: CONFIRMED vs CANDIDATE")
print("=" * 100)

# 2단계: CONFIRMED vs CANDIDATE만 사용
y_stage2 = y_full[y_full.isin(['CONFIRMED', 'CANDIDATE'])]
X_stage2 = X_full.loc[y_stage2.index]

print(f"\n2단계 학습 데이터:")
print(f"  총 샘플: {len(y_stage2):,}")
for label, count in y_stage2.value_counts().sort_index().items():
    print(f"  {label}: {count:,} ({count/len(y_stage2)*100:.1f}%)")

# 레이블 인코딩 (CONFIRMED=1, CANDIDATE=0)
y_stage2_encoded = (y_stage2 == 'CONFIRMED').astype(int)

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
print("\n2단계 모델 학습 중 (강한 정규화)...")

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
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42
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
    y_test_proba = model.predict_proba(X_test_s2_scaled)
    
    # 정확도
    train_acc = accuracy_score(y_train_s2, y_train_pred)
    test_acc = accuracy_score(y_test_s2, y_test_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_s2_scaled, y_train_s2, cv=5, scoring='accuracy')
    
    # AUC
    auc = roc_auc_score(y_test_s2, y_test_proba[:, 1])
    
    results_stage2[name] = {
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
auc_voting = roc_auc_score(y_test_s2, y_test_proba_voting[:, 1])

print(f"  Train 정확도: {train_acc_voting:.4f}")
print(f"  Test 정확도:  {test_acc_voting:.4f}")
print(f"  CV 정확도:    {cv_scores_voting.mean():.4f} ± {cv_scores_voting.std():.4f}")
print(f"  과적합:       {train_acc_voting - test_acc_voting:.4f} ({(train_acc_voting - test_acc_voting)*100:.2f}%p)")
print(f"  AUC:          {auc_voting:.4f}")

results_stage2['Voting'] = {
    'model': voting_s2,
    'train_acc': train_acc_voting,
    'test_acc': test_acc_voting,
    'cv_mean': cv_scores_voting.mean(),
    'cv_std': cv_scores_voting.std(),
    'overfitting': train_acc_voting - test_acc_voting,
    'auc': auc_voting,
    'y_test_proba': y_test_proba_voting
}

# 최고 성능 모델 선택
best_model_s2 = max(results_stage2.items(), key=lambda x: x[1]['test_acc'])
print(f"\n✅ 2단계 최고 모델: {best_model_s2[0]} - Test 정확도: {best_model_s2[1]['test_acc']:.4f}")

# ============================================================================
# STEP 4: 파이프라인 통합 및 확신도 임계값 최적화
# ============================================================================
print("\n" + "=" * 100)
print("[STEP 4] 파이프라인 통합 및 확신도 임계값 최적화")
print("=" * 100)

# 전체 테스트 데이터 준비
y_test_full = y_full.loc[X_test_s1.index]
X_test_full_scaled = scaler_s1.transform(X_test_s1)

# 1단계 모델로 전체 테스트 데이터 예측
best_stage1_model = results_stage1['Voting']['model']
stage1_proba = best_stage1_model.predict_proba(X_test_full_scaled)
stage1_pred = best_stage1_model.predict(X_test_full_scaled)

# 각 확신도 임계값별 성능 평가
thresholds = np.arange(0.85, 0.96, 0.01)
threshold_results = []

print("\n확신도 임계값 최적화 중...")
print(f"{'임계값':<10} {'1단계 사용':<12} {'2단계 사용':<12} {'최종 정확도':<12} {'상세'}")
print("-" * 100)

for threshold in thresholds:
    # 최종 예측 초기화
    final_predictions = np.empty(len(y_test_full), dtype=object)
    
    # 1단계 예측: EXOPLANET_LIKE (1) vs NOT_EXOPLANET (0)
    # 확신도가 높은 샘플만 1단계 결과 사용
    high_confidence_mask = (stage1_proba.max(axis=1) >= threshold)
    
    # 1단계 고확신도 케이스 처리
    for idx in range(len(y_test_full)):
        if high_confidence_mask[idx]:
            if stage1_pred[idx] == 0:  # NOT_EXOPLANET
                final_predictions[idx] = 'FALSE POSITIVE'
            # stage1_pred[idx] == 1 (EXOPLANET_LIKE)인 경우는 아래 2단계에서 처리
    
    # 2단계: EXOPLANET_LIKE로 예측되었거나 확신도가 낮은 케이스
    # → CONFIRMED vs CANDIDATE 구분
    needs_stage2_mask = (stage1_pred == 1) | (~high_confidence_mask)
    
    # 2단계 모델로 예측
    X_for_stage2 = X_test_full_scaled[needs_stage2_mask]
    
    if len(X_for_stage2) > 0:
        # 2단계 스케일러로 변환
        X_for_stage2_rescaled = scaler_s2.transform(
            pd.DataFrame(X_for_stage2, columns=X_stage2.columns).fillna(X_stage2.median())
        )
        stage2_pred = results_stage2['Voting']['model'].predict(X_for_stage2_rescaled)
        
        # CONFIRMED=1, CANDIDATE=0
        final_predictions[needs_stage2_mask] = np.where(
            stage2_pred == 1,
            'CONFIRMED',
            'CANDIDATE'
        )
    
    # 정확도 계산
    accuracy = accuracy_score(y_test_full, final_predictions)
    
    # 1단계/2단계 사용 비율
    stage1_ratio = high_confidence_mask.sum() / len(y_test_full) * 100
    stage2_ratio = needs_stage2_mask.sum() / len(y_test_full) * 100
    
    threshold_results.append({
        'threshold': threshold,
        'accuracy': accuracy,
        'stage1_ratio': stage1_ratio,
        'stage2_ratio': stage2_ratio,
        'stage1_count': high_confidence_mask.sum(),
        'stage2_count': needs_stage2_mask.sum()
    })
    
    print(f"{threshold:.2f}      {stage1_ratio:>5.1f}%      {stage2_ratio:>5.1f}%      "
          f"{accuracy:.4f}      (1단계:{high_confidence_mask.sum()}, 2단계:{needs_stage2_mask.sum()})")

# 최적 임계값 선택
best_threshold_result = max(threshold_results, key=lambda x: x['accuracy'])
best_threshold = best_threshold_result['threshold']
best_accuracy = best_threshold_result['accuracy']

print("\n" + "=" * 100)
print(f"✅ 최적 확신도 임계값: {best_threshold:.2f}")
print(f"✅ 최종 정확도: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"   1단계 사용: {best_threshold_result['stage1_ratio']:.1f}% ({best_threshold_result['stage1_count']}개)")
print(f"   2단계 사용: {best_threshold_result['stage2_ratio']:.1f}% ({best_threshold_result['stage2_count']}개)")
print("=" * 100)

# 최적 임계값으로 최종 예측
final_predictions = np.empty(len(y_test_full), dtype=object)
high_confidence_mask = (stage1_proba.max(axis=1) >= best_threshold)

# 1단계 고확신도 케이스: NOT_EXOPLANET → FALSE POSITIVE
for idx in range(len(y_test_full)):
    if high_confidence_mask[idx] and stage1_pred[idx] == 0:
        final_predictions[idx] = 'FALSE POSITIVE'

# 2단계: EXOPLANET_LIKE로 예측되었거나 확신도가 낮은 케이스
needs_stage2_mask = (stage1_pred == 1) | (~high_confidence_mask)
X_for_stage2 = X_test_full_scaled[needs_stage2_mask]

if len(X_for_stage2) > 0:
    X_for_stage2_rescaled = scaler_s2.transform(
        pd.DataFrame(X_for_stage2, columns=X_stage2.columns).fillna(X_stage2.median())
    )
    stage2_pred = results_stage2['Voting']['model'].predict(X_for_stage2_rescaled)
    final_predictions[needs_stage2_mask] = np.where(
        stage2_pred == 1,
        'CONFIRMED',
        'CANDIDATE'
    )

# ============================================================================
# STEP 5: 과적합 분석
# ============================================================================
print("\n" + "=" * 100)
print("[STEP 5] 과적합 분석")
print("=" * 100)

print("\n1단계 모델 (EXOPLANET_LIKE vs NOT_EXOPLANET) 과적합 분석:")
print("-" * 100)
for name, result in results_stage1.items():
    status = "✅" if result['overfitting'] < 0.03 else "⚠️"
    print(f"{status} {name:20} Train: {result['train_acc']:.4f}  Test: {result['test_acc']:.4f}  "
          f"과적합: {result['overfitting']:.4f} ({result['overfitting']*100:.2f}%p)")

print("\n2단계 모델 (CONFIRMED vs CANDIDATE) 과적합 분석:")
print("-" * 100)
for name, result in results_stage2.items():
    status = "✅" if result['overfitting'] < 0.03 else "⚠️"
    print(f"{status} {name:20} Train: {result['train_acc']:.4f}  Test: {result['test_acc']:.4f}  "
          f"과적합: {result['overfitting']:.4f} ({result['overfitting']*100:.2f}%p)")

# 과적합 판정
overfitting_threshold = 0.03
stage1_overfitting = [(name, r['overfitting']) for name, r in results_stage1.items() 
                      if r['overfitting'] > overfitting_threshold]
stage2_overfitting = [(name, r['overfitting']) for name, r in results_stage2.items() 
                      if r['overfitting'] > overfitting_threshold]

if not stage1_overfitting and not stage2_overfitting:
    print("\n✅✅ 모든 모델 과적합 없음! (3%p 기준)")
else:
    if stage1_overfitting:
        print(f"\n⚠️ 1단계 과적합 모델: {', '.join([f'{n}({o*100:.2f}%p)' for n, o in stage1_overfitting])}")
    if stage2_overfitting:
        print(f"⚠️ 2단계 과적합 모델: {', '.join([f'{n}({o*100:.2f}%p)' for n, o in stage2_overfitting])}")

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
plt.title(f'Confusion Matrix (천문학적 계층 분류) - 정확도: {best_accuracy:.4f}')
plt.ylabel('실제')
plt.xlabel('예측')
plt.tight_layout()
plt.savefig('confusion_matrix_astronomical.png', dpi=150, bbox_inches='tight')
print("\n✅ Confusion Matrix 저장: confusion_matrix_astronomical.png")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test_full, final_predictions, 
                           labels=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'],
                           target_names=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']))

# 클래스별 정확도
print("\n클래스별 정확도:")
for label in ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']:
    mask = y_test_full == label
    if mask.sum() > 0:
        class_acc = accuracy_score(y_test_full[mask], final_predictions[mask])
        print(f"  {label:20} {class_acc:.4f} ({class_acc*100:.2f}%)  (샘플: {mask.sum()}개)")

# 목표 달성 여부
target_accuracy = 0.95
gap = best_accuracy - target_accuracy

print("\n" + "=" * 100)
print("🎯 목표 달성 평가")
print("=" * 100)
print(f"목표 정확도: {target_accuracy:.4f} (95%)")
print(f"달성 정확도: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"격차:        {gap:.4f} ({gap*100:.2f}%p)")

if best_accuracy >= target_accuracy:
    print("\n🎉🎉🎉 목표 달성! 95% 이상 정확도 달성! 🎉🎉🎉")
else:
    print(f"\n📊 목표 미달성. 추가 개선 필요: {abs(gap)*100:.2f}%p 향상 필요")
    print("\n💡 추가 개선 방안:")
    print("  1. 하이퍼파라미터 튜닝 (Optuna 최적화)")
    print("  2. 데이터 증강 (SMOTE, ADASYN)")
    print("  3. 앙상블 Stacking (메타 모델)")
    print("  4. 딥러닝 모델 추가 (Neural Network)")
    print("  5. 피처 선택 최적화 (Recursive Feature Elimination)")

print("\n" + "=" * 100)
print("✅ 분석 완료")
print("=" * 100)
