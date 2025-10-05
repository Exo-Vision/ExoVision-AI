"""
외계행성 판별 최적화 모델 (95% 목표)
- 특징 엔지니어링
- 하이퍼파라미터 튜닝
- 스태킹 앙상블
- 과적합 방지 강화
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

import time

print("="*100)
print("🚀 외계행성 판별 고급 최적화 모델 (목표: 95%)")
print("="*100)

# ============================================================================
# 1. 데이터 로드 및 고급 전처리
# ============================================================================

print("\n" + "="*100)
print("📂 1. 데이터 로드 및 고급 전처리")
print("="*100)

df = pd.read_csv('datasets/exoplanets.csv')
print(f"원본 데이터: {df.shape}")

# 이진 분류
df_binary = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
print(f"이진 분류 데이터: {df_binary.shape[0]} 샘플")

# 핵심 특징
base_features = [
    'koi_prad', 'koi_teq', 'koi_insol', 'koi_period', 'koi_sma', 'koi_impact',
    'koi_eccen', 'koi_depth', 'koi_duration', 'koi_srad', 'koi_smass',
    'koi_steff', 'koi_slogg', 'koi_smet', 'ra', 'dec'
]

# ============================================================================
# 특징 엔지니어링 (새로운 특징 생성)
# ============================================================================

print("\n🔧 특징 엔지니어링:")

df_fe = df_binary[base_features + ['koi_disposition']].copy()

# 1. 행성-항성 비율 특징
df_fe['planet_star_ratio'] = df_fe['koi_prad'] / (df_fe['koi_srad'] * 109)  # R_earth / R_sun
print("  ✅ planet_star_ratio: 행성/항성 반지름 비율")

# 2. 궤도 에너지 특징
df_fe['orbital_energy'] = df_fe['koi_smass'] / df_fe['koi_sma']  # M/a (에너지 대리 변수)
print("  ✅ orbital_energy: 궤도 에너지 (M/a)")

# 3. 통과 신호 강도
df_fe['transit_signal'] = df_fe['koi_depth'] * df_fe['koi_duration']  # 신호 적분
print("  ✅ transit_signal: 통과 신호 적분 (depth × duration)")

# 4. Habitable Zone 지표
# 지구와 비슷한 입사 플럭스 (0.5~1.5)
df_fe['habitable_flux'] = np.abs(df_fe['koi_insol'] - 1.0)
print("  ✅ habitable_flux: Habitable Zone 지표 (|insol - 1.0|)")

# 5. 항성 밀도 대리 변수
df_fe['stellar_density'] = df_fe['koi_smass'] / (df_fe['koi_srad']**3)
print("  ✅ stellar_density: 항성 밀도 (M/R³)")

# 6. 행성 밀도 대리 변수 (간접적 추정)
df_fe['planet_density_proxy'] = df_fe['koi_prad'] / np.sqrt(df_fe['koi_teq'] + 1)
print("  ✅ planet_density_proxy: 행성 밀도 대리 변수")

# 7. 로그 스케일 변환 (왜도 감소)
df_fe['log_period'] = np.log10(df_fe['koi_period'] + 1)
df_fe['log_depth'] = np.log10(df_fe['koi_depth'] + 1)
df_fe['log_insol'] = np.log10(df_fe['koi_insol'] + 1)
print("  ✅ log 변환: period, depth, insol")

# 8. 궤도 안정성 지표
df_fe['orbit_stability'] = df_fe['koi_impact'] * (1 - df_fe['koi_eccen'])
print("  ✅ orbit_stability: 궤도 안정성 지표")

# 결측치 제거
df_clean = df_fe.dropna()
print(f"\n결측치 제거 후: {df_clean.shape[0]} 샘플 ({df_clean.shape[0]/df_binary.shape[0]*100:.1f}%)")

# 특징과 레이블 분리
feature_cols = [col for col in df_clean.columns if col != 'koi_disposition']
X = df_clean[feature_cols]
y = df_clean['koi_disposition']

# 레이블 인코딩
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\n최종 데이터:")
print(f"  • 특징: {X.shape[1]}개 (기본 {len(base_features)}개 + 엔지니어링 {X.shape[1]-len(base_features)}개)")
print(f"  • 샘플: {X.shape[0]}개")
print(f"  • 레이블: CONFIRMED={np.sum(y_encoded==1)}, FALSE POSITIVE={np.sum(y_encoded==0)}")

# Train/Test 분할 (90/10)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.1,
    random_state=42,
    stratify=y_encoded
)

print(f"\n데이터 분할 (90/10):")
print(f"  • 학습: {X_train.shape[0]} 샘플")
print(f"  • 테스트: {X_test.shape[0]} 샘플")

# 특징 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✅ 고급 전처리 완료!")

# ============================================================================
# 2. 최적화된 개별 모델 (하이퍼파라미터 튜닝 적용)
# ============================================================================

print("\n" + "="*100)
print("🎯 2. 최적화된 개별 모델 학습")
print("="*100)

model_results = []

def evaluate_optimized_model(name, model, X_train, y_train, X_test, y_test):
    """최적화된 모델 평가"""
    
    print(f"\n{'='*100}")
    print(f"🔹 {name} (최적화)")
    print(f"{'='*100}")
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    overfitting = train_acc - test_acc
    
    print(f"⏱️ 학습 시간: {train_time:.2f}초")
    print(f"📊 학습 정확도: {train_acc*100:.2f}%")
    print(f"📊 테스트 정확도: {test_acc*100:.2f}%")
    print(f"⚠️ 과적합 정도: {overfitting*100:.2f}%p", end="")
    
    if overfitting > 0.05:
        print(" (⚠️ 과적합 경고!)")
    elif overfitting > 0.02:
        print(" (⚡ 약간 과적합)")
    else:
        print(" (✅ 양호)")
    
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"🔄 교차 검증 (5-fold): {cv_mean*100:.2f}% ± {cv_std*100:.2f}%")
    
    model_results.append({
        'name': name,
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'overfitting': overfitting
    })
    
    return model, test_acc

# ============================================================================
# 최적화된 LightGBM (가장 성능 좋았던 모델)
# ============================================================================

lgbm_optimized = LGBMClassifier(
    n_estimators=500,        # 증가
    max_depth=10,            # 증가
    learning_rate=0.03,      # 감소 (과적합 방지)
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=20,    # 증가 (과적합 방지)
    reg_alpha=0.5,           # L1 증가
    reg_lambda=2.0,          # L2 증가
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgbm_optimized, lgbm_opt_acc = evaluate_optimized_model(
    "LightGBM", lgbm_optimized, X_train_scaled, y_train, X_test_scaled, y_test
)

# ============================================================================
# 최적화된 CatBoost
# ============================================================================

catboost_optimized = CatBoostClassifier(
    iterations=500,
    depth=10,
    learning_rate=0.03,
    l2_leaf_reg=5.0,
    random_strength=1.5,
    bagging_temperature=1.0,
    class_weights=[1, 1],
    random_state=42,
    verbose=0
)

catboost_optimized, catboost_opt_acc = evaluate_optimized_model(
    "CatBoost", catboost_optimized, X_train_scaled, y_train, X_test_scaled, y_test
)

# ============================================================================
# 최적화된 XGBoost
# ============================================================================

xgb_optimized = XGBClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    gamma=0.2,
    reg_alpha=0.5,
    reg_lambda=2.0,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

xgb_optimized, xgb_opt_acc = evaluate_optimized_model(
    "XGBoost", xgb_optimized, X_train_scaled, y_train, X_test_scaled, y_test
)

# ============================================================================
# 최적화된 Random Forest
# ============================================================================

rf_optimized = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

rf_optimized, rf_opt_acc = evaluate_optimized_model(
    "Random Forest", rf_optimized, X_train_scaled, y_train, X_test_scaled, y_test
)

# ============================================================================
# 3. Stacking Ensemble (메타 러닝)
# ============================================================================

print("\n" + "="*100)
print("🏗️ 3. Stacking Ensemble 구성")
print("="*100)

# Base 모델들
base_models = [
    ('lgbm', lgbm_optimized),
    ('catboost', catboost_optimized),
    ('xgb', xgb_optimized),
    ('rf', rf_optimized)
]

# Meta 모델 (Logistic Regression)
meta_model = LogisticRegression(
    C=1.0,
    max_iter=1000,
    random_state=42
)

# Stacking Classifier
stacking_clf = StackingClassifier(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5,
    n_jobs=-1
)

print("\n🏗️ Stacking Ensemble 학습 중...")
print("  • Base 모델: LightGBM, CatBoost, XGBoost, Random Forest")
print("  • Meta 모델: Logistic Regression")
print("  • CV: 5-fold")

start_time = time.time()
stacking_clf.fit(X_train_scaled, y_train)
stacking_train_time = time.time() - start_time

y_train_pred_stacking = stacking_clf.predict(X_train_scaled)
y_test_pred_stacking = stacking_clf.predict(X_test_scaled)

stacking_train_acc = accuracy_score(y_train, y_train_pred_stacking)
stacking_test_acc = accuracy_score(y_test, y_test_pred_stacking)
stacking_overfitting = stacking_train_acc - stacking_test_acc

print(f"\n⏱️ Stacking 학습 시간: {stacking_train_time:.2f}초")
print(f"📊 Stacking 학습 정확도: {stacking_train_acc*100:.2f}%")
print(f"📊 Stacking 테스트 정확도: {stacking_test_acc*100:.2f}%")
print(f"⚠️ Stacking 과적합 정도: {stacking_overfitting*100:.2f}%p", end="")

if stacking_overfitting > 0.05:
    print(" (⚠️ 과적합 경고!)")
elif stacking_overfitting > 0.02:
    print(" (⚡ 약간 과적합)")
else:
    print(" (✅ 양호)")

# Stacking 교차 검증
print("\n🔄 Stacking 교차 검증 (5-fold)...")
stacking_cv_scores = cross_val_score(stacking_clf, X_train_scaled, y_train, cv=5, scoring='accuracy')
stacking_cv_mean = stacking_cv_scores.mean()
stacking_cv_std = stacking_cv_scores.std()

print(f"🔄 Stacking 교차 검증: {stacking_cv_mean*100:.2f}% ± {stacking_cv_std*100:.2f}%")

print(f"\n📊 Stacking 상세 분류 리포트:")
print(classification_report(y_test, y_test_pred_stacking, 
                            target_names=['FALSE POSITIVE', 'CONFIRMED'],
                            digits=4))

# ============================================================================
# 4. Soft Voting Ensemble
# ============================================================================

print("\n" + "="*100)
print("🗳️ 4. Soft Voting Ensemble")
print("="*100)

voting_clf_optimized = VotingClassifier(
    estimators=base_models,
    voting='soft',
    n_jobs=-1
)

print("\n🗳️ Voting Ensemble 학습 중...")
start_time = time.time()
voting_clf_optimized.fit(X_train_scaled, y_train)
voting_train_time = time.time() - start_time

y_train_pred_voting = voting_clf_optimized.predict(X_train_scaled)
y_test_pred_voting = voting_clf_optimized.predict(X_test_scaled)

voting_train_acc = accuracy_score(y_train, y_train_pred_voting)
voting_test_acc = accuracy_score(y_test, y_test_pred_voting)
voting_overfitting = voting_train_acc - voting_test_acc

print(f"\n⏱️ Voting 학습 시간: {voting_train_time:.2f}초")
print(f"📊 Voting 학습 정확도: {voting_train_acc*100:.2f}%")
print(f"📊 Voting 테스트 정확도: {voting_test_acc*100:.2f}%")
print(f"⚠️ Voting 과적합 정도: {voting_overfitting*100:.2f}%p", end="")

if voting_overfitting > 0.05:
    print(" (⚠️ 과적합 경고!)")
elif voting_overfitting > 0.02:
    print(" (⚡ 약간 과적합)")
else:
    print(" (✅ 양호)")

# ============================================================================
# 5. 최종 결과
# ============================================================================

print("\n" + "="*100)
print("📈 5. 최종 결과 요약")
print("="*100)

# 개별 모델 성능
sorted_results = sorted(model_results, key=lambda x: x['test_acc'], reverse=True)

print("\n🏆 개별 모델 테스트 정확도:")
print("-"*100)
for i, result in enumerate(sorted_results, 1):
    print(f"  {i}. {result['name']:<25} : {result['test_acc']*100:>6.2f}% "
          f"(CV: {result['cv_mean']*100:.2f}% ± {result['cv_std']*100:.2f}%)")

print(f"\n🎯 앙상블 모델 테스트 정확도:")
print("-"*100)
print(f"  1. Stacking Ensemble          : {stacking_test_acc*100:>6.2f}% "
      f"(CV: {stacking_cv_mean*100:.2f}% ± {stacking_cv_std*100:.2f}%)")
print(f"  2. Soft Voting Ensemble       : {voting_test_acc*100:>6.2f}%")

# 최고 성능 모델
best_single = sorted_results[0]
best_ensemble_acc = max(stacking_test_acc, voting_test_acc)
best_ensemble_name = "Stacking" if stacking_test_acc > voting_test_acc else "Voting"

print(f"\n🥇 최고 단일 모델:")
print(f"  • 모델: {best_single['name']}")
print(f"  • 테스트 정확도: {best_single['test_acc']*100:.2f}%")

print(f"\n🏆 최고 앙상블 모델:")
print(f"  • 모델: {best_ensemble_name} Ensemble")
print(f"  • 테스트 정확도: {best_ensemble_acc*100:.2f}%")

# 개선 정도
improvement = (best_ensemble_acc - best_single['test_acc']) * 100
print(f"\n📊 앙상블 개선:")
if improvement > 0:
    print(f"  • 앙상블이 최고 단일 모델보다 {improvement:.2f}%p 높음 ⬆️")
elif improvement < 0:
    print(f"  • 앙상블이 최고 단일 모델보다 {abs(improvement):.2f}%p 낮음 ⬇️")
else:
    print(f"  • 앙상블과 최고 단일 모델이 동일 ➡️")

# 목표 달성 확인
target_acc = 0.95
print(f"\n🎯 목표 달성 여부:")
print(f"  • 목표 정확도: {target_acc*100:.0f}%")
print(f"  • 달성 정확도: {best_ensemble_acc*100:.2f}%")

if best_ensemble_acc >= target_acc:
    print(f"  • 결과: ✅ 목표 달성!")
elif best_ensemble_acc >= target_acc - 0.02:
    print(f"  • 결과: 🔥 거의 달성 (목표까지 {(target_acc - best_ensemble_acc)*100:.2f}%p)")
else:
    print(f"  • 결과: ⚡ 목표에 근접 (목표까지 {(target_acc - best_ensemble_acc)*100:.2f}%p)")

# 혼동 행렬
print(f"\n📊 {best_ensemble_name} Ensemble 혼동 행렬:")
best_predictions = y_test_pred_stacking if best_ensemble_name == "Stacking" else y_test_pred_voting
cm = confusion_matrix(y_test, best_predictions)
print("\n실제\\예측   FALSE POSITIVE   CONFIRMED")
print("-"*45)
print(f"FALSE POS   {cm[0,0]:>8}          {cm[0,1]:>8}")
print(f"CONFIRMED   {cm[1,0]:>8}          {cm[1,1]:>8}")

precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n📊 주요 메트릭:")
print(f"  • Accuracy (정확도): {best_ensemble_acc*100:.2f}%")
print(f"  • Precision (정밀도): {precision*100:.2f}%")
print(f"  • Recall (재현율): {recall*100:.2f}%")
print(f"  • F1-Score: {f1*100:.2f}%")

print("\n" + "="*100)
print("🎉 최적화된 외계행성 판별 모델 완성!")
print("="*100)
