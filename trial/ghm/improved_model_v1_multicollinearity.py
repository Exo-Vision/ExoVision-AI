"""
개선 버전 1: 다중공선성 제거
- 중복도 높은 특징 제거 (상관도 >0.95)
- habitable_flux, orbit_stability, planet_density_proxy 제거
- 목표: 일반화 성능 1-2%p 향상
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import StackingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

import time

print("="*100)
print("🚀 개선 버전 1: 다중공선성 제거")
print("="*100)

# ============================================================================
# 1. 데이터 로드 및 전처리
# ============================================================================

print("\n" + "="*100)
print("📂 1. 데이터 로드 및 전처리")
print("="*100)

df = pd.read_csv('datasets/exoplanets.csv')
df_binary = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
print(f"이진 분류 데이터: {df_binary.shape[0]} 샘플")

base_features = [
    'koi_prad', 'koi_teq', 'koi_insol', 'koi_period', 'koi_sma', 'koi_impact',
    'koi_eccen', 'koi_depth', 'koi_duration', 'koi_srad', 'koi_smass',
    'koi_steff', 'koi_slogg', 'koi_smet', 'ra', 'dec'
]

# ============================================================================
# 특징 엔지니어링 (다중공선성 제거)
# ============================================================================

print("\n🔧 특징 엔지니어링 (다중공선성 제거 버전):")

df_fe = df_binary[base_features + ['koi_disposition']].copy()

# 1. 행성-항성 비율 특징
df_fe['planet_star_ratio'] = df_fe['koi_prad'] / (df_fe['koi_srad'] * 109)
print("  ✅ planet_star_ratio: 행성/항성 반지름 비율")

# 2. 궤도 에너지 특징
df_fe['orbital_energy'] = df_fe['koi_smass'] / df_fe['koi_sma']
print("  ✅ orbital_energy: 궤도 에너지 (M/a)")

# 3. 통과 신호 강도
df_fe['transit_signal'] = df_fe['koi_depth'] * df_fe['koi_duration']
print("  ✅ transit_signal: 통과 신호 적분")

# 4. ❌ habitable_flux 제거 (koi_insol과 1.000 상관)
print("  ❌ habitable_flux: 제거됨 (koi_insol과 완전 중복)")

# 5. 항성 밀도 대리 변수
df_fe['stellar_density'] = df_fe['koi_smass'] / (df_fe['koi_srad']**3)
print("  ✅ stellar_density: 항성 밀도 (M/R³)")

# 6. ❌ planet_density_proxy 제거 (koi_prad와 0.995 상관)
print("  ❌ planet_density_proxy: 제거됨 (koi_prad와 99.5% 중복)")

# 7. 로그 스케일 변환 (log_insol 제거)
df_fe['log_period'] = np.log10(df_fe['koi_period'] + 1)
df_fe['log_depth'] = np.log10(df_fe['koi_depth'] + 1)
print("  ✅ log 변환: period, depth (insol 제외)")
print("  ❌ log_insol: 제거됨 (koi_teq, orbital_energy와 높은 상관)")

# 8. ❌ orbit_stability 제거 (koi_impact와 1.000 상관)
print("  ❌ orbit_stability: 제거됨 (koi_impact와 완전 중복)")

# 결측치 제거
df_fe = df_fe.dropna()

print(f"\n📊 제거된 특징: 4개")
print("   • habitable_flux (koi_insol과 1.000 상관)")
print("   • planet_density_proxy (koi_prad와 0.995 상관)")
print("   • log_insol (koi_teq와 0.907 상관)")
print("   • orbit_stability (koi_impact와 1.000 상관)")

X = df_fe.drop('koi_disposition', axis=1)
y = df_fe['koi_disposition']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\n최종 데이터:")
print(f"  • 특징: {X.shape[1]}개 (기존 26개 → {X.shape[1]}개)")
print(f"  • 샘플: {X.shape[0]}개")
print(f"  • 레이블: CONFIRMED={np.sum(y_encoded==1)}, FALSE POSITIVE={np.sum(y_encoded==0)}")

# Train/Test 분할
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

print(f"\n✅ 전처리 완료!")

# ============================================================================
# 2. 개별 모델 학습 (동일한 하이퍼파라미터)
# ============================================================================

print("\n" + "="*100)
print("🎯 2. 개별 모델 학습 (다중공선성 제거 효과 검증)")
print("="*100)

model_results = []

def evaluate_model(name, model, X_train, y_train, X_test, y_test):
    """모델 평가"""
    
    print(f"\n{'='*100}")
    print(f"🔹 {name}")
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
    
    print(f"🔄 5-Fold CV: {cv_mean*100:.2f}% ± {cv_std*100:.2f}%")
    
    model_results.append({
        'name': name,
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'overfitting': overfitting
    })
    
    return model

# CatBoost
print("\n⏳ CatBoost 학습 중...")
catboost_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=2.0,
    random_state=42,
    verbose=False
)
evaluate_model("CatBoost", catboost_model, X_train_scaled, y_train, X_test_scaled, y_test)

# XGBoost
print("\n⏳ XGBoost 학습 중...")
xgboost_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=2.0,
    random_state=42,
    eval_metric='logloss'
)
evaluate_model("XGBoost", xgboost_model, X_train_scaled, y_train, X_test_scaled, y_test)

# LightGBM
print("\n⏳ LightGBM 학습 중...")
lightgbm_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.5,
    reg_lambda=2.0,
    random_state=42,
    verbose=-1
)
evaluate_model("LightGBM", lightgbm_model, X_train_scaled, y_train, X_test_scaled, y_test)

# ============================================================================
# 3. 스태킹 앙상블
# ============================================================================

print("\n" + "="*100)
print("🎯 3. 스태킹 앙상블")
print("="*100)

base_learners = [
    ('catboost', catboost_model),
    ('xgboost', xgboost_model),
    ('lightgbm', lightgbm_model)
]

meta_learner = LogisticRegression(max_iter=1000, random_state=42)

stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1
)

print("\n⏳ Stacking Ensemble 학습 중...")
start_time = time.time()
stacking_clf.fit(X_train_scaled, y_train)
train_time = time.time() - start_time

y_train_pred = stacking_clf.predict(X_train_scaled)
y_test_pred = stacking_clf.predict(X_test_scaled)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)
overfitting = train_acc - test_acc

print(f"\n⏱️ 학습 시간: {train_time:.2f}초")
print(f"📊 학습 정확도: {train_acc*100:.2f}%")
print(f"📊 테스트 정확도: {test_acc*100:.2f}%")
print(f"⚠️ 과적합 정도: {overfitting*100:.2f}%p")

stacking_cv_scores = cross_val_score(stacking_clf, X_train_scaled, y_train, cv=5, scoring='accuracy')
print(f"🔄 5-Fold CV: {stacking_cv_scores.mean()*100:.2f}% ± {stacking_cv_scores.std()*100:.2f}%")

# ============================================================================
# 4. 결과 비교 (이전 버전과 비교)
# ============================================================================

print("\n" + "="*100)
print("📊 4. 성능 비교 (이전 vs 개선 버전)")
print("="*100)

print("\n" + "="*100)
print("모델                     학습 정확도    테스트 정확도    CV 정확도       과적합")
print("="*100)

for result in model_results:
    print(f"{result['name']:20s}   {result['train_acc']*100:6.2f}%      {result['test_acc']*100:6.2f}%      "
          f"{result['cv_mean']*100:6.2f}%±{result['cv_std']*100:4.2f}%   {result['overfitting']*100:5.2f}%p")

print(f"{'Stacking Ensemble':20s}   {train_acc*100:6.2f}%      {test_acc*100:6.2f}%      "
      f"{stacking_cv_scores.mean()*100:6.2f}%±{stacking_cv_scores.std()*100:4.2f}%   {overfitting*100:5.2f}%p")

print("\n" + "="*100)
print("💡 개선 효과 분석")
print("="*100)

print("\n🔍 이전 버전 (26개 특징, 다중공선성 있음):")
print("   • CatBoost: 92.24% 테스트 정확도")
print("   • Stacking: 92.24% 테스트 정확도")
print("   • 과적합: 5.09%p")

print(f"\n✨ 개선 버전 ({X.shape[1]}개 특징, 다중공선성 제거):")
best_result = max(model_results, key=lambda x: x['test_acc'])
print(f"   • {best_result['name']}: {best_result['test_acc']*100:.2f}% 테스트 정확도")
print(f"   • Stacking: {test_acc*100:.2f}% 테스트 정확도")
print(f"   • 과적합: {overfitting*100:.2f}%p")

improvement = test_acc - 0.9224
print(f"\n📈 성능 향상: {improvement*100:+.2f}%p")

if improvement > 0:
    print("   ✅ 다중공선성 제거로 일반화 성능이 향상되었습니다!")
elif improvement > -0.005:
    print("   ⚡ 성능이 비슷하지만 모델이 더 단순해졌습니다 (더 안정적)")
else:
    print("   ⚠️ 성능이 약간 하락했습니다. 제거한 특징에 유용한 정보가 있었을 수 있습니다.")

print("\n" + "="*100)
print("✅ 다중공선성 제거 완료!")
print("="*100)
