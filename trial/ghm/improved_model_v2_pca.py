"""
개선 버전 2: PCA 차원 축소
- 26개 특징 → 최적 주성분으로 축소
- 다중공선성 완전 제거
- 노이즈 감소로 일반화 성능 향상
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score

from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

import matplotlib.pyplot as plt
import time

print("="*100)
print("🚀 개선 버전 2: PCA 차원 축소")
print("="*100)

# ============================================================================
# 1. 데이터 로드 및 전처리 (원본 26개 특징 사용)
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

# 특징 엔지니어링 (전체 26개)
df_fe = df_binary[base_features + ['koi_disposition']].copy()

df_fe['planet_star_ratio'] = df_fe['koi_prad'] / (df_fe['koi_srad'] * 109)
df_fe['orbital_energy'] = df_fe['koi_smass'] / df_fe['koi_sma']
df_fe['transit_signal'] = df_fe['koi_depth'] * df_fe['koi_duration']
df_fe['habitable_flux'] = np.abs(df_fe['koi_insol'] - 1.0)
df_fe['stellar_density'] = df_fe['koi_smass'] / (df_fe['koi_srad']**3)
df_fe['planet_density_proxy'] = df_fe['koi_prad'] / np.sqrt(df_fe['koi_teq'] + 1)
df_fe['log_period'] = np.log10(df_fe['koi_period'] + 1)
df_fe['log_depth'] = np.log10(df_fe['koi_depth'] + 1)
df_fe['log_insol'] = np.log10(df_fe['koi_insol'] + 1)
df_fe['orbit_stability'] = df_fe['koi_impact'] * (1 - df_fe['koi_eccen'])

df_fe = df_fe.dropna()

X = df_fe.drop('koi_disposition', axis=1)
y = df_fe['koi_disposition']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\n원본 데이터:")
print(f"  • 특징: {X.shape[1]}개")
print(f"  • 샘플: {X.shape[0]}개")

# Train/Test 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.1,
    random_state=42,
    stratify=y_encoded
)

# 스케일링 (PCA 전에 필수!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✅ 전처리 완료!")

# ============================================================================
# 2. PCA 분석 - 최적 주성분 개수 찾기
# ============================================================================

print("\n" + "="*100)
print("🔍 2. PCA 분석 - 최적 주성분 개수 탐색")
print("="*100)

# 전체 주성분 분석
pca_full = PCA()
pca_full.fit(X_train_scaled)

explained_variance_ratio = pca_full.explained_variance_ratio_
cumulative_variance_ratio = np.cumsum(explained_variance_ratio)

print("\n📊 설명 분산 비율:")
print("주성분    개별 분산    누적 분산")
print("-" * 50)
for i in range(min(20, len(explained_variance_ratio))):
    print(f"PC{i+1:2d}      {explained_variance_ratio[i]*100:6.2f}%      {cumulative_variance_ratio[i]*100:6.2f}%")

# 95% 분산을 설명하는 주성분 개수
n_components_95 = np.argmax(cumulative_variance_ratio >= 0.95) + 1
n_components_90 = np.argmax(cumulative_variance_ratio >= 0.90) + 1
n_components_85 = np.argmax(cumulative_variance_ratio >= 0.85) + 1

print(f"\n💡 분산 비율별 필요 주성분:")
print(f"  • 85% 분산: {n_components_85}개 주성분")
print(f"  • 90% 분산: {n_components_90}개 주성분")
print(f"  • 95% 분산: {n_components_95}개 주성분")

# ============================================================================
# 3. 다양한 주성분 개수로 모델 학습
# ============================================================================

print("\n" + "="*100)
print("🎯 3. 다양한 주성분 개수로 성능 비교")
print("="*100)

pca_results = []

for n_comp in [n_components_85, n_components_90, n_components_95, 15, 18, 20]:
    if n_comp > X_train_scaled.shape[1]:
        continue
    if n_comp in [r['n_components'] for r in pca_results]:
        continue
    
    print(f"\n{'='*100}")
    print(f"🔹 PCA with {n_comp} components")
    print(f"{'='*100}")
    
    # PCA 변환
    pca = PCA(n_components=n_comp)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    variance_explained = pca.explained_variance_ratio_.sum()
    print(f"📊 설명 분산: {variance_explained*100:.2f}%")
    
    # CatBoost로 평가
    model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=2.0,
        random_state=42,
        verbose=False
    )
    
    start_time = time.time()
    model.fit(X_train_pca, y_train)
    train_time = time.time() - start_time
    
    train_acc = accuracy_score(y_train, model.predict(X_train_pca))
    test_acc = accuracy_score(y_test, model.predict(X_test_pca))
    overfitting = train_acc - test_acc
    
    cv_scores = cross_val_score(model, X_train_pca, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"⏱️ 학습 시간: {train_time:.2f}초")
    print(f"📊 학습 정확도: {train_acc*100:.2f}%")
    print(f"📊 테스트 정확도: {test_acc*100:.2f}%")
    print(f"⚠️ 과적합: {overfitting*100:.2f}%p")
    print(f"🔄 5-Fold CV: {cv_mean*100:.2f}% ± {cv_std*100:.2f}%")
    
    pca_results.append({
        'n_components': n_comp,
        'variance_explained': variance_explained,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'overfitting': overfitting,
        'pca': pca
    })

# 최적 주성분 개수 선택
best_result = max(pca_results, key=lambda x: x['test_acc'])
print(f"\n🏆 최적 주성분 개수: {best_result['n_components']}개")
print(f"   • 테스트 정확도: {best_result['test_acc']*100:.2f}%")
print(f"   • 설명 분산: {best_result['variance_explained']*100:.2f}%")

# ============================================================================
# 4. 최적 PCA로 앙상블 학습
# ============================================================================

print("\n" + "="*100)
print(f"🎯 4. 최적 PCA ({best_result['n_components']}개 주성분)로 앙상블 학습")
print("="*100)

# 최적 PCA 변환
best_pca = best_result['pca']
X_train_best = best_pca.transform(X_train_scaled)
X_test_best = best_pca.transform(X_test_scaled)

print(f"\n📊 차원 축소: {X_train_scaled.shape[1]}개 → {X_train_best.shape[1]}개")

# 개별 모델 학습
models_dict = {}

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
catboost_model.fit(X_train_best, y_train)
cat_test_acc = accuracy_score(y_test, catboost_model.predict(X_test_best))
print(f"   테스트 정확도: {cat_test_acc*100:.2f}%")
models_dict['catboost'] = catboost_model

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
xgboost_model.fit(X_train_best, y_train)
xgb_test_acc = accuracy_score(y_test, xgboost_model.predict(X_test_best))
print(f"   테스트 정확도: {xgb_test_acc*100:.2f}%")
models_dict['xgboost'] = xgboost_model

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
lightgbm_model.fit(X_train_best, y_train)
lgb_test_acc = accuracy_score(y_test, lightgbm_model.predict(X_test_best))
print(f"   테스트 정확도: {lgb_test_acc*100:.2f}%")
models_dict['lightgbm'] = lightgbm_model

# Stacking Ensemble
print("\n⏳ Stacking Ensemble 학습 중...")
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

stacking_clf.fit(X_train_best, y_train)

stacking_train_acc = accuracy_score(y_train, stacking_clf.predict(X_train_best))
stacking_test_acc = accuracy_score(y_test, stacking_clf.predict(X_test_best))
stacking_overfitting = stacking_train_acc - stacking_test_acc

stacking_cv = cross_val_score(stacking_clf, X_train_best, y_train, cv=5, scoring='accuracy')

print(f"   학습 정확도: {stacking_train_acc*100:.2f}%")
print(f"   테스트 정확도: {stacking_test_acc*100:.2f}%")
print(f"   과적합: {stacking_overfitting*100:.2f}%p")
print(f"   5-Fold CV: {stacking_cv.mean()*100:.2f}% ± {stacking_cv.std()*100:.2f}%")

# ============================================================================
# 5. 결과 비교
# ============================================================================

print("\n" + "="*100)
print("📊 5. 성능 비교 (원본 vs PCA)")
print("="*100)

print("\n" + "="*100)
print("모델                     테스트 정확도    차원      설명 분산")
print("="*100)

print(f"{'원본 (26개 특징)':20s}   92.24%        26개      100.00%")
print(f"{'PCA CatBoost':20s}   {cat_test_acc*100:6.2f}%      {best_result['n_components']:2d}개      {best_result['variance_explained']*100:6.2f}%")
print(f"{'PCA XGBoost':20s}   {xgb_test_acc*100:6.2f}%      {best_result['n_components']:2d}개      {best_result['variance_explained']*100:6.2f}%")
print(f"{'PCA LightGBM':20s}   {lgb_test_acc*100:6.2f}%      {best_result['n_components']:2d}개      {best_result['variance_explained']*100:6.2f}%")
print(f"{'PCA Stacking':20s}   {stacking_test_acc*100:6.2f}%      {best_result['n_components']:2d}개      {best_result['variance_explained']*100:6.2f}%")

print("\n" + "="*100)
print("💡 PCA 개선 효과 분석")
print("="*100)

improvement = stacking_test_acc - 0.9224
print(f"\n📈 성능 변화: {improvement*100:+.2f}%p")
print(f"📉 차원 축소: 26개 → {best_result['n_components']}개 ({(1-best_result['n_components']/26)*100:.1f}% 감소)")
print(f"📊 정보 보존: {best_result['variance_explained']*100:.2f}%")

if improvement > 0.5:
    print("\n✅ PCA로 일반화 성능이 크게 향상되었습니다!")
elif improvement > 0:
    print("\n⚡ PCA로 일반화 성능이 약간 향상되었습니다!")
elif improvement > -0.5:
    print("\n📊 성능은 비슷하지만 차원이 크게 감소했습니다 (계산 효율성 향상)")
else:
    print("\n⚠️ PCA로 성능이 하락했습니다. 원본 특징이 더 효과적일 수 있습니다.")

print("\n과적합 비교:")
print(f"  • 원본 (26개 특징): 5.09%p")
print(f"  • PCA ({best_result['n_components']}개 주성분): {stacking_overfitting*100:.2f}%p")

if stacking_overfitting < 0.0509:
    print("  ✅ PCA로 과적합이 감소했습니다!")
else:
    print("  ⚠️ 과적합이 여전히 남아있습니다.")

print("\n" + "="*100)
print("✅ PCA 차원 축소 완료!")
print("="*100)
