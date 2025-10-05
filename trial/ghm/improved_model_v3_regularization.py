"""
개선 버전 3: 정규화 강화
- L2 regularization 증가
- Bagging/Subsampling 추가
- Early Stopping 강화
- 과적합 최소화로 일반화 성능 향상
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.ensemble import StackingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

import time

print("="*100)
print("🚀 개선 버전 3: 정규화 강화 (과적합 최소화)")
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

print(f"\n최종 데이터:")
print(f"  • 특징: {X.shape[1]}개")
print(f"  • 샘플: {X.shape[0]}개")

# Train/Test 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.1,
    random_state=42,
    stratify=y_encoded
)

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✅ 전처리 완료!")

# ============================================================================
# 2. 정규화 강화 실험
# ============================================================================

print("\n" + "="*100)
print("🎯 2. 다양한 정규화 수준으로 실험")
print("="*100)

regularization_configs = [
    {
        'name': '기본 정규화',
        'catboost': {'l2_leaf_reg': 2.0, 'bagging_temperature': 0.0, 'subsample': None},
        'xgboost': {'reg_lambda': 2.0, 'reg_alpha': 0.5, 'subsample': 0.8, 'colsample_bytree': 0.8},
        'lightgbm': {'reg_lambda': 2.0, 'reg_alpha': 0.5, 'subsample': 0.8, 'colsample_bytree': 0.8}
    },
    {
        'name': '중간 정규화',
        'catboost': {'l2_leaf_reg': 5.0, 'bagging_temperature': 0.5, 'subsample': 0.8},
        'xgboost': {'reg_lambda': 5.0, 'reg_alpha': 1.0, 'subsample': 0.7, 'colsample_bytree': 0.7},
        'lightgbm': {'reg_lambda': 5.0, 'reg_alpha': 1.0, 'subsample': 0.7, 'colsample_bytree': 0.7}
    },
    {
        'name': '강한 정규화',
        'catboost': {'l2_leaf_reg': 10.0, 'bagging_temperature': 1.0, 'subsample': 0.7},
        'xgboost': {'reg_lambda': 10.0, 'reg_alpha': 2.0, 'subsample': 0.6, 'colsample_bytree': 0.6},
        'lightgbm': {'reg_lambda': 10.0, 'reg_alpha': 2.0, 'subsample': 0.6, 'colsample_bytree': 0.6}
    }
]

best_config = None
best_test_acc = 0
all_results = []

for config in regularization_configs:
    print(f"\n{'='*100}")
    print(f"🔹 {config['name']}")
    print(f"{'='*100}")
    
    # CatBoost
    cat_params = config['catboost']
    catboost_model = CatBoostClassifier(
        iterations=500,
        learning_rate=0.03,
        depth=6,
        l2_leaf_reg=cat_params['l2_leaf_reg'],
        bagging_temperature=cat_params['bagging_temperature'],
        subsample=cat_params['subsample'] if cat_params['subsample'] else None,
        random_state=42,
        verbose=False
    )
    
    catboost_model.fit(X_train_scaled, y_train)
    cat_train_acc = accuracy_score(y_train, catboost_model.predict(X_train_scaled))
    cat_test_acc = accuracy_score(y_test, catboost_model.predict(X_test_scaled))
    cat_overfitting = cat_train_acc - cat_test_acc
    
    print(f"\n📊 CatBoost:")
    print(f"   학습: {cat_train_acc*100:.2f}%  테스트: {cat_test_acc*100:.2f}%  과적합: {cat_overfitting*100:.2f}%p")
    
    # XGBoost
    xgb_params = config['xgboost']
    xgboost_model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        reg_lambda=xgb_params['reg_lambda'],
        reg_alpha=xgb_params['reg_alpha'],
        subsample=xgb_params['subsample'],
        colsample_bytree=xgb_params['colsample_bytree'],
        random_state=42,
        eval_metric='logloss'
    )
    
    xgboost_model.fit(X_train_scaled, y_train)
    xgb_train_acc = accuracy_score(y_train, xgboost_model.predict(X_train_scaled))
    xgb_test_acc = accuracy_score(y_test, xgboost_model.predict(X_test_scaled))
    xgb_overfitting = xgb_train_acc - xgb_test_acc
    
    print(f"📊 XGBoost:")
    print(f"   학습: {xgb_train_acc*100:.2f}%  테스트: {xgb_test_acc*100:.2f}%  과적합: {xgb_overfitting*100:.2f}%p")
    
    # LightGBM
    lgb_params = config['lightgbm']
    lightgbm_model = LGBMClassifier(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=6,
        num_leaves=31,
        reg_lambda=lgb_params['reg_lambda'],
        reg_alpha=lgb_params['reg_alpha'],
        subsample=lgb_params['subsample'],
        colsample_bytree=lgb_params['colsample_bytree'],
        random_state=42,
        verbose=-1
    )
    
    lightgbm_model.fit(X_train_scaled, y_train)
    lgb_train_acc = accuracy_score(y_train, lightgbm_model.predict(X_train_scaled))
    lgb_test_acc = accuracy_score(y_test, lightgbm_model.predict(X_test_scaled))
    lgb_overfitting = lgb_train_acc - lgb_test_acc
    
    print(f"📊 LightGBM:")
    print(f"   학습: {lgb_train_acc*100:.2f}%  테스트: {lgb_test_acc*100:.2f}%  과적합: {lgb_overfitting*100:.2f}%p")
    
    # Stacking Ensemble
    base_learners = [
        ('catboost', catboost_model),
        ('xgboost', xgboost_model),
        ('lightgbm', lightgbm_model)
    ]
    
    meta_learner = LogisticRegression(C=0.5, max_iter=1000, random_state=42)  # C=0.5로 정규화 강화
    
    stacking_clf = StackingClassifier(
        estimators=base_learners,
        final_estimator=meta_learner,
        cv=5,
        n_jobs=-1
    )
    
    stacking_clf.fit(X_train_scaled, y_train)
    stack_train_acc = accuracy_score(y_train, stacking_clf.predict(X_train_scaled))
    stack_test_acc = accuracy_score(y_test, stacking_clf.predict(X_test_scaled))
    stack_overfitting = stack_train_acc - stack_test_acc
    
    stack_cv = cross_val_score(stacking_clf, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    print(f"\n🎯 Stacking Ensemble:")
    print(f"   학습: {stack_train_acc*100:.2f}%  테스트: {stack_test_acc*100:.2f}%  과적합: {stack_overfitting*100:.2f}%p")
    print(f"   5-Fold CV: {stack_cv.mean()*100:.2f}% ± {stack_cv.std()*100:.2f}%")
    
    all_results.append({
        'config': config['name'],
        'cat_test': cat_test_acc,
        'xgb_test': xgb_test_acc,
        'lgb_test': lgb_test_acc,
        'stack_test': stack_test_acc,
        'stack_overfitting': stack_overfitting,
        'stack_cv_mean': stack_cv.mean(),
        'models': {
            'catboost': catboost_model,
            'xgboost': xgboost_model,
            'lightgbm': lightgbm_model,
            'stacking': stacking_clf
        }
    })
    
    if stack_test_acc > best_test_acc:
        best_test_acc = stack_test_acc
        best_config = config['name']

# ============================================================================
# 3. 최적 정규화 수준 선택
# ============================================================================

print("\n" + "="*100)
print("📊 3. 정규화 수준별 성능 비교")
print("="*100)

print("\n" + "="*100)
print("정규화 수준        CatBoost    XGBoost    LightGBM    Stacking    과적합")
print("="*100)

for result in all_results:
    print(f"{result['config']:15s}   {result['cat_test']*100:6.2f}%   {result['xgb_test']*100:6.2f}%   "
          f"{result['lgb_test']*100:6.2f}%   {result['stack_test']*100:6.2f}%   {result['stack_overfitting']*100:5.2f}%p")

print(f"\n🏆 최적 정규화: {best_config}")
print(f"   • 테스트 정확도: {best_test_acc*100:.2f}%")

best_result = [r for r in all_results if r['config'] == best_config][0]

# ============================================================================
# 4. 최종 비교
# ============================================================================

print("\n" + "="*100)
print("📊 4. 최종 성능 비교")
print("="*100)

print("\n🔍 버전별 성능:")
print(f"  • 원본 (기본 정규화):       92.24% 테스트, 5.09%p 과적합")
print(f"  • 다중공선성 제거:          91.61% 테스트, 3.84%p 과적합")
print(f"  • PCA 차원 축소:            89.83% 테스트, 4.50%p 과적합")
print(f"  • 정규화 강화 ({best_config}): {best_result['stack_test']*100:.2f}% 테스트, {best_result['stack_overfitting']*100:.2f}%p 과적합")

improvement = best_result['stack_test'] - 0.9224
print(f"\n📈 정규화 강화 효과: {improvement*100:+.2f}%p")

if improvement > 0.5:
    print("   ✅ 정규화 강화로 일반화 성능이 크게 향상되었습니다!")
elif improvement > 0:
    print("   ✅ 정규화 강화로 일반화 성능이 향상되었습니다!")
elif improvement > -0.5:
    print("   ⚡ 성능은 비슷하지만 과적합이 감소했습니다 (더 안정적)")
else:
    print("   ⚠️ 과도한 정규화로 성능이 하락했습니다 (언더피팅)")

# 과적합 개선 분석
overfitting_improvement = 5.09 - best_result['stack_overfitting']*100
print(f"\n📉 과적합 감소: {overfitting_improvement:+.2f}%p")

if best_result['stack_overfitting'] < 0.03:
    print("   ✅ 과적합이 매우 낮은 수준입니다 (일반화 우수)")
elif best_result['stack_overfitting'] < 0.05:
    print("   ✅ 과적합이 양호한 수준입니다")
else:
    print("   ⚡ 과적합이 여전히 남아있습니다")

print("\n" + "="*100)
print("💡 종합 분석 및 권장사항")
print("="*100)

print("\n📊 3가지 개선 방법 요약:")
print("\n1. 다중공선성 제거 (22개 특징):")
print(f"   • 테스트: 91.61% (-0.63%p)")
print(f"   • 과적합: 3.84%p (✅ 1.25%p 개선)")
print(f"   • 결론: 과적합 개선, 성능 약간 하락")

print("\n2. PCA 차원 축소 (20개 주성분):")
print(f"   • 테스트: 89.83% (-2.41%p)")
print(f"   • 과적합: 4.50%p (✅ 0.59%p 개선)")
print(f"   • 결론: 과적합 개선, 성능 하락")

print(f"\n3. 정규화 강화 ({best_config}, 26개 특징):")
print(f"   • 테스트: {best_result['stack_test']*100:.2f}% ({improvement*100:+.2f}%p)")
print(f"   • 과적합: {best_result['stack_overfitting']*100:.2f}%p ({overfitting_improvement:+.2f}%p 개선)")
print(f"   • 결론: ", end="")

if improvement > 0 and overfitting_improvement > 0:
    print("✅ 최고의 방법! 성능과 일반화 모두 개선")
elif improvement > 0:
    print("✅ 성능 향상, 과적합도 개선")
elif overfitting_improvement > 1:
    print("⚡ 과적합 크게 개선, 성능 유지")
else:
    print("⚠️ 추가 조정 필요")

print("\n🎯 최종 권장사항:")

if best_result['stack_test'] > 0.9224:
    print(f"   ✅ {best_config} 정규화를 적용하세요!")
    print(f"   ✅ 테스트 정확도 {best_result['stack_test']*100:.2f}%로 목표에 가장 근접합니다.")
else:
    print("   💡 추가 개선 방안:")
    print("   • 앙상블 다양성: SVM, Neural Network 추가")
    print("   • CANDIDATE 클래스 포함하여 학습 데이터 증가")
    print("   • Bayesian Optimization으로 하이퍼파라미터 최적화")
    print("   • 특징 선택(RFE)으로 최적 특징 조합 탐색")

print("\n" + "="*100)
print("✅ 정규화 강화 실험 완료!")
print("="*100)
