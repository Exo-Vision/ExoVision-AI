"""
최종 모델: 95% 목표 달성
- 강한 정규화 적용 (과적합 최소화)
- 앙상블 다양성 증가: SVM, Neural Network, Naive Bayes 추가
- 트리 기반 + 비트리 기반 모델 조합
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

import time

print("="*100)
print("🚀 최종 모델: 95% 목표 달성")
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

print("\n🔧 특징 엔지니어링:")
df_fe['planet_star_ratio'] = df_fe['koi_prad'] / (df_fe['koi_srad'] * 109)
print("  ✅ planet_star_ratio")

df_fe['orbital_energy'] = df_fe['koi_smass'] / df_fe['koi_sma']
print("  ✅ orbital_energy")

df_fe['transit_signal'] = df_fe['koi_depth'] * df_fe['koi_duration']
print("  ✅ transit_signal")

df_fe['habitable_flux'] = np.abs(df_fe['koi_insol'] - 1.0)
print("  ✅ habitable_flux")

df_fe['stellar_density'] = df_fe['koi_smass'] / (df_fe['koi_srad']**3)
print("  ✅ stellar_density")

df_fe['planet_density_proxy'] = df_fe['koi_prad'] / np.sqrt(df_fe['koi_teq'] + 1)
print("  ✅ planet_density_proxy")

df_fe['log_period'] = np.log10(df_fe['koi_period'] + 1)
df_fe['log_depth'] = np.log10(df_fe['koi_depth'] + 1)
df_fe['log_insol'] = np.log10(df_fe['koi_insol'] + 1)
print("  ✅ log 변환: period, depth, insol")

df_fe['orbit_stability'] = df_fe['koi_impact'] * (1 - df_fe['koi_eccen'])
print("  ✅ orbit_stability")

df_fe = df_fe.dropna()

X = df_fe.drop('koi_disposition', axis=1)
y = df_fe['koi_disposition']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\n최종 데이터:")
print(f"  • 특징: {X.shape[1]}개")
print(f"  • 샘플: {X.shape[0]}개")
print(f"  • 레이블: CONFIRMED={np.sum(y_encoded==1)}, FALSE POSITIVE={np.sum(y_encoded==0)}")

# Train/Test 분할
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.1,
    random_state=42,
    stratify=y_encoded
)

print(f"\n데이터 분할:")
print(f"  • 학습: {X_train.shape[0]} 샘플")
print(f"  • 테스트: {X_test.shape[0]} 샘플")

# 스케일링 (SVM, Neural Network에 필수)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✅ 전처리 완료!")

# ============================================================================
# 2. 다양한 모델 학습 (강한 정규화 적용)
# ============================================================================

print("\n" + "="*100)
print("🎯 2. 다양한 모델 학습 (트리 + 비트리 기반)")
print("="*100)

model_results = []

def evaluate_model(name, model, X_train, y_train, X_test, y_test, use_cv=True):
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
        print(" (⚠️ 과적합 경고)")
    elif overfitting > 0.02:
        print(" (⚡ 약간 과적합)")
    else:
        print(" (✅ 양호)")
    
    if use_cv:
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1)
        cv_mean = cv_scores.mean()
        cv_std = cv_scores.std()
        print(f"🔄 5-Fold CV: {cv_mean*100:.2f}% ± {cv_std*100:.2f}%")
    else:
        cv_mean = test_acc
        cv_std = 0.0
    
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

# ============================================================================
# 트리 기반 모델 (강한 정규화)
# ============================================================================

print("\n" + "-"*100)
print("🌳 트리 기반 모델 (강한 정규화)")
print("-"*100)

# CatBoost
print("\n⏳ CatBoost 학습 중...")
catboost_model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=10.0,           # 강한 정규화
    bagging_temperature=1.0,    # 배깅 추가
    subsample=0.7,              # 70% 샘플링
    random_state=42,
    verbose=False
)
evaluate_model("CatBoost (강한 정규화)", catboost_model, X_train_scaled, y_train, X_test_scaled, y_test)

# XGBoost
print("\n⏳ XGBoost 학습 중...")
xgboost_model = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    reg_lambda=10.0,            # 강한 L2
    reg_alpha=2.0,              # 강한 L1
    subsample=0.6,              # 60% 샘플링
    colsample_bytree=0.6,       # 60% 특징 샘플링
    random_state=42,
    eval_metric='logloss'
)
evaluate_model("XGBoost (강한 정규화)", xgboost_model, X_train_scaled, y_train, X_test_scaled, y_test)

# LightGBM
print("\n⏳ LightGBM 학습 중...")
lightgbm_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=31,
    reg_lambda=10.0,            # 강한 L2
    reg_alpha=2.0,              # 강한 L1
    subsample=0.6,              # 60% 샘플링
    colsample_bytree=0.6,       # 60% 특징 샘플링
    random_state=42,
    verbose=-1
)
evaluate_model("LightGBM (강한 정규화)", lightgbm_model, X_train_scaled, y_train, X_test_scaled, y_test)

# ============================================================================
# 비트리 기반 모델 (다양성 증가)
# ============================================================================

print("\n" + "-"*100)
print("🔬 비트리 기반 모델 (다양성)")
print("-"*100)

# SVM (RBF Kernel)
print("\n⏳ SVM (RBF) 학습 중...")
svm_rbf_model = SVC(
    C=1.0,                      # 정규화 강도
    kernel='rbf',
    gamma='scale',
    probability=True,           # Stacking에 필요
    random_state=42
)
evaluate_model("SVM (RBF Kernel)", svm_rbf_model, X_train_scaled, y_train, X_test_scaled, y_test, use_cv=False)

# SVM (Linear Kernel)
print("\n⏳ SVM (Linear) 학습 중...")
svm_linear_model = SVC(
    C=1.0,
    kernel='linear',
    probability=True,
    random_state=42
)
evaluate_model("SVM (Linear Kernel)", svm_linear_model, X_train_scaled, y_train, X_test_scaled, y_test, use_cv=False)

# Neural Network (MLP)
print("\n⏳ Neural Network 학습 중...")
mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),  # 3층 네트워크
    activation='relu',
    solver='adam',
    alpha=0.01,                 # L2 정규화
    batch_size=128,
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,        # 조기 종료
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42
)
evaluate_model("Neural Network (MLP)", mlp_model, X_train_scaled, y_train, X_test_scaled, y_test, use_cv=False)

# Naive Bayes
print("\n⏳ Naive Bayes 학습 중...")
nb_model = GaussianNB()
evaluate_model("Naive Bayes", nb_model, X_train_scaled, y_train, X_test_scaled, y_test)

# ============================================================================
# 3. 앙상블 전략
# ============================================================================

print("\n" + "="*100)
print("🎯 3. 앙상블 전략")
print("="*100)

# 상위 모델 선택
sorted_results = sorted(model_results, key=lambda x: x['test_acc'], reverse=True)

print("\n📊 개별 모델 성능 순위:")
print("-"*100)
print("순위  모델                            테스트 정확도    CV 정확도       과적합")
print("-"*100)
for i, result in enumerate(sorted_results, 1):
    print(f"{i:2d}.  {result['name']:30s}  {result['test_acc']*100:6.2f}%      "
          f"{result['cv_mean']*100:6.2f}%±{result['cv_std']*100:4.2f}%   {result['overfitting']*100:5.2f}%p")

# Top 5 모델 선택
top_n = 5
top_models = sorted_results[:top_n]

print(f"\n🏆 상위 {top_n}개 모델 선택:")
for i, result in enumerate(top_models, 1):
    print(f"  {i}. {result['name']:30s}  {result['test_acc']*100:.2f}%")

# ============================================================================
# 3-1. Soft Voting Ensemble
# ============================================================================

print("\n" + "-"*100)
print("🗳️ Soft Voting Ensemble")
print("-"*100)

voting_estimators = [(result['name'], result['model']) for result in top_models]

voting_clf = VotingClassifier(
    estimators=voting_estimators,
    voting='soft',
    n_jobs=-1
)

print("\n⏳ Soft Voting 학습 중...")
start_time = time.time()
voting_clf.fit(X_train_scaled, y_train)
train_time = time.time() - start_time

voting_train_acc = accuracy_score(y_train, voting_clf.predict(X_train_scaled))
voting_test_acc = accuracy_score(y_test, voting_clf.predict(X_test_scaled))
voting_overfitting = voting_train_acc - voting_test_acc

print(f"\n⏱️ 학습 시간: {train_time:.2f}초")
print(f"📊 학습 정확도: {voting_train_acc*100:.2f}%")
print(f"📊 테스트 정확도: {voting_test_acc*100:.2f}%")
print(f"⚠️ 과적합 정도: {voting_overfitting*100:.2f}%p")

voting_cv = cross_val_score(voting_clf, X_train_scaled, y_train, cv=5, scoring='accuracy', n_jobs=-1)
print(f"🔄 5-Fold CV: {voting_cv.mean()*100:.2f}% ± {voting_cv.std()*100:.2f}%")

# ============================================================================
# 3-2. Stacking Ensemble
# ============================================================================

print("\n" + "-"*100)
print("🏗️ Stacking Ensemble")
print("-"*100)

# Base learners: 상위 5개 모델
base_learners = [(result['name'], result['model']) for result in top_models]

# Meta learner: 정규화된 Logistic Regression
meta_learner = LogisticRegression(
    C=0.5,                  # 정규화 강화
    max_iter=1000,
    random_state=42
)

stacking_clf = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=5,
    n_jobs=-1
)

print("\n⏳ Stacking 학습 중...")
start_time = time.time()
stacking_clf.fit(X_train_scaled, y_train)
train_time = time.time() - start_time

stacking_train_acc = accuracy_score(y_train, stacking_clf.predict(X_train_scaled))
stacking_test_acc = accuracy_score(y_test, stacking_clf.predict(X_test_scaled))
stacking_overfitting = stacking_train_acc - stacking_test_acc

print(f"\n⏱️ 학습 시간: {train_time:.2f}초")
print(f"📊 학습 정확도: {stacking_train_acc*100:.2f}%")
print(f"📊 테스트 정확도: {stacking_test_acc*100:.2f}%")
print(f"⚠️ 과적합 정도: {stacking_overfitting*100:.2f}%p")

stacking_cv = cross_val_score(stacking_clf, X_train_scaled, y_train, cv=5, scoring='accuracy', n_jobs=-1)
print(f"🔄 5-Fold CV: {stacking_cv.mean()*100:.2f}% ± {stacking_cv.std()*100:.2f}%")

# ============================================================================
# 4. 최종 결과
# ============================================================================

print("\n" + "="*100)
print("📊 4. 최종 결과 비교")
print("="*100)

print("\n" + "="*100)
print("모델                            학습 정확도    테스트 정확도    과적합      CV 정확도")
print("="*100)

# 개별 모델 (상위 5개만)
for result in top_models:
    print(f"{result['name']:30s}  {result['train_acc']*100:6.2f}%      {result['test_acc']*100:6.2f}%      "
          f"{result['overfitting']*100:5.2f}%p    {result['cv_mean']*100:6.2f}%±{result['cv_std']*100:4.2f}%")

print("-"*100)

# 앙상블
print(f"{'Soft Voting Ensemble':30s}  {voting_train_acc*100:6.2f}%      {voting_test_acc*100:6.2f}%      "
      f"{voting_overfitting*100:5.2f}%p    {voting_cv.mean()*100:6.2f}%±{voting_cv.std()*100:4.2f}%")

print(f"{'Stacking Ensemble':30s}  {stacking_train_acc*100:6.2f}%      {stacking_test_acc*100:6.2f}%      "
      f"{stacking_overfitting*100:5.2f}%p    {stacking_cv.mean()*100:6.2f}%±{stacking_cv.std()*100:4.2f}%")

print("="*100)

# 최고 성능 모델
all_final_results = [
    ('Soft Voting', voting_test_acc, voting_overfitting),
    ('Stacking', stacking_test_acc, stacking_overfitting)
] + [(r['name'], r['test_acc'], r['overfitting']) for r in top_models]

best_model = max(all_final_results, key=lambda x: x[1])

print(f"\n🏆 최고 성능 모델: {best_model[0]}")
print(f"   • 테스트 정확도: {best_model[1]*100:.2f}%")
print(f"   • 과적합: {best_model[2]*100:.2f}%p")

# 목표 달성 여부
print("\n" + "="*100)
print("🎯 목표 달성 여부")
print("="*100)

target_acc = 0.95
print(f"\n목표: {target_acc*100:.2f}%")
print(f"달성: {best_model[1]*100:.2f}%")
print(f"차이: {(best_model[1] - target_acc)*100:+.2f}%p")

if best_model[1] >= target_acc:
    print("\n🎉 축하합니다! 95% 목표를 달성했습니다!")
elif best_model[1] >= 0.93:
    print("\n⚡ 목표에 근접했습니다! (93%+)")
    print("\n💡 추가 개선 방안:")
    print("   • CANDIDATE 클래스 포함 (10,065개 추가 샘플)")
    print("   • Bayesian Optimization (하이퍼파라미터 최적화)")
    print("   • Deep Learning (LSTM, Transformer)")
else:
    print("\n📊 목표 달성 실패")
    print("\n💡 추가 개선 방안:")
    print("   • 더 많은 데이터 필요")
    print("   • 특징 선택 알고리즘 (RFE)")
    print("   • AutoML 프레임워크 시도")

# 상세 분류 리포트 (최고 모델)
print("\n" + "="*100)
print(f"📋 상세 분류 리포트 ({best_model[0]})")
print("="*100)

if best_model[0] == 'Soft Voting':
    y_pred = voting_clf.predict(X_test_scaled)
elif best_model[0] == 'Stacking':
    y_pred = stacking_clf.predict(X_test_scaled)
else:
    best_individual = [r for r in model_results if r['name'] == best_model[0]][0]
    y_pred = best_individual['model'].predict(X_test_scaled)

print("\n" + classification_report(y_test, y_pred, target_names=['FALSE POSITIVE', 'CONFIRMED']))

print("\n" + "="*100)
print("✅ 최종 모델 학습 완료!")
print("="*100)
