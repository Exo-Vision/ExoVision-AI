"""
3-클래스 분류 모델
- CONFIRMED, FALSE POSITIVE, CANDIDATE 모두 정답 레이블로 사용
- 전체 21,271개 데이터 활용
- 강한 정규화 + 앙상블 다양성
- 목표: 3-클래스 분류 정확도 95%
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

import time

print("="*100)
print("🚀 3-클래스 분류 모델 (CONFIRMED / FALSE POSITIVE / CANDIDATE)")
print("="*100)

# ============================================================================
# 1. 데이터 로드 및 전처리
# ============================================================================

print("\n" + "="*100)
print("📂 1. 데이터 로드 (전체 3-클래스)")
print("="*100)

df = pd.read_csv('datasets/exoplanets.csv')
print(f"전체 데이터: {df.shape[0]} 샘플")

# 클래스 분포 확인
print("\n📊 클래스 분포:")
class_counts = df['koi_disposition'].value_counts()
for label, count in class_counts.items():
    print(f"  • {label:20s}: {count:5d}개 ({count/len(df)*100:5.1f}%)")

base_features = [
    'koi_prad', 'koi_teq', 'koi_insol', 'koi_period', 'koi_sma', 'koi_impact',
    'koi_eccen', 'koi_depth', 'koi_duration', 'koi_srad', 'koi_smass',
    'koi_steff', 'koi_slogg', 'koi_smet', 'ra', 'dec'
]

# ============================================================================
# 특징 엔지니어링
# ============================================================================

print("\n🔧 특징 엔지니어링:")

df_fe = df[base_features + ['koi_disposition']].copy()

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

# 결측치 제거
df_clean = df_fe.dropna()

print(f"\n결측치 제거 후: {df_clean.shape[0]} 샘플 ({df_clean.shape[0]/df.shape[0]*100:.1f}%)")

print("\n📊 최종 클래스 분포:")
final_class_counts = df_clean['koi_disposition'].value_counts()
for label, count in final_class_counts.items():
    print(f"  • {label:20s}: {count:5d}개 ({count/len(df_clean)*100:5.1f}%)")

# 특징과 레이블 분리
X = df_clean.drop('koi_disposition', axis=1)
y = df_clean['koi_disposition']

# 레이블 인코딩
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\n최종 데이터:")
print(f"  • 특징: {X.shape[1]}개")
print(f"  • 샘플: {X.shape[0]}개")
print(f"\n레이블 매핑:")
for i, label in enumerate(le.classes_):
    count = np.sum(y_encoded == i)
    print(f"  • class {i} = {label:20s}: {count:5d}개")

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

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✅ 전처리 완료!")

# ============================================================================
# 2. 개별 모델 학습 (3-클래스 분류)
# ============================================================================

print("\n" + "="*100)
print("🎯 2. 개별 모델 학습 (3-클래스 분류)")
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
        print(f"🔄 5-Fold CV: 생략 (시간 절약)")
    
    model_results.append({
        'name': name,
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'overfitting': overfitting,
        'predictions': y_test_pred
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
    l2_leaf_reg=10.0,
    bagging_temperature=1.0,
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
    reg_lambda=10.0,
    reg_alpha=2.0,
    subsample=0.6,
    colsample_bytree=0.6,
    random_state=42,
    eval_metric='mlogloss',
    n_jobs=-1
)
evaluate_model("XGBoost", xgboost_model, X_train_scaled, y_train, X_test_scaled, y_test)

# LightGBM
print("\n⏳ LightGBM 학습 중...")
lightgbm_model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=31,
    reg_lambda=10.0,
    reg_alpha=2.0,
    subsample=0.6,
    colsample_bytree=0.6,
    random_state=42,
    verbose=-1,
    n_jobs=-1
)
evaluate_model("LightGBM", lightgbm_model, X_train_scaled, y_train, X_test_scaled, y_test)

# ============================================================================
# 비트리 기반 모델
# ============================================================================

print("\n" + "-"*100)
print("🔬 비트리 기반 모델")
print("-"*100)

# SVM (RBF)
print("\n⏳ SVM (RBF) 학습 중...")
svm_rbf_model = SVC(
    C=1.0,
    kernel='rbf',
    gamma='scale',
    probability=True,
    random_state=42
)
evaluate_model("SVM (RBF)", svm_rbf_model, X_train_scaled, y_train, X_test_scaled, y_test, use_cv=False)

# Neural Network
print("\n⏳ Neural Network 학습 중...")
mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64, 32),
    activation='relu',
    solver='adam',
    alpha=0.01,
    batch_size=128,
    learning_rate_init=0.001,
    max_iter=500,
    early_stopping=True,
    validation_fraction=0.1,
    n_iter_no_change=20,
    random_state=42
)
evaluate_model("Neural Network", mlp_model, X_train_scaled, y_train, X_test_scaled, y_test, use_cv=False)

# ============================================================================
# 3. 앙상블
# ============================================================================

print("\n" + "="*100)
print("🎯 3. 앙상블")
print("="*100)

# 상위 모델 선택
sorted_results = sorted(model_results, key=lambda x: x['test_acc'], reverse=True)

print("\n📊 개별 모델 성능 순위:")
print("-"*100)
print("순위  모델                     테스트 정확도    CV 정확도       과적합")
print("-"*100)
for i, result in enumerate(sorted_results, 1):
    print(f"{i:2d}.  {result['name']:22s}  {result['test_acc']*100:6.2f}%      "
          f"{result['cv_mean']*100:6.2f}%±{result['cv_std']*100:4.2f}%   {result['overfitting']*100:5.2f}%p")

# Top 5 선택
top_n = min(5, len(model_results))
top_models = sorted_results[:top_n]

print(f"\n🏆 상위 {top_n}개 모델 선택:")
for i, result in enumerate(top_models, 1):
    print(f"  {i}. {result['name']:22s}  {result['test_acc']*100:.2f}%")

# Soft Voting
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

# Stacking
print("\n" + "-"*100)
print("🏗️ Stacking Ensemble")
print("-"*100)

base_learners = [(result['name'], result['model']) for result in top_models]

meta_learner = LogisticRegression(
    C=0.5,
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
print("📊 4. 최종 결과 (3-클래스 분류)")
print("="*100)

print("\n" + "="*100)
print("모델                     학습 정확도    테스트 정확도    과적합      CV 정확도")
print("="*100)

# 개별 모델 (상위 5개)
for result in top_models:
    print(f"{result['name']:22s}  {result['train_acc']*100:6.2f}%      {result['test_acc']*100:6.2f}%      "
          f"{result['overfitting']*100:5.2f}%p    {result['cv_mean']*100:6.2f}%±{result['cv_std']*100:4.2f}%")

print("-"*100)

# 앙상블
print(f"{'Soft Voting':22s}  {voting_train_acc*100:6.2f}%      {voting_test_acc*100:6.2f}%      "
      f"{voting_overfitting*100:5.2f}%p    {voting_cv.mean()*100:6.2f}%±{voting_cv.std()*100:4.2f}%")

print(f"{'Stacking':22s}  {stacking_train_acc*100:6.2f}%      {stacking_test_acc*100:6.2f}%      "
      f"{stacking_overfitting*100:5.2f}%p    {stacking_cv.mean()*100:6.2f}%±{stacking_cv.std()*100:4.2f}%")

print("="*100)

# 최고 성능
all_final_results = [
    ('Soft Voting', voting_test_acc, voting_overfitting),
    ('Stacking', stacking_test_acc, stacking_overfitting)
] + [(r['name'], r['test_acc'], r['overfitting']) for r in top_models]

best_model, best_acc, best_overfit = max(all_final_results, key=lambda x: x[1])

print(f"\n🏆 최고 성능 모델: {best_model}")
print(f"   • 테스트 정확도: {best_acc*100:.2f}%")
print(f"   • 과적합: {best_overfit*100:.2f}%p")

# 목표 달성 여부
print("\n" + "="*100)
print("🎯 목표 달성 여부")
print("="*100)

target_acc = 0.95
print(f"\n목표: {target_acc*100:.0f}%")
print(f"달성: {best_acc*100:.2f}%")
print(f"차이: {(best_acc - target_acc)*100:+.2f}%p")

if best_acc >= target_acc:
    print("\n🎉 축하합니다! 95% 목표를 달성했습니다!")
elif best_acc >= 0.90:
    print("\n⚡ 90% 이상 달성! 목표에 근접했습니다!")
else:
    print("\n📊 추가 개선 필요")

# 상세 분류 리포트
print("\n" + "="*100)
print(f"📋 상세 분류 리포트 ({best_model})")
print("="*100)

if best_model == 'Soft Voting':
    y_pred = voting_clf.predict(X_test_scaled)
elif best_model == 'Stacking':
    y_pred = stacking_clf.predict(X_test_scaled)
else:
    best_individual = [r for r in model_results if r['name'] == best_model][0]
    y_pred = best_individual['predictions']

print("\n" + classification_report(y_test, y_pred, target_names=le.classes_))

# 혼동 행렬
print("\n📊 혼동 행렬:")
cm = confusion_matrix(y_test, y_pred)
print("\n실제\\예측      ", end="")
for label in le.classes_:
    print(f"{label[:10]:>12s}", end="")
print()
print("-" * (15 + 12 * len(le.classes_)))

for i, label in enumerate(le.classes_):
    print(f"{label[:12]:12s}  ", end="")
    for j in range(len(le.classes_)):
        print(f"{cm[i,j]:>12d}", end="")
    print()

# 클래스별 정확도
print("\n📊 클래스별 정확도:")
for i, label in enumerate(le.classes_):
    class_acc = cm[i,i] / cm[i].sum() if cm[i].sum() > 0 else 0
    print(f"  • {label:20s}: {class_acc*100:.2f}% ({cm[i,i]}/{cm[i].sum()})")

print("\n" + "="*100)
print("✅ 3-클래스 분류 모델 완료!")
print("="*100)
