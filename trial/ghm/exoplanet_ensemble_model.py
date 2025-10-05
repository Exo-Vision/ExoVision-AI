"""
외계행성 판별 고성능 앙상블 모델
- 목표: 90% 중반 정확도 달성
- 과적합 방지: 교차 검증, 학습 곡선 분석
- 앙상블: 최고 성능 모델들 조합
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 앙상블 모델들
from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)

# 부스팅 모델들
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

# 기타
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
import time

print("="*100)
print("🚀 외계행성 판별 고성능 앙상블 모델")
print("="*100)

# ============================================================================
# 1. 데이터 로드 및 전처리
# ============================================================================

print("\n" + "="*100)
print("📂 1. 데이터 로드 및 전처리")
print("="*100)

# 데이터 로드
df = pd.read_csv('datasets/exoplanets.csv')
print(f"원본 데이터: {df.shape}")

# 레이블 분포 확인
print(f"\n레이블 분포:")
print(df['koi_disposition'].value_counts())

# 이진 분류: CONFIRMED vs FALSE POSITIVE (가장 명확한 분류)
print("\n🎯 분류 방식: 이진 분류 (CONFIRMED vs FALSE POSITIVE)")
df_binary = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
print(f"이진 분류 데이터: {df_binary.shape[0]} 샘플")

# 외계행성 판별에 중요한 특징 선택
print("\n🔍 외계행성 판별 중요 특징 선택:")

# 핵심 특징 (완성도 높고 물리적으로 중요한 컬럼)
key_features = [
    # 행성 특성
    'koi_prad',       # 행성 반지름 - 크기 구별 (94.5%)
    'koi_teq',        # 평형 온도 - Habitable Zone (90.2%)
    'koi_insol',      # 입사 플럭스 - 에너지 수준 (90.5%)
    
    # 궤도 특성
    'koi_period',     # 궤도 주기 - 궤도 특성 (99.3%)
    'koi_sma',        # 반장축 - 궤도 거리 (86.7%)
    'koi_impact',     # 충격 매개변수 - 궤도 기하학 (88.0%)
    'koi_eccen',      # 궤도 이심률 - 궤도 모양 (99.8%)
    
    # 통과 신호 특성
    'koi_depth',      # 통과 깊이 - 신호 강도 (96.9%)
    'koi_duration',   # 통과 지속시간 - 신호 특성 (99.3%)
    
    # 항성 특성
    'koi_srad',       # 항성 반지름 - 항성 크기 (95.3%)
    'koi_smass',      # 항성 질량 - 항성 질량 (87.2%)
    'koi_steff',      # 유효 온도 - 항성 온도 (92.5%)
    'koi_slogg',      # 표면 중력 - 항성 밀도 (87.5%)
    'koi_smet',       # 금속성 - 행성 형성 (100%)
    
    # 위치 정보
    'ra',             # 적경 (100%)
    'dec',            # 적위 (100%)
]

print(f"선택된 특징: {len(key_features)}개")
for i, feat in enumerate(key_features, 1):
    completeness = df_binary[feat].notna().sum() / len(df_binary) * 100
    print(f"  {i:2d}. {feat:<20} - 완성도 {completeness:5.1f}%")

# 결측치 처리: 완전한 데이터만 사용
df_clean = df_binary[key_features + ['koi_disposition']].dropna()
print(f"\n결측치 제거 후: {df_clean.shape[0]} 샘플 ({df_clean.shape[0]/df_binary.shape[0]*100:.1f}%)")

# 특징과 레이블 분리
X = df_clean[key_features]
y = df_clean['koi_disposition']

# 레이블 인코딩 (CONFIRMED=1, FALSE POSITIVE=0)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

print(f"\n최종 데이터:")
print(f"  • 특징: {X.shape}")
print(f"  • 레이블: CONFIRMED={np.sum(y_encoded==1)}, FALSE POSITIVE={np.sum(y_encoded==0)}")

# Train/Test 분할 (90/10)
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, 
    test_size=0.1,  # 10% 테스트
    random_state=42,
    stratify=y_encoded  # 레이블 비율 유지
)

print(f"\n데이터 분할 (90/10):")
print(f"  • 학습: {X_train.shape[0]} 샘플")
print(f"  • 테스트: {X_test.shape[0]} 샘플")

# 특징 스케일링 (트리 기반 모델에는 필수는 아니지만 일부 모델에 도움)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"\n✅ 전처리 완료!")

# ============================================================================
# 2. 개별 모델 학습 및 평가
# ============================================================================

print("\n" + "="*100)
print("🤖 2. 개별 모델 학습 및 평가 (과적합 검사 포함)")
print("="*100)

# 모델 결과 저장
model_results = []

def evaluate_model_with_overfitting_check(name, model, X_train, y_train, X_test, y_test, use_scaled=False):
    """모델 평가 및 과적합 검사"""
    
    print(f"\n{'='*100}")
    print(f"🔹 {name}")
    print(f"{'='*100}")
    
    start_time = time.time()
    
    # 학습
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    # 예측
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)
    
    # 정확도
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    # 과적합 검사
    overfitting = train_acc - test_acc
    
    print(f"⏱️ 학습 시간: {train_time:.2f}초")
    print(f"📊 학습 정확도: {train_acc*100:.2f}%")
    print(f"📊 테스트 정확도: {test_acc*100:.2f}%")
    print(f"⚠️ 과적합 정도: {overfitting*100:.2f}%p", end="")
    
    if overfitting > 0.05:  # 5% 이상 차이
        print(" (⚠️ 과적합 경고!)")
    elif overfitting > 0.02:  # 2-5% 차이
        print(" (⚡ 약간 과적합)")
    else:
        print(" (✅ 양호)")
    
    # 교차 검증 (5-fold)
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    cv_mean = cv_scores.mean()
    cv_std = cv_scores.std()
    
    print(f"🔄 교차 검증 (5-fold): {cv_mean*100:.2f}% ± {cv_std*100:.2f}%")
    
    # 상세 리포트
    print(f"\n상세 분류 리포트:")
    print(classification_report(y_test, y_test_pred, 
                                target_names=['FALSE POSITIVE', 'CONFIRMED'],
                                digits=4))
    
    # 결과 저장
    model_results.append({
        'name': name,
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'cv_mean': cv_mean,
        'cv_std': cv_std,
        'overfitting': overfitting,
        'train_time': train_time
    })
    
    return model, test_acc

# ============================================================================
# 2.1 Random Forest (과적합 방지 파라미터)
# ============================================================================

rf_model = RandomForestClassifier(
    n_estimators=300,        # 트리 개수 증가
    max_depth=15,            # 깊이 제한 (과적합 방지)
    min_samples_split=10,    # 분할 최소 샘플 수
    min_samples_leaf=5,      # 리프 최소 샘플 수
    max_features='sqrt',     # 특징 샘플링
    class_weight='balanced', # 클래스 불균형 처리
    random_state=42,
    n_jobs=-1
)

rf_model, rf_acc = evaluate_model_with_overfitting_check(
    "Random Forest", rf_model, X_train_scaled, y_train, X_test_scaled, y_test, use_scaled=True
)

# ============================================================================
# 2.2 Extra Trees (Random Forest보다 더 무작위성)
# ============================================================================

et_model = ExtraTreesClassifier(
    n_estimators=300,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    class_weight='balanced',
    random_state=42,
    n_jobs=-1
)

et_model, et_acc = evaluate_model_with_overfitting_check(
    "Extra Trees", et_model, X_train_scaled, y_train, X_test_scaled, y_test, use_scaled=True
)

# ============================================================================
# 2.3 XGBoost (강력한 부스팅)
# ============================================================================

xgb_model = XGBClassifier(
    n_estimators=300,
    max_depth=8,             # 깊이 제한
    learning_rate=0.05,      # 낮은 학습률
    subsample=0.8,           # 샘플 서브샘플링
    colsample_bytree=0.8,    # 특징 서브샘플링
    min_child_weight=5,      # 과적합 방지
    gamma=0.1,               # 분할 정규화
    reg_alpha=0.1,           # L1 정규화
    reg_lambda=1.0,          # L2 정규화
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

xgb_model, xgb_acc = evaluate_model_with_overfitting_check(
    "XGBoost", xgb_model, X_train_scaled, y_train, X_test_scaled, y_test, use_scaled=True
)

# ============================================================================
# 2.4 LightGBM (빠르고 효율적)
# ============================================================================

lgbm_model = LGBMClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_samples=10,    # 과적합 방지
    reg_alpha=0.1,
    reg_lambda=1.0,
    class_weight='balanced',
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgbm_model, lgbm_acc = evaluate_model_with_overfitting_check(
    "LightGBM", lgbm_model, X_train_scaled, y_train, X_test_scaled, y_test, use_scaled=True
)

# ============================================================================
# 2.5 CatBoost (범주형 특징 처리 우수)
# ============================================================================

catboost_model = CatBoostClassifier(
    iterations=300,
    depth=8,
    learning_rate=0.05,
    l2_leaf_reg=3.0,         # L2 정규화
    random_strength=1.0,     # 무작위성 추가
    bagging_temperature=1.0,
    class_weights=[1, 1],
    random_state=42,
    verbose=0
)

catboost_model, catboost_acc = evaluate_model_with_overfitting_check(
    "CatBoost", catboost_model, X_train_scaled, y_train, X_test_scaled, y_test, use_scaled=True
)

# ============================================================================
# 2.6 Gradient Boosting (안정적)
# ============================================================================

gb_model = GradientBoostingClassifier(
    n_estimators=300,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_split=10,
    min_samples_leaf=5,
    max_features='sqrt',
    random_state=42
)

gb_model, gb_acc = evaluate_model_with_overfitting_check(
    "Gradient Boosting", gb_model, X_train_scaled, y_train, X_test_scaled, y_test, use_scaled=True
)

# ============================================================================
# 3. 앙상블 모델 구성
# ============================================================================

print("\n" + "="*100)
print("🎯 3. 앙상블 모델 구성")
print("="*100)

# 모델 성능 정렬
sorted_results = sorted(model_results, key=lambda x: x['test_acc'], reverse=True)

print("\n📊 개별 모델 성능 순위:")
print("-"*100)
print(f"{'순위':<5} {'모델':<25} {'테스트 정확도':<15} {'CV 정확도':<20} {'과적합':<15} {'학습시간':<10}")
print("-"*100)

for i, result in enumerate(sorted_results, 1):
    print(f"{i:<5} {result['name']:<25} {result['test_acc']*100:>6.2f}%      "
          f"{result['cv_mean']*100:>6.2f}% ± {result['cv_std']*100:>4.2f}%   "
          f"{result['overfitting']*100:>6.2f}%p     {result['train_time']:>6.2f}초")

# 상위 5개 모델로 Voting Classifier 구성
top_5_models = sorted_results[:5]

print(f"\n🏆 상위 5개 모델로 Soft Voting Classifier 구성:")
for i, result in enumerate(top_5_models, 1):
    print(f"  {i}. {result['name']} (테스트 정확도: {result['test_acc']*100:.2f}%)")

voting_clf = VotingClassifier(
    estimators=[(result['name'], result['model']) for result in top_5_models],
    voting='soft',  # 확률 기반 소프트 보팅
    n_jobs=-1
)

print("\n🔄 앙상블 모델 학습 중...")
start_time = time.time()
voting_clf.fit(X_train_scaled, y_train)
ensemble_train_time = time.time() - start_time

# 앙상블 예측
y_train_pred_ensemble = voting_clf.predict(X_train_scaled)
y_test_pred_ensemble = voting_clf.predict(X_test_scaled)

# 앙상블 정확도
ensemble_train_acc = accuracy_score(y_train, y_train_pred_ensemble)
ensemble_test_acc = accuracy_score(y_test, y_test_pred_ensemble)
ensemble_overfitting = ensemble_train_acc - ensemble_test_acc

print(f"⏱️ 앙상블 학습 시간: {ensemble_train_time:.2f}초")
print(f"📊 앙상블 학습 정확도: {ensemble_train_acc*100:.2f}%")
print(f"📊 앙상블 테스트 정확도: {ensemble_test_acc*100:.2f}%")
print(f"⚠️ 앙상블 과적합 정도: {ensemble_overfitting*100:.2f}%p", end="")

if ensemble_overfitting > 0.05:
    print(" (⚠️ 과적합 경고!)")
elif ensemble_overfitting > 0.02:
    print(" (⚡ 약간 과적합)")
else:
    print(" (✅ 양호)")

# 앙상블 교차 검증
print("\n🔄 앙상블 교차 검증 (5-fold)...")
ensemble_cv_scores = cross_val_score(voting_clf, X_train_scaled, y_train, cv=5, scoring='accuracy')
ensemble_cv_mean = ensemble_cv_scores.mean()
ensemble_cv_std = ensemble_cv_scores.std()

print(f"🔄 앙상블 교차 검증: {ensemble_cv_mean*100:.2f}% ± {ensemble_cv_std*100:.2f}%")

# 상세 리포트
print(f"\n📊 앙상블 상세 분류 리포트:")
print(classification_report(y_test, y_test_pred_ensemble, 
                            target_names=['FALSE POSITIVE', 'CONFIRMED'],
                            digits=4))

# ============================================================================
# 4. 최종 결과 요약
# ============================================================================

print("\n" + "="*100)
print("📈 4. 최종 결과 요약")
print("="*100)

print("\n🏆 개별 모델 테스트 정확도:")
print("-"*100)
for i, result in enumerate(sorted_results, 1):
    print(f"  {i}. {result['name']:<25} : {result['test_acc']*100:>6.2f}%")

print(f"\n🎯 앙상블 모델 테스트 정확도:")
print("-"*100)
print(f"  Soft Voting Ensemble (Top 5) : {ensemble_test_acc*100:>6.2f}%")

# 최고 성능 모델
best_single = sorted_results[0]
print(f"\n🥇 최고 단일 모델:")
print(f"  • 모델: {best_single['name']}")
print(f"  • 테스트 정확도: {best_single['test_acc']*100:.2f}%")
print(f"  • CV 정확도: {best_single['cv_mean']*100:.2f}% ± {best_single['cv_std']*100:.2f}%")
print(f"  • 과적합 정도: {best_single['overfitting']*100:.2f}%p")

# 개선 정도
improvement = (ensemble_test_acc - best_single['test_acc']) * 100
print(f"\n📊 앙상블 개선:")
if improvement > 0:
    print(f"  • 앙상블이 최고 단일 모델보다 {improvement:.2f}%p 높음 ⬆️")
elif improvement < 0:
    print(f"  • 앙상블이 최고 단일 모델보다 {abs(improvement):.2f}%p 낮음 ⬇️")
else:
    print(f"  • 앙상블과 최고 단일 모델이 동일 ➡️")

# 목표 달성 확인
target_acc = 0.95  # 90% 중반 목표
print(f"\n🎯 목표 달성 여부:")
print(f"  • 목표 정확도: {target_acc*100:.0f}%")
print(f"  • 달성 정확도: {ensemble_test_acc*100:.2f}%")

if ensemble_test_acc >= target_acc:
    print(f"  • 결과: ✅ 목표 달성!")
elif ensemble_test_acc >= target_acc - 0.02:
    print(f"  • 결과: 🔥 거의 달성 (목표까지 {(target_acc - ensemble_test_acc)*100:.2f}%p)")
else:
    print(f"  • 결과: ⚡ 추가 최적화 필요 (목표까지 {(target_acc - ensemble_test_acc)*100:.2f}%p)")

print("\n" + "="*100)
print("✅ 모델 학습 및 평가 완료!")
print("="*100)

# 혼동 행렬 출력
print("\n📊 앙상블 혼동 행렬:")
cm = confusion_matrix(y_test, y_test_pred_ensemble)
print("\n실제\\예측   FALSE POSITIVE   CONFIRMED")
print("-"*45)
print(f"FALSE POS   {cm[0,0]:>8}          {cm[0,1]:>8}")
print(f"CONFIRMED   {cm[1,0]:>8}          {cm[1,1]:>8}")

# 정밀도, 재현율 계산
precision = cm[1,1] / (cm[1,1] + cm[0,1]) if (cm[1,1] + cm[0,1]) > 0 else 0
recall = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

print(f"\n📊 주요 메트릭:")
print(f"  • Precision (정밀도): {precision*100:.2f}%")
print(f"  • Recall (재현율): {recall*100:.2f}%")
print(f"  • F1-Score: {f1*100:.2f}%")

print("\n" + "="*100)
print("🎉 외계행성 판별 모델 완성!")
print("="*100)
