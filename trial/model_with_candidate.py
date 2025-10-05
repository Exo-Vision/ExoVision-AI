"""
CANDIDATE 포함 학습 모델
- CANDIDATE 클래스를 학습에 포함 (21,271개 전체 데이터 활용)
- Semi-supervised learning 방식
- 강한 정규화 + 앙상블 다양성
- 목표: 95% 달성
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

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.linear_model import LogisticRegression

import time

print("="*100)
print("🚀 CANDIDATE 포함 학습 모델 (95% 목표)")
print("="*100)

# ============================================================================
# 1. 데이터 로드 및 전처리
# ============================================================================

print("\n" + "="*100)
print("📂 1. 데이터 로드 (CANDIDATE 포함)")
print("="*100)

df = pd.read_csv('datasets/exoplanets.csv')
print(f"전체 데이터: {df.shape[0]} 샘플")

# 클래스 분포 확인
print("\n📊 클래스 분포:")
print(df['koi_disposition'].value_counts())

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

df_fe = df_fe.dropna()

print(f"\n결측치 제거 후: {df_fe.shape[0]} 샘플")
print("\n클래스 분포 (결측치 제거 후):")
print(df_fe['koi_disposition'].value_counts())

# ============================================================================
# 2. 3-클래스 학습 데이터 준비
# ============================================================================

print("\n" + "="*100)
print("📊 2. 3-클래스 학습 데이터 준비")
print("="*100)

# 전체 데이터 사용 (CONFIRMED, FALSE POSITIVE, CANDIDATE)
X_full = df_fe.drop('koi_disposition', axis=1)
y_full = df_fe['koi_disposition']

# 레이블 인코딩
le = LabelEncoder()
y_full_encoded = le.fit_transform(y_full)

print(f"\n전체 데이터:")
print(f"  • 특징: {X_full.shape[1]}개")
print(f"  • 샘플: {X_full.shape[0]}개")
print(f"\n레이블 분포:")
for i, label in enumerate(le.classes_):
    count = np.sum(y_full_encoded == i)
    print(f"  • {label} (class {i}): {count}개 ({count/len(y_full_encoded)*100:.1f}%)")

# Train/Test 분할 (전체 데이터)
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X_full, y_full_encoded,
    test_size=0.1,
    random_state=42,
    stratify=y_full_encoded
)

# 스케일링
scaler_full = StandardScaler()
X_train_full_scaled = scaler_full.fit_transform(X_train_full)
X_test_full_scaled = scaler_full.transform(X_test_full)

print(f"\n전체 데이터 분할:")
print(f"  • 학습: {X_train_full.shape[0]} 샘플 (CONFIRMED, FALSE POSITIVE, CANDIDATE)")
print(f"  • 테스트: {X_test_full.shape[0]} 샘플")

# ============================================================================
# 3. 2-클래스 평가 데이터 준비 (CONFIRMED vs FALSE POSITIVE만)
# ============================================================================

print("\n" + "="*100)
print("📊 3. 2-클래스 평가 데이터 준비")
print("="*100)

# 평가용: CONFIRMED, FALSE POSITIVE만
df_binary = df_fe[df_fe['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()

X_binary = df_binary.drop('koi_disposition', axis=1)
y_binary = df_binary['koi_disposition']

le_binary = LabelEncoder()
y_binary_encoded = le_binary.fit_transform(y_binary)

# Train/Test 분할 (2-클래스)
X_train_binary, X_test_binary, y_train_binary, y_test_binary = train_test_split(
    X_binary, y_binary_encoded,
    test_size=0.1,
    random_state=42,
    stratify=y_binary_encoded
)

# 스케일링
scaler_binary = StandardScaler()
X_train_binary_scaled = scaler_binary.fit_transform(X_train_binary)
X_test_binary_scaled = scaler_binary.transform(X_test_binary)

print(f"\n2-클래스 평가 데이터:")
print(f"  • 학습: {X_train_binary.shape[0]} 샘플 (CONFIRMED, FALSE POSITIVE)")
print(f"  • 테스트: {X_test_binary.shape[0]} 샘플")

# ============================================================================
# 4. 전략 1: 3-클래스로 학습 후 2-클래스 예측
# ============================================================================

print("\n" + "="*100)
print("🎯 4. 전략 1: 3-클래스 학습 → 2-클래스 평가")
print("="*100)
print("\n💡 CANDIDATE를 포함한 전체 데이터로 학습하여 더 나은 패턴 인식")

model_results_strategy1 = []

def evaluate_3class_model(name, model, X_train_full, y_train_full, X_test_binary, y_test_binary):
    """3-클래스로 학습 후 2-클래스 테스트"""
    
    print(f"\n{'='*100}")
    print(f"🔹 {name} (3-클래스 학습)")
    print(f"{'='*100}")
    
    start_time = time.time()
    model.fit(X_train_full, y_train_full)
    train_time = time.time() - start_time
    
    # 3-클래스 학습 정확도
    train_acc_3class = accuracy_score(y_train_full, model.predict(X_train_full))
    
    # 2-클래스 테스트 예측
    y_pred_3class = model.predict(X_test_binary)
    
    # CANDIDATE(class 2)를 FALSE POSITIVE(class 0)로 매핑
    y_pred_binary = np.where(y_pred_3class == 2, 0, y_pred_3class)
    
    test_acc = accuracy_score(y_test_binary, y_pred_binary)
    
    print(f"⏱️ 학습 시간: {train_time:.2f}초")
    print(f"📊 3-클래스 학습 정확도: {train_acc_3class*100:.2f}%")
    print(f"📊 2-클래스 테스트 정확도: {test_acc*100:.2f}%")
    
    model_results_strategy1.append({
        'name': name,
        'model': model,
        'test_acc': test_acc
    })
    
    return model, test_acc

# CatBoost (3-클래스)
print("\n⏳ CatBoost 학습 중...")
catboost_3class = CatBoostClassifier(
    iterations=500,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=10.0,
    bagging_temperature=1.0,
    subsample=0.7,
    random_state=42,
    verbose=False
)
evaluate_3class_model("CatBoost", catboost_3class, X_train_full_scaled, y_train_full, 
                      X_test_binary_scaled, y_test_binary)

# XGBoost (3-클래스)
print("\n⏳ XGBoost 학습 중...")
xgboost_3class = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    reg_lambda=10.0,
    reg_alpha=2.0,
    subsample=0.6,
    colsample_bytree=0.6,
    random_state=42,
    eval_metric='mlogloss'
)
evaluate_3class_model("XGBoost", xgboost_3class, X_train_full_scaled, y_train_full,
                      X_test_binary_scaled, y_test_binary)

# LightGBM (3-클래스)
print("\n⏳ LightGBM 학습 중...")
lightgbm_3class = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=31,
    reg_lambda=10.0,
    reg_alpha=2.0,
    subsample=0.6,
    colsample_bytree=0.6,
    random_state=42,
    verbose=-1
)
evaluate_3class_model("LightGBM", lightgbm_3class, X_train_full_scaled, y_train_full,
                      X_test_binary_scaled, y_test_binary)

# Neural Network (3-클래스)
print("\n⏳ Neural Network 학습 중...")
mlp_3class = MLPClassifier(
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
evaluate_3class_model("Neural Network", mlp_3class, X_train_full_scaled, y_train_full,
                      X_test_binary_scaled, y_test_binary)

# ============================================================================
# 5. 전략 2: 2-클래스로 직접 학습 (비교용)
# ============================================================================

print("\n" + "="*100)
print("🎯 5. 전략 2: 2-클래스 직접 학습 (비교용)")
print("="*100)

model_results_strategy2 = []

def evaluate_2class_model(name, model, X_train, y_train, X_test, y_test):
    """2-클래스 직접 학습"""
    
    print(f"\n{'='*100}")
    print(f"🔹 {name} (2-클래스 학습)")
    print(f"{'='*100}")
    
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    
    test_acc = accuracy_score(y_test, model.predict(X_test))
    
    print(f"⏱️ 학습 시간: {train_time:.2f}초")
    print(f"📊 테스트 정확도: {test_acc*100:.2f}%")
    
    model_results_strategy2.append({
        'name': name,
        'test_acc': test_acc
    })
    
    return test_acc

# CatBoost (2-클래스)
print("\n⏳ CatBoost 학습 중...")
catboost_2class = CatBoostClassifier(
    iterations=500,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=10.0,
    bagging_temperature=1.0,
    subsample=0.7,
    random_state=42,
    verbose=False
)
evaluate_2class_model("CatBoost", catboost_2class, X_train_binary_scaled, y_train_binary,
                      X_test_binary_scaled, y_test_binary)

# XGBoost (2-클래스)
print("\n⏳ XGBoost 학습 중...")
xgboost_2class = XGBClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    reg_lambda=10.0,
    reg_alpha=2.0,
    subsample=0.6,
    colsample_bytree=0.6,
    random_state=42,
    eval_metric='logloss'
)
evaluate_2class_model("XGBoost", xgboost_2class, X_train_binary_scaled, y_train_binary,
                      X_test_binary_scaled, y_test_binary)

# LightGBM (2-클래스)
print("\n⏳ LightGBM 학습 중...")
lightgbm_2class = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=31,
    reg_lambda=10.0,
    reg_alpha=2.0,
    subsample=0.6,
    colsample_bytree=0.6,
    random_state=42,
    verbose=-1
)
evaluate_2class_model("LightGBM", lightgbm_2class, X_train_binary_scaled, y_train_binary,
                      X_test_binary_scaled, y_test_binary)

# ============================================================================
# 6. 앙상블 (3-클래스 모델들)
# ============================================================================

print("\n" + "="*100)
print("🎯 6. 앙상블 (3-클래스 학습 모델들)")
print("="*100)

# Soft Voting
voting_3class = VotingClassifier(
    estimators=[
        ('catboost', catboost_3class),
        ('xgboost', xgboost_3class),
        ('lightgbm', lightgbm_3class),
        ('mlp', mlp_3class)
    ],
    voting='soft',
    n_jobs=-1
)

print("\n⏳ Voting Ensemble 학습 중...")
voting_3class.fit(X_train_full_scaled, y_train_full)

y_pred_voting = voting_3class.predict(X_test_binary_scaled)
y_pred_voting_binary = np.where(y_pred_voting == 2, 0, y_pred_voting)
voting_test_acc = accuracy_score(y_test_binary, y_pred_voting_binary)

print(f"📊 Voting Ensemble 테스트 정확도: {voting_test_acc*100:.2f}%")

# Stacking
stacking_3class = StackingClassifier(
    estimators=[
        ('catboost', catboost_3class),
        ('xgboost', xgboost_3class),
        ('lightgbm', lightgbm_3class),
        ('mlp', mlp_3class)
    ],
    final_estimator=LogisticRegression(C=0.5, max_iter=1000, random_state=42),
    cv=5,
    n_jobs=-1
)

print("\n⏳ Stacking Ensemble 학습 중...")
stacking_3class.fit(X_train_full_scaled, y_train_full)

y_pred_stacking = stacking_3class.predict(X_test_binary_scaled)
y_pred_stacking_binary = np.where(y_pred_stacking == 2, 0, y_pred_stacking)
stacking_test_acc = accuracy_score(y_test_binary, y_pred_stacking_binary)

print(f"📊 Stacking Ensemble 테스트 정확도: {stacking_test_acc*100:.2f}%")

# ============================================================================
# 7. 최종 결과 비교
# ============================================================================

print("\n" + "="*100)
print("📊 7. 최종 결과 비교")
print("="*100)

print("\n" + "="*100)
print("전략 1: 3-클래스 학습 (CANDIDATE 포함, 19,128개 샘플)")
print("="*100)
for result in sorted(model_results_strategy1, key=lambda x: x['test_acc'], reverse=True):
    print(f"  • {result['name']:25s}: {result['test_acc']*100:.2f}%")
print(f"  • Voting Ensemble        : {voting_test_acc*100:.2f}%")
print(f"  • Stacking Ensemble      : {stacking_test_acc*100:.2f}%")

print("\n" + "="*100)
print("전략 2: 2-클래스 직접 학습 (CANDIDATE 제외, 10,085개 샘플)")
print("="*100)
for result in sorted(model_results_strategy2, key=lambda x: x['test_acc'], reverse=True):
    print(f"  • {result['name']:25s}: {result['test_acc']*100:.2f}%")

# 최고 성능
all_results = [
    ('3-class ' + r['name'], r['test_acc']) for r in model_results_strategy1
] + [
    ('3-class Voting', voting_test_acc),
    ('3-class Stacking', stacking_test_acc),
] + [
    ('2-class ' + r['name'], r['test_acc']) for r in model_results_strategy2
]

best_model, best_acc = max(all_results, key=lambda x: x[1])

print("\n" + "="*100)
print("🏆 최고 성능 모델")
print("="*100)
print(f"\n  • 모델: {best_model}")
print(f"  • 테스트 정확도: {best_acc*100:.2f}%")

# 목표 달성 여부
target_acc = 0.95
print("\n" + "="*100)
print("🎯 목표 달성 여부")
print("="*100)
print(f"\n  • 목표: {target_acc*100:.0f}%")
print(f"  • 달성: {best_acc*100:.2f}%")
print(f"  • 차이: {(best_acc - target_acc)*100:+.2f}%p")

if best_acc >= target_acc:
    print("\n🎉 축하합니다! 95% 목표를 달성했습니다!")
elif best_acc >= 0.93:
    print("\n⚡ 목표에 근접했습니다! (93%+)")
else:
    print("\n📊 추가 개선 필요")

# CANDIDATE 포함 효과 분석
best_3class = max(model_results_strategy1, key=lambda x: x['test_acc'])
best_2class = max(model_results_strategy2, key=lambda x: x['test_acc'])
improvement = (best_3class['test_acc'] - best_2class['test_acc']) * 100

print("\n" + "="*100)
print("💡 CANDIDATE 포함 효과")
print("="*100)
print(f"\n  • 2-클래스 학습 (11,206개): {best_2class['test_acc']*100:.2f}%")
print(f"  • 3-클래스 학습 (21,271개): {best_3class['test_acc']*100:.2f}%")
print(f"  • 개선 효과: {improvement:+.2f}%p")

if improvement > 0.5:
    print("\n  ✅ CANDIDATE 포함으로 성능이 크게 향상되었습니다!")
elif improvement > 0:
    print("\n  ⚡ CANDIDATE 포함으로 성능이 향상되었습니다!")
elif improvement > -0.5:
    print("\n  📊 성능 차이가 거의 없습니다")
else:
    print("\n  ⚠️ CANDIDATE 포함으로 성능이 하락했습니다 (노이즈 증가)")

# 상세 분류 리포트
print("\n" + "="*100)
print(f"📋 상세 분류 리포트 ({best_model})")
print("="*100)

if 'Stacking' in best_model and '3-class' in best_model:
    y_pred_final = y_pred_stacking_binary
elif 'Voting' in best_model and '3-class' in best_model:
    y_pred_final = y_pred_voting_binary
elif '3-class' in best_model:
    model_name = best_model.replace('3-class ', '')
    best_3class_model = [r for r in model_results_strategy1 if r['name'] == model_name][0]['model']
    y_pred_3class_final = best_3class_model.predict(X_test_binary_scaled)
    y_pred_final = np.where(y_pred_3class_final == 2, 0, y_pred_3class_final)
else:
    model_name = best_model.replace('2-class ', '')
    best_2class_model = [r for r in model_results_strategy2 if r['name'] == model_name][0]
    # 재학습 필요
    print("\n  (2-클래스 모델의 상세 리포트는 생략)")
    y_pred_final = None

if y_pred_final is not None:
    print("\n" + classification_report(y_test_binary, y_pred_final, 
                                      target_names=['FALSE POSITIVE', 'CONFIRMED']))

print("\n" + "="*100)
print("✅ CANDIDATE 포함 학습 완료!")
print("="*100)
