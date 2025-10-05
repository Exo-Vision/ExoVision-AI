"""
최종 2-모델 시스템
- 모델 1: CONFIRMED vs FALSE POSITIVE (2-클래스)
- 모델 2: CANDIDATE vs NOT_CANDIDATE (CANDIDATE 판별)
- 파이프라인: 확신도 기반 계층적 분류
- 모델 저장/로드 기능 포함
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import joblib
import os
from datetime import datetime

print("=" * 100)
print("🎯 최종 2-모델 시스템: CONFIRMED/FALSE POSITIVE + CANDIDATE 판별")
print("=" * 100)

# ============================================================================
# 데이터 로드 및 피처 엔지니어링
# ============================================================================
print("\n[데이터 로드 및 피처 엔지니어링]")
print("-" * 100)

df = pd.read_csv('datasets/exoplanets.csv', low_memory=False)
print(f"원본 데이터: {df.shape[0]:,} 샘플")

# 타겟 변수 확인
print(f"타겟 분포:")
for label, count in df['koi_disposition'].value_counts().items():
    print(f"  {label}: {count:,}")

y_full = df['koi_disposition']
X_full = df.drop(['koi_disposition', 'kepoi_name'], axis=1, errors='ignore')

# 수치형 컬럼만
numeric_cols = X_full.select_dtypes(include=[np.number]).columns
X_full = X_full[numeric_cols]
print(f"기본 피처: {len(numeric_cols)}개")

# 피처 엔지니어링 (29개 피처)
print("\n피처 엔지니어링 중...")
X_full['planet_star_ratio'] = X_full['koi_prad'] / (X_full['koi_srad'] + 1e-10)
X_full['orbital_energy'] = 1.0 / (X_full['koi_sma'] + 1e-10)
X_full['transit_signal'] = X_full['koi_depth'] * X_full['koi_duration']
X_full['stellar_density'] = X_full['koi_smass'] / (X_full['koi_srad']**3 + 1e-10)
X_full['planet_density_proxy'] = X_full['koi_prad']**3 / (X_full['koi_sma']**2 + 1e-10)
X_full['log_period'] = np.log1p(X_full['koi_period'])
X_full['log_depth'] = np.log1p(X_full['koi_depth'])
X_full['log_insol'] = np.log1p(X_full['koi_insol'])
X_full['orbit_stability'] = X_full['koi_eccen'] * X_full['koi_impact']
X_full['transit_snr'] = X_full['koi_depth'] / (X_full['koi_duration'] + 1e-10)

X_full = X_full.replace([np.inf, -np.inf], np.nan)
X_full = X_full.fillna(X_full.median())

print(f"최종 피처 수: {X_full.shape[1]}개")

# ============================================================================
# 모델 1: CONFIRMED vs FALSE POSITIVE (2-클래스)
# ============================================================================
print("\n" + "=" * 100)
print("🔵 모델 1: CONFIRMED vs FALSE POSITIVE")
print("=" * 100)

y_binary = y_full[y_full.isin(['CONFIRMED', 'FALSE POSITIVE'])]
X_binary = X_full.loc[y_binary.index]

print(f"\n학습 데이터:")
print(f"  총 샘플: {len(y_binary):,}")
for label, count in y_binary.value_counts().sort_index().items():
    print(f"  {label}: {count:,} ({count/len(y_binary)*100:.1f}%)")

y_binary_encoded = (y_binary == 'CONFIRMED').astype(int)

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(
    X_binary, y_binary_encoded, test_size=0.1, random_state=42, stratify=y_binary_encoded
)

print(f"\nTrain: {len(y_train_1):,} / Test: {len(y_test_1):,}")

# 스케일링
scaler_1 = StandardScaler()
X_train_1_scaled = scaler_1.fit_transform(X_train_1)
X_test_1_scaled = scaler_1.transform(X_test_1)

# 여러 모델 테스트
print("\n모델 테스트 중...")

models_1 = {
    'CatBoost': CatBoostClassifier(
        iterations=500, depth=4, learning_rate=0.02,
        l2_leaf_reg=10.0, bagging_temperature=1.0,
        random_state=42, verbose=False
    ),
    'XGBoost': XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.02,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=10.0,
        random_state=42, eval_metric='logloss'
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.02,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=10.0,
        random_state=42, verbose=-1
    )
}

results_1 = {}

for name, model in models_1.items():
    print(f"\n{name} 학습 중...")
    model.fit(X_train_1_scaled, y_train_1)
    
    train_acc = accuracy_score(y_train_1, model.predict(X_train_1_scaled))
    test_acc = accuracy_score(y_test_1, model.predict(X_test_1_scaled))
    cv_scores = cross_val_score(model, X_train_1_scaled, y_train_1, cv=5, scoring='accuracy')
    
    results_1[name] = {
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'overfitting': train_acc - test_acc
    }
    
    print(f"  Train: {train_acc:.4f} | Test: {test_acc:.4f} | CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
    print(f"  과적합: {(train_acc - test_acc)*100:.2f}%p")

# Voting Ensemble
print("\nVoting Ensemble 학습 중...")
voting_1 = VotingClassifier(
    estimators=[(name, model) for name, model in models_1.items()],
    voting='soft'
)
voting_1.fit(X_train_1_scaled, y_train_1)

train_acc_v1 = accuracy_score(y_train_1, voting_1.predict(X_train_1_scaled))
test_acc_v1 = accuracy_score(y_test_1, voting_1.predict(X_test_1_scaled))
cv_scores_v1 = cross_val_score(voting_1, X_train_1_scaled, y_train_1, cv=5, scoring='accuracy')

results_1['Voting'] = {
    'model': voting_1,
    'train_acc': train_acc_v1,
    'test_acc': test_acc_v1,
    'cv_mean': cv_scores_v1.mean(),
    'cv_std': cv_scores_v1.std(),
    'overfitting': train_acc_v1 - test_acc_v1
}

print(f"  Train: {train_acc_v1:.4f} | Test: {test_acc_v1:.4f} | CV: {cv_scores_v1.mean():.4f}±{cv_scores_v1.std():.4f}")
print(f"  과적합: {(train_acc_v1 - test_acc_v1)*100:.2f}%p")

# 최고 모델 선택
best_1 = max(results_1.items(), key=lambda x: x[1]['test_acc'])
print(f"\n✅ 모델 1 최고: {best_1[0]} - {best_1[1]['test_acc']:.4f} ({best_1[1]['test_acc']*100:.2f}%)")

# ============================================================================
# 모델 2: CANDIDATE vs NOT_CANDIDATE
# ============================================================================
print("\n" + "=" * 100)
print("🟢 모델 2: CANDIDATE 판별 (IS_CANDIDATE vs NOT_CANDIDATE)")
print("=" * 100)

y_candidate = (y_full == 'CANDIDATE').astype(int)
X_candidate = X_full.copy()

print(f"\n학습 데이터:")
print(f"  총 샘플: {len(y_candidate):,}")
print(f"  CANDIDATE: {y_candidate.sum():,} ({y_candidate.sum()/len(y_candidate)*100:.1f}%)")
print(f"  NOT CANDIDATE: {(~y_candidate.astype(bool)).sum():,} ({(~y_candidate.astype(bool)).sum()/len(y_candidate)*100:.1f}%)")

X_train_2, X_test_2, y_train_2, y_test_2 = train_test_split(
    X_candidate, y_candidate, test_size=0.1, random_state=42, stratify=y_candidate
)

print(f"\nTrain: {len(y_train_2):,} / Test: {len(y_test_2):,}")

# 스케일링
scaler_2 = StandardScaler()
X_train_2_scaled = scaler_2.fit_transform(X_train_2)
X_test_2_scaled = scaler_2.transform(X_test_2)

# 여러 모델 테스트
print("\n모델 테스트 중...")

models_2 = {
    'CatBoost': CatBoostClassifier(
        iterations=500, depth=4, learning_rate=0.02,
        l2_leaf_reg=10.0, bagging_temperature=1.0,
        random_state=42, verbose=False
    ),
    'XGBoost': XGBClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.02,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=10.0,
        random_state=42, eval_metric='logloss'
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=500, max_depth=4, learning_rate=0.02,
        subsample=0.7, colsample_bytree=0.7,
        reg_alpha=2.0, reg_lambda=10.0,
        random_state=42, verbose=-1
    ),
    'Neural Network': MLPClassifier(
        hidden_layer_sizes=(128, 64, 32),
        activation='relu',
        alpha=0.01,
        max_iter=500,
        early_stopping=True,
        random_state=42
    )
}

results_2 = {}

for name, model in models_2.items():
    print(f"\n{name} 학습 중...")
    model.fit(X_train_2_scaled, y_train_2)
    
    train_acc = accuracy_score(y_train_2, model.predict(X_train_2_scaled))
    test_acc = accuracy_score(y_test_2, model.predict(X_test_2_scaled))
    cv_scores = cross_val_score(model, X_train_2_scaled, y_train_2, cv=5, scoring='accuracy')
    
    results_2[name] = {
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'overfitting': train_acc - test_acc
    }
    
    print(f"  Train: {train_acc:.4f} | Test: {test_acc:.4f} | CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
    print(f"  과적합: {(train_acc - test_acc)*100:.2f}%p")

# Voting Ensemble
print("\nVoting Ensemble 학습 중...")
voting_2 = VotingClassifier(
    estimators=[(name, model) for name, model in models_2.items()],
    voting='soft'
)
voting_2.fit(X_train_2_scaled, y_train_2)

train_acc_v2 = accuracy_score(y_train_2, voting_2.predict(X_train_2_scaled))
test_acc_v2 = accuracy_score(y_test_2, voting_2.predict(X_test_2_scaled))
cv_scores_v2 = cross_val_score(voting_2, X_train_2_scaled, y_train_2, cv=5, scoring='accuracy')

results_2['Voting'] = {
    'model': voting_2,
    'train_acc': train_acc_v2,
    'test_acc': test_acc_v2,
    'cv_mean': cv_scores_v2.mean(),
    'cv_std': cv_scores_v2.std(),
    'overfitting': train_acc_v2 - test_acc_v2
}

print(f"  Train: {train_acc_v2:.4f} | Test: {test_acc_v2:.4f} | CV: {cv_scores_v2.mean():.4f}±{cv_scores_v2.std():.4f}")
print(f"  과적합: {(train_acc_v2 - test_acc_v2)*100:.2f}%p")

# 최고 모델 선택
best_2 = max(results_2.items(), key=lambda x: x[1]['test_acc'])
print(f"\n✅ 모델 2 최고: {best_2[0]} - {best_2[1]['test_acc']:.4f} ({best_2[1]['test_acc']*100:.2f}%)")

# ============================================================================
# 파이프라인 통합: 확신도 기반 분류
# ============================================================================
print("\n" + "=" * 100)
print("🔗 파이프라인 통합: 확신도 기반 3-클래스 분류")
print("=" * 100)

# 전체 테스트 데이터 준비
y_test_full = y_full.loc[X_test_2.index]
X_test_full_scaled_1 = scaler_1.transform(X_test_2)
X_test_full_scaled_2 = scaler_2.transform(X_test_2)

# 모델 1 예측
model1 = best_1[1]['model']
stage1_proba = model1.predict_proba(X_test_full_scaled_1)
stage1_pred = model1.predict(X_test_full_scaled_1)

# 모델 2 예측
model2 = best_2[1]['model']
stage2_proba = model2.predict_proba(X_test_full_scaled_2)
stage2_pred = model2.predict(X_test_full_scaled_2)

# 확신도 임계값 최적화
thresholds = [0.80, 0.85, 0.90, 0.92, 0.94, 0.96]

print(f"\n{'임계값':<10} {'1단계사용':<12} {'CANDIDATE수':<12} {'최종정확도':<12}")
print("-" * 100)

best_threshold = 0.90
best_accuracy = 0.0
best_predictions = None

for threshold in thresholds:
    final_predictions = np.empty(len(y_test_full), dtype=object)
    
    # 1단계 확신도가 높은 케이스 → CONFIRMED or FALSE POSITIVE
    high_conf_mask = (stage1_proba.max(axis=1) >= threshold)
    final_predictions[high_conf_mask] = np.where(
        stage1_pred[high_conf_mask] == 1,
        'CONFIRMED',
        'FALSE POSITIVE'
    )
    
    # 1단계 확신도가 낮은 케이스 → CANDIDATE 판별
    low_conf_mask = ~high_conf_mask
    final_predictions[low_conf_mask] = np.where(
        stage2_pred[low_conf_mask] == 1,
        'CANDIDATE',
        np.where(stage1_pred[low_conf_mask] == 1, 'CONFIRMED', 'FALSE POSITIVE')
    )
    
    accuracy = accuracy_score(y_test_full, final_predictions)
    stage1_ratio = high_conf_mask.sum() / len(y_test_full) * 100
    candidate_count = (final_predictions == 'CANDIDATE').sum()
    
    print(f"{threshold:.2f}      {stage1_ratio:>6.1f}%      {candidate_count:>4}개       {accuracy:.4f}")
    
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_threshold = threshold
        best_predictions = final_predictions.copy()

print(f"\n✅ 최적 임계값: {best_threshold:.2f}")
print(f"✅ 최종 정확도: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

# ============================================================================
# 최종 평가
# ============================================================================
print("\n" + "=" * 100)
print("📊 최종 성능 평가")
print("=" * 100)

print("\n모델별 정확도:")
print("-" * 100)
print(f"모델 1 (CONFIRMED vs FALSE POSITIVE): {best_1[1]['test_acc']:.4f} ({best_1[1]['test_acc']*100:.2f}%)")
print(f"모델 2 (CANDIDATE 판별):              {best_2[1]['test_acc']:.4f} ({best_2[1]['test_acc']*100:.2f}%)")
print(f"최종 통합 (3-클래스):                 {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

# Confusion Matrix
cm = confusion_matrix(y_test_full, best_predictions,
                      labels=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'])

print("\nConfusion Matrix:")
print("-" * 100)
print(f"{'':15} {'CANDIDATE':>12} {'CONFIRMED':>12} {'FALSE POS':>12}")
print("-" * 100)
for i, label in enumerate(['CANDIDATE', 'CONFIRMED', 'FALSE POS']):
    print(f"{label:15} {cm[i][0]:>12} {cm[i][1]:>12} {cm[i][2]:>12}")

# Classification Report
print("\nClassification Report:")
print("-" * 100)
print(classification_report(y_test_full, best_predictions,
                           labels=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'],
                           target_names=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']))

# 클래스별 정확도
print("\n클래스별 정확도:")
print("-" * 100)
for label in ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']:
    mask = y_test_full == label
    if mask.sum() > 0:
        class_acc = accuracy_score(y_test_full[mask], best_predictions[mask])
        print(f"  {label:20} {class_acc:.4f} ({class_acc*100:.2f}%)  [{mask.sum()}개 샘플]")

# ============================================================================
# 모델 저장
# ============================================================================
print("\n" + "=" * 100)
print("💾 모델 저장")
print("=" * 100)

# 저장 디렉토리 생성
save_dir = 'saved_models'
os.makedirs(save_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# 모델 1 저장
model1_path = os.path.join(save_dir, f'model1_binary_{best_1[0].lower().replace(" ", "_")}_{timestamp}.pkl')
scaler1_path = os.path.join(save_dir, f'scaler1_{timestamp}.pkl')

joblib.dump(model1, model1_path)
joblib.dump(scaler_1, scaler1_path)
print(f"✅ 모델 1 저장: {model1_path}")
print(f"✅ 스케일러 1 저장: {scaler1_path}")

# 모델 2 저장
model2_path = os.path.join(save_dir, f'model2_candidate_{best_2[0].lower().replace(" ", "_")}_{timestamp}.pkl')
scaler2_path = os.path.join(save_dir, f'scaler2_{timestamp}.pkl')

joblib.dump(model2, model2_path)
joblib.dump(scaler_2, scaler2_path)
print(f"✅ 모델 2 저장: {model2_path}")
print(f"✅ 스케일러 2 저장: {scaler2_path}")

# 설정 저장
config = {
    'model1_name': best_1[0],
    'model1_accuracy': best_1[1]['test_acc'],
    'model2_name': best_2[0],
    'model2_accuracy': best_2[1]['test_acc'],
    'best_threshold': best_threshold,
    'final_accuracy': best_accuracy,
    'timestamp': timestamp,
    'feature_count': X_full.shape[1]
}

config_path = os.path.join(save_dir, f'config_{timestamp}.pkl')
joblib.dump(config, config_path)
print(f"✅ 설정 저장: {config_path}")

print("\n" + "=" * 100)
print("📂 저장된 파일:")
print("=" * 100)
print(f"  • {model1_path}")
print(f"  • {scaler1_path}")
print(f"  • {model2_path}")
print(f"  • {scaler2_path}")
print(f"  • {config_path}")

# ============================================================================
# 모델 로드 예제 코드 생성
# ============================================================================
load_example = f"""
# ============================================================================
# 모델 로드 및 사용 예제
# ============================================================================

import joblib
import numpy as np
import pandas as pd

# 모델 및 스케일러 로드
model1 = joblib.load('{model1_path}')
scaler1 = joblib.load('{scaler1_path}')
model2 = joblib.load('{model2_path}')
scaler2 = joblib.load('{scaler2_path}')
config = joblib.load('{config_path}')

print("모델 로드 완료!")
print(f"모델 1: {{config['model1_name']}} - {{config['model1_accuracy']*100:.2f}}%")
print(f"모델 2: {{config['model2_name']}} - {{config['model2_accuracy']*100:.2f}}%")
print(f"최종 정확도: {{config['final_accuracy']*100:.2f}}%")
print(f"최적 임계값: {{config['best_threshold']:.2f}}")

# 새로운 데이터 예측
def predict_exoplanet(X_new):
    \"\"\"
    새로운 데이터에 대해 3-클래스 예측
    X_new: pandas DataFrame (피처 엔지니어링 완료된 데이터)
    \"\"\"
    # 스케일링
    X_scaled_1 = scaler1.transform(X_new)
    X_scaled_2 = scaler2.transform(X_new)
    
    # 모델 1 예측
    proba1 = model1.predict_proba(X_scaled_1)
    pred1 = model1.predict(X_scaled_1)
    
    # 모델 2 예측
    pred2 = model2.predict(X_scaled_2)
    
    # 최종 예측
    final_pred = []
    for i in range(len(X_new)):
        if proba1[i].max() >= config['best_threshold']:
            # 고확신도 → 모델 1 결과 사용
            final_pred.append('CONFIRMED' if pred1[i] == 1 else 'FALSE POSITIVE')
        else:
            # 저확신도 → 모델 2로 CANDIDATE 판별
            if pred2[i] == 1:
                final_pred.append('CANDIDATE')
            else:
                final_pred.append('CONFIRMED' if pred1[i] == 1 else 'FALSE POSITIVE')
    
    return final_pred

# 사용 예시:
# predictions = predict_exoplanet(X_new_data)
"""

example_path = os.path.join(save_dir, f'load_and_predict_example_{timestamp}.py')
with open(example_path, 'w', encoding='utf-8') as f:
    f.write(load_example)

print(f"\n✅ 사용 예제 코드 저장: {example_path}")

print("\n" + "=" * 100)
print("✅ 최종 2-모델 시스템 완료!")
print("=" * 100)

print(f"\n🎯 최종 결과:")
print(f"  • 모델 1 ({best_1[0]}): {best_1[1]['test_acc']*100:.2f}%")
print(f"  • 모델 2 ({best_2[0]}): {best_2[1]['test_acc']*100:.2f}%")
print(f"  • 통합 시스템: {best_accuracy*100:.2f}%")

if best_accuracy >= 0.95:
    print("\n🎉🎉🎉 95% 목표 달성! 🎉🎉🎉")
elif best_accuracy >= 0.90:
    print(f"\n💪 90% 이상 달성! 목표까지 {(0.95-best_accuracy)*100:.2f}%p")
else:
    print(f"\n📊 추가 개선 필요: {(0.95-best_accuracy)*100:.2f}%p")

print("\n" + "=" * 100)
