"""
직접 3-클래스 분류 시스템 (단일 모델 접근)
- 2-모델 시스템의 복잡성 제거
- CatBoost 다중 클래스 분류
- SMOTE + 클래스 가중치
- 44개 고급 피처
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from imblearn.over_sampling import SMOTE

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier

import joblib
import os
from datetime import datetime

print("=" * 100)
print("🎯 직접 3-클래스 분류 시스템")
print("=" * 100)

# ============================================================================
# 데이터 로드 및 고급 피처 엔지니어링
# ============================================================================
print("\n[1] 데이터 로드 및 고급 피처 엔지니어링")
print("-" * 100)

df = pd.read_csv('datasets/exoplanets_integrated.csv', low_memory=False)
print(f"원본 데이터: {df.shape[0]:,} 샘플")

# REFUTED 제거 (22개뿐) + NaN 제거
df = df[df['koi_disposition'] != 'REFUTED']
df = df.dropna(subset=['koi_disposition'])
print(f"REFUTED 및 NaN 제거 후: {df.shape[0]:,} 샘플")

y_full = df['koi_disposition']
X_full = df.drop(['koi_disposition', 'kepoi_name'], axis=1, errors='ignore')

print(f"\n타겟 분포:")
for label, count in y_full.value_counts().items():
    print(f"  {label}: {count:,} ({count/len(y_full)*100:.1f}%)")

# 수치형 컬럼만
numeric_cols = X_full.select_dtypes(include=[np.number]).columns
X_full = X_full[numeric_cols]

# 고급 피처 엔지니어링 (44개)
print("\n피처 엔지니어링 중 (44개)...")

# 기존 피처 (10개)
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

# 고급 피처 (15개)
X_full['habitable_zone_score'] = 1.0 / (1.0 + np.abs(X_full['koi_insol'] - 1.0))
X_full['temp_habitable_score'] = 1.0 / (1.0 + np.abs(X_full['koi_teq'] - 288) / 100)
X_full['roche_limit'] = 2.46 * X_full['koi_srad'] * (X_full['koi_smass'] / (X_full['koi_prad'] / 109.0))**(1/3)
X_full['hill_sphere'] = X_full['koi_sma'] * (X_full['koi_smass'] / 3.0)**(1/3)
X_full['transit_probability'] = X_full['koi_srad'] / (X_full['koi_sma'] * 215.032 + 1e-10)
X_full['improved_snr'] = (X_full['koi_depth'] * np.sqrt(X_full['koi_duration'])) / (X_full['koi_period'] + 1e-10)
X_full['stability_index'] = (1 - X_full['koi_eccen']) * (1 - X_full['koi_impact'])
X_full['mass_ratio'] = (X_full['koi_prad'] / 109.0)**3 / (X_full['koi_smass'] + 1e-10)
X_full['tidal_heating'] = X_full['koi_eccen'] / (X_full['koi_sma']**3 + 1e-10)
X_full['duration_ratio'] = X_full['koi_duration'] / (X_full['koi_period'] + 1e-10)
X_full['radiation_balance'] = X_full['koi_insol'] * (X_full['koi_prad']**2) / (X_full['koi_sma']**2 + 1e-10)
X_full['age_metallicity'] = X_full['koi_sage'] * (X_full['koi_smet'] + 2.5)
X_full['depth_size_ratio'] = X_full['koi_depth'] / (X_full['koi_prad']**2 + 1e-10)
X_full['kepler_ratio'] = X_full['koi_period']**2 / (X_full['koi_sma']**3 + 1e-10)
X_full['depth_variability'] = X_full['koi_depth'] / (X_full['koi_duration'] * X_full['koi_period'] + 1e-10)

# Inf, NaN 처리
X_full = X_full.replace([np.inf, -np.inf], np.nan)
X_full = X_full.fillna(X_full.median())

print(f"최종 피처 수: {X_full.shape[1]}개")

# ============================================================================
# 레이블 인코딩
# ============================================================================
le = LabelEncoder()
y_encoded = le.fit_transform(y_full)

print(f"\n레이블 매핑:")
for i, label in enumerate(le.classes_):
    print(f"  {i}: {label} ({(y_encoded == i).sum():,}개)")

# ============================================================================
# Train/Test Split
# ============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_encoded, test_size=0.15, random_state=42, stratify=y_encoded
)

print(f"\nTrain: {len(y_train):,} / Test: {len(y_test):,}")

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# SMOTE 데이터 증강
# ============================================================================
print("\n" + "=" * 100)
print("📈 SMOTE 데이터 증강")
print("=" * 100)

print("\n증강 전 분포:")
for i, label in enumerate(le.classes_):
    count = (y_train == i).sum()
    print(f"  {label}: {count:,} ({count/len(y_train)*100:.1f}%)")

smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train_scaled, y_train)

print(f"\n증강 후 분포:")
for i, label in enumerate(le.classes_):
    count = (y_train_smote == i).sum()
    print(f"  {label}: {count:,} ({count/len(y_train_smote)*100:.1f}%)")

print(f"\n총 샘플: {len(y_train):,} → {len(y_train_smote):,} (+{len(y_train_smote)-len(y_train):,})")

# ============================================================================
# 다중 클래스 분류 모델 (클래스 가중치 자동 조정)
# ============================================================================
print("\n" + "=" * 100)
print("🤖 다중 클래스 분류 모델 학습")
print("=" * 100)

# 클래스 가중치 계산
class_counts = np.bincount(y_train)
total = len(y_train)
class_weights = {i: total / (len(class_counts) * count) for i, count in enumerate(class_counts)}

print("\n클래스 가중치:")
for i, label in enumerate(le.classes_):
    print(f"  {label}: {class_weights[i]:.3f}")

models = {}

# CatBoost (최적화) - subsample 제거하고 MVS Sampling 사용
print("\n[1] CatBoost 학습 중...")
models['CatBoost'] = CatBoostClassifier(
    iterations=2000,
    depth=8,
    learning_rate=0.01,
    l2_leaf_reg=20.0,
    bagging_temperature=0.5,
    border_count=128,
    bootstrap_type='Bernoulli',  # subsample 대신
    subsample=0.85,
    class_weights=list(class_weights.values()),
    random_state=42,
    verbose=False
)
models['CatBoost'].fit(X_train_smote, y_train_smote)

train_pred_cb = models['CatBoost'].predict(X_train_smote)
test_pred_cb = models['CatBoost'].predict(X_test_scaled)
train_acc_cb = accuracy_score(y_train_smote, train_pred_cb)
test_acc_cb = accuracy_score(y_test, test_pred_cb)
cv_scores_cb = cross_val_score(models['CatBoost'], X_train_smote, y_train_smote, cv=5, scoring='accuracy', n_jobs=-1)

print(f"  Train: {train_acc_cb:.4f} ({train_acc_cb*100:.2f}%)")
print(f"  Test: {test_acc_cb:.4f} ({test_acc_cb*100:.2f}%)")
print(f"  CV: {cv_scores_cb.mean():.4f}±{cv_scores_cb.std():.4f}")
print(f"  과적합: {(train_acc_cb - test_acc_cb)*100:.2f}%p")

# XGBoost
print("\n[2] XGBoost 학습 중...")
models['XGBoost'] = XGBClassifier(
    n_estimators=2000,
    max_depth=8,
    learning_rate=0.01,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=8.0,
    reg_lambda=20.0,
    random_state=42,
    eval_metric='mlogloss',
    n_jobs=-1
)
models['XGBoost'].fit(X_train_smote, y_train_smote)

train_pred_xgb = models['XGBoost'].predict(X_train_smote)
test_pred_xgb = models['XGBoost'].predict(X_test_scaled)
train_acc_xgb = accuracy_score(y_train_smote, train_pred_xgb)
test_acc_xgb = accuracy_score(y_test, test_pred_xgb)
cv_scores_xgb = cross_val_score(models['XGBoost'], X_train_smote, y_train_smote, cv=5, scoring='accuracy', n_jobs=-1)

print(f"  Train: {train_acc_xgb:.4f} ({train_acc_xgb*100:.2f}%)")
print(f"  Test: {test_acc_xgb:.4f} ({test_acc_xgb*100:.2f}%)")
print(f"  CV: {cv_scores_xgb.mean():.4f}±{cv_scores_xgb.std():.4f}")
print(f"  과적합: {(train_acc_xgb - test_acc_xgb)*100:.2f}%p")

# LightGBM
print("\n[3] LightGBM 학습 중...")
models['LightGBM'] = LGBMClassifier(
    n_estimators=2000,
    max_depth=8,
    learning_rate=0.01,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=8.0,
    reg_lambda=20.0,
    class_weight='balanced',
    random_state=42,
    verbose=-1,
    n_jobs=-1
)
models['LightGBM'].fit(X_train_smote, y_train_smote)

train_pred_lgb = models['LightGBM'].predict(X_train_smote)
test_pred_lgb = models['LightGBM'].predict(X_test_scaled)
train_acc_lgb = accuracy_score(y_train_smote, train_pred_lgb)
test_acc_lgb = accuracy_score(y_test, test_pred_lgb)
cv_scores_lgb = cross_val_score(models['LightGBM'], X_train_smote, y_train_smote, cv=5, scoring='accuracy', n_jobs=-1)

print(f"  Train: {train_acc_lgb:.4f} ({train_acc_lgb*100:.2f}%)")
print(f"  Test: {test_acc_lgb:.4f} ({test_acc_lgb*100:.2f}%)")
print(f"  CV: {cv_scores_lgb.mean():.4f}±{cv_scores_lgb.std():.4f}")
print(f"  과적합: {(train_acc_lgb - test_acc_lgb)*100:.2f}%p")

# Voting Ensemble
print("\n[4] Voting Ensemble 학습 중...")
voting = VotingClassifier(
    estimators=[
        ('catboost', models['CatBoost']),
        ('xgboost', models['XGBoost']),
        ('lightgbm', models['LightGBM'])
    ],
    voting='soft',
    weights=[2, 1, 1]  # CatBoost에 높은 가중치
)
voting.fit(X_train_smote, y_train_smote)

train_pred_voting = voting.predict(X_train_smote)
test_pred_voting = voting.predict(X_test_scaled)
train_acc_voting = accuracy_score(y_train_smote, train_pred_voting)
test_acc_voting = accuracy_score(y_test, test_pred_voting)
cv_scores_voting = cross_val_score(voting, X_train_smote, y_train_smote, cv=5, scoring='accuracy', n_jobs=-1)

print(f"  Train: {train_acc_voting:.4f} ({train_acc_voting*100:.2f}%)")
print(f"  Test: {test_acc_voting:.4f} ({test_acc_voting*100:.2f}%)")
print(f"  CV: {cv_scores_voting.mean():.4f}±{cv_scores_voting.std():.4f}")
print(f"  과적합: {(train_acc_voting - test_acc_voting)*100:.2f}%p")

# ============================================================================
# 최고 모델 선택
# ============================================================================
results = {
    'CatBoost': test_acc_cb,
    'XGBoost': test_acc_xgb,
    'LightGBM': test_acc_lgb,
    'Voting': test_acc_voting
}

best_model_name = max(results, key=results.get)
best_model = models.get(best_model_name, voting)
best_accuracy = results[best_model_name]

print("\n" + "=" * 100)
print(f"✅ 최고 모델: {best_model_name} - {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print("=" * 100)

# 최종 예측
y_pred = best_model.predict(X_test_scaled)
y_pred_labels = le.inverse_transform(y_pred)
y_test_labels = le.inverse_transform(y_test)

# ============================================================================
# 최종 평가
# ============================================================================
print("\n" + "=" * 100)
print("📊 최종 성능 평가")
print("=" * 100)

print(f"\n최종 정확도: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")

# Confusion Matrix
cm = confusion_matrix(y_test_labels, y_pred_labels,
                      labels=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'])

print("\nConfusion Matrix:")
print("-" * 100)
print(f"{'':15} {'CANDIDATE':>12} {'CONFIRMED':>12} {'FALSE POS':>12}")
print("-" * 100)
for i, label in enumerate(['CANDIDATE', 'CONFIRMED', 'FALSE POS']):
    print(f"{label:15} {cm[i][0]:>12} {cm[i][1]:>12} {cm[i][2]:>12}")

print("\nClassification Report:")
print("-" * 100)
print(classification_report(y_test_labels, y_pred_labels,
                           labels=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'],
                           target_names=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']))

# 클래스별 정확도
print("\n클래스별 정확도:")
print("-" * 100)
for label in ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']:
    mask = y_test_labels == label
    if mask.sum() > 0:
        class_acc = accuracy_score(y_test_labels[mask], y_pred_labels[mask])
        print(f"  {label:20} {class_acc:.4f} ({class_acc*100:.2f}%)  [{mask.sum():,}개 샘플]")

# ============================================================================
# 모델 저장
# ============================================================================
print("\n" + "=" * 100)
print("💾 모델 저장")
print("=" * 100)

save_dir = 'saved_models'
os.makedirs(save_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

model_path = os.path.join(save_dir, f'model_3class_{timestamp}.pkl')
scaler_path = os.path.join(save_dir, f'scaler_3class_{timestamp}.pkl')
encoder_path = os.path.join(save_dir, f'encoder_3class_{timestamp}.pkl')

joblib.dump(best_model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(le, encoder_path)

config = {
    'model_name': best_model_name,
    'accuracy': best_accuracy,
    'timestamp': timestamp,
    'feature_count': X_full.shape[1],
    'classes': le.classes_.tolist()
}

config_path = os.path.join(save_dir, f'config_3class_{timestamp}.pkl')
joblib.dump(config, config_path)

print("✅ 모델 저장 완료")
print(f"  • {model_path}")
print(f"  • {scaler_path}")
print(f"  • {encoder_path}")
print(f"  • {config_path}")

print("\n" + "=" * 100)
print("🎯 최종 결과")
print("=" * 100)
print(f"모델: {best_model_name}")
print(f"정확도: {best_accuracy*100:.2f}%")

if best_accuracy >= 0.95:
    print("\n🎉🎉🎉 95% 목표 달성! 🎉🎉🎉")
elif best_accuracy >= 0.90:
    print(f"\n💪 90% 이상 달성! 목표까지 {(0.95-best_accuracy)*100:.2f}%p")
else:
    print(f"\n📊 현재 수준: {best_accuracy*100:.2f}% (목표: 95.00%)")
    print(f"   개선 필요: {(0.95-best_accuracy)*100:.2f}%p")

print("\n" + "=" * 100)
