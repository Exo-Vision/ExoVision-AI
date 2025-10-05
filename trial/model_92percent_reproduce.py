"""
92% 모델 재현 (PCA 제외)
- 26개 피처 (기본 16 + 엔지니어링 10)
- 강한 정규화
- Voting Ensemble
- CONFIRMED vs FALSE POSITIVE (2-클래스)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 100)
print("🎯 92% 모델 재현 (PCA 제외, 강한 정규화)")
print("=" * 100)

# ============================================================================
# 데이터 로드 및 피처 엔지니어링
# ============================================================================
print("\n[데이터 로드 및 피처 엔지니어링]")
print("-" * 100)

df = pd.read_csv('datasets/exoplanets.csv')
print(f"원본 데이터: {df.shape[0]:,} 샘플")

# 타겟 분리
y_full = df['koi_disposition']
X_full = df.drop(['koi_disposition', 'kepoi_name'], axis=1, errors='ignore')

# 수치형 컬럼만 선택
numeric_cols = X_full.select_dtypes(include=[np.number]).columns
X_full = X_full[numeric_cols]
print(f"기본 피처: {len(numeric_cols)}개")

# ============================================================================
# 피처 엔지니어링 (92% 달성 모델과 동일)
# ============================================================================
print("\n피처 엔지니어링 중...")

# 1. 행성-항성 비율
X_full['planet_star_ratio'] = X_full['koi_prad'] / (X_full['koi_srad'] + 1e-10)

# 2. 궤도 에너지
X_full['orbital_energy'] = 1.0 / (X_full['koi_sma'] + 1e-10)

# 3. 통과 신호 강도
X_full['transit_signal'] = X_full['koi_depth'] * X_full['koi_duration']

# 4. 항성 밀도
X_full['stellar_density'] = X_full['koi_smass'] / (X_full['koi_srad']**3 + 1e-10)

# 5. 행성 밀도 프록시
X_full['planet_density_proxy'] = X_full['koi_prad']**3 / (X_full['koi_sma']**2 + 1e-10)

# 6. Log 변환
X_full['log_period'] = np.log1p(X_full['koi_period'])
X_full['log_depth'] = np.log1p(X_full['koi_depth'])
X_full['log_insol'] = np.log1p(X_full['koi_insol'])

# 7. 궤도 안정성
X_full['orbit_stability'] = X_full['koi_eccen'] * X_full['koi_impact']

# 8. Transit SNR
X_full['transit_snr'] = X_full['koi_depth'] / (X_full['koi_duration'] + 1e-10)

# NaN 처리
X_full = X_full.replace([np.inf, -np.inf], np.nan)
X_full = X_full.fillna(X_full.median())

print(f"최종 피처 수: {X_full.shape[1]}개")
print(f"추가된 피처: planet_star_ratio, orbital_energy, transit_signal, stellar_density,")
print(f"            planet_density_proxy, log_period, log_depth, log_insol,")
print(f"            orbit_stability, transit_snr")

# ============================================================================
# 2-클래스 분류: CONFIRMED vs FALSE POSITIVE
# ============================================================================
print("\n" + "=" * 100)
print("[2-클래스 분류: CONFIRMED vs FALSE POSITIVE]")
print("=" * 100)

# CANDIDATE 제외
y_binary = y_full[y_full.isin(['CONFIRMED', 'FALSE POSITIVE'])]
X_binary = X_full.loc[y_binary.index]

print(f"\n학습 데이터:")
print(f"  총 샘플: {len(y_binary):,}")
for label, count in y_binary.value_counts().sort_index().items():
    print(f"  {label}: {count:,} ({count/len(y_binary)*100:.1f}%)")

# 레이블 인코딩 (CONFIRMED=1, FALSE POSITIVE=0)
y_binary_encoded = (y_binary == 'CONFIRMED').astype(int)

# Train/Test 분할 (90/10)
X_train, X_test, y_train, y_test = train_test_split(
    X_binary, y_binary_encoded, test_size=0.1, random_state=42, stratify=y_binary_encoded
)

print(f"\nTrain: {len(y_train):,} / Test: {len(y_test):,}")

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ============================================================================
# 모델 학습 (강한 정규화)
# ============================================================================
print("\n모델 학습 중 (강한 정규화)...")
print("-" * 100)

models = {
    'XGBoost': XGBClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=2.0,
        reg_lambda=10.0,
        random_state=42,
        eval_metric='logloss'
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=500,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.7,
        colsample_bytree=0.7,
        reg_alpha=2.0,
        reg_lambda=10.0,
        random_state=42,
        verbose=-1
    ),
    'CatBoost': CatBoostClassifier(
        iterations=500,
        depth=4,
        learning_rate=0.02,
        l2_leaf_reg=10.0,
        bagging_temperature=1.0,
        random_state=42,
        verbose=False
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=500,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
}

results = {}

for name, model in models.items():
    print(f"\n{name} 학습 중...")
    
    # 학습
    model.fit(X_train_scaled, y_train)
    
    # 예측
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)
    y_test_proba = model.predict_proba(X_test_scaled)
    
    # 정확도
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy')
    
    # AUC
    auc = roc_auc_score(y_test, y_test_proba[:, 1])
    
    results[name] = {
        'model': model,
        'train_acc': train_acc,
        'test_acc': test_acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'overfitting': train_acc - test_acc,
        'auc': auc
    }
    
    print(f"  Train: {train_acc:.4f} | Test: {test_acc:.4f} | CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f}")
    print(f"  과적합: {train_acc - test_acc:.4f} ({(train_acc - test_acc)*100:.2f}%p) | AUC: {auc:.4f}")

# ============================================================================
# Voting Ensemble
# ============================================================================
print("\n" + "=" * 100)
print("Voting Ensemble 학습 중...")
print("=" * 100)

voting = VotingClassifier(
    estimators=[(name, model) for name, model in models.items()],
    voting='soft'
)

voting.fit(X_train_scaled, y_train)

y_train_pred_voting = voting.predict(X_train_scaled)
y_test_pred_voting = voting.predict(X_test_scaled)
y_test_proba_voting = voting.predict_proba(X_test_scaled)

train_acc_voting = accuracy_score(y_train, y_train_pred_voting)
test_acc_voting = accuracy_score(y_test, y_test_pred_voting)
cv_scores_voting = cross_val_score(voting, X_train_scaled, y_train, cv=5, scoring='accuracy')
auc_voting = roc_auc_score(y_test, y_test_proba_voting[:, 1])

print(f"\n  Train: {train_acc_voting:.4f} | Test: {test_acc_voting:.4f} | CV: {cv_scores_voting.mean():.4f}±{cv_scores_voting.std():.4f}")
print(f"  과적합: {train_acc_voting - test_acc_voting:.4f} ({(train_acc_voting - test_acc_voting)*100:.2f}%p) | AUC: {auc_voting:.4f}")

results['Voting'] = {
    'model': voting,
    'train_acc': train_acc_voting,
    'test_acc': test_acc_voting,
    'cv_mean': cv_scores_voting.mean(),
    'cv_std': cv_scores_voting.std(),
    'overfitting': train_acc_voting - test_acc_voting,
    'auc': auc_voting
}

# ============================================================================
# 결과 요약
# ============================================================================
print("\n" + "=" * 100)
print("📊 결과 요약")
print("=" * 100)

results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Train Acc': [r['train_acc'] for r in results.values()],
    'Test Acc': [r['test_acc'] for r in results.values()],
    'CV Mean': [r['cv_mean'] for r in results.values()],
    'CV Std': [r['cv_std'] for r in results.values()],
    'Overfitting': [r['overfitting'] for r in results.values()],
    'AUC': [r['auc'] for r in results.values()]
})

results_df = results_df.sort_values('Test Acc', ascending=False)
print(results_df.to_string(index=False))

# 최고 모델
best_model = max(results.items(), key=lambda x: x[1]['test_acc'])
print(f"\n✅ 최고 모델: {best_model[0]}")
print(f"   Test 정확도: {best_model[1]['test_acc']:.4f} ({best_model[1]['test_acc']*100:.2f}%)")
print(f"   과적합: {best_model[1]['overfitting']:.4f} ({best_model[1]['overfitting']*100:.2f}%p)")
print(f"   AUC: {best_model[1]['auc']:.4f}")

# ============================================================================
# 과적합 분석
# ============================================================================
print("\n" + "=" * 100)
print("🔍 과적합 분석")
print("=" * 100)

overfitting_threshold = 0.03
print(f"과적합 기준: {overfitting_threshold*100:.1f}%p 이상")

for name, r in results.items():
    status = "✅" if r['overfitting'] < overfitting_threshold else "⚠️"
    print(f"{status} {name:15} 과적합: {r['overfitting']:.4f} ({r['overfitting']*100:.2f}%p)")

overfitting_models = [name for name, r in results.items() if r['overfitting'] >= overfitting_threshold]
if overfitting_models:
    print(f"\n⚠️ 과적합 모델: {', '.join(overfitting_models)}")
else:
    print("\n✅ 모든 모델 과적합 없음!")

# ============================================================================
# Confusion Matrix 및 상세 평가
# ============================================================================
print("\n" + "=" * 100)
print("📈 최종 평가")
print("=" * 100)

# 최고 모델로 예측
best_model_obj = best_model[1]['model']
y_test_pred_best = best_model_obj.predict(X_test_scaled)

# Confusion Matrix
cm = confusion_matrix(y_test, y_test_pred_best)
class_names = ['FALSE POSITIVE', 'CONFIRMED']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title(f'Confusion Matrix - {best_model[0]} ({best_model[1]["test_acc"]:.4f})')
plt.ylabel('실제')
plt.xlabel('예측')
plt.tight_layout()
plt.savefig('confusion_matrix_92percent.png', dpi=150, bbox_inches='tight')
print("\n✅ Confusion Matrix 저장: confusion_matrix_92percent.png")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_test_pred_best, target_names=class_names))

# 클래스별 정확도
print("\n클래스별 정확도:")
for i, label in enumerate(class_names):
    mask = y_test == i
    if mask.sum() > 0:
        class_acc = accuracy_score(y_test[mask], y_test_pred_best[mask])
        print(f"  {label:20} {class_acc:.4f} ({class_acc*100:.2f}%)  [{mask.sum()}개 샘플]")

# ============================================================================
# 목표 달성 평가
# ============================================================================
print("\n" + "=" * 100)
print("🎯 목표 달성 평가")
print("=" * 100)

target_acc = 0.90
best_acc = best_model[1]['test_acc']
gap = best_acc - target_acc

print(f"목표: {target_acc*100:.2f}%")
print(f"달성: {best_acc*100:.2f}%")
print(f"격차: {gap*100:+.2f}%p")

if best_acc >= 0.95:
    print("\n🎉🎉🎉 95% 달성! 🎉🎉🎉")
elif best_acc >= 0.92:
    print("\n🎊🎊 92% 이상 달성! 우수한 성능! 🎊🎊")
elif best_acc >= 0.90:
    print("\n💪 90% 이상 달성! 목표 달성!")
else:
    print(f"\n📊 추가 개선 필요: {(0.90 - best_acc)*100:.2f}%p")

print("\n" + "=" * 100)
print("✅ 분석 완료")
print("=" * 100)
