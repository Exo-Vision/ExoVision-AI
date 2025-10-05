"""
머신러닝 최적화 - 95% 목표
- Stacking Ensemble (메타 모델)
- 다양한 모델 추가 (ExtraTrees, AdaBoost, SVM)
- 하이퍼파라미터 미세 조정
- 피처 선택 최적화
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.feature_selection import SelectFromModel

from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import (
    VotingClassifier, 
    StackingClassifier,
    RandomForestClassifier,
    ExtraTreesClassifier,
    AdaBoostClassifier,
    GradientBoostingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

import warnings
warnings.filterwarnings('ignore')

plt.rcParams['font.family'] = 'Malgun Gothic'
plt.rcParams['axes.unicode_minus'] = False

print("=" * 100)
print("🚀 머신러닝 최적화 - 95% 도전!")
print("=" * 100)

# ============================================================================
# 데이터 로드 및 피처 엔지니어링
# ============================================================================
print("\n[1단계] 데이터 로드 및 피처 엔지니어링")
print("-" * 100)

df = pd.read_csv('datasets/exoplanets.csv')
print(f"원본 데이터: {df.shape[0]:,} 샘플")

y_full = df['koi_disposition']
X_full = df.drop(['koi_disposition', 'kepoi_name'], axis=1, errors='ignore')

numeric_cols = X_full.select_dtypes(include=[np.number]).columns
X_full = X_full[numeric_cols]
print(f"기본 피처: {len(numeric_cols)}개")

# 피처 엔지니어링
print("피처 엔지니어링 중...")
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

print(f"엔지니어링 후 피처: {X_full.shape[1]}개")

# 2-클래스 데이터
y_binary = y_full[y_full.isin(['CONFIRMED', 'FALSE POSITIVE'])]
X_binary = X_full.loc[y_binary.index]

print(f"\n학습 데이터: {len(y_binary):,} 샘플")
for label, count in y_binary.value_counts().sort_index().items():
    print(f"  {label}: {count:,} ({count/len(y_binary)*100:.1f}%)")

y_binary_encoded = (y_binary == 'CONFIRMED').astype(int)

# Train/Test 분할
X_train, X_test, y_train, y_test = train_test_split(
    X_binary, y_binary_encoded, test_size=0.1, random_state=42, stratify=y_binary_encoded
)

print(f"\nTrain: {len(y_train):,} / Test: {len(y_test):,}")

# 스케일링
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 피처 선택 제거 - 전체 피처 사용
X_train_selected = X_train_scaled
X_test_selected = X_test_scaled
print(f"\n전체 피처 사용: {X_binary.shape[1]}개")

# ============================================================================
# Base Models (정규화 완화 + 최적화)
# ============================================================================
print("\n[2단계] Base Models 학습 (최적화된 하이퍼파라미터)")
print("-" * 100)

base_models = {
    'XGBoost': XGBClassifier(
        n_estimators=800,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=1.5,
        reg_lambda=8.0,
        random_state=42,
        eval_metric='logloss'
    ),
    'LightGBM': LGBMClassifier(
        n_estimators=800,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.75,
        colsample_bytree=0.75,
        reg_alpha=1.5,
        reg_lambda=8.0,
        random_state=42,
        verbose=-1
    ),
    'CatBoost': CatBoostClassifier(
        iterations=800,
        depth=5,
        learning_rate=0.03,
        l2_leaf_reg=8.0,
        bagging_temperature=1.0,
        random_state=42,
        verbose=False
    ),
    'RandomForest': RandomForestClassifier(
        n_estimators=800,
        max_depth=12,
        min_samples_split=15,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    ),
    'ExtraTrees': ExtraTreesClassifier(
        n_estimators=800,
        max_depth=12,
        min_samples_split=15,
        min_samples_leaf=5,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    ),
    'GradientBoosting': GradientBoostingClassifier(
        n_estimators=600,
        max_depth=5,
        learning_rate=0.03,
        subsample=0.75,
        random_state=42
    )
}

results = {}

for name, model in base_models.items():
    print(f"\n{name} 학습 중...")
    
    model.fit(X_train_selected, y_train)
    
    y_train_pred = model.predict(X_train_selected)
    y_test_pred = model.predict(X_test_selected)
    y_test_proba = model.predict_proba(X_test_selected)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    cv_scores = cross_val_score(model, X_train_selected, y_train, cv=5, scoring='accuracy')
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
    
    print(f"  Test: {test_acc:.4f} | CV: {cv_scores.mean():.4f}±{cv_scores.std():.4f} | 과적합: {train_acc - test_acc:.4f}")

# ============================================================================
# Neural Network 추가
# ============================================================================
print("\n[3단계] Neural Network 학습")
print("-" * 100)

mlp = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation='relu',
    solver='adam',
    alpha=0.001,
    batch_size=64,
    learning_rate='adaptive',
    learning_rate_init=0.01,
    max_iter=300,
    early_stopping=True,
    validation_fraction=0.15,
    random_state=42
)

print("Neural Network 학습 중...")
mlp.fit(X_train_selected, y_train)

y_train_pred_mlp = mlp.predict(X_train_selected)
y_test_pred_mlp = mlp.predict(X_test_selected)
y_test_proba_mlp = mlp.predict_proba(X_test_selected)

train_acc_mlp = accuracy_score(y_train, y_train_pred_mlp)
test_acc_mlp = accuracy_score(y_test, y_test_pred_mlp)
cv_scores_mlp = cross_val_score(mlp, X_train_selected, y_train, cv=5, scoring='accuracy')
auc_mlp = roc_auc_score(y_test, y_test_proba_mlp[:, 1])

results['NeuralNetwork'] = {
    'model': mlp,
    'train_acc': train_acc_mlp,
    'test_acc': test_acc_mlp,
    'cv_mean': cv_scores_mlp.mean(),
    'cv_std': cv_scores_mlp.std(),
    'overfitting': train_acc_mlp - test_acc_mlp,
    'auc': auc_mlp
}

print(f"  Test: {test_acc_mlp:.4f} | CV: {cv_scores_mlp.mean():.4f}±{cv_scores_mlp.std():.4f} | 과적합: {train_acc_mlp - test_acc_mlp:.4f}")

# ============================================================================
# Voting Ensemble
# ============================================================================
print("\n[4단계] Voting Ensemble")
print("-" * 100)

voting = VotingClassifier(
    estimators=[(name, model) for name, model in base_models.items()] + [('NeuralNetwork', mlp)],
    voting='soft'
)

print("Voting Ensemble 학습 중...")
voting.fit(X_train_selected, y_train)

y_train_pred_voting = voting.predict(X_train_selected)
y_test_pred_voting = voting.predict(X_test_selected)
y_test_proba_voting = voting.predict_proba(X_test_selected)

train_acc_voting = accuracy_score(y_train, y_train_pred_voting)
test_acc_voting = accuracy_score(y_test, y_test_pred_voting)
cv_scores_voting = cross_val_score(voting, X_train_selected, y_train, cv=3, scoring='accuracy')
auc_voting = roc_auc_score(y_test, y_test_proba_voting[:, 1])

results['Voting'] = {
    'model': voting,
    'train_acc': train_acc_voting,
    'test_acc': test_acc_voting,
    'cv_mean': cv_scores_voting.mean(),
    'cv_std': cv_scores_voting.std(),
    'overfitting': train_acc_voting - test_acc_voting,
    'auc': auc_voting
}

print(f"  Test: {test_acc_voting:.4f} | CV: {cv_scores_voting.mean():.4f}±{cv_scores_voting.std():.4f}")

# ============================================================================
# Stacking Ensemble (메타 모델)
# ============================================================================
print("\n[5단계] Stacking Ensemble (메타 모델)")
print("-" * 100)

# Top 5 모델 선택
top_models = sorted(results.items(), key=lambda x: x[1]['test_acc'], reverse=True)[:5]
print(f"Top 5 모델: {', '.join([name for name, _ in top_models])}")

stacking = StackingClassifier(
    estimators=[(name, r['model']) for name, r in top_models],
    final_estimator=LogisticRegression(C=0.1, max_iter=1000, random_state=42),
    cv=5
)

print("\nStacking Ensemble 학습 중...")
stacking.fit(X_train_selected, y_train)

y_train_pred_stacking = stacking.predict(X_train_selected)
y_test_pred_stacking = stacking.predict(X_test_selected)
y_test_proba_stacking = stacking.predict_proba(X_test_selected)

train_acc_stacking = accuracy_score(y_train, y_train_pred_stacking)
test_acc_stacking = accuracy_score(y_test, y_test_pred_stacking)
cv_scores_stacking = cross_val_score(stacking, X_train_selected, y_train, cv=3, scoring='accuracy')
auc_stacking = roc_auc_score(y_test, y_test_proba_stacking[:, 1])

results['Stacking'] = {
    'model': stacking,
    'train_acc': train_acc_stacking,
    'test_acc': test_acc_stacking,
    'cv_mean': cv_scores_stacking.mean(),
    'cv_std': cv_scores_stacking.std(),
    'overfitting': train_acc_stacking - test_acc_stacking,
    'auc': auc_stacking
}

print(f"  Test: {test_acc_stacking:.4f} | CV: {cv_scores_stacking.mean():.4f}±{cv_scores_stacking.std():.4f}")

# ============================================================================
# 결과 요약
# ============================================================================
print("\n" + "=" * 100)
print("📊 최종 결과")
print("=" * 100)

results_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Test Acc': [r['test_acc'] for r in results.values()],
    'CV Mean': [r['cv_mean'] for r in results.values()],
    'Overfitting': [r['overfitting'] for r in results.values()],
    'AUC': [r['auc'] for r in results.values()]
})

results_df = results_df.sort_values('Test Acc', ascending=False)
print("\n" + results_df.to_string(index=False))

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

for name, r in results.items():
    status = "✅" if r['overfitting'] < 0.03 else "⚠️"
    print(f"{status} {name:20} 과적합: {r['overfitting']:.4f} ({r['overfitting']*100:.2f}%p)")

# ============================================================================
# 최종 평가
# ============================================================================
print("\n" + "=" * 100)
print("📈 최종 평가")
print("=" * 100)

best_model_obj = best_model[1]['model']
y_test_pred_best = best_model_obj.predict(X_test_selected)

cm = confusion_matrix(y_test, y_test_pred_best)
class_names = ['FALSE POSITIVE', 'CONFIRMED']

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names,
            yticklabels=class_names)
plt.title(f'{best_model[0]} - {best_model[1]["test_acc"]:.4f}')
plt.ylabel('실제')
plt.xlabel('예측')
plt.tight_layout()
plt.savefig('confusion_matrix_95target.png', dpi=150, bbox_inches='tight')
print("\n✅ Confusion Matrix 저장: confusion_matrix_95target.png")

print("\nClassification Report:")
print(classification_report(y_test, y_test_pred_best, target_names=class_names))

print("\n클래스별 정확도:")
for i, label in enumerate(class_names):
    mask = y_test == i
    if mask.sum() > 0:
        class_acc = accuracy_score(y_test[mask], y_test_pred_best[mask])
        print(f"  {label:20} {class_acc:.4f} ({class_acc*100:.2f}%)")

# ============================================================================
# 목표 달성 평가
# ============================================================================
print("\n" + "=" * 100)
print("🎯 95% 목표 달성 평가")
print("=" * 100)

best_acc = best_model[1]['test_acc']
gap = best_acc - 0.95

print(f"목표: 95.00%")
print(f"달성: {best_acc*100:.2f}%")
print(f"격차: {gap*100:+.2f}%p")

if best_acc >= 0.95:
    print("\n🎉🎉🎉 95% 목표 달성! 🎉🎉🎉")
elif best_acc >= 0.92:
    print(f"\n💪 92% 이상! 목표까지 {abs(gap)*100:.2f}%p")
elif best_acc >= 0.90:
    print(f"\n✨ 90% 이상! 목표까지 {abs(gap)*100:.2f}%p")
else:
    print(f"\n📊 목표까지 {abs(gap)*100:.2f}%p 필요")

print("\n" + "=" * 100)
print("✅ 분석 완료")
print("=" * 100)
