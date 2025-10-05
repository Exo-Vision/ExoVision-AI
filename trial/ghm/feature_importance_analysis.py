"""
외계행성 판별 모델 Feature Importance 분석
- 각 모델의 특징 중요도 확인
- 특정 컬럼 편향 검사
- 균형잡힌 모델 평가
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

import matplotlib.pyplot as plt
import seaborn as sns

print("="*100)
print("🔍 Feature Importance 분석")
print("="*100)

# ============================================================================
# 1. 데이터 로드 및 전처리 (이전과 동일)
# ============================================================================

print("\n📂 데이터 준비...")
df = pd.read_csv('datasets/exoplanets.csv')
df_binary = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()

# 기본 특징
base_features = [
    'koi_prad', 'koi_teq', 'koi_insol', 'koi_period', 'koi_sma', 'koi_impact',
    'koi_eccen', 'koi_depth', 'koi_duration', 'koi_srad', 'koi_smass',
    'koi_steff', 'koi_slogg', 'koi_smet', 'ra', 'dec'
]

# 특징 엔지니어링
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

df_clean = df_fe.dropna()

feature_cols = [col for col in df_clean.columns if col != 'koi_disposition']
X = df_clean[feature_cols]
y = df_clean['koi_disposition']

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.1, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"특징 수: {len(feature_cols)}개")
print(f"학습 샘플: {X_train.shape[0]}개")

# ============================================================================
# 2. 모델별 Feature Importance 추출
# ============================================================================

print("\n" + "="*100)
print("🤖 모델 학습 및 Feature Importance 추출")
print("="*100)

# 저장용 딕셔너리
feature_importance_dict = {}

# ============================================================================
# 2.1 CatBoost (최고 성능 모델)
# ============================================================================

print("\n🔹 CatBoost...")
catboost_model = CatBoostClassifier(
    iterations=500,
    depth=10,
    learning_rate=0.03,
    l2_leaf_reg=5.0,
    random_state=42,
    verbose=0
)

catboost_model.fit(X_train_scaled, y_train)
catboost_acc = accuracy_score(y_test, catboost_model.predict(X_test_scaled))

# Feature importance 추출
catboost_importance = catboost_model.feature_importances_
feature_importance_dict['CatBoost'] = dict(zip(feature_cols, catboost_importance))

print(f"정확도: {catboost_acc*100:.2f}%")

# ============================================================================
# 2.2 XGBoost
# ============================================================================

print("\n🔹 XGBoost...")
xgb_model = XGBClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.03,
    random_state=42,
    n_jobs=-1,
    eval_metric='logloss'
)

xgb_model.fit(X_train_scaled, y_train)
xgb_acc = accuracy_score(y_test, xgb_model.predict(X_test_scaled))

# Feature importance 추출
xgb_importance = xgb_model.feature_importances_
feature_importance_dict['XGBoost'] = dict(zip(feature_cols, xgb_importance))

print(f"정확도: {xgb_acc*100:.2f}%")

# ============================================================================
# 2.3 LightGBM
# ============================================================================

print("\n🔹 LightGBM...")
lgbm_model = LGBMClassifier(
    n_estimators=500,
    max_depth=10,
    learning_rate=0.03,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

lgbm_model.fit(X_train_scaled, y_train)
lgbm_acc = accuracy_score(y_test, lgbm_model.predict(X_test_scaled))

# Feature importance 추출
lgbm_importance = lgbm_model.feature_importances_
feature_importance_dict['LightGBM'] = dict(zip(feature_cols, lgbm_importance))

print(f"정확도: {lgbm_acc*100:.2f}%")

# ============================================================================
# 2.4 Random Forest
# ============================================================================

print("\n🔹 Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(X_train_scaled, y_train)
rf_acc = accuracy_score(y_test, rf_model.predict(X_test_scaled))

# Feature importance 추출
rf_importance = rf_model.feature_importances_
feature_importance_dict['Random Forest'] = dict(zip(feature_cols, rf_importance))

print(f"정확도: {rf_acc*100:.2f}%")

# ============================================================================
# 3. Feature Importance 분석
# ============================================================================

print("\n" + "="*100)
print("📊 Feature Importance 상세 분석")
print("="*100)

# 각 모델별 Top 10 특징
print("\n🏆 각 모델별 Top 10 중요 특징:")
print("="*100)

for model_name, importance_dict in feature_importance_dict.items():
    print(f"\n🔹 {model_name}")
    print("-"*100)
    
    # 정규화 (합이 100%가 되도록)
    total_importance = sum(importance_dict.values())
    normalized_importance = {k: (v/total_importance)*100 for k, v in importance_dict.items()}
    
    # 정렬
    sorted_features = sorted(normalized_importance.items(), key=lambda x: x[1], reverse=True)
    
    # Top 10 출력
    for i, (feature, importance) in enumerate(sorted_features[:10], 1):
        print(f"  {i:2d}. {feature:<30} {importance:>6.2f}%")
    
    # 나머지 특징들의 합
    remaining_importance = sum([imp for feat, imp in sorted_features[10:]])
    print(f"  {'기타 (16개)':>33} {remaining_importance:>6.2f}%")
    
    # 편향 검사
    top1_importance = sorted_features[0][1]
    top3_importance = sum([imp for feat, imp in sorted_features[:3]])
    top5_importance = sum([imp for feat, imp in sorted_features[:5]])
    
    print(f"\n  📊 집중도 분석:")
    print(f"     - Top 1 특징 비율: {top1_importance:.2f}%", end="")
    if top1_importance > 30:
        print(" ⚠️ 과도한 집중!")
    elif top1_importance > 20:
        print(" ⚡ 높은 편")
    else:
        print(" ✅ 양호")
    
    print(f"     - Top 3 특징 비율: {top3_importance:.2f}%", end="")
    if top3_importance > 60:
        print(" ⚠️ 과도한 집중!")
    elif top3_importance > 50:
        print(" ⚡ 높은 편")
    else:
        print(" ✅ 양호")
    
    print(f"     - Top 5 특징 비율: {top5_importance:.2f}%", end="")
    if top5_importance > 70:
        print(" ⚠️ 과도한 집중!")
    elif top5_importance > 60:
        print(" ⚡ 높은 편")
    else:
        print(" ✅ 양호")

# ============================================================================
# 4. 특징 중요도 비교 (모델 간)
# ============================================================================

print("\n" + "="*100)
print("🔄 모델 간 Feature Importance 비교")
print("="*100)

# 모든 모델의 평균 중요도 계산
avg_importance = {}
for feature in feature_cols:
    importances = []
    for model_name, importance_dict in feature_importance_dict.items():
        # 정규화
        total = sum(importance_dict.values())
        normalized = (importance_dict[feature] / total) * 100
        importances.append(normalized)
    
    avg_importance[feature] = {
        'mean': np.mean(importances),
        'std': np.std(importances),
        'values': importances
    }

# 평균 중요도 정렬
sorted_avg_importance = sorted(avg_importance.items(), key=lambda x: x[1]['mean'], reverse=True)

print("\n📊 평균 Feature Importance (4개 모델 평균):")
print("-"*100)
print(f"{'순위':<5} {'특징':<30} {'평균 중요도':<15} {'표준편차':<15} {'일관성':<10}")
print("-"*100)

for i, (feature, stats) in enumerate(sorted_avg_importance[:15], 1):
    consistency = "높음" if stats['std'] < 2 else ("중간" if stats['std'] < 4 else "낮음")
    consistency_emoji = "✅" if stats['std'] < 2 else ("⚡" if stats['std'] < 4 else "⚠️")
    
    print(f"{i:<5} {feature:<30} {stats['mean']:>6.2f}%        {stats['std']:>6.2f}%        "
          f"{consistency_emoji} {consistency}")

# ============================================================================
# 5. 특징 중요도 균형 평가
# ============================================================================

print("\n" + "="*100)
print("⚖️ Feature Importance 균형 평가")
print("="*100)

# Gini 계수 계산 (불평등 지수)
def calculate_gini_coefficient(importances):
    """Gini 계수: 0 (완전 균등) ~ 1 (완전 불균등)"""
    importances = np.sort(np.array(importances))
    n = len(importances)
    index = np.arange(1, n + 1)
    return (2 * np.sum(index * importances)) / (n * np.sum(importances)) - (n + 1) / n

print("\n📊 각 모델의 특징 중요도 불균형 지수 (Gini Coefficient):")
print("-"*100)
print(f"{'모델':<25} {'Gini 계수':<15} {'균형 평가':<20} {'정확도':<10}")
print("-"*100)

gini_results = []
accuracies = {
    'CatBoost': catboost_acc,
    'XGBoost': xgb_acc,
    'LightGBM': lgbm_acc,
    'Random Forest': rf_acc
}

for model_name, importance_dict in feature_importance_dict.items():
    importances = list(importance_dict.values())
    gini = calculate_gini_coefficient(importances)
    
    # 균형 평가
    if gini < 0.3:
        balance = "✅ 매우 균형적"
    elif gini < 0.5:
        balance = "⚡ 다소 불균형"
    else:
        balance = "⚠️ 심각한 불균형"
    
    accuracy = accuracies[model_name]
    gini_results.append((model_name, gini, balance, accuracy))
    
    print(f"{model_name:<25} {gini:>6.4f}         {balance:<20} {accuracy*100:>6.2f}%")

print("\n💡 해석:")
print("  • Gini 계수 < 0.3: 특징들이 균형있게 사용됨 (좋음)")
print("  • Gini 계수 0.3-0.5: 일부 특징에 치우침 (보통)")
print("  • Gini 계수 > 0.5: 소수 특징에 과도하게 의존 (나쁨)")

# ============================================================================
# 6. 시각화 (Feature Importance)
# ============================================================================

print("\n" + "="*100)
print("📈 Feature Importance 시각화")
print("="*100)

# Top 15 특징만 시각화
top_n = 15
top_features = [feat for feat, stats in sorted_avg_importance[:top_n]]

# 각 모델의 Top 15 특징 중요도 추출
importance_df = pd.DataFrame()
for model_name, importance_dict in feature_importance_dict.items():
    total = sum(importance_dict.values())
    normalized = {k: (v/total)*100 for k, v in importance_dict.items()}
    
    model_importances = [normalized.get(feat, 0) for feat in top_features]
    importance_df[model_name] = model_importances

importance_df.index = top_features

# 플롯 생성
fig, axes = plt.subplots(2, 2, figsize=(20, 16))
fig.suptitle('Feature Importance 비교 (상위 15개 특징)', fontsize=20, fontweight='bold')

models = list(feature_importance_dict.keys())
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

for idx, (ax, model_name) in enumerate(zip(axes.flat, models)):
    importance_values = importance_df[model_name].sort_values(ascending=True)
    
    bars = ax.barh(range(len(importance_values)), importance_values, color=colors[idx], alpha=0.7)
    ax.set_yticks(range(len(importance_values)))
    ax.set_yticklabels(importance_values.index, fontsize=10)
    ax.set_xlabel('중요도 (%)', fontsize=12)
    ax.set_title(f'{model_name} (정확도: {accuracies[model_name]*100:.2f}%)', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    # 값 표시
    for i, (bar, value) in enumerate(zip(bars, importance_values)):
        ax.text(value + 0.3, i, f'{value:.1f}%', va='center', fontsize=9)

plt.tight_layout()
plt.savefig('feature_importance_comparison.png', dpi=150, bbox_inches='tight')
print("\n✅ 시각화 저장: feature_importance_comparison.png")

# ============================================================================
# 7. 평균 Feature Importance 시각화
# ============================================================================

fig, ax = plt.subplots(figsize=(12, 10))

avg_importances = [stats['mean'] for feat, stats in sorted_avg_importance[:top_n]]
std_importances = [stats['std'] for feat, stats in sorted_avg_importance[:top_n]]
feature_names = [feat for feat, stats in sorted_avg_importance[:top_n]]

# 내림차순 정렬
indices = np.argsort(avg_importances)
avg_importances_sorted = [avg_importances[i] for i in indices]
std_importances_sorted = [std_importances[i] for i in indices]
feature_names_sorted = [feature_names[i] for i in indices]

bars = ax.barh(range(len(avg_importances_sorted)), avg_importances_sorted, 
               xerr=std_importances_sorted, color='#3498db', alpha=0.7, 
               error_kw={'linewidth': 2, 'ecolor': '#e74c3c'})

ax.set_yticks(range(len(feature_names_sorted)))
ax.set_yticklabels(feature_names_sorted, fontsize=11)
ax.set_xlabel('평균 중요도 (%) ± 표준편차', fontsize=13, fontweight='bold')
ax.set_title('평균 Feature Importance (4개 모델)', fontsize=16, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# 값 표시
for i, (bar, value, std) in enumerate(zip(bars, avg_importances_sorted, std_importances_sorted)):
    ax.text(value + std + 0.5, i, f'{value:.1f}%', va='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('feature_importance_average.png', dpi=150, bbox_inches='tight')
print("✅ 시각화 저장: feature_importance_average.png")

# ============================================================================
# 8. 최종 권장사항
# ============================================================================

print("\n" + "="*100)
print("💡 최종 분석 및 권장사항")
print("="*100)

# 평균 Gini 계수
avg_gini = np.mean([result[1] for result in gini_results])
avg_accuracy = np.mean([result[3] for result in gini_results])

print(f"\n📊 전체 요약:")
print(f"  • 평균 Gini 계수: {avg_gini:.4f}")
print(f"  • 평균 정확도: {avg_accuracy*100:.2f}%")

# 상위 3개 특징의 평균 비율
top3_avg_importance = sum([stats['mean'] for feat, stats in sorted_avg_importance[:3]])
print(f"  • Top 3 특징 평균 비율: {top3_avg_importance:.2f}%")

print(f"\n🔍 모델 편향 진단:")

if avg_gini < 0.3 and top3_avg_importance < 40:
    print("  ✅ 매우 우수: 특징들이 균형있게 사용되고 있습니다!")
    print("     - 다양한 특징을 활용하여 robustness가 높습니다.")
    print("     - 과적합 위험이 낮습니다.")
    
elif avg_gini < 0.5 and top3_avg_importance < 50:
    print("  ⚡ 양호: 일부 특징에 치우쳐 있지만 허용 가능한 수준입니다.")
    print("     - 주요 특징들이 실제로 중요한 물리량일 수 있습니다.")
    print("     - 교차 검증으로 과적합을 계속 모니터링하세요.")
    
else:
    print("  ⚠️ 주의 필요: 소수 특징에 과도하게 의존하고 있습니다!")
    print("     - 특정 특징에 대한 과적합 위험이 있습니다.")
    print("     - 정규화 강화 또는 특징 선택 재검토가 필요합니다.")

print(f"\n🎯 가장 중요한 특징 Top 5:")
for i, (feature, stats) in enumerate(sorted_avg_importance[:5], 1):
    print(f"  {i}. {feature:<30} {stats['mean']:>6.2f}% ± {stats['std']:>4.2f}%")

print(f"\n📚 물리적 해석:")

# Top 특징들의 물리적 의미
feature_meanings = {
    'koi_depth': '통과 깊이 - 행성 크기 직접 측정',
    'koi_duration': '통과 지속시간 - 궤도 기하학',
    'koi_prad': '행성 반지름 - 행성 크기',
    'koi_period': '궤도 주기 - 궤도 특성',
    'transit_signal': '통과 신호 적분 - 신호 강도',
    'koi_impact': '충격 매개변수 - 궤도 정렬',
    'koi_sma': '반장축 - 궤도 거리',
    'koi_teq': '평형 온도 - 행성 온도',
    'planet_star_ratio': '행성/항성 비율 - 상대 크기',
    'koi_insol': '입사 플럭스 - 에너지 수준'
}

for i, (feature, stats) in enumerate(sorted_avg_importance[:5], 1):
    meaning = feature_meanings.get(feature, '특징 엔지니어링 변수')
    print(f"  {i}. {feature}: {meaning}")

print("\n" + "="*100)
print("✅ Feature Importance 분석 완료!")
print("="*100)
