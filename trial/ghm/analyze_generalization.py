"""
일반화 성능 분석 스크립트
- 데이터 분포 분석
- 학습 곡선 분석
- 클래스 불균형 확인
- 특징 간 상관관계 분석
- 모델 복잡도 vs 성능 분석
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score
from catboost import CatBoostClassifier
import warnings
warnings.filterwarnings('ignore')

print("="*100)
print("🔍 일반화 성능 분석")
print("="*100)

# ============================================================================
# 1. 데이터 로드 및 분석
# ============================================================================

print("\n" + "="*100)
print("📂 1. 데이터 분석")
print("="*100)

df = pd.read_csv('datasets/exoplanets.csv')
df_binary = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()

print(f"\n📊 클래스 분포:")
print(df_binary['koi_disposition'].value_counts())
print(f"\n비율:")
for label, count in df_binary['koi_disposition'].value_counts().items():
    print(f"  • {label}: {count/len(df_binary)*100:.2f}%")

# 클래스 불균형 비율
class_ratio = df_binary['koi_disposition'].value_counts().max() / df_binary['koi_disposition'].value_counts().min()
print(f"\n⚖️ 클래스 불균형 비율: {class_ratio:.2f}:1")

if class_ratio > 1.5:
    print("  ⚠️ 클래스 불균형이 존재합니다 (>1.5:1)")
else:
    print("  ✅ 클래스가 균형적입니다 (≤1.5:1)")

# ============================================================================
# 2. 특징 상관관계 분석
# ============================================================================

print("\n" + "="*100)
print("🔗 2. 특징 상관관계 분석")
print("="*100)

base_features = [
    'koi_prad', 'koi_teq', 'koi_insol', 'koi_period', 'koi_sma', 'koi_impact',
    'koi_eccen', 'koi_depth', 'koi_duration', 'koi_srad', 'koi_smass',
    'koi_steff', 'koi_slogg', 'koi_smet', 'ra', 'dec'
]

# 특징 엔지니어링 (동일하게 적용)
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

# 상관관계 매트릭스 계산
X = df_fe.drop('koi_disposition', axis=1)
corr_matrix = X.corr().abs()

# 높은 상관관계 찾기 (>0.8)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if corr_matrix.iloc[i, j] > 0.8:
            high_corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))

print(f"\n🔴 높은 상관관계 특징 쌍 (>0.8):")
if high_corr_pairs:
    for feat1, feat2, corr in sorted(high_corr_pairs, key=lambda x: -x[2]):
        print(f"  • {feat1:30s} ↔ {feat2:30s}: {corr:.3f}")
    print(f"\n  ⚠️ {len(high_corr_pairs)}개의 고상관 특징 쌍 발견!")
    print(f"  💡 다중공선성이 모델 성능을 저하시킬 수 있습니다.")
else:
    print("  ✅ 높은 상관관계 특징 없음 (다중공선성 없음)")

# 중간 상관관계 (0.6~0.8)
medium_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if 0.6 < corr_matrix.iloc[i, j] <= 0.8:
            medium_corr_pairs.append((
                corr_matrix.columns[i],
                corr_matrix.columns[j],
                corr_matrix.iloc[i, j]
            ))

print(f"\n🟡 중간 상관관계 특징 쌍 (0.6~0.8):")
if medium_corr_pairs:
    print(f"  • {len(medium_corr_pairs)}개의 중상관 특징 쌍 발견")
    print(f"  ⚡ 일부 중복 정보가 있을 수 있습니다.")
else:
    print("  ✅ 중간 상관관계 특징 없음")

# ============================================================================
# 3. 학습 곡선 분석 (데이터 크기 vs 성능)
# ============================================================================

print("\n" + "="*100)
print("📈 3. 학습 곡선 분석")
print("="*100)

y = df_fe['koi_disposition']
le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.1, random_state=42, stratify=y_encoded
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# CatBoost로 학습 곡선 생성
print("\n⏳ 학습 곡선 생성 중... (약 1분 소요)")
model = CatBoostClassifier(
    iterations=500,
    learning_rate=0.03,
    depth=6,
    l2_leaf_reg=2.0,
    random_state=42,
    verbose=False
)

train_sizes, train_scores, val_scores = learning_curve(
    model, X_train_scaled, y_train,
    train_sizes=np.linspace(0.1, 1.0, 10),
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
train_std = train_scores.std(axis=1)
val_mean = val_scores.mean(axis=1)
val_std = val_scores.std(axis=1)

print("\n📊 학습 곡선 결과:")
print("\n샘플 수       학습 정확도      검증 정확도      격차")
print("-" * 70)
for i, size in enumerate(train_sizes):
    gap = (train_mean[i] - val_mean[i]) * 100
    print(f"{int(size):6d}개     {train_mean[i]*100:6.2f}% ± {train_std[i]*100:4.2f}%   "
          f"{val_mean[i]*100:6.2f}% ± {val_std[i]*100:4.2f}%   {gap:5.2f}%p")

# 격차 분석
final_gap = (train_mean[-1] - val_mean[-1]) * 100
print(f"\n최종 격차 (전체 데이터): {final_gap:.2f}%p")

if final_gap > 5:
    print("  ⚠️ 높은 격차: 과적합 의심")
elif final_gap > 2:
    print("  ⚡ 중간 격차: 약간의 과적합")
else:
    print("  ✅ 낮은 격차: 양호한 일반화")

# 수렴 여부 확인
last_3_val = val_mean[-3:]
val_improvement = (last_3_val[-1] - last_3_val[0]) * 100
print(f"\n마지막 3개 구간 검증 성능 향상: {val_improvement:.2f}%p")

if val_improvement > 1:
    print("  📈 여전히 성능 향상 중 - 더 많은 데이터가 도움될 수 있습니다")
elif val_improvement > 0:
    print("  📊 약간 향상 중 - 데이터가 충분할 수 있습니다")
else:
    print("  📉 성능 정체 - 데이터보다 모델/특징 개선이 필요합니다")

# ============================================================================
# 4. 모델 복잡도 vs 성능 분석
# ============================================================================

print("\n" + "="*100)
print("🎯 4. 모델 복잡도 vs 성능 분석")
print("="*100)

print("\n⏳ 다양한 복잡도로 모델 학습 중...")

complexities = [
    {'iterations': 100, 'depth': 4, 'name': '매우 단순'},
    {'iterations': 200, 'depth': 5, 'name': '단순'},
    {'iterations': 500, 'depth': 6, 'name': '중간'},
    {'iterations': 1000, 'depth': 7, 'name': '복잡'},
    {'iterations': 1500, 'depth': 8, 'name': '매우 복잡'},
]

print("\n복잡도       학습 정확도    테스트 정확도    격차      과적합")
print("-" * 75)

best_test_acc = 0
best_complexity = None

for config in complexities:
    model = CatBoostClassifier(
        iterations=config['iterations'],
        learning_rate=0.03,
        depth=config['depth'],
        l2_leaf_reg=2.0,
        random_state=42,
        verbose=False
    )
    
    model.fit(X_train_scaled, y_train)
    train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
    test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
    gap = (train_acc - test_acc) * 100
    
    overfitting_status = "⚠️ 높음" if gap > 5 else "⚡ 중간" if gap > 2 else "✅ 낮음"
    
    print(f"{config['name']:12s} {train_acc*100:6.2f}%       {test_acc*100:6.2f}%       "
          f"{gap:5.2f}%p   {overfitting_status}")
    
    if test_acc > best_test_acc:
        best_test_acc = test_acc
        best_complexity = config['name']

print(f"\n🏆 최고 테스트 성능: {best_test_acc*100:.2f}% ({best_complexity} 복잡도)")

# ============================================================================
# 5. 종합 분석 및 권장사항
# ============================================================================

print("\n" + "="*100)
print("💡 5. 일반화 성능 향상 권장사항")
print("="*100)

recommendations = []

# 1. 클래스 불균형 문제
if class_ratio > 1.5:
    recommendations.append({
        'priority': '🔴 높음',
        'issue': '클래스 불균형',
        'detail': f'비율 {class_ratio:.2f}:1',
        'solution': [
            '• SMOTE/ADASYN 등 오버샘플링 기법 적용',
            '• class_weight="balanced" 파라미터 사용',
            '• focal loss 같은 불균형 손실함수 사용'
        ]
    })

# 2. 다중공선성 문제
if len(high_corr_pairs) > 0:
    recommendations.append({
        'priority': '🔴 높음',
        'issue': '다중공선성 (높은 특징 상관관계)',
        'detail': f'{len(high_corr_pairs)}개의 고상관 특징 쌍',
        'solution': [
            '• PCA로 차원 축소하여 독립적인 주성분 사용',
            '• VIF(Variance Inflation Factor) 기반 특징 선택',
            '• 상관도 높은 특징 쌍 중 하나 제거',
            '• Ridge/Lasso 정규화로 다중공선성 완화'
        ]
    })

# 3. 과적합 문제
if final_gap > 3:
    recommendations.append({
        'priority': '🟡 중간',
        'issue': '과적합 (Train-Validation 격차)',
        'detail': f'격차 {final_gap:.2f}%p',
        'solution': [
            '• 더 강한 정규화 (L2 증가, Dropout 추가)',
            '• 조기 종료(Early Stopping) 강화',
            '• 데이터 증강(Data Augmentation)',
            '• K-Fold 교차검증 fold 수 증가 (5→10)'
        ]
    })

# 4. 데이터 부족 문제
if val_improvement > 0.5:
    recommendations.append({
        'priority': '🟢 낮음',
        'issue': '데이터 규모',
        'detail': '학습 곡선이 수렴하지 않음',
        'solution': [
            '• CANDIDATE 클래스 포함하여 학습 데이터 증가',
            '• 다른 외계행성 데이터셋 병합 (K2, TESS 등)',
            '• 준지도 학습(Semi-supervised learning) 고려'
        ]
    })

# 5. 특징 엔지니어링 개선
if len(medium_corr_pairs) > 10:
    recommendations.append({
        'priority': '🟡 중간',
        'issue': '특징 중복성',
        'detail': f'{len(medium_corr_pairs)}개의 중상관 특징 쌍',
        'solution': [
            '• 특징 선택 알고리즘 적용 (RFE, SelectKBest)',
            '• AutoML로 최적 특징 조합 탐색',
            '• 물리적으로 독립적인 특징 우선 선택'
        ]
    })

# 6. 앙상블 다양성 향상
recommendations.append({
    'priority': '🟢 낮음',
    'issue': '앙상블 다양성 부족',
    'detail': '트리 기반 모델만 사용 중',
    'solution': [
        '• SVM, Neural Network 등 다른 모델군 추가',
        '• 베이지안 최적화로 하이퍼파라미터 다양화',
        '• Bagging에서 특징 서브샘플링 강화'
    ]
})

# 출력
print("\n" + "="*100)
for i, rec in enumerate(recommendations, 1):
    print(f"\n{rec['priority']} {i}. {rec['issue']}")
    print(f"{'':8s}📊 상태: {rec['detail']}")
    print(f"{'':8s}💡 해결책:")
    for sol in rec['solution']:
        print(f"{'':12s}{sol}")

print("\n" + "="*100)
print("✅ 분석 완료!")
print("="*100)
