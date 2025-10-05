"""
저장된 2-모델 시스템 로드 및 예측
- 입력: 29개 피처 (19개 기본 + 10개 엔지니어링)
- 출력: 3-클래스 (CONFIRMED/FALSE POSITIVE/CANDIDATE)
- 기존 모델과 완전 호환
"""

import pandas as pd
import numpy as np
import joblib
import os
import glob
from datetime import datetime

print("=" * 100)
print("🔍 저장된 모델 로드 및 예측 시스템")
print("=" * 100)

# ============================================================================
# 최신 모델 자동 로드
# ============================================================================
print("\n[1] 저장된 모델 검색")
print("-" * 100)

save_dir = '../../saved_models'

# 최신 config 파일 찾기
config_files = glob.glob(os.path.join(save_dir, 'config_*.pkl'))
if not config_files:
    print("❌ 저장된 모델이 없습니다!")
    exit(1)

latest_config_path = max(config_files, key=os.path.getctime)
print(f"최신 config: {os.path.basename(latest_config_path)}")

# Config 로드
config = joblib.load(latest_config_path)
timestamp = config['timestamp']

print(f"\n모델 정보:")
print(f"  타임스탬프: {timestamp}")
print(f"  모델 1: {config['model1_name']} ({config['model1_accuracy']*100:.2f}%)")
print(f"  모델 2: {config['model2_name']} ({config['model2_accuracy']*100:.2f}%)")
print(f"  최종 정확도: {config['final_accuracy']*100:.2f}%")
print(f"  임계값: {config['best_threshold']}")
print(f"  피처 수: {config['feature_count']}개")

# ============================================================================
# 모델 및 스케일러 로드
# ============================================================================
print("\n[2] 모델 및 스케일러 로드")
print("-" * 100)

model1_path = os.path.join(save_dir, f'model1_ultra_{timestamp}.pkl')
scaler1_path = os.path.join(save_dir, f'scaler1_{timestamp}.pkl')
model2_path = os.path.join(save_dir, f'model2_ultra_{timestamp}.pkl')
scaler2_path = os.path.join(save_dir, f'scaler2_{timestamp}.pkl')

try:
    model1 = joblib.load(model1_path)
    scaler1 = joblib.load(scaler1_path)
    model2 = joblib.load(model2_path)
    scaler2 = joblib.load(scaler2_path)
    print("✅ 모든 모델 및 스케일러 로드 완료")
except Exception as e:
    print(f"❌ 로드 실패: {e}")
    exit(1)

# ============================================================================
# 피처 엔지니어링 함수 (입력 형식 유지)
# ============================================================================
def engineer_features(X_raw):
    """
    원본 19개 피처 → 44개 피처로 확장
    기존 모델 입력 형식 완전 호환
    """
    X = X_raw.copy()
    
    # 기본 10개 피처
    X['planet_star_ratio'] = X['koi_prad'] / (X['koi_srad'] + 1e-10)
    X['orbital_energy'] = 1.0 / (X['koi_sma'] + 1e-10)
    X['transit_signal'] = X['koi_depth'] * X['koi_duration']
    X['stellar_density'] = X['koi_smass'] / (X['koi_srad']**3 + 1e-10)
    X['planet_density_proxy'] = X['koi_prad']**3 / (X['koi_sma']**2 + 1e-10)
    X['log_period'] = np.log1p(X['koi_period'])
    X['log_depth'] = np.log1p(X['koi_depth'])
    X['log_insol'] = np.log1p(X['koi_insol'])
    X['orbit_stability'] = X['koi_eccen'] * X['koi_impact']
    X['transit_snr'] = X['koi_depth'] / (X['koi_duration'] + 1e-10)
    
    # 고급 15개 피처
    X['habitable_zone_score'] = 1.0 / (1.0 + np.abs(X['koi_insol'] - 1.0))
    X['temp_habitable_score'] = 1.0 / (1.0 + np.abs(X['koi_teq'] - 288) / 100)
    X['roche_limit'] = 2.46 * X['koi_srad'] * (X['koi_smass'] / (X['koi_prad'] / 109.0))**(1/3)
    X['hill_sphere'] = X['koi_sma'] * (X['koi_smass'] / 3.0)**(1/3)
    X['transit_probability'] = X['koi_srad'] / (X['koi_sma'] * 215.032 + 1e-10)
    X['improved_snr'] = (X['koi_depth'] * np.sqrt(X['koi_duration'])) / (X['koi_period'] + 1e-10)
    X['stability_index'] = (1 - X['koi_eccen']) * (1 - X['koi_impact'])
    X['mass_ratio'] = (X['koi_prad'] / 109.0)**3 / (X['koi_smass'] + 1e-10)
    X['tidal_heating'] = X['koi_eccen'] / (X['koi_sma']**3 + 1e-10)
    X['duration_ratio'] = X['koi_duration'] / (X['koi_period'] + 1e-10)
    X['radiation_balance'] = X['koi_insol'] * (X['koi_prad']**2) / (X['koi_sma']**2 + 1e-10)
    X['age_metallicity'] = X['koi_sage'] * (X['koi_smet'] + 2.5)
    X['depth_size_ratio'] = X['koi_depth'] / (X['koi_prad']**2 + 1e-10)
    X['kepler_ratio'] = X['koi_period']**2 / (X['koi_sma']**3 + 1e-10)
    X['depth_variability'] = X['koi_depth'] / (X['koi_duration'] * X['koi_period'] + 1e-10)
    
    # Inf, NaN 처리
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    return X

# ============================================================================
# 예측 함수 (2-모델 파이프라인)
# ============================================================================
def predict(X_raw):
    """
    2-모델 파이프라인 예측
    입력: 원본 19개 피처
    출력: 3-클래스 (CONFIRMED/FALSE POSITIVE/CANDIDATE)
    """
    # 피처 엔지니어링
    X_engineered = engineer_features(X_raw)
    
    # 스케일링
    X_scaled_1 = scaler1.transform(X_engineered)
    X_scaled_2 = scaler2.transform(X_engineered)
    
    # 모델 1 예측 (CONFIRMED vs FALSE POSITIVE)
    stage1_proba = model1.predict_proba(X_scaled_1)
    stage1_pred = model1.predict(X_scaled_1)
    
    # 모델 2 예측 (CANDIDATE 판별)
    stage2_pred = model2.predict(X_scaled_2)
    
    # 확신도 기반 최종 예측
    threshold = config['best_threshold']
    final_predictions = np.empty(len(X_raw), dtype=object)
    
    high_conf_mask = (stage1_proba.max(axis=1) >= threshold)
    final_predictions[high_conf_mask] = np.where(
        stage1_pred[high_conf_mask] == 1,
        'CONFIRMED',
        'FALSE POSITIVE'
    )
    
    low_conf_mask = ~high_conf_mask
    final_predictions[low_conf_mask] = np.where(
        stage2_pred[low_conf_mask] == 1,
        'CANDIDATE',
        np.where(stage1_pred[low_conf_mask] == 1, 'CONFIRMED', 'FALSE POSITIVE')
    )
    
    return final_predictions, stage1_proba.max(axis=1)

# ============================================================================
# 테스트: 데이터 로드 및 예측
# ============================================================================
print("\n[3] 테스트 데이터 로드 및 예측")
print("-" * 100)

# 데이터 로드
df = pd.read_csv('../../dataset/preprocessed_all.csv', low_memory=False)
print(f"데이터 로드: {df.shape[0]:,} 샘플")

# 타겟 분리
y_true = df['koi_disposition']
X_raw = df.drop(['koi_disposition', 'kepoi_name'], axis=1, errors='ignore')

# 수치형 컬럼만 (원본 19개)
numeric_cols = X_raw.select_dtypes(include=[np.number]).columns
X_raw = X_raw[numeric_cols]

print(f"입력 피처: {X_raw.shape[1]}개 (원본)")

# 예측 실행
print("\n예측 중...")
y_pred, confidence = predict(X_raw)

# ============================================================================
# 결과 평가
# ============================================================================
print("\n[4] 예측 결과")
print("-" * 100)

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 전체 정확도
accuracy = accuracy_score(y_true, y_pred)
print(f"\n✅ 전체 정확도: {accuracy:.4f} ({accuracy*100:.2f}%)")

# 예측 분포
print("\n예측 분포:")
for label in ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']:
    count = (y_pred == label).sum()
    print(f"  {label}: {count:,} ({count/len(y_pred)*100:.1f}%)")

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred,
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
print(classification_report(y_true, y_pred,
                           labels=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE'],
                           target_names=['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']))

# 클래스별 정확도
print("\n클래스별 정확도:")
print("-" * 100)
for label in ['CANDIDATE', 'CONFIRMED', 'FALSE POSITIVE']:
    mask = y_true == label
    if mask.sum() > 0:
        class_acc = accuracy_score(y_true[mask], y_pred[mask])
        print(f"  {label:20} {class_acc:.4f} ({class_acc*100:.2f}%)  [{mask.sum():,}개 샘플]")

# 확신도 분석
print("\n확신도 분석:")
print("-" * 100)
print(f"  평균 확신도: {confidence.mean():.4f}")
print(f"  중앙값: {np.median(confidence):.4f}")
print(f"  최소값: {confidence.min():.4f}")
print(f"  최대값: {confidence.max():.4f}")
print(f"  고확신(≥{config['best_threshold']}): {(confidence >= config['best_threshold']).sum():,}개 ({(confidence >= config['best_threshold']).sum()/len(confidence)*100:.1f}%)")

# ============================================================================
# 샘플 예측 (처음 10개)
# ============================================================================
print("\n[5] 샘플 예측 (처음 10개)")
print("-" * 100)
print(f"{'실제':<15} {'예측':<15} {'확신도':<10} {'일치':<10}")
print("-" * 100)

for i in range(min(10, len(y_true))):
    match = "✅" if y_true.iloc[i] == y_pred[i] else "❌"
    print(f"{y_true.iloc[i]:<15} {y_pred[i]:<15} {confidence[i]:.4f}    {match}")

# ============================================================================
# 새로운 데이터 예측 예시
# ============================================================================
print("\n[6] 새로운 데이터 예측 함수")
print("-" * 100)

def predict_new_samples(csv_path):
    """
    새로운 CSV 파일에서 데이터를 읽어 예측
    CSV는 19개 원본 피처만 포함하면 됨
    """
    # 데이터 로드
    df_new = pd.read_csv(csv_path, low_memory=False)
    
    # 타겟 제거 (있다면)
    if 'koi_disposition' in df_new.columns:
        df_new = df_new.drop('koi_disposition', axis=1)
    if 'kepoi_name' in df_new.columns:
        names = df_new['kepoi_name']
        df_new = df_new.drop('kepoi_name', axis=1)
    else:
        names = None
    
    # 수치형 컬럼만
    numeric_cols = df_new.select_dtypes(include=[np.number]).columns
    X_new = df_new[numeric_cols]
    
    # 예측
    predictions, confidences = predict(X_new)
    
    # 결과 데이터프레임
    result_df = pd.DataFrame({
        'kepoi_name': names if names is not None else range(len(predictions)),
        'prediction': predictions,
        'confidence': confidences
    })
    
    return result_df

print("사용 예시:")
print("  result = predict_new_samples('new_data.csv')")
print("  result.to_csv('predictions.csv', index=False)")

print("\n" + "=" * 100)
print("✅ 모델 로드 및 예측 시스템 준비 완료")
print("=" * 100)
print(f"\n입력 형식: 19개 원본 피처")
print(f"출력 형식: 3-클래스 (CONFIRMED/FALSE POSITIVE/CANDIDATE)")
print(f"정확도: {accuracy*100:.2f}%")
print(f"임계값: {config['best_threshold']}")
print("\n" + "=" * 100)
