"""
실제 데이터로 에러 컬럼 통합 예제
"""

import pandas as pd
import numpy as np

# K2 데이터 샘플 확인
print("=" * 100)
print("K2 데이터셋 에러 컬럼 실제 값 예시")
print("=" * 100)
print()

df_k2 = pd.read_csv('datasets/k2.csv', nrows=5)

# 궤도 주기 예시
print("🔹 궤도 주기 (pl_orbper) 예시:")
print()
for i in range(3):
    orbper = df_k2.loc[i, 'pl_orbper']
    err1 = df_k2.loc[i, 'pl_orbpererr1']
    err2 = df_k2.loc[i, 'pl_orbpererr2']
    lim = df_k2.loc[i, 'pl_orbperlim']
    
    print(f"행 {i+1}:")
    print(f"  측정값: {orbper:.8f} days")
    print(f"  상위 불확실성 (err1): +{err1:.8f} days")
    print(f"  하위 불확실성 (err2): {err2:.8f} days")
    print(f"  제한 플래그 (lim): {lim}")
    
    # 실제 범위 계산
    upper_bound = orbper + err1
    lower_bound = orbper + err2  # err2는 음수
    
    print(f"  → 실제 값 범위: {lower_bound:.8f} ~ {upper_bound:.8f} days")
    
    # 전략 1: 평균 에러
    avg_error = (abs(err1) + abs(err2)) / 2
    print(f"  → 평균 에러: ±{avg_error:.8f} days")
    
    # 전략 3: 최대 에러
    max_error = max(abs(err1), abs(err2))
    print(f"  → 최대 에러: ±{max_error:.8f} days")
    print()

print()
print("=" * 100)

# Kepler 데이터 샘플 확인
print("Kepler 데이터셋 에러 컬럼 실제 값 예시")
print("=" * 100)
print()

df_kepler = pd.read_csv('datasets/kepler.csv', nrows=5)

print("🔹 궤도 주기 (koi_period) 예시:")
print()
for i in range(3):
    period = df_kepler.loc[i, 'koi_period']
    err1 = df_kepler.loc[i, 'koi_period_err1']
    err2 = df_kepler.loc[i, 'koi_period_err2']
    
    print(f"행 {i+1}:")
    print(f"  측정값: {period:.8f} days")
    print(f"  상위 불확실성 (err1): +{err1:.10f} days")
    print(f"  하위 불확실성 (err2): {err2:.10f} days")
    print(f"  제한 플래그 (lim): 없음 (Kepler는 lim 컬럼 없음)")
    
    # 실제 범위 계산
    upper_bound = period + err1
    lower_bound = period + err2
    
    print(f"  → 실제 값 범위: {lower_bound:.8f} ~ {upper_bound:.8f} days")
    
    # 전략 1: 평균 에러
    avg_error = (abs(err1) + abs(err2)) / 2
    print(f"  → 평균 에러: ±{avg_error:.10f} days")
    print()

print()
print("=" * 100)
print("실제 통합 함수 적용 예제")
print("=" * 100)
print()


def merge_error_columns_demo(df, base_col, dataset_name):
    """에러 컬럼 통합 데모 함수"""
    
    # 데이터셋별 에러 컬럼명 패턴
    if dataset_name == 'kepler':
        err1_col = base_col + '_err1'
        err2_col = base_col + '_err2'
        lim_col = None
    else:  # k2, tess
        err1_col = base_col + 'err1'
        err2_col = base_col + 'err2'
        lim_col = base_col + 'lim'
    
    # 통합 전략 적용
    df_result = df.copy()
    
    # 전략 1: 평균 에러
    df_result[base_col + '_error_avg'] = (
        df_result[err1_col].abs() + df_result[err2_col].abs()
    ) / 2
    
    # 전략 3: 최대 에러
    df_result[base_col + '_error_max'] = np.maximum(
        df_result[err1_col].abs(),
        df_result[err2_col].abs()
    )
    
    # 전략 5: 분리 보존
    df_result[base_col + '_error_upper'] = df_result[err1_col].abs()
    df_result[base_col + '_error_lower'] = df_result[err2_col].abs()
    
    # lim 플래그가 있으면 추가
    if lim_col and lim_col in df.columns:
        df_result[base_col + '_limit_flag'] = df_result[lim_col]
    
    return df_result


# K2 데이터에 적용
print("📊 K2 데이터 통합 결과:")
print()
df_k2_merged = merge_error_columns_demo(df_k2.head(3), 'pl_orbper', 'k2')

result_cols = [
    'pl_orbper', 
    'pl_orbper_error_avg', 
    'pl_orbper_error_max',
    'pl_orbper_error_upper',
    'pl_orbper_error_lower',
    'pl_orbper_limit_flag'
]

print(df_k2_merged[result_cols].to_string())
print()
print()

# Kepler 데이터에 적용
print("📊 Kepler 데이터 통합 결과:")
print()
df_kepler_merged = merge_error_columns_demo(df_kepler.head(3), 'koi_period', 'kepler')

result_cols_kepler = [
    'koi_period',
    'koi_period_error_avg',
    'koi_period_error_max',
    'koi_period_error_upper',
    'koi_period_error_lower'
]

print(df_kepler_merged[result_cols_kepler].to_string())
print()
print()

print("=" * 100)
print("✅ 실제 데이터 분석 완료!")
print()
print("💡 요약:")
print("  - K2/TESS: err1, err2, lim 3개 컬럼 → 1~2개 통합 컬럼으로 축소 가능")
print("  - Kepler: _err1, _err2 2개 컬럼 → 1~2개 통합 컬럼으로 축소 가능")
print("  - 머신러닝용: 평균 에러 권장 (단순, 해석 가능)")
print("  - 정밀 분석용: 분리 보존 권장 (정보 손실 없음)")
