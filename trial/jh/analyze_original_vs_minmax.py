"""
원본 값 vs min/max 분석
원본 값이 (min + max) / 2와 같은지 확인
"""

from pathlib import Path

import numpy as np
import pandas as pd

# 데이터 로드
DATA_DIR = Path(__file__).parent.parent / "datasets"
INPUT_PATH = DATA_DIR / "all_missions_with_ranges.csv"

print("=" * 100)
print("원본 값 vs min/max 관계 분석")
print("=" * 100)

df = pd.read_csv(INPUT_PATH, low_memory=False)

params = [
    "pl_orbper",
    "pl_rade",
    "pl_trandep",
    "pl_trandurh",
    "pl_insol",
    "pl_eqt",
    "st_teff",
    "st_logg",
    "st_rad",
]

print("\n" + "=" * 100)
print("원본 = (min + max) / 2 인지 확인")
print("=" * 100)

for param in params:
    orig_col = param
    min_col = f"{param}_min"
    max_col = f"{param}_max"

    if orig_col in df.columns and min_col in df.columns and max_col in df.columns:
        # 유효한 데이터만 (모두 NaN이 아닌 행)
        valid_mask = df[orig_col].notna() & df[min_col].notna() & df[max_col].notna()
        valid_df = df[valid_mask]

        if len(valid_df) == 0:
            print(f"\n{param}: 유효한 데이터 없음")
            continue

        # (min + max) / 2 계산
        calculated_mid = (valid_df[min_col] + valid_df[max_col]) / 2
        original = valid_df[orig_col]

        # 차이 계산
        diff = original - calculated_mid
        abs_diff = abs(diff)

        # 상대 오차 (%)
        rel_error = (abs_diff / original.abs()) * 100

        # 통계
        print(f"\n{param}:")
        print(f"  유효한 데이터: {len(valid_df):,}개")
        print(f"  평균 절대 차이: {abs_diff.mean():.6f}")
        print(f"  최대 절대 차이: {abs_diff.max():.6f}")
        print(f"  평균 상대 오차: {rel_error.mean():.4f}%")
        print(f"  최대 상대 오차: {rel_error.max():.4f}%")

        # 정확히 일치하는 비율
        exact_match = (abs_diff < 0.001).sum()
        exact_match_pct = (exact_match / len(valid_df)) * 100
        print(
            f"  정확히 일치 (차이 < 0.001): {exact_match:,}개 ({exact_match_pct:.2f}%)"
        )

        # 거의 일치하는 비율 (상대 오차 < 1%)
        approx_match = (rel_error < 1.0).sum()
        approx_match_pct = (approx_match / len(valid_df)) * 100
        print(
            f"  거의 일치 (상대오차 < 1%): {approx_match:,}개 ({approx_match_pct:.2f}%)"
        )

        # 샘플 5개 출력
        print(f"  샘플 5개:")
        sample = valid_df.head(5)
        for idx in sample.index:
            orig = sample.loc[idx, orig_col]
            min_v = sample.loc[idx, min_col]
            max_v = sample.loc[idx, max_col]
            mid = (min_v + max_v) / 2
            diff_v = orig - mid
            print(
                f"    원본: {orig:10.4f}, min: {min_v:10.4f}, max: {max_v:10.4f}, (min+max)/2: {mid:10.4f}, 차이: {diff_v:8.4f}"
            )

# ============================================================================
# 결론
# ============================================================================

print("\n" + "=" * 100)
print("📊 결론")
print("=" * 100)

print(
    """
분석 결과를 보고 판단:

1. 만약 대부분의 파라미터에서 원본 ≈ (min+max)/2 라면:
   ✅ 원본 값 제거 가능
   → 피처 수: 27개 → 18개 (9개 감소)
   → 장점: 모델 간소화, 학습 속도 향상
   → 단점: 비대칭 에러 정보 손실

2. 만약 차이가 크다면:
   ❌ 원본 값 유지 필요
   → 피처 수: 27개 유지
   → 장점: 완전한 정보 보존
   → 단점: 피처 수 증가

💡 추천:
  - 두 버전 모두 테스트해보고 성능 비교
  - 원본만, min/max만, 둘 다 사용 → 3가지 실험
"""
)

# ============================================================================
# 정보 이론적 분석
# ============================================================================

print("\n" + "=" * 100)
print("🔍 정보 이론적 분석")
print("=" * 100)

print("\n원본, min, max 3개 값이 제공하는 정보:")
print("  1. 원본 (best estimate): 가장 신뢰할 만한 값")
print("  2. min (lower bound): 최소 가능 값")
print("  3. max (upper bound): 최대 가능 값")

print("\nmin, max만 사용하면:")
print("  ✅ 범위 정보는 보존됨")
print("  ✅ 불확실성 정보도 보존됨 (max - min)")
print("  ⚠️  중심값은 계산 가능 ((min+max)/2)")
print("  ❌ 비대칭 에러 정보는 손실 가능")

print("\n비대칭 에러란?")
print("  - err1 (상한 에러) ≠ |err2| (하한 에러)")
print("  - 예: 값 = 100, err1 = +50, err2 = -20")
print("  - min = 80, max = 150")
print("  - (min+max)/2 = 115 ≠ 100 (원본)")

print("\n→ 위 분석 결과를 보고 비대칭 에러가 얼마나 흔한지 확인!")
