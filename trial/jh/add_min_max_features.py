"""
병합된 데이터에 err 값을 이용한 min/max 범위 추가
lim 값에 따라 다르게 처리
"""

from pathlib import Path

import numpy as np
import pandas as pd

# 데이터 로드
DATA_DIR = Path(__file__).parent.parent / "datasets"
MERGED_PATH = DATA_DIR / "all_missions_merged.csv"
OUTPUT_PATH = DATA_DIR / "all_missions_with_ranges.csv"

print("=" * 100)
print("err 값을 이용한 min/max 범위 추가")
print("=" * 100)

merged_df = pd.read_csv(MERGED_PATH, low_memory=False)

print(f"\n원본 데이터: {len(merged_df)} 행, {len(merged_df.columns)} 컬럼")

# ============================================================================
# 처리할 파라미터 목록 - 모든 주요 물리 파라미터
# ============================================================================
params_to_process = [
    # 행성 파라미터
    "pl_orbper",  # Orbital Period (days)
    "pl_rade",  # Planet Radius (Earth Radius)
    "pl_trandep",  # Transit Depth (ppm)
    "pl_trandurh",  # Transit Duration (hours)
    "pl_insol",  # Insolation Flux (Earth Flux)
    "pl_eqt",  # Equilibrium Temperature (K)
    # 항성 파라미터
    "st_teff",  # Stellar Effective Temperature (K)
    "st_logg",  # Stellar Surface Gravity (log10(cm/s²))
    "st_rad",  # Stellar Radius (Solar Radius)
]

print(f"\n처리할 파라미터: {params_to_process}")

# ============================================================================
# min/max 계산 함수
# ============================================================================


def calculate_min_max(df, param_name):
    """
    lim 값에 따라 min, max 범위 계산
    - lim = NA/0: min = col + err2, max = col + err1
    - lim = 1: min = col + err2, max = col (상한 제한)
    - lim = -1: min = col, max = col + err1 (하한 제한)

    Kepler는 lim이 없으므로 항상 양방향 에러 적용
    """

    print(f"\n{'='*100}")
    print(f"처리 중: {param_name}")
    print(f"{'='*100}")

    # 기본값으로 원본 값 복사
    min_col_name = f"{param_name}_min"
    max_col_name = f"{param_name}_max"

    df[min_col_name] = df[param_name].copy()
    df[max_col_name] = df[param_name].copy()

    # err 컬럼이 있는지 확인
    err1_col = f"{param_name}err1"
    err2_col = f"{param_name}err2"
    lim_col = f"{param_name}lim"

    has_err1 = err1_col in df.columns
    has_err2 = err2_col in df.columns
    has_lim = lim_col in df.columns

    print(f"  컬럼 존재 여부:")
    print(f"    - {param_name}: {'✅' if param_name in df.columns else '❌'}")
    print(f"    - {err1_col}: {'✅' if has_err1 else '❌'}")
    print(f"    - {err2_col}: {'✅' if has_err2 else '❌'}")
    print(f"    - {lim_col}: {'✅' if has_lim else '❌'}")

    if not has_err1 or not has_err2:
        print(f"  ⚠️  에러 컬럼이 없어서 min=max=값으로 설정")
        return df

    # 미션별 처리 통계
    stats = {}

    for mission in ["Kepler", "K2", "TESS"]:
        mission_mask = df["mission"] == mission
        mission_count = mission_mask.sum()

        if mission_count == 0:
            continue

        print(f"\n  {mission} ({mission_count}개 행):")

        # 해당 미션의 행만 필터링
        mission_idx = df[mission_mask].index

        col_val = df.loc[mission_idx, param_name]
        err1 = df.loc[mission_idx, err1_col]
        err2 = df.loc[mission_idx, err2_col]

        # 유효한 데이터만 처리
        valid_mask = col_val.notna() & err1.notna() & err2.notna()
        valid_idx = mission_idx[valid_mask]

        if len(valid_idx) == 0:
            print(f"    - 유효한 데이터 없음")
            continue

        print(f"    - 유효한 데이터: {len(valid_idx)}/{mission_count}")

        # 기본값: 양방향 에러 (lim = NA 또는 0)
        min_val = col_val.loc[valid_idx] + err2.loc[valid_idx]
        max_val = col_val.loc[valid_idx] + err1.loc[valid_idx]

        # lim 값이 있는 경우 (K2/TESS)
        if has_lim and mission in ["K2", "TESS"]:
            lim = df.loc[valid_idx, lim_col]

            # lim = 1: 상한 제한 (max = col)
            upper_limit_mask = lim == 1
            upper_limit_count = upper_limit_mask.sum()
            if upper_limit_count > 0:
                max_val.loc[upper_limit_mask] = col_val.loc[valid_idx][upper_limit_mask]
                print(f"    - 상한 제한 (lim=1): {upper_limit_count}개")

            # lim = -1: 하한 제한 (min = col)
            lower_limit_mask = lim == -1
            lower_limit_count = lower_limit_mask.sum()
            if lower_limit_count > 0:
                min_val.loc[lower_limit_mask] = col_val.loc[valid_idx][lower_limit_mask]
                print(f"    - 하한 제한 (lim=-1): {lower_limit_count}개")

            # lim = 0 또는 NA: 양방향 (이미 기본값으로 설정됨)
            both_limit_count = len(valid_idx) - upper_limit_count - lower_limit_count
            print(f"    - 양방향 에러 (lim=0/NA): {both_limit_count}개")
        else:
            # Kepler는 항상 양방향
            print(f"    - 양방향 에러 (Kepler): {len(valid_idx)}개")

        # 값 할당
        df.loc[valid_idx, min_col_name] = min_val
        df.loc[valid_idx, max_col_name] = max_val

        # 통계 출력
        print(f"    - 평균 범위: [{min_val.mean():.4f}, {max_val.mean():.4f}]")
        print(f"    - 원본 평균: {col_val.loc[valid_idx].mean():.4f}")

        stats[mission] = {
            "count": len(valid_idx),
            "min_avg": min_val.mean(),
            "max_avg": max_val.mean(),
            "orig_avg": col_val.loc[valid_idx].mean(),
        }

    return df


# ============================================================================
# 각 파라미터 처리
# ============================================================================

for param in params_to_process:
    merged_df = calculate_min_max(merged_df, param)

# ============================================================================
# 결과 확인
# ============================================================================

print("\n" + "=" * 100)
print("처리 결과 요약")
print("=" * 100)

print(f"\n추가된 컬럼:")
new_cols = [
    col for col in merged_df.columns if col.endswith("_min") or col.endswith("_max")
]
for col in sorted(new_cols):
    print(f"  - {col}")

print(f"\n최종 데이터: {len(merged_df)} 행, {len(merged_df.columns)} 컬럼")

# ============================================================================
# 샘플 데이터 확인
# ============================================================================

print("\n" + "=" * 100)
print("샘플 데이터 확인 (각 미션별 1개씩)")
print("=" * 100)

for mission in ["Kepler", "K2", "TESS"]:
    print(f"\n{mission}:")
    sample = merged_df[merged_df["mission"] == mission].head(1)

    for param in params_to_process:
        if param in sample.columns:
            val = sample[param].values[0]
            min_val = sample[f"{param}_min"].values[0]
            max_val = sample[f"{param}_max"].values[0]

            if pd.notna(val):
                err1_col = f"{param}err1"
                err2_col = f"{param}err2"
                err1 = (
                    sample[err1_col].values[0] if err1_col in sample.columns else np.nan
                )
                err2 = (
                    sample[err2_col].values[0] if err2_col in sample.columns else np.nan
                )

                print(f"  {param}:")
                print(f"    원본: {val:.4f}")
                print(f"    err1: {err1:.4f}, err2: {err2:.4f}")
                print(f"    범위: [{min_val:.4f}, {max_val:.4f}]")
                print(f"    폭: {max_val - min_val:.4f}")

# ============================================================================
# 저장
# ============================================================================

print("\n" + "=" * 100)
print(f"데이터 저장 중: {OUTPUT_PATH}")
print("=" * 100)

merged_df.to_csv(OUTPUT_PATH, index=False)
print("✅ 저장 완료!")

print(
    f"\n원본: {len(merged_df)} 행 × {len(pd.read_csv(MERGED_PATH, nrows=1).columns)} 컬럼"
)
print(f"처리 후: {len(merged_df)} 행 × {len(merged_df.columns)} 컬럼")
print(
    f"추가된 컬럼: {len(merged_df.columns) - len(pd.read_csv(MERGED_PATH, nrows=1).columns)}개"
)
