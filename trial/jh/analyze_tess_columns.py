"""
TESS와 Kepler+K2 병합 데이터셋의 컬럼 비교 분석
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "datasets"
TESS_PATH = DATA_DIR / "tess.csv"
MERGED_PATH = DATA_DIR / "kepler_k2_merged.csv"


def analyze_columns():
    """컬럼 비교 분석"""
    print("=" * 80)
    print("TESS vs Kepler+K2 컬럼 비교 분석")
    print("=" * 80)

    # 데이터 로드
    tess_df = pd.read_csv(TESS_PATH, low_memory=False)
    merged_df = pd.read_csv(MERGED_PATH, low_memory=False)

    tess_cols = set(tess_df.columns)
    merged_cols = set(merged_df.columns)

    print(f"\n1. 기본 정보")
    print("-" * 80)
    print(f"TESS 데이터: {tess_df.shape}")
    print(f"  - 컬럼 수: {len(tess_cols)}")
    print(f"  - 행 수: {len(tess_df)}")

    print(f"\nKepler+K2 병합 데이터: {merged_df.shape}")
    print(f"  - 컬럼 수: {len(merged_cols)}")
    print(f"  - 행 수: {len(merged_df)}")

    # TESS 고유 컬럼
    print(f"\n2. TESS 데이터셋 주요 컬럼")
    print("-" * 80)
    print("\nTESS 전체 컬럼:")
    for i, col in enumerate(sorted(tess_cols), 1):
        print(f"  {i:2d}. {col}")

    # 컬럼 분류
    print(f"\n3. TESS 컬럼 분류")
    print("-" * 80)

    # Identification
    id_cols = [
        c
        for c in tess_cols
        if any(x in c.lower() for x in ["toi", "tid", "rowid", "ctoi"])
    ]
    print(f"\n[식별자 관련] ({len(id_cols)}개):")
    for col in sorted(id_cols):
        print(f"  - {col}")

    # Position
    pos_cols = [
        c for c in tess_cols if any(x in c.lower() for x in ["ra", "dec", "pm"])
    ]
    print(f"\n[위치/좌표 관련] ({len(pos_cols)}개):")
    for col in sorted(pos_cols):
        print(f"  - {col}")

    # Planet parameters
    pl_cols = [c for c in tess_cols if c.startswith("pl_")]
    print(f"\n[행성 파라미터] ({len(pl_cols)}개):")
    for col in sorted(pl_cols):
        print(f"  - {col}")

    # Stellar parameters
    st_cols = [c for c in tess_cols if c.startswith("st_")]
    print(f"\n[항성 파라미터] ({len(st_cols)}개):")
    for col in sorted(st_cols):
        print(f"  - {col}")

    # Dates
    date_cols = [
        c for c in tess_cols if any(x in c.lower() for x in ["created", "update"])
    ]
    print(f"\n[날짜 관련] ({len(date_cols)}개):")
    for col in sorted(date_cols):
        print(f"  - {col}")

    # 매핑 가능한 컬럼 분석
    print(f"\n4. Kepler+K2와 매핑 가능한 TESS 컬럼")
    print("-" * 80)

    # 행성 파라미터 매핑
    planet_mapping = {
        "pl_orbper": "koi_period",
        "pl_orbpererr1": "koi_period_err1",
        "pl_orbpererr2": "koi_period_err2",
        "pl_tranmid": "koi_time0",
        "pl_tranmiderr1": "koi_time0_err1",
        "pl_tranmiderr2": "koi_time0_err2",
        "pl_trandurh": "koi_duration",
        "pl_trandurherr1": "koi_duration_err1",
        "pl_trandurherr2": "koi_duration_err2",
        "pl_trandep": "koi_depth",  # 주의: TESS는 ppm
        "pl_trandeperr1": "koi_depth_err1",
        "pl_trandeperr2": "koi_depth_err2",
        "pl_rade": "koi_prad",
        "pl_radeerr1": "koi_prad_err1",
        "pl_radeerr2": "koi_prad_err2",
        "pl_insol": "koi_insol",
        "pl_eqt": "koi_teq",
        "pl_eqterr1": "koi_teq_err1",
        "pl_eqterr2": "koi_teq_err2",
    }

    stellar_mapping = {
        "st_teff": "koi_steff",
        "st_tefferr1": "koi_steff_err1",
        "st_tefferr2": "koi_steff_err2",
        "st_logg": "koi_slogg",
        "st_loggerr1": "koi_slogg_err1",
        "st_loggerr2": "koi_slogg_err2",
        "st_rad": "koi_srad",
        "st_raderr1": "koi_srad_err1",
        "st_raderr2": "koi_srad_err2",
        "st_dist": None,  # Kepler에는 없음
    }

    position_mapping = {
        "ra": "ra",
        "dec": "dec",
        "st_pmra": None,  # Kepler에는 없음
        "st_pmdec": None,
    }

    print("\n[행성 파라미터 매핑]:")
    for tess_col, kepler_col in planet_mapping.items():
        exists = "✅" if tess_col in tess_cols else "❌"
        mapped = "✅" if kepler_col and kepler_col in merged_cols else "❌"
        print(
            f"  {exists} {tess_col:20s} -> {mapped} {kepler_col if kepler_col else 'N/A'}"
        )

    print("\n[항성 파라미터 매핑]:")
    for tess_col, kepler_col in stellar_mapping.items():
        exists = "✅" if tess_col in tess_cols else "❌"
        mapped = "✅" if kepler_col and kepler_col in merged_cols else "❌"
        print(
            f"  {exists} {tess_col:20s} -> {mapped} {kepler_col if kepler_col else 'N/A'}"
        )

    # TESS 고유 컬럼
    print(f"\n5. TESS 고유 컬럼 (Kepler+K2에 없는 것)")
    print("-" * 80)

    tess_only = [
        "toi",
        "toipfx",
        "tid",
        "ctoi_alias",
        "tfopwg_disp",
        "st_tmag",
        "st_dist",
        "st_pmra",
        "st_pmdec",
        "toi_created",
    ]

    for col in tess_only:
        if col in tess_cols:
            non_null = tess_df[col].notna().sum()
            pct = (non_null / len(tess_df)) * 100
            print(f"  - {col:20s}: {non_null:5d}/{len(tess_df)} ({pct:5.1f}%) 값 존재")

    # Kepler+K2 고유 컬럼
    print(f"\n6. Kepler+K2 고유 컬럼 (TESS에 없는 중요한 것)")
    print("-" * 80)

    important_kepler_cols = [
        "kepid",
        "kepoi_name",
        "kepler_name",
        "koi_disposition",
        "koi_score",
        "koi_fpflag_nt",
        "koi_fpflag_ss",
        "koi_fpflag_co",
        "koi_fpflag_ec",
        "koi_impact",
        "koi_eccen",
        "koi_sma",
        "koi_incl",
        "koi_kepmag",
        "data_source",
    ]

    for col in important_kepler_cols:
        if col in merged_cols:
            non_null = merged_df[col].notna().sum()
            pct = (non_null / len(merged_df)) * 100
            print(
                f"  - {col:20s}: {non_null:5d}/{len(merged_df)} ({pct:5.1f}%) 값 존재"
            )

    # 단위 확인
    print(f"\n7. 단위 확인 - 주의 필요!")
    print("-" * 80)

    print("\n웹사이트 문서 확인:")
    print("  TESS pl_trandep: ppm (parts per million)")
    print("  Kepler koi_depth: ppm (parts per million)")
    print("  K2 pl_trandep: % (percent)")
    print("\n  ⚠️  단위 변환 필요 여부 확인:")

    # TESS 샘플 데이터 확인
    tess_sample = tess_df[tess_df["pl_trandep"].notna()].head(5)
    print(f"\n  TESS pl_trandep 샘플값:")
    for idx, val in enumerate(tess_sample["pl_trandep"].values, 1):
        print(f"    {idx}. {val:.4f}")

    print(f"\n  TESS pl_trandep 통계:")
    print(f"    평균: {tess_df['pl_trandep'].mean():.2f}")
    print(
        f"    범위: [{tess_df['pl_trandep'].min():.2f}, {tess_df['pl_trandep'].max():.2f}]"
    )

    kepler_k2_depth = merged_df["koi_depth"].dropna()
    print(f"\n  Kepler+K2 koi_depth 통계:")
    print(f"    평균: {kepler_k2_depth.mean():.2f}")
    print(f"    범위: [{kepler_k2_depth.min():.2f}, {kepler_k2_depth.max():.2f}]")

    # TESS 값이 ppm인지 % 인지 판단
    tess_depth_mean = tess_df["pl_trandep"].mean()
    if tess_depth_mean > 100:
        print(
            f"\n  ✅ TESS pl_trandep은 ppm 단위입니다 (평균 {tess_depth_mean:.2f} ppm)"
        )
        print(f"     → 단위 변환 불필요!")
    else:
        print(
            f"\n  ⚠️  TESS pl_trandep이 % 단위일 수 있습니다 (평균 {tess_depth_mean:.4f})"
        )
        print(f"     → ppm 변환 필요: % * 10000")

    # Disposition 확인
    print(f"\n8. Disposition 값 확인")
    print("-" * 80)

    print("\nTESS tfopwg_disp 값:")
    if "tfopwg_disp" in tess_df.columns:
        print(tess_df["tfopwg_disp"].value_counts())

    print("\nKepler+K2 koi_disposition 값:")
    print(merged_df["koi_disposition"].value_counts())

    # 병합 전략 제안
    print(f"\n9. 병합 전략 제안")
    print("=" * 80)

    print(
        """
📋 TESS 데이터 병합 시 고려사항:

1. ✅ 매핑 가능한 컬럼:
   - 행성 파라미터: pl_orbper, pl_rade, pl_eqt, pl_insol, pl_trandep 등
   - 항성 파라미터: st_teff, st_logg, st_rad 등
   - 위치 정보: ra, dec

2. ⚠️  단위 확인 필요:
   - pl_trandep (Transit Depth): ppm 확인 필요
   - pl_trandurh (Transit Duration): hours (동일)
   - pl_orbper (Orbital Period): days (동일)

3. 🆕 TESS 고유 정보:
   - toi, tid: TESS 식별자
   - tfopwg_disp: TESS Follow-up Working Group Disposition
   - st_tmag: TESS magnitude
   - st_dist: 거리 (Kepler에는 없음)
   - st_pmra, st_pmdec: Proper motion (Kepler에는 없음)

4. 📊 Disposition 매핑:
   - TESS: PC, CP, FP, FA, APC, KP
   - Kepler: CONFIRMED, CANDIDATE, FALSE POSITIVE
   - 매핑 규칙 정의 필요

5. 🔄 병합 방식:
   Option A: Kepler+K2 컬럼 구조 유지, TESS를 변환하여 추가
   Option B: 공통 컬럼만 사용하여 새로운 통합 데이터셋 생성
   
   → 추천: Option A (일관성 유지)
"""
    )


if __name__ == "__main__":
    analyze_columns()
