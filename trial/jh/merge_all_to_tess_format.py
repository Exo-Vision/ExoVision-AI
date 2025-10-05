"""
TESS 데이터셋 구조를 기준으로 Kepler+K2 데이터를 병합하는 스크립트
TESS 컬럼 형식에 맞춰 모든 데이터를 통일합니다.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# 데이터 경로 설정
DATA_DIR = Path(__file__).parent.parent / "datasets"
KEPLER_PATH = DATA_DIR / "kepler.csv"
K2_PATH = DATA_DIR / "k2.csv"
TESS_PATH = DATA_DIR / "tess.csv"
OUTPUT_PATH = DATA_DIR / "all_missions_merged.csv"


def load_datasets():
    """데이터셋 로드"""
    print("=" * 80)
    print("데이터셋 로딩 중...")
    print("=" * 80)

    kepler_df = pd.read_csv(KEPLER_PATH)
    k2_df = pd.read_csv(K2_PATH)
    tess_df = pd.read_csv(TESS_PATH, low_memory=False)

    print(f"Kepler 데이터셋: {kepler_df.shape}")
    print(f"K2 데이터셋: {k2_df.shape}")
    print(f"TESS 데이터셋: {tess_df.shape}")

    return kepler_df, k2_df, tess_df


def create_tess_format_mapping():
    """
    TESS 형식으로 변환하기 위한 매핑 정의
    """

    # Kepler -> TESS 매핑
    kepler_to_tess = {
        # 행성 파라미터
        "koi_period": "pl_orbper",
        "koi_period_err1": "pl_orbpererr1",
        "koi_period_err2": "pl_orbpererr2",
        "koi_time0": "pl_tranmid",
        "koi_time0_err1": "pl_tranmiderr1",
        "koi_time0_err2": "pl_tranmiderr2",
        "koi_duration": "pl_trandurh",  # hours
        "koi_duration_err1": "pl_trandurherr1",
        "koi_duration_err2": "pl_trandurherr2",
        "koi_depth": "pl_trandep",  # ppm (동일)
        "koi_depth_err1": "pl_trandeperr1",
        "koi_depth_err2": "pl_trandeperr2",
        "koi_prad": "pl_rade",
        "koi_prad_err1": "pl_radeerr1",
        "koi_prad_err2": "pl_radeerr2",
        "koi_insol": "pl_insol",
        "koi_teq": "pl_eqt",
        "koi_teq_err1": "pl_eqterr1",
        "koi_teq_err2": "pl_eqterr2",
        # 항성 파라미터
        "koi_steff": "st_teff",
        "koi_steff_err1": "st_tefferr1",
        "koi_steff_err2": "st_tefferr2",
        "koi_slogg": "st_logg",
        "koi_slogg_err1": "st_loggerr1",
        "koi_slogg_err2": "st_loggerr2",
        "koi_srad": "st_rad",
        "koi_srad_err1": "st_raderr1",
        "koi_srad_err2": "st_radlerr2",
        # 위치
        "ra": "ra",
        "dec": "dec",
        # Disposition
        "koi_disposition": "label",
    }

    # K2 -> TESS 매핑 (K2는 이미 pl_*, st_* 형식 사용)
    k2_to_tess = {
        # 행성 파라미터
        "pl_orbper": "pl_orbper",
        "pl_orbpererr1": "pl_orbpererr1",
        "pl_orbpererr2": "pl_orbpererr2",
        "pl_tranmid": "pl_tranmid",
        "pl_tranmiderr1": "pl_tranmiderr1",
        "pl_tranmiderr2": "pl_tranmiderr2",
        "pl_trandur": "pl_trandurh",  # hours
        "pl_trandurerr1": "pl_trandurherr1",
        "pl_trandurerr2": "pl_trandurherr2",
        "pl_trandep": "pl_trandep",  # % -> ppm 변환 필요
        "pl_trandeperr1": "pl_trandeperr1",
        "pl_trandeperr2": "pl_trandeperr2",
        "pl_rade": "pl_rade",
        "pl_radeerr1": "pl_radeerr1",
        "pl_radeerr2": "pl_radeerr2",
        "pl_insol": "pl_insol",
        "pl_eqt": "pl_eqt",
        "pl_eqterr1": "pl_eqterr1",
        "pl_eqterr2": "pl_eqterr2",
        # 항성 파라미터
        "st_teff": "st_teff",
        "st_tefferr1": "st_tefferr1",
        "st_tefferr2": "st_tefferr2",
        "st_logg": "st_logg",
        "st_loggerr1": "st_loggerr1",
        "st_loggerr2": "st_loggerr2",
        "st_rad": "st_rad",
        "st_raderr1": "st_raderr1",
        "st_raderr2": "st_raderr2",
        # 위치
        "ra": "ra",
        "dec": "dec",
        # Disposition
        "disposition": "label",
    }

    return kepler_to_tess, k2_to_tess


def convert_kepler_to_tess_format(kepler_df, mapping):
    """Kepler 데이터를 TESS 형식으로 변환"""
    print("\n" + "=" * 80)
    print("Kepler 데이터를 TESS 형식으로 변환 중...")
    print("=" * 80)

    tess_format = pd.DataFrame()

    # 매핑된 컬럼 변환
    for kepler_col, tess_col in mapping.items():
        if kepler_col in kepler_df.columns:
            tess_format[tess_col] = kepler_df[kepler_col]

    # Disposition 변환: Kepler -> TESS 형식
    if "label" in tess_format.columns:
        disposition_map = {
            "CONFIRMED": "CP",
            "CANDIDATE": "PC",
            "FALSE POSITIVE": "FP",
            "REFUTED": "FP",
        }
        tess_format["label"] = tess_format["label"].replace(disposition_map)

    # Kepler 고유 식별자 추가 (별도 컬럼)
    tess_format["kepler_kepid"] = kepler_df["kepid"]
    tess_format["kepler_kepoi_name"] = kepler_df["kepoi_name"]
    tess_format["kepler_name"] = kepler_df["kepler_name"]

    # Kepler 고유 파라미터 추가
    tess_format["kepler_koi_score"] = kepler_df["koi_score"]
    tess_format["kepler_fpflag_nt"] = kepler_df["koi_fpflag_nt"]
    tess_format["kepler_fpflag_ss"] = kepler_df["koi_fpflag_ss"]
    tess_format["kepler_fpflag_co"] = kepler_df["koi_fpflag_co"]
    tess_format["kepler_fpflag_ec"] = kepler_df["koi_fpflag_ec"]
    tess_format["kepler_impact"] = kepler_df["koi_impact"]
    tess_format["kepler_eccen"] = kepler_df["koi_eccen"]
    tess_format["kepler_sma"] = kepler_df["koi_sma"]
    tess_format["kepler_incl"] = kepler_df["koi_incl"]
    tess_format["kepler_kepmag"] = kepler_df["koi_kepmag"]
    tess_format["kepler_model_snr"] = kepler_df["koi_model_snr"]
    tess_format["kepler_quarters"] = kepler_df["koi_quarters"]

    # 데이터 소스 표시
    tess_format["mission"] = "Kepler"

    print(f"변환된 Kepler 데이터: {tess_format.shape}")
    print(
        f"  - TESS 형식 컬럼: {len([c for c in tess_format.columns if c.startswith(('pl_', 'st_', 'ra', 'dec', 'tfopwg'))])}개"
    )
    print(
        f"  - Kepler 고유 컬럼: {len([c for c in tess_format.columns if c.startswith('kepler_')])}개"
    )

    return tess_format


def convert_k2_to_tess_format(k2_df, mapping):
    """K2 데이터를 TESS 형식으로 변환"""
    print("\n" + "=" * 80)
    print("K2 데이터를 TESS 형식으로 변환 중...")
    print("=" * 80)

    tess_format = pd.DataFrame()

    # 매핑된 컬럼 변환
    for k2_col, tess_col in mapping.items():
        if k2_col in k2_df.columns:
            tess_format[tess_col] = k2_df[k2_col]

    # Transit Depth 단위 변환: % -> ppm
    if "pl_trandep" in tess_format.columns:
        print("\n⚠️  Transit Depth 단위 변환: % -> ppm")
        before_mean = tess_format["pl_trandep"].mean()
        tess_format["pl_trandep"] = tess_format["pl_trandep"] * 10000
        if "pl_trandeperr1" in tess_format.columns:
            tess_format["pl_trandeperr1"] = tess_format["pl_trandeperr1"] * 10000
        if "pl_trandeperr2" in tess_format.columns:
            tess_format["pl_trandeperr2"] = tess_format["pl_trandeperr2"] * 10000
        after_mean = tess_format["pl_trandep"].mean()
        print(f"   변환 전 평균: {before_mean:.4f} %")
        print(f"   변환 후 평균: {after_mean:.2f} ppm")

    # Disposition 변환: K2 -> TESS 형식
    if "label" in tess_format.columns:
        disposition_map = {
            "CONFIRMED": "CP",
            "CANDIDATE": "PC",
            "FALSE POSITIVE": "FP",
            "REFUTED": "FP",
        }
        tess_format["label"] = tess_format["label"].replace(disposition_map)

    # K2 고유 식별자 추가 (별도 컬럼)
    tess_format["k2_name"] = k2_df["k2_name"]
    tess_format["k2_pl_name"] = k2_df["pl_name"]
    tess_format["k2_epic_hostname"] = k2_df["epic_hostname"]
    tess_format["k2_epic_candname"] = k2_df["epic_candname"]

    # K2 고유 파라미터 추가
    if "pl_masse" in k2_df.columns:
        tess_format["k2_pl_masse"] = k2_df["pl_masse"]
    if "pl_bmasse" in k2_df.columns:
        tess_format["k2_pl_bmasse"] = k2_df["pl_bmasse"]
    if "pl_orbeccen" in k2_df.columns:
        tess_format["k2_orbeccen"] = k2_df["pl_orbeccen"]
    if "pl_orbincl" in k2_df.columns:
        tess_format["k2_orbincl"] = k2_df["pl_orbincl"]
    if "pl_imppar" in k2_df.columns:
        tess_format["k2_imppar"] = k2_df["pl_imppar"]
    if "sy_kepmag" in k2_df.columns:
        tess_format["k2_kepmag"] = k2_df["sy_kepmag"]

    # 데이터 소스 표시
    tess_format["mission"] = "K2"

    print(f"변환된 K2 데이터: {tess_format.shape}")
    print(
        f"  - TESS 형식 컬럼: {len([c for c in tess_format.columns if c.startswith(('pl_', 'st_', 'ra', 'dec', 'tfopwg'))])}개"
    )
    print(
        f"  - K2 고유 컬럼: {len([c for c in tess_format.columns if c.startswith('k2_')])}개"
    )

    return tess_format


def prepare_tess_data(tess_df):
    """TESS 데이터 준비 (Disposition 매핑)"""
    print("\n" + "=" * 80)
    print("TESS 데이터 준비 중...")
    print("=" * 80)

    tess_prepared = tess_df.copy()

    # Disposition 매핑
    disposition_map = {
        "APC": "PC",  # Ambiguous PC -> PC
        "CP": "CP",  # Confirmed Planet
        "FA": "FP",  # False Alarm -> FP
        "FP": "FP",  # False Positive
        "KP": "CP",  # Known Planet -> CP
        "PC": "PC",  # Planet Candidate
    }

    print("\nDisposition 매핑:")
    print("  APC (Ambiguous PC) -> PC")
    print("  CP (Confirmed Planet) -> CP")
    print("  FA (False Alarm) -> FP")
    print("  FP (False Positive) -> FP")
    print("  KP (Known Planet) -> CP")
    print("  PC (Planet Candidate) -> PC")

    before_counts = tess_prepared["tfopwg_disp"].value_counts()
    tess_prepared["label"] = tess_prepared["tfopwg_disp"].replace(disposition_map)
    after_counts = tess_prepared["label"].value_counts()

    print("\n변환 전:")
    print(before_counts)
    print("\n변환 후:")
    print(after_counts)

    # 데이터 소스 표시
    tess_prepared["mission"] = "TESS"

    # TESS 고유 식별자 컬럼 추가
    tess_prepared["tess_toi"] = tess_df["toi"]
    tess_prepared["tess_tid"] = tess_df["tid"]

    print(f"\n준비된 TESS 데이터: {tess_prepared.shape}")

    return tess_prepared


def merge_all_datasets(kepler_tess, k2_tess, tess_prepared):
    """모든 데이터셋 병합"""
    print("\n" + "=" * 80)
    print("데이터셋 병합 중...")
    print("=" * 80)

    # 모든 컬럼 수집
    all_cols = set()
    all_cols.update(kepler_tess.columns)
    all_cols.update(k2_tess.columns)
    all_cols.update(tess_prepared.columns)

    # 각 데이터셋에 없는 컬럼을 NaN으로 추가
    for col in all_cols:
        if col not in kepler_tess.columns:
            kepler_tess[col] = np.nan
        if col not in k2_tess.columns:
            k2_tess[col] = np.nan
        if col not in tess_prepared.columns:
            tess_prepared[col] = np.nan

    # 컬럼 순서 정렬 (mission과 label을 앞으로)
    priority_cols = ["mission", "label"]
    tess_cols = sorted(
        [
            c
            for c in all_cols
            if c.startswith(("pl_", "st_", "ra", "dec", "toi", "tid"))
            and c not in priority_cols
        ]
    )
    mission_specific_cols = sorted(
        [
            c
            for c in all_cols
            if c.startswith(("kepler_", "k2_", "tess_")) and c not in priority_cols
        ]
    )
    other_cols = sorted(
        [
            c
            for c in all_cols
            if c not in priority_cols + tess_cols + mission_specific_cols
        ]
    )

    final_col_order = priority_cols + tess_cols + mission_specific_cols + other_cols

    # 컬럼 순서 맞추기
    kepler_tess = kepler_tess[final_col_order]
    k2_tess = k2_tess[final_col_order]
    tess_prepared = tess_prepared[final_col_order]

    # 병합
    merged_df = pd.concat(
        [kepler_tess, k2_tess, tess_prepared], axis=0, ignore_index=True
    )

    print(f"\n병합 완료:")
    print(f"  전체 데이터: {len(merged_df)} 행, {len(merged_df.columns)} 컬럼")
    print(f"    - Kepler: {len(kepler_tess)} 행")
    print(f"    - K2: {len(k2_tess)} 행")
    print(f"    - TESS: {len(tess_prepared)} 행")

    print(f"\n컬럼 구성:")
    print(f"  - 공통 TESS 형식 컬럼: {len(tess_cols)}개")
    print(f"  - 미션별 고유 컬럼: {len(mission_specific_cols)}개")
    print(
        f"    * Kepler 고유: {len([c for c in mission_specific_cols if c.startswith('kepler_')])}개"
    )
    print(
        f"    * K2 고유: {len([c for c in mission_specific_cols if c.startswith('k2_')])}개"
    )
    print(
        f"    * TESS 고유: {len([c for c in mission_specific_cols if c.startswith('tess_')])}개"
    )

    return merged_df


def print_summary(merged_df):
    """병합 결과 요약"""
    print("\n" + "=" * 80)
    print("병합 결과 요약")
    print("=" * 80)

    print(f"\n1. 데이터 규모")
    print("-" * 80)
    print(f"전체: {len(merged_df)} 행, {len(merged_df.columns)} 컬럼")

    print(f"\n2. 미션별 분포")
    print("-" * 80)
    mission_counts = merged_df["mission"].value_counts()
    for mission, count in mission_counts.items():
        pct = (count / len(merged_df)) * 100
        print(f"  {mission:10s}: {count:6d} ({pct:5.2f}%)")

    print(f"\n3. Disposition 분포 (통일된 형식)")
    print("-" * 80)
    disp_counts = merged_df["label"].value_counts()
    for disp, count in disp_counts.items():
        pct = (count / len(merged_df)) * 100
        desc = {
            "PC": "Planet Candidate",
            "CP": "Confirmed Planet",
            "FP": "False Positive",
        }.get(disp, disp)
        print(f"  {disp} ({desc:20s}): {count:6d} ({pct:5.2f}%)")

    print(f"\n4. 주요 TESS 형식 컬럼 결측치 비율")
    print("-" * 80)
    important_cols = [
        "pl_orbper",
        "pl_rade",
        "pl_eqt",
        "pl_insol",
        "pl_trandep",
        "pl_trandurh",
        "st_teff",
        "st_logg",
        "st_rad",
    ]

    for col in important_cols:
        if col in merged_df.columns:
            missing = merged_df[col].isna().sum()
            pct = (missing / len(merged_df)) * 100
            print(f"  {col:20s}: {missing:6d} ({pct:5.2f}%)")


def save_dataset(merged_df):
    """데이터셋 저장"""
    print("\n" + "=" * 80)
    print(f"데이터셋 저장 중: {OUTPUT_PATH}")
    print("=" * 80)

    merged_df.to_csv(OUTPUT_PATH, index=False)
    print("✅ 저장 완료!")


def main():
    """메인 실행 함수"""
    print("\n" + "🚀" * 40)
    print("TESS 형식 기반 Kepler+K2+TESS 데이터셋 병합")
    print("🚀" * 40 + "\n")

    # 1. 데이터 로드
    kepler_df, k2_df, tess_df = load_datasets()

    # 2. 매핑 생성
    kepler_to_tess, k2_to_tess = create_tess_format_mapping()

    # 3. Kepler 데이터를 TESS 형식으로 변환
    kepler_tess = convert_kepler_to_tess_format(kepler_df, kepler_to_tess)

    # 4. K2 데이터를 TESS 형식으로 변환
    k2_tess = convert_k2_to_tess_format(k2_df, k2_to_tess)

    # 5. TESS 데이터 준비
    tess_prepared = prepare_tess_data(tess_df)

    # 6. 모든 데이터셋 병합
    merged_df = merge_all_datasets(kepler_tess, k2_tess, tess_prepared)

    # 7. 요약 출력
    print_summary(merged_df)

    # 8. 저장
    save_dataset(merged_df)

    print("\n" + "🎉" * 40)
    print("병합 완료!")
    print("🎉" * 40 + "\n")


if __name__ == "__main__":
    main()
