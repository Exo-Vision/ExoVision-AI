"""
Kepler와 K2 데이터셋을 병합하는 스크립트
Kepler 데이터셋을 기준으로 K2 데이터를 병합합니다.
"""

from pathlib import Path

import numpy as np
import pandas as pd

# 데이터 경로 설정
DATA_DIR = Path(__file__).parent.parent / "datasets"
KEPLER_PATH = DATA_DIR / "kepler.csv"
K2_PATH = DATA_DIR / "k2.csv"
OUTPUT_PATH = DATA_DIR / "kepler_k2_merged.csv"


def load_datasets():
    """데이터셋 로드"""
    print("데이터셋 로딩 중...")
    kepler_df = pd.read_csv(KEPLER_PATH)
    k2_df = pd.read_csv(K2_PATH)

    print(f"Kepler 데이터셋: {kepler_df.shape}")
    print(f"K2 데이터셋: {k2_df.shape}")

    return kepler_df, k2_df


def create_column_mapping():
    """
    Kepler와 K2 데이터셋의 컬럼 매핑 생성
    Kepler 컬럼 -> K2 컬럼 매핑
    """

    # 공통 컬럼 매핑 (유사한 의미를 가진 컬럼들)
    column_mapping = {
        # Identification
        "kepid": None,  # Kepler에만 존재
        "kepoi_name": "epic_candname",  # KOI name -> EPIC candidate name
        "kepler_name": "pl_name",  # Kepler name -> Planet name
        # Disposition
        "koi_disposition": "disposition",
        "koi_pdisposition": None,  # Kepler에만 존재
        "koi_score": None,  # Kepler에만 존재
        # Flags
        "koi_fpflag_nt": None,  # Kepler에만 존재
        "koi_fpflag_ss": None,
        "koi_fpflag_co": None,
        "koi_fpflag_ec": None,
        # Orbital Period
        "koi_period": "pl_orbper",
        "koi_period_err1": "pl_orbpererr1",
        "koi_period_err2": "pl_orbpererr2",
        # Transit Time
        "koi_time0bk": None,  # Kepler에만 존재 (offset BJD)
        "koi_time0": "pl_tranmid",
        "koi_time0_err1": "pl_tranmiderr1",
        "koi_time0_err2": "pl_tranmiderr2",
        # Eccentricity
        "koi_eccen": "pl_orbeccen",
        "koi_eccen_err1": "pl_orbeccenerr1",
        "koi_eccen_err2": "pl_orbeccenerr2",
        # Impact Parameter
        "koi_impact": "pl_imppar",
        "koi_impact_err1": "pl_impparerr1",
        "koi_impact_err2": "pl_impparerr2",
        # Duration
        "koi_duration": None,  # Kepler는 hours, K2는 pl_trandur (hours)
        "koi_duration_err1": "pl_trandurerr1",
        "koi_duration_err2": "pl_trandurerr2",
        # Depth
        "koi_depth": None,  # Kepler는 ppm, K2는 pl_trandep (%)
        "koi_depth_err1": "pl_trandeperr1",
        "koi_depth_err2": "pl_trandeperr2",
        # Planet Radius
        "koi_prad": "pl_rade",  # Earth radii
        "koi_prad_err1": "pl_radeerr1",
        "koi_prad_err2": "pl_radeerr2",
        # Semi-major axis
        "koi_sma": "pl_orbsmax",
        "koi_sma_err1": "pl_orbsmaxerr1",
        "koi_sma_err2": "pl_orbsmaxerr2",
        # Inclination
        "koi_incl": "pl_orbincl",
        "koi_incl_err1": "pl_orbinclerr1",
        "koi_incl_err2": "pl_orbinclerr2",
        # Temperature
        "koi_teq": "pl_eqt",
        "koi_teq_err1": "pl_eqterr1",
        "koi_teq_err2": "pl_eqterr2",
        # Insolation
        "koi_insol": "pl_insol",
        # Stellar Parameters
        "koi_steff": "st_teff",
        "koi_steff_err1": "st_tefferr1",
        "koi_steff_err2": "st_tefferr2",
        "koi_slogg": "st_logg",
        "koi_slogg_err1": "st_loggerr1",
        "koi_slogg_err2": "st_loggerr2",
        "koi_smet": "st_met",
        "koi_smet_err1": "st_meterr1",
        "koi_smet_err2": "st_meterr2",
        "koi_srad": "st_rad",
        "koi_srad_err1": "st_raderr1",
        "koi_srad_err2": "st_radlerr2",
        "koi_smass": "st_mass",
        "koi_smass_err1": "st_masserr1",
        "koi_smass_err2": "st_masserr2",
        # Position
        "ra": "ra",
        "dec": "dec",
        # Magnitudes
        "koi_kepmag": "sy_kepmag",
        "koi_jmag": "sy_jmag",
        "koi_hmag": "sy_hmag",
        "koi_kmag": "sy_kmag",
    }

    return column_mapping


def convert_k2_to_kepler_format(k2_df, column_mapping):
    """
    K2 데이터를 Kepler 형식으로 변환
    """
    print("\nK2 데이터를 Kepler 형식으로 변환 중...")

    # 새로운 데이터프레임 생성
    k2_converted = pd.DataFrame()

    # 각 Kepler 컬럼에 대해 K2 컬럼 매핑
    for kepler_col, k2_col in column_mapping.items():
        if k2_col is not None and k2_col in k2_df.columns:
            k2_converted[kepler_col] = k2_df[k2_col]
        else:
            # 매핑되는 컬럼이 없으면 NaN으로 채움
            k2_converted[kepler_col] = np.nan

    # K2 고유 식별자 추가
    if "epic_hostname" in k2_df.columns:
        k2_converted["epic_hostname"] = k2_df["epic_hostname"]
    if "k2_name" in k2_df.columns:
        k2_converted["k2_name"] = k2_df["k2_name"]

    # 데이터 소스 표시
    k2_converted["data_source"] = "K2"

    # Transit Depth 변환: K2는 %, Kepler는 ppm
    if "pl_trandep" in k2_df.columns:
        # % to ppm: multiply by 10000
        k2_converted["koi_depth"] = k2_df["pl_trandep"] * 10000
        if "pl_trandeperr1" in k2_df.columns:
            k2_converted["koi_depth_err1"] = k2_df["pl_trandeperr1"] * 10000
        if "pl_trandeperr2" in k2_df.columns:
            k2_converted["koi_depth_err2"] = k2_df["pl_trandeperr2"] * 10000

    # Transit Duration: K2에서 pl_trandur로 매핑
    if "pl_trandur" in k2_df.columns:
        k2_converted["koi_duration"] = k2_df["pl_trandur"]

    print(f"변환된 K2 데이터: {k2_converted.shape}")

    return k2_converted


def merge_datasets(kepler_df, k2_converted):
    """
    Kepler와 변환된 K2 데이터셋 병합
    """
    print("\n데이터셋 병합 중...")

    # Kepler 데이터에 소스 표시
    kepler_df["data_source"] = "Kepler"

    # Kepler에만 있는 컬럼 중 K2에 없는 것들은 K2에 추가
    kepler_only_cols = set(kepler_df.columns) - set(k2_converted.columns)
    for col in kepler_only_cols:
        if col != "data_source":
            k2_converted[col] = np.nan

    # K2에만 있는 컬럼 중 Kepler에 없는 것들은 Kepler에 추가
    k2_only_cols = set(k2_converted.columns) - set(kepler_df.columns)
    for col in k2_only_cols:
        if col != "data_source":
            kepler_df[col] = np.nan

    # 컬럼 순서 정렬
    all_cols = sorted(set(kepler_df.columns) | set(k2_converted.columns))

    # 두 데이터프레임의 컬럼 순서를 동일하게 맞춤
    kepler_df = kepler_df[all_cols]
    k2_converted = k2_converted[all_cols]

    # 병합
    merged_df = pd.concat([kepler_df, k2_converted], axis=0, ignore_index=True)

    # REFUTED를 FALSE POSITIVE로 변경
    if "koi_disposition" in merged_df.columns:
        refuted_count = (merged_df["koi_disposition"] == "REFUTED").sum()
        if refuted_count > 0:
            print(f"\nREFUTED를 FALSE POSITIVE로 변경: {refuted_count}개")
            merged_df["koi_disposition"] = merged_df["koi_disposition"].replace(
                "REFUTED", "FALSE POSITIVE"
            )

    print(f"병합된 데이터셋: {merged_df.shape}")
    print(f"  - Kepler: {len(kepler_df)} 행")
    print(f"  - K2: {len(k2_converted)} 행")

    return merged_df


def save_merged_dataset(merged_df):
    """병합된 데이터셋 저장"""
    print(f"\n병합된 데이터셋 저장 중: {OUTPUT_PATH}")
    merged_df.to_csv(OUTPUT_PATH, index=False)
    print("저장 완료!")


def print_summary(merged_df):
    """병합 결과 요약 출력"""
    print("\n" + "=" * 50)
    print("병합 결과 요약")
    print("=" * 50)

    print(f"\n전체 데이터: {len(merged_df)} 행, {len(merged_df.columns)} 컬럼")

    # 데이터 소스별 통계
    print("\n데이터 소스별 분포:")
    print(merged_df["data_source"].value_counts())

    # Disposition별 통계
    if "koi_disposition" in merged_df.columns:
        print("\nDisposition 분포:")
        print(merged_df["koi_disposition"].value_counts())

    # 결측치 비율
    print("\n주요 컬럼 결측치 비율:")
    important_cols = ["koi_period", "koi_prad", "koi_teq", "koi_steff", "koi_srad"]
    for col in important_cols:
        if col in merged_df.columns:
            missing_pct = (merged_df[col].isna().sum() / len(merged_df)) * 100
            print(f"  {col}: {missing_pct:.2f}%")


def main():
    """메인 실행 함수"""
    print("=" * 50)
    print("Kepler와 K2 데이터셋 병합 시작")
    print("=" * 50)

    # 1. 데이터셋 로드
    kepler_df, k2_df = load_datasets()

    # 2. 컬럼 매핑 생성
    column_mapping = create_column_mapping()

    # 3. K2 데이터를 Kepler 형식으로 변환
    k2_converted = convert_k2_to_kepler_format(k2_df, column_mapping)

    # 4. 데이터셋 병합
    merged_df = merge_datasets(kepler_df, k2_converted)

    # 5. 저장
    save_merged_dataset(merged_df)

    # 6. 요약 출력
    print_summary(merged_df)

    print("\n" + "=" * 50)
    print("병합 완료!")
    print("=" * 50)


if __name__ == "__main__":
    main()
