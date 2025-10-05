"""
사용하는 칼럼
pl_rade
pl_orbper
pl_trandep
pl_trandurh
pl_insol
st_teff
st_tmag
st_dist
st_rad
st_logg

koi_model_snr
koi_quarters

---
칼럼 수정
1.
st_dens = st_rad / st_logg

2.
err1, err2, lim 사용해서 수정

3. lim 칼럼
1, 0 으로 변경

"""

import numpy as np
import pandas as pd

koi_datasets = pd.read_csv("C:/workspace/nasa/datasets/kepler.csv")
k2_datasets = pd.read_csv("C:/workspace/nasa/datasets/k2.csv")
tess_datasets = pd.read_csv("C:/workspace/nasa/datasets/tess.csv")

tess_columns = [
    "pl_rade",
    "pl_orbper",
    "pl_trandep",
    "pl_trandurh",
    "pl_insol",
    "st_teff",
    # "st_tmag",
    # "st_dist",
    "st_rad",
    "st_logg",
]

tess_pre_datasets = tess_datasets[tess_columns].copy()


def tess_cal_min_max(df, apply_df, param_name):
    """
    lim 값에 따라 min, max 범위 계산
    - lim = NA/0: min = col + err2, max = col + err1
    - lim = 1: min = col + err2, max = col (상한 제한)
    - lim = -1: min = col, max = col + err1 (하한 제한)
    """
    if (
        param_name not in df.columns
        or f"{param_name}err1" not in df.columns
        or f"{param_name}err2" not in df.columns
    ):
        return apply_df

    col_val = df[param_name]
    err1 = df[f"{param_name}err1"]  # 항상 양수
    err2 = df[f"{param_name}err2"]  # 항상 음수

    # 기본값: lim이 NA 또는 0인 경우
    min_val = col_val + err2
    max_val = col_val + err1

    # lim 값 확인
    if f"{param_name}lim" in df.columns:
        lim = df[f"{param_name}lim"]

        # lim = 1: 상한 제한 (max = col)
        mask_upper = lim == 1
        max_val = np.where(mask_upper, col_val, max_val)

        # lim = -1: 하한 제한 (min = col)
        mask_lower = lim == -1
        min_val = np.where(mask_lower, col_val, min_val)

    apply_df[f"{param_name}_min"] = min_val
    apply_df[f"{param_name}_max"] = max_val

    return apply_df


for col in tess_columns:
    tess_pre_datasets = tess_cal_min_max(
        tess_datasets,
        tess_pre_datasets,
        col,
    )

# 원본 칼럼 제거 (min/max만 유지)
tess_pre_datasets = tess_pre_datasets.drop(columns=tess_columns)

tess_pre_datasets["st_dens_min"] = (
    tess_pre_datasets["st_rad_min"] / tess_pre_datasets["st_logg_max"]
)
tess_pre_datasets["st_dens_max"] = (
    tess_pre_datasets["st_rad_max"] / tess_pre_datasets["st_logg_min"]
)
tess_pre_datasets = tess_pre_datasets.drop(
    columns=[
        "st_rad_min",
        "st_rad_max",
        "st_logg_min",
        "st_logg_max",
    ]
)

# snr도 min/max로 계산
tess_pre_datasets["snr_min"] = (
    tess_datasets["pl_trandep"]
    / (tess_datasets["pl_trandeperr1"] + abs(tess_datasets["pl_trandeperr2"]))
    * 2
)
tess_pre_datasets["snr_max"] = tess_pre_datasets[
    "snr_min"
]  # snr은 계산값이므로 min=max

tess_pre_datasets["mission"] = "TESS"

# label에서 apc->pc, cp, fa -> fp, fp, kp->cp, pc
# pc: 후보, cp: 확정, fp: 거짓
tess_pre_datasets["label"] = tess_datasets["tfopwg_disp"].replace(
    {"APC": "PC", "CP": "CP", "FA": "FP", "FP": "FP", "KP": "CP", "PC": "PC"}
)

print(tess_pre_datasets.columns)
print(len(tess_pre_datasets.columns))

# kepler에는 lim 없네
# KOI
# koi_columns = [
#     "koi_model_snr",
#     "koi_quarters",
# ]
# KOI -> TESS
koi_to_tess_map = {
    "koi_prad": "pl_rade",
    "koi_period": "pl_orbper",
    "koi_depth": "pl_trandep",
    "koi_duration": "pl_trandurh",
    "koi_insol": "pl_insol",
    "koi_steff": "st_teff",
    # "koi_kepmag": "st_tmag",
    # "koi_sma": "st_dist",
    "koi_srad": "st_rad",
    "koi_slogg": "st_logg",
    "koi_model_snr": "snr",
}
""""pl_trandep" -> "(pl_rade·R⊕)/(st_rad·R☉))² × 10^6","""

koi_pre_datasets = koi_datasets[koi_to_tess_map.keys()].copy()
koi_pre_datasets.rename(columns=koi_to_tess_map, inplace=True)


def kepler_cal_min_max(df, param_name):
    """
    Kepler 데이터는 lim이 없으므로 기본 min/max만 계산
    min = col + err2, max = col + err1
    """
    if f"{param_name}_err1" not in df.columns or f"{param_name}_err2" not in df.columns:
        # err 칼럼이 없으면 min=max=col (범위 없음)
        return df[param_name], df[param_name]

    col_val = df[param_name]
    err1 = df[f"{param_name}_err1"]  # 항상 양수
    err2 = df[f"{param_name}_err2"]  # 항상 음수

    min_val = col_val + err2
    max_val = col_val + err1

    return min_val, max_val


for col, val in koi_to_tess_map.items():
    if col in koi_datasets.columns:
        min_val, max_val = kepler_cal_min_max(koi_datasets, col)
        koi_pre_datasets[f"{val}_min"] = min_val
        koi_pre_datasets[f"{val}_max"] = max_val

# 원본 칼럼 제거 (min/max만 유지)
original_cols_to_drop = [
    val for col, val in koi_to_tess_map.items() if val in koi_pre_datasets.columns
]
koi_pre_datasets = koi_pre_datasets.drop(columns=original_cols_to_drop)

koi_pre_datasets["st_dens_min"] = (
    koi_pre_datasets["st_rad_min"] / koi_pre_datasets["st_logg_max"]
)
koi_pre_datasets["st_dens_max"] = (
    koi_pre_datasets["st_rad_max"] / koi_pre_datasets["st_logg_min"]
)
koi_pre_datasets = koi_pre_datasets.drop(
    columns=[
        "st_rad_min",
        "st_rad_max",
        "st_logg_min",
        "st_logg_max",
    ]
)
koi_pre_datasets["mission"] = "kepler"
koi_pre_datasets["label"] = koi_datasets["koi_disposition"].replace(
    {"CANDIDATE": "PC", "CONFIRMED": "CP", "FALSE POSITIVE": "FP"}
)

print(koi_pre_datasets.columns)
print(len(koi_pre_datasets.columns))

# K2 -> TESS
k2_to_tess_map = {
    "pl_rade": "pl_rade",
    "pl_orbper": "pl_orbper",
    "pl_trandep": "pl_trandep",
    "pl_trandur": "pl_trandurh",
    "pl_insol": "pl_insol",
    "st_teff": "st_teff",
    # "st_tmag": "st_tmag",
    # "sy_dist": "st_dist",  # lim 없
    "st_rad": "st_rad",
    "st_logg": "st_logg",
}
# snr 없음

k2_pre_datasets = k2_datasets[k2_to_tess_map.keys()].copy()
k2_pre_datasets.rename(columns=k2_to_tess_map, inplace=True)

# K2의 pl_trandep은 %단위이므로 ppm으로 변환 (Kepler, TESS와 단위 통일)
# 1% = 10,000 ppm
k2_pre_datasets["pl_trandep"] = k2_pre_datasets["pl_trandep"] * 10000

# k2_pre_datasets["st_distlim"] = 0


def k2_cal_min_max(df, param_name):
    """
    K2 데이터의 min/max 계산 (lim 포함)
    - lim = NA/0: min = col + err2, max = col + err1
    - lim = 1: min = col + err2, max = col (상한 제한)
    - lim = -1: min = col, max = col + err1 (하한 제한)
    """
    col_val = df[param_name]
    err1 = df[f"{param_name}err1"]  # 항상 양수
    err2 = df[f"{param_name}err2"]  # 항상 음수

    # 기본값
    min_val = col_val + err2
    max_val = col_val + err1

    # lim 값 확인
    if f"{param_name}lim" in df.columns:
        lim = df[f"{param_name}lim"]

        # lim = 1: 상한 제한 (max = col)
        mask_upper = lim == 1
        max_val = np.where(mask_upper, col_val, max_val)

        # lim = -1: 하한 제한 (min = col)
        mask_lower = lim == -1
        min_val = np.where(mask_lower, col_val, min_val)

    return min_val, max_val


for col, val in k2_to_tess_map.items():
    if col in k2_datasets.columns:
        min_val, max_val = k2_cal_min_max(k2_datasets, col)
        k2_pre_datasets[f"{val}_min"] = min_val
        k2_pre_datasets[f"{val}_max"] = max_val

# 원본 칼럼 제거 (min/max만 유지)
original_cols_to_drop = [
    val for col, val in k2_to_tess_map.items() if val in k2_pre_datasets.columns
]
k2_pre_datasets = k2_pre_datasets.drop(columns=original_cols_to_drop)

k2_pre_datasets["st_dens_min"] = (
    k2_pre_datasets["st_rad_min"] / k2_pre_datasets["st_logg_max"]
)
k2_pre_datasets["st_dens_max"] = (
    k2_pre_datasets["st_rad_max"] / k2_pre_datasets["st_logg_min"]
)
k2_pre_datasets = k2_pre_datasets.drop(
    columns=[
        "st_rad_min",
        "st_rad_max",
        "st_logg_min",
        "st_logg_max",
    ]
)

# snr도 min/max로 계산
k2_pre_datasets["snr_min"] = (
    k2_datasets["pl_trandep"]
    / (k2_datasets["pl_trandeperr1"] + abs(k2_datasets["pl_trandeperr2"]))
    * 2
)
k2_pre_datasets["snr_max"] = k2_pre_datasets["snr_min"]  # snr은 계산값이므로 min=max

k2_pre_datasets["mission"] = "k2"
k2_pre_datasets["label"] = koi_datasets["koi_disposition"].replace(
    {"CANDIDATE": "PC", "CONFIRMED": "CP", "FALSE POSITIVE": "FP", "REFUTED": "FP"}
)

print(k2_pre_datasets.columns)
print(len(k2_pre_datasets.columns))


# 데이터 통합
combined_df = pd.concat(
    [koi_pre_datasets, k2_pre_datasets, tess_pre_datasets], ignore_index=True
)

# pl_trandurh의 min/max 절대값 처리
combined_df["pl_trandurh_min"] = combined_df["pl_trandurh_min"].abs()
combined_df["pl_trandurh_max"] = combined_df["pl_trandurh_max"].abs()

# ========================================================================
# Phase 1: NA 값 복구 - 계산 가능한 칼럼들
# ========================================================================
print("\n" + "=" * 70)
print("Phase 1: NA 값 복구 (계산 가능한 칼럼)")
print("=" * 70)

# 각 데이터셋에 mission 마커 추가하여 재구성
koi_datasets["mission_temp"] = "kepler"
k2_datasets["mission_temp"] = "k2"
tess_datasets["mission_temp"] = "TESS"

# 원본 데이터 통합
combined_original = pd.concat(
    [koi_datasets, k2_datasets, tess_datasets], ignore_index=True
)

# 1. st_dens_min, st_dens_max 복구 (st_rad / st_logg)
print("\n1. st_dens_min, st_dens_max 복구 중...")
before_na_min = combined_df["st_dens_min"].isna().sum()
before_na_max = combined_df["st_dens_max"].isna().sum()

# Kepler
kepler_mask = (
    combined_df["st_dens_min"].isna() | combined_df["st_dens_max"].isna()
) & (combined_df["mission"] == "kepler")
kepler_orig_mask = combined_original["mission_temp"] == "kepler"
kepler_orig_data = combined_original[kepler_orig_mask].reset_index(drop=True)
kepler_combined_indices = combined_df[kepler_mask].index.tolist()

kepler_start = 0
for i, idx in enumerate(kepler_combined_indices):
    if i < len(kepler_orig_data):
        st_rad = kepler_orig_data.loc[i, "koi_srad"]
        st_logg = kepler_orig_data.loc[i, "koi_slogg"]
        st_rad_err1 = kepler_orig_data.loc[i, "koi_srad_err1"]
        st_rad_err2 = kepler_orig_data.loc[i, "koi_srad_err2"]
        st_logg_err1 = kepler_orig_data.loc[i, "koi_slogg_err1"]
        st_logg_err2 = kepler_orig_data.loc[i, "koi_slogg_err2"]

        if pd.notna(st_rad) and pd.notna(st_logg) and st_logg != 0:
            # st_dens_min = st_rad_min / st_logg_max
            # st_dens_max = st_rad_max / st_logg_min
            if (
                pd.notna(st_rad_err1)
                and pd.notna(st_rad_err2)
                and pd.notna(st_logg_err1)
                and pd.notna(st_logg_err2)
            ):
                st_rad_min = st_rad + st_rad_err2
                st_rad_max = st_rad + st_rad_err1
                st_logg_min = st_logg + st_logg_err2
                st_logg_max = st_logg + st_logg_err1

                if st_logg_max != 0 and st_logg_min != 0:
                    combined_df.loc[idx, "st_dens_min"] = st_rad_min / st_logg_max
                    combined_df.loc[idx, "st_dens_max"] = st_rad_max / st_logg_min

# K2
k2_mask = (combined_df["st_dens_min"].isna() | combined_df["st_dens_max"].isna()) & (
    combined_df["mission"] == "k2"
)
k2_orig_mask = combined_original["mission_temp"] == "k2"
k2_orig_data = combined_original[k2_orig_mask].reset_index(drop=True)
k2_combined_indices = combined_df[k2_mask].index.tolist()

for i, idx in enumerate(k2_combined_indices):
    if i < len(k2_orig_data):
        st_rad = k2_orig_data.loc[i, "st_rad"]
        st_logg = k2_orig_data.loc[i, "st_logg"]
        st_rad_err1 = k2_orig_data.loc[i, "st_raderr1"]
        st_rad_err2 = k2_orig_data.loc[i, "st_raderr2"]
        st_logg_err1 = k2_orig_data.loc[i, "st_loggerr1"]
        st_logg_err2 = k2_orig_data.loc[i, "st_loggerr2"]

        if pd.notna(st_rad) and pd.notna(st_logg) and st_logg != 0:
            if (
                pd.notna(st_rad_err1)
                and pd.notna(st_rad_err2)
                and pd.notna(st_logg_err1)
                and pd.notna(st_logg_err2)
            ):
                st_rad_min = st_rad + st_rad_err2
                st_rad_max = st_rad + st_rad_err1
                st_logg_min = st_logg + st_logg_err2
                st_logg_max = st_logg + st_logg_err1

                if st_logg_max != 0 and st_logg_min != 0:
                    combined_df.loc[idx, "st_dens_min"] = st_rad_min / st_logg_max
                    combined_df.loc[idx, "st_dens_max"] = st_rad_max / st_logg_min

# TESS
tess_mask = (combined_df["st_dens_min"].isna() | combined_df["st_dens_max"].isna()) & (
    combined_df["mission"] == "TESS"
)
tess_orig_mask = combined_original["mission_temp"] == "TESS"
tess_orig_data = combined_original[tess_orig_mask].reset_index(drop=True)
tess_combined_indices = combined_df[tess_mask].index.tolist()

for i, idx in enumerate(tess_combined_indices):
    if i < len(tess_orig_data):
        st_rad = tess_orig_data.loc[i, "st_rad"]
        st_logg = tess_orig_data.loc[i, "st_logg"]
        st_rad_err1 = tess_orig_data.loc[i, "st_raderr1"]
        st_rad_err2 = tess_orig_data.loc[i, "st_raderr2"]
        st_logg_err1 = tess_orig_data.loc[i, "st_loggerr1"]
        st_logg_err2 = tess_orig_data.loc[i, "st_loggerr2"]

        if pd.notna(st_rad) and pd.notna(st_logg) and st_logg != 0:
            if (
                pd.notna(st_rad_err1)
                and pd.notna(st_rad_err2)
                and pd.notna(st_logg_err1)
                and pd.notna(st_logg_err2)
            ):
                st_rad_min = st_rad + st_rad_err2
                st_rad_max = st_rad + st_rad_err1
                st_logg_min = st_logg + st_logg_err2
                st_logg_max = st_logg + st_logg_err1

                if st_logg_max != 0 and st_logg_min != 0:
                    combined_df.loc[idx, "st_dens_min"] = st_rad_min / st_logg_max
                    combined_df.loc[idx, "st_dens_max"] = st_rad_max / st_logg_min

after_na_min = combined_df["st_dens_min"].isna().sum()
after_na_max = combined_df["st_dens_max"].isna().sum()
recovered_min = before_na_min - after_na_min
recovered_max = before_na_max - after_na_max
print(f"   복구 전 NA (min): {before_na_min}개")
print(f"   복구 후 NA (min): {after_na_min}개")
print(
    f"   ✅ 복구된 개수 (min): {recovered_min}개 ({(recovered_min/before_na_min*100 if before_na_min > 0 else 0):.1f}%)"
)
print(f"   복구 전 NA (max): {before_na_max}개")
print(f"   복구 후 NA (max): {after_na_max}개")
print(
    f"   ✅ 복구된 개수 (max): {recovered_max}개 ({(recovered_max/before_na_max*100 if before_na_max > 0 else 0):.1f}%)"
)

# mission_temp 칼럼 제거
koi_datasets.drop(columns=["mission_temp"], inplace=True)
k2_datasets.drop(columns=["mission_temp"], inplace=True)
tess_datasets.drop(columns=["mission_temp"], inplace=True)

print("\n" + "=" * 70)
print("Phase 1 복구 완료!")
print("=" * 70)

# ========================================================================
# Phase 2: NA 값 통계적 대체
# ========================================================================
print("\n" + "=" * 70)
print("Phase 2: NA 값 통계적 대체")
print("=" * 70)

# 1. pl_orbper - NA가 적으므로 (0.71%) 행 제거
print("\n1. pl_orbper_min, pl_orbper_max NA 처리 (행 제거)...")
before_rows = len(combined_df)
before_na_min = combined_df["pl_orbper_min"].isna().sum()
before_na_max = combined_df["pl_orbper_max"].isna().sum()
combined_df = combined_df[
    combined_df["pl_orbper_min"].notna() & combined_df["pl_orbper_max"].notna()
]
removed = before_rows - len(combined_df)
print(f"   제거 전: {before_rows}개 행")
print(f"   제거 후: {len(combined_df)}개 행")
print(f"   ✅ {removed}개 행 제거 (pl_orbper NA)")

# 2. 나머지 칼럼들 - 미션별 중앙값으로 대체
na_columns = [
    "pl_insol_min",
    "pl_insol_max",
    "snr_min",
    "snr_max",
    "pl_rade_min",
    "pl_rade_max",
    "st_teff_min",
    "st_teff_max",
    "pl_trandurh_min",
    "pl_trandurh_max",
    "st_dens_min",
    "st_dens_max",
    "pl_trandep_min",
    "pl_trandep_max",
]

print("\n2. 미션별 중앙값으로 NA 대체...")
for col in na_columns:
    before_na = combined_df[col].isna().sum()
    if before_na > 0:
        # 미션별 중앙값 계산
        mission_medians = combined_df.groupby("mission")[col].median()

        # 각 미션별로 NA를 중앙값으로 대체
        for mission in combined_df["mission"].unique():
            mask = (combined_df["mission"] == mission) & (combined_df[col].isna())
            if mask.sum() > 0 and mission in mission_medians:
                combined_df.loc[mask, col] = mission_medians[mission]

        after_na = combined_df[col].isna().sum()
        recovered = before_na - after_na
        print(
            f"   {col:20s}: {before_na:4d} → {after_na:4d} NA ({recovered:4d}개 복구)"
        )

# 3. 여전히 NA가 남아있는 경우 (미션별 중앙값도 NA인 경우) - 전체 중앙값으로 대체
print("\n3. 전체 중앙값으로 남은 NA 대체...")
for col in na_columns:
    before_na = combined_df[col].isna().sum()
    if before_na > 0:
        overall_median = combined_df[col].median()
        combined_df[col].fillna(overall_median, inplace=True)
        after_na = combined_df[col].isna().sum()
        recovered = before_na - after_na
        if recovered > 0:
            print(f"   {col:20s}: {recovered}개 추가 복구 (전체 중앙값)")

print("\n" + "=" * 70)
print("Phase 2 복구 완료!")
print("=" * 70)

# 2. 이상치 제거 (99.7 percentile + 2개 이상 칼럼에서 이상치일 때만 제거)
print("\n이상치 제거 중...")
# max 값을 사용하여 이상치 판단
outlier_columns = ["pl_rade_max", "pl_orbper_max", "st_dens_max", "snr_max"]

# 원본 데이터 크기
original_size = len(combined_df)

# 각 칼럼별로 임계값을 먼저 계산 (99.7 percentile - 더 완화)
PERCENTILE = 0.997  # 상위 0.3%
MIN_OUTLIER_COLUMNS = 2  # 최소 2개 칼럼에서 이상치여야 제거

thresholds = {}
for col in outlier_columns:
    thresholds[col] = combined_df[col].quantile(PERCENTILE)
    print(f"  {col} 임계값: {thresholds[col]:.2f} (상위 {(1-PERCENTILE)*100:.1f}%)")

# 각 행마다 몇 개 칼럼에서 이상치인지 카운트
outlier_count_per_row = pd.Series([0] * len(combined_df), index=combined_df.index)

for col in outlier_columns:
    is_outlier = combined_df[col] > thresholds[col]
    outlier_count_per_row += is_outlier.astype(int)
    # 각 칼럼별로 몇 개가 이상치인지 확인
    outliers = is_outlier.sum()
    print(f"  {col}: {outliers}개 이상치 감지")

# 이상치 중복 분석
print("\n이상치 중복 분석:")
for i in range(1, 5):
    count = (outlier_count_per_row == i).sum()
    if count > 0:
        print(f"  {i}개 칼럼에서 이상치인 행: {count}개")

# 2개 이상 칼럼에서 이상치인 행만 제거
mask = outlier_count_per_row < MIN_OUTLIER_COLUMNS
combined_df = combined_df[mask]

total_removed = original_size - len(combined_df)
print(
    f"\n제거 기준: {MIN_OUTLIER_COLUMNS}개 이상 칼럼에서 상위 {(1-PERCENTILE)*100:.1f}% 초과"
)
print(f"전체 제거된 행: {total_removed}개")
print(f"제거 비율: {(total_removed/original_size)*100:.2f}%")

print(f"\n최종 데이터: {len(combined_df)} rows")

# 3. NA 값 확인
print("\n" + "=" * 60)
print("NA 값 분석")
print("=" * 60)

na_counts = combined_df.isna().sum()
na_percentage = combined_df.isna().sum() / len(combined_df) * 100

print(
    f"\n전체 데이터 크기: {len(combined_df)} rows × {len(combined_df.columns)} columns"
)
print(f"총 셀 개수: {len(combined_df) * len(combined_df.columns):,}")

# NA가 있는 칼럼만 출력
has_na = na_counts[na_counts > 0].sort_values(ascending=False)

if len(has_na) > 0:
    print(f"\nNA 값이 있는 칼럼: {len(has_na)}개")
    print("-" * 60)
    for col, count in has_na.items():
        pct = na_percentage[col]
        print(f"  {col:30s}: {count:5d}개 ({pct:5.2f}%)")

    print("\nNA 값 요약:")
    print(f"  총 NA 개수: {na_counts.sum():,}")
    print(
        f"  전체 대비: {(na_counts.sum() / (len(combined_df) * len(combined_df.columns)) * 100):.2f}%"
    )

    # 행 단위 NA 분석
    na_per_row = combined_df.isna().sum(axis=1)
    rows_with_na = (na_per_row > 0).sum()
    print(
        f"\nNA가 포함된 행: {rows_with_na}개 ({(rows_with_na/len(combined_df)*100):.2f}%)"
    )
    print(
        f"NA가 없는 완전한 행: {len(combined_df) - rows_with_na}개 ({((len(combined_df) - rows_with_na)/len(combined_df)*100):.2f}%)"
    )

    # NA 개수별 행 분포
    print("\n행당 NA 개수 분포:")
    for i in sorted(na_per_row.unique()):
        if i > 0:
            count = (na_per_row == i).sum()
            print(f"  {int(i):2d}개 NA: {count:5d}개 행")
else:
    print("\n✅ NA 값이 없습니다! 모든 데이터가 완전합니다.")

# 저장
print("\n" + "=" * 60)
print("데이터 저장 중...")
combined_df.to_csv("C:/workspace/nasa/datasets/preprocessed_all.csv", index=False)
print("✅ C:/workspace/nasa/datasets/preprocessed_all.csv 저장 완료")
