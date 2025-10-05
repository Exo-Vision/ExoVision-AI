import pandas as pd
import numpy as np

# CSV 불러오기
kepler = pd.read_csv("datasets/kepler.csv")
print("원본 데이터 개수:", len(kepler))

# 라벨 매핑 (Kepler → TESS)
label_mapping = {
    "CANDIDATE": "PC",
    "CONFIRMED": "CP",
    "FALSE POSITIVE": "FP"
}
kepler["tfopwg_disp"] = kepler["koi_disposition"].map(label_mapping)

# 사용할 값 + err 컬럼 매핑 (Kepler → TESS)
col_map = {
    "pl_rade": ("koi_prad", "koi_prad_err1", "koi_prad_err2"),
    "pl_orbper": ("koi_period", "koi_period_err1", "koi_period_err2"),
    "pl_trandep": ("koi_depth", "koi_depth_err1", "koi_depth_err2"),
    "pl_trandurh": ("koi_duration", "koi_duration_err1", "koi_duration_err2"),
    "pl_tranmid": ("koi_time0bk", "koi_time0bk_err1", "koi_time0bk_err2"),
    "pl_insol": ("koi_insol", "koi_insol_err1", "koi_insol_err2"),
    "pl_eqt": ("koi_teq", "koi_teq_err1", "koi_teq_err2"),
    "st_teff": ("koi_steff", "koi_steff_err1", "koi_steff_err2"),
    "st_loggerr1": ("koi_slogg_err1", None, None),
    "st_loggerr2": ("koi_slogg_err2", None, None),
    "st_rad": ("koi_srad", "koi_srad_err1", "koi_srad_err2"),
    "st_tmag": ("koi_kepmag", "koi_kepmag", "koi_kepmag"),  # 오류 없으면 그대로
    "st_dist": ("koi_sma", "koi_sma_err1", "koi_sma_err2")  # 단순 예시
}

kepler_clean = pd.DataFrame()
kepler_clean["tfopwg_disp"] = kepler["tfopwg_disp"]

# 신뢰도 계산 함수
def calc_conf(val, err1, err2):
    # NaN 또는 값이 0이면 제외
    if pd.isna(val) or pd.isna(err1) or pd.isna(err2) or val == 0:
        return np.nan
    rel_err = (abs(err1) + abs(err2)) / (2 * abs(val))  # 절대값 사용
    return max(0, 1 - rel_err)

# 컬럼 처리
for new_col, (val_col, err1_col, err2_col) in col_map.items():
    kepler_clean[new_col] = kepler[val_col]
    # 상대 오차 기반 confidence
    if err1_col and err2_col:
        kepler_clean[f"{new_col}_conf"] = kepler.apply(lambda row: calc_conf(row[val_col], row[err1_col], row[err2_col]), axis=1)
    else:
        kepler_clean[f"{new_col}_conf"] = 1.0  # 오류 없는 경우 기본 1

# confidence 평균 계산
conf_cols = [c for c in kepler_clean.columns if c.endswith("_conf")]
kepler_clean["confidence"] = kepler_clean[conf_cols].mean(axis=1, skipna=True)

# confidence 0.5 미만 제거
kepler_clean = kepler_clean[kepler_clean["confidence"] >= 0.5]

# 개별 err, conf 컬럼 제거
drop_cols = [c for c in kepler_clean.columns if c.endswith("_conf")]
kepler_clean = kepler_clean.drop(columns=drop_cols)

# 항성 밀도 생성 (근사)
kepler_clean["st_dens"] = kepler_clean["st_loggerr1"] / kepler_clean["st_rad"]

# 필요 없는 컬럼 제거
kepler_clean = kepler_clean.drop(columns=["st_loggerr1", "st_loggerr2"])

# TESS 기준 컬럼 순서
tess_columns = [
    "tfopwg_disp","pl_rade","pl_radeerr1","pl_radeerr2",
    "pl_orbper","pl_orbpererr1","pl_orbpererr2",
    "pl_trandep","pl_trandeperr1","pl_trandeperr2",
    "pl_trandurh","pl_trandurherr1","pl_trandurherr2",
    "pl_tranmid","pl_tranmiderr1","pl_tranmiderr2",
    "pl_insol","pl_insolerr1","pl_insolerr2",
    "pl_eqt","pl_eqterr1","pl_eqterr2",
    "st_teff","st_tefferr1","st_tefferr2",
    "st_loggerr1","st_loggerr2",
    "st_rad","st_raderr1","st_raderr2",
    "st_tmag","st_tmagerr1","st_tmagerr2",
    "st_dist","st_disterr1","st_disterr2",
    "confidence","st_dens"
]

# 컬럼 없는 경우 대비
for col in tess_columns:
    if col not in kepler_clean.columns:
        kepler_clean[col] = np.nan

kepler_clean = kepler_clean[tess_columns]

# 전처리 후 데이터 개수
print("전처리 후 데이터 개수:", len(kepler_clean))

# 구분 컬럼 추가
kepler_clean["source"] = "Kepler"

# CSV 저장
kepler_clean.to_csv("datasets/kepler_cleaned.csv", index=False)
print("CSV 저장 완료: datasets/kepler_cleaned.csv")
