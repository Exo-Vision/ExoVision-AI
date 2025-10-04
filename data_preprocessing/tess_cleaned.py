import pandas as pd
import numpy as np

# CSV 불러오기
tess = pd.read_csv('datasets/tess.csv')
print("원본 데이터 개수:", len(tess))

# 사용할 컬럼
columns = [
    "tfopwg_disp", "pl_rade", "pl_radeerr1", "pl_radeerr2",
    "pl_orbper", "pl_orbpererr1", "pl_orbpererr2",
    "pl_trandep", "pl_trandeperr1", "pl_trandeperr2",
    "pl_trandurh", "pl_trandurherr1", "pl_trandurherr2",
    "pl_tranmid", "pl_tranmiderr1", "pl_tranmiderr2",
    "pl_insol", "pl_insolerr1", "pl_insolerr2",
    "pl_eqt", "pl_eqterr1", "pl_eqterr2",
    "st_teff", "st_tefferr1", "st_tefferr2",
    "st_logg", "st_loggerr1", "st_loggerr2",
    "st_rad", "st_raderr1", "st_raderr2",
    "st_tmag", "st_tmagerr1", "st_tmagerr2",
    "st_dist", "st_disterr1", "st_disterr2"
]

tess_clean = tess[columns].copy()

# Limit Flag 컬럼 (있으면)
limit_flags = {
    "pl_rade": "pl_radelim",
    "pl_orbper": "pl_orbperlim",
    "pl_trandep": "pl_trandeplim",
    "pl_trandurh": "pl_trandurhlim",
    "pl_tranmid": "pl_tranmidlim",
    "pl_insol": "pl_insollim",
    "pl_eqt": "pl_eqtlim",
    "st_teff": "st_tefflim",
    "st_logg": "st_logglim",
    "st_rad": "st_radlim",
    "st_tmag": "st_tmaglim",
    "st_dist": "st_distlim"
}

# 신뢰도 계산 함수
def calc_conf(row, col_prefix):
    val = row[col_prefix]
    err1 = row.get(f"{col_prefix}err1", np.nan)
    err2 = row.get(f"{col_prefix}err2", np.nan)
    lim_flag = row.get(limit_flags.get(col_prefix, ""), 0)

    if pd.isna(val) or pd.isna(err1) or pd.isna(err2) or lim_flag == 1:
        return np.nan
    rel_err = (abs(err1) + abs(err2)) / (2 * val)
    return max(0, 1 - rel_err)

# 신뢰도 컬럼 생성
for col in ["pl_rade", "pl_orbper", "pl_trandep", "pl_trandurh",
            "pl_tranmid", "pl_insol", "pl_eqt",
            "st_teff", "st_logg", "st_rad", "st_tmag", "st_dist"]:
    tess_clean[f"{col}_conf"] = tess_clean.apply(lambda row: calc_conf(row, col), axis=1)

# confidence 평균
conf_cols = [c for c in tess_clean.columns if c.endswith("_conf")]
tess_clean["confidence"] = tess_clean[conf_cols].mean(axis=1, skipna=True)

# tfopwg_disp가 PC 또는 CP만 남기기
tess_clean = tess_clean[tess_clean["tfopwg_disp"].isin(["PC", "CP"])]

# confidence 너무 낮은 데이터 제거 (0.5 미만)
tess_clean = tess_clean[tess_clean["confidence"] >= 0.5]

# 필요없는 개별 err, conf 컬럼 제거
drop_cols = [c for c in tess_clean.columns if c.endswith("_err1") or c.endswith("_err2") or c.endswith("_conf")]
tess_clean = tess_clean.drop(columns=drop_cols)

# 새로운 파생 변수 생성: 항성 밀도 근사
tess_clean["st_dens"] = tess_clean["st_logg"] / tess_clean["st_rad"]
tess_clean = tess_clean.drop(columns=["st_logg"])
tess_clean = tess_clean.dropna(subset=["st_rad"])

# CSV로 저장
tess_clean.to_csv("datasets/tess_cleaned.csv", index=False)

print("전처리 후 데이터 개수:", len(tess_clean))
print("tfopwg_disp 분포:\n", tess_clean["tfopwg_disp"].value_counts())
print("전처리 완료! CSV 저장됨: datasets/tess_cleaned.csv")
