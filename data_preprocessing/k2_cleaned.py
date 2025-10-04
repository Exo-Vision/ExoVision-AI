import pandas as pd
import numpy as np

# K2 데이터 로드
k2 = pd.read_csv("datasets/k2.csv")

def process_with_errors(df, error_threshold=0.5):
    """
    값(value) + 에러(err1, err2) 형태의 컬럼을 자동 처리:
    - *_corrected : 보정된 값
    - *_relerr    : 상대 오차
    - 신뢰도 낮으면 np.nan 으로 대체
    """
    new_df = df.copy()

    for col in df.columns:
        if col.endswith("err1"):  # 예: pl_radeerr1
            base = col.replace("err1", "")
            err1 = df[col]
            err2 = df.get(base + "err2", None)
            value = df.get(base, None)

            if value is None or err2 is None:
                continue

            # 상대 오차 계산
            rel_err = (err1.abs() + err2.abs()) / (2 * value.abs().replace(0, np.nan))
            new_df[base + "_relerr"] = rel_err

            # 보정값 계산 (비대칭 오차 고려 → 중앙값 근사)
            corrected = value + (err1 - err2) / 2
            new_df[base + "_corrected"] = corrected

            # threshold 이상이면 NaN 처리
            new_df.loc[rel_err > error_threshold, base + "_corrected"] = np.nan

    # 전체 신뢰도 점수 = 상대 오차 평균
    relerr_cols = [c for c in new_df.columns if c.endswith("_relerr")]
    if relerr_cols:
        new_df["reliability_score"] = new_df[relerr_cols].mean(axis=1)

    return new_df

# 에러 처리
k2 = process_with_errors(k2, error_threshold=0.5)

# 라벨 매핑
label_mapping = {
    "CANDIDATE": "PC",
    "CONFIRMED": "CP",
    "FALSE POSITIVE": "FP",
    "REFUTED": "FA"
}
k2["disposition"] = k2["disposition"].map(label_mapping)

# 컬럼명 통일
k2_renamed = k2.rename(columns={
    "disposition": "tfopwg_disp",
    "pl_trandur": "pl_trandurh",
    "sy_tmag": "st_tmag",
    "sy_tmagerr1": "st_tmagerr1",
    "sy_tmagerr2": "st_tmagerr2",
    "sy_dist": "st_dist",
    "sy_disterr1": "st_disterr1",
    "sy_disterr2": "st_disterr2",
})

# 원하는 컬럼 순서
columns_to_keep = [
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

# confidence 기본값 채우기
k2_renamed["confidence"] = 1.0

# 신뢰도 기준 필터링
k2_filtered = k2_renamed[k2_renamed.get("reliability_score", 0) < 0.3]

# 원하는 컬럼만 남기기
k2_clean = k2_filtered.reindex(columns=columns_to_keep)

# 라벨 없는 행 제거
k2_clean = k2_clean.dropna(subset=["tfopwg_disp"])

# CSV로 저장
k2_clean.to_csv("datasets/k2_cleaned_filtered.csv", index=False)

print("원본 데이터 개수 :", len(k2))
print("신뢰도 필터링 후 개수 :", len(k2_clean))
print(k2_clean["tfopwg_disp"].value_counts())
