import pandas as pd

# CSV 파일 불러오기
kepler = pd.read_csv("datasets/kepler.csv")
k2 = pd.read_csv("datasets/k2.csv")

# kepler -> TESS 매핑
kepler_tess = pd.DataFrame({
    "toi": kepler["kepoi_name"],
    "tfopwg_disp": kepler["koi_disposition"],
    "pl_orbper": kepler["koi_period"],
    "pl_orbpererr1": kepler["koi_period_err1"],
    "pl_orbpererr2": kepler["koi_period_err2"],
    "pl_trandurh": kepler["koi_duration"],
    "pl_trandurherr1": kepler["koi_duration_err1"],
    "pl_trandurherr2": kepler["koi_duration_err2"],
    "pl_trandep": kepler["koi_depth"],
    "pl_trandeperr1": kepler["koi_depth_err1"],
    "pl_trandeperr2": kepler["koi_depth_err2"],
    "pl_rade": kepler["koi_prad"],
    "pl_radeerr1": kepler["koi_prad_err1"],
    "pl_radeerr2": kepler["koi_prad_err2"],
    "pl_insol": kepler["koi_insol"],
    "pl_eqt": kepler["koi_teq"],
    "st_teff": kepler["koi_steff"],
    "st_logg": kepler["koi_slogg"],
    "st_rad": kepler["koi_srad"],
    "ra": kepler["ra"],
    "dec": kepler["dec"]
})

# k2 -> TESS 매핑
k2_tess = pd.DataFrame({
    "toi": k2["pl_name"],
    "tfopwg_disp": k2["disposition"],
    "pl_orbper": k2["pl_orbper"],
    "pl_orbpererr1": k2["pl_orbpererr1"],
    "pl_orbpererr2": k2["pl_orbpererr2"],
    "pl_trandurh": k2["pl_trandur"],
    "pl_trandurherr1": k2["pl_trandurerr1"],
    "pl_trandurherr2": k2["pl_trandurerr2"],
    "pl_trandep": k2["pl_trandep"],
    "pl_trandeperr1": k2["pl_trandeperr1"],
    "pl_trandeperr2": k2["pl_trandeperr2"],
    "pl_rade": k2["pl_rade"],
    "pl_radeerr1": k2["pl_radeerr1"],
    "pl_radeerr2": k2["pl_radeerr2"],
    "pl_insol": k2["pl_insol"],
    "pl_eqt": k2["pl_eqt"],
    "st_teff": k2["st_teff"],
    "st_logg": k2["st_logg"],
    "st_rad": k2["st_rad"],
    "ra": k2["ra"],
    "dec": k2["dec"]
})

# 두 데이터를 합치기
feature_table = pd.concat([kepler_tess, k2_tess], ignore_index=True)

# CSV로 저장
feature_table.to_csv("datasets/feature_table.csv", index=False)

print("Feature table 생성 및 CSV 저장 완료!")
