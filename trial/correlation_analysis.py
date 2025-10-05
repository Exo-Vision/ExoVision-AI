import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# 전처리 완료 CSV 불러오기
tess_clean = pd.read_csv("datasets/final_exoplanets.csv")

# 외계행성 판별에 유용한 컬럼 선택
cols_for_corr = [
    "pl_rade",  # 행성 반지름
    "pl_orbper",  # 궤도 주기
    "pl_trandep",  # 트랜싯 깊이
    "pl_trandurh",  # 트랜싯 지속 시간
    "pl_tranmid",  # 트랜싯 중간 시점
    "pl_insol",  # 복사 에너지 (Insolation)
    "pl_eqt",  # 행성 평형 온도
    "st_teff",  # 항성 유효온도
    # "st_logg",  # 항성 표면 중력
    # "st_rad",  # 항성 반지름
    "st_dens",  # 항성 밀도
    "st_tmag",  # 항성 TESS magnitude
    "st_dist",  # 항성 거리
]

# 선택한 컬럼 데이터만 추출
data_corr = tess_clean[cols_for_corr]

# 상관계수 계산
corr_matrix = data_corr.corr()

# 히트맵 시각화
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation of Key Columns in Final Dataset")
# plt.show()
plt.savefig("corr.png")
