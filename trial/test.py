"""
pl_pnum
pl_rade
pl_orbper
pl_trandep
pl_trandurh
pl_insol → 별에 의해 transit 크기와 뭔가 관계가 있을거같은데?
st_teff
st_tmag
st_dist
st_dens → 이건 st_rad / st_logg 계산해서 밀도로 변경한 칼럼
"""

import pandas as pd

# KOI 데이터셋 불러오기 (Kepler)
print("=== KOI 데이터셋 로드 및 kepid 그룹 카운트 확인 ===")
kepler = pd.read_csv("datasets/kepler.csv")

print(f"\n전체 데이터 수: {len(kepler)}")
print(f"\n데이터 컬럼: {kepler.columns.tolist()}")

# kepid가 있는지 확인
if "kepid" in kepler.columns:
    print(f"\nkepid 컬럼 존재 ✓")

    # kepid로 그룹화하여 카운트
    kepid_counts = kepler.groupby("kepid").size()

    print(f"\n고유한 kepid 수: {kepid_counts.shape[0]}")
    print(f"평균 행성 수/별: {kepid_counts.mean():.2f}")
    print(f"최대 행성 수/별: {kepid_counts.max()}")
    print(f"최소 행성 수/별: {kepid_counts.min()}")

    print(f"\n행성 수별 별 분포:")
    print(kepid_counts.value_counts().sort_index().head(10))

    print(f"\n가장 많은 행성을 가진 별 TOP 5:")
    print(kepid_counts.sort_values(ascending=False).head())

else:
    print("\nkepid 컬럼이 없습니다!")
    print(
        f"사용 가능한 ID 컬럼: {[col for col in kepler.columns if 'id' in col.lower()]}"
    )
