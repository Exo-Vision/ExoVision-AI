import json

import numpy as np
import pandas as pd

# CSV 파일 읽기
exoplanets = pd.read_csv("output/exoplanets.csv", low_memory=False)

# 결과를 저장할 딕셔너리
result = {}

# 각 미션별로 처리
missions = ["K2", "Kepler", "TESS"]

for mission in missions:
    # 미션 데이터 필터링
    mission_data = exoplanets[exoplanets["mission"] == mission].copy()

    # 각 미션별로 컬럼 선택
    if mission == "Kepler":
        period_col = "koi_period"
        radius_col = "koi_prad"
        depth_col = "koi_depth"
    elif mission == "K2":
        period_col = "pl_orbper"
        radius_col = "pl_rade"
        depth_col = "pl_trandep"
    else:  # TESS
        period_col = "pl_orbper"
        radius_col = "pl_rade"
        depth_col = "pl_trandep"

    # 필요한 컬럼만 선택하고 NaN 제거
    scatter_data = mission_data[[period_col, radius_col, depth_col]].dropna()

    # 데이터가 없으면 건너뛰기
    if len(scatter_data) == 0:
        continue

    # 각 데이터 포인트를 리스트로 저장
    data_points = []
    for idx, row in scatter_data.iterrows():
        data_points.append(
            {
                "period": round(float(row[period_col]), 3),
                "planet_radius": round(float(row[radius_col]), 3),
                "transit_depth": round(float(row[depth_col]), 3),
            }
        )

    # 결과 저장
    result[mission] = {"count": len(data_points), "data": data_points}

# JSON 파일로 저장
with open("output/mission_scatter_data.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)

print("✓ JSON 파일 저장 완료: output/mission_scatter_data.json")

# 각 미션별 통계 정보 출력
print("\n=== 통계 정보 ===")
for mission in missions:
    if mission in result:
        print(f"\n{mission}:")
        print(f"  데이터 포인트 개수: {result[mission]['count']}")
        data = result[mission]["data"]
        periods = [d["period"] for d in data]
        radii = [d["planet_radius"] for d in data]
        depths = [d["transit_depth"] for d in data]
        print(f"  Period 범위: {min(periods):.3f} ~ {max(periods):.3f}")
        print(f"  Planet Radius 범위: {min(radii):.3f} ~ {max(radii):.3f}")
        print(f"  Transit Depth 범위: {min(depths):.3f} ~ {max(depths):.3f}")
        print(f"  Transit Depth 범위: {min(depths):.3f} ~ {max(depths):.3f}")
