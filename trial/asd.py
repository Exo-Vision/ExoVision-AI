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

    # PlanetRadius와 TransitDuration 컬럼 선택
    if mission == "Kepler":
        radius_col = "koi_prad"
        duration_col = "koi_duration"
    elif mission == "K2":
        radius_col = "pl_rade"
        duration_col = "pl_trandur"
    else:  # TESS
        radius_col = "pl_rade"
        duration_col = "pl_trandurh"

    # NaN 값 제거
    radius_data = mission_data[radius_col].dropna()
    duration_data = mission_data[duration_col].dropna()

    # 데이터가 없으면 건너뛰기
    if len(radius_data) == 0 or len(duration_data) == 0:
        continue

    # 범위 구간 설정 (8개 범주)
    # PlanetRadius: 0-2, 2-4, 4-6, 6-8, 8-10, 10-15, 15-20, 20+
    radius_bins = [0, 2, 4, 6, 8, 10, 15, 20, float("inf")]
    # TransitDuration: 0-1, 1-2, 2-3, 3-4, 4-5, 5-7, 7-10, 10+
    duration_bins = [0, 1, 2, 3, 4, 5, 7, 10, float("inf")]

    # PlanetRadius 구간별 카운트
    radius_counts = (
        pd.cut(radius_data, bins=radius_bins, include_lowest=True)
        .value_counts()
        .sort_index()
    )
    radius_dict = {}
    for interval, count in radius_counts.items():
        if interval.right == float("inf"):
            key = f"({interval.left}+]"
        else:
            key = f"({interval.left} ~ {interval.right}]"
        radius_dict[key] = int(count)

    # TransitDuration 구간별 카운트
    duration_counts = (
        pd.cut(duration_data, bins=duration_bins, include_lowest=True)
        .value_counts()
        .sort_index()
    )
    duration_dict = {}
    for interval, count in duration_counts.items():
        if interval.right == float("inf"):
            key = f"({interval.left}+]"
        else:
            key = f"({interval.left} ~ {interval.right}]"
        duration_dict[key] = int(count)

    # TransitDuration 중앙값 계산 (소수점 3자리)
    duration_median = round(float(duration_data.median()), 3)

    # 결과 저장
    result[mission] = {
        "PlanetRadiusCount": radius_dict,
        "TransitDurationCount": duration_dict,
        "TransitDurationMedian": duration_median,
    }

# JSON 형식으로 출력
print(json.dumps(result, indent=2, ensure_ascii=False))

# JSON 파일로 저장
with open("output/mission_analysis.json", "w", encoding="utf-8") as f:
    json.dump(result, f, indent=2, ensure_ascii=False)
    json.dump(result, f, indent=2, ensure_ascii=False)
