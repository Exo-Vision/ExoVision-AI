"""
병합된 데이터셋의 컬럼 분석 - 모델 학습에 불필요한 컬럼 식별
"""

from pathlib import Path

import pandas as pd

# 데이터 로드
DATA_DIR = Path(__file__).parent.parent / "datasets"
MERGED_PATH = DATA_DIR / "all_missions_merged.csv"

print("=" * 100)
print("컬럼 분석 - 모델 학습에 필요/불필요 컬럼 분류")
print("=" * 100)

merged_df = pd.read_csv(MERGED_PATH, low_memory=False)

print(f"\n전체 데이터: {len(merged_df)} 행, {len(merged_df.columns)} 컬럼")

# ============================================================================
# 컬럼 분류
# ============================================================================

# 1. 식별자/메타데이터 컬럼 (모델 학습 불필요)
identifier_cols = []
metadata_cols = []

# 2. 에러/불확실성 컬럼 (err1, err2, lim)
error_cols = []

# 3. 주요 물리 파라미터 (모델 학습 필요)
planet_params = []
stellar_params = []

# 4. 위치/시간 정보 (모델 학습 불필요)
position_time_cols = []

# 5. 미션별 고유 컬럼 (선택적)
mission_specific_cols = []

# 6. 기타
other_cols = []

for col in merged_df.columns:
    # 식별자
    if col in ["mission", "label"]:
        identifier_cols.append(col)

    # TESS 고유 식별자
    elif col in ["toi", "tid", "tess_toi", "tess_tid", "tfopwg_disp"]:
        identifier_cols.append(col)

    # Kepler 고유 식별자
    elif (
        col.startswith("kepler_kepid")
        or col.startswith("kepler_kepoi")
        or col.startswith("kepler_name")
    ):
        identifier_cols.append(col)

    # K2 고유 식별자
    elif (
        col.startswith("k2_name")
        or col.startswith("k2_pl_name")
        or col.startswith("k2_epic")
    ):
        identifier_cols.append(col)

    # 에러 컬럼
    elif "err1" in col or "err2" in col or "lim" in col:
        error_cols.append(col)

    # 위치 정보
    elif col in ["ra", "dec"]:
        position_time_cols.append(col)

    # 시간 정보 (Transit Midpoint)
    elif "tranmid" in col.lower() or "time0" in col.lower():
        position_time_cols.append(col)

    # 행성 물리 파라미터
    elif col.startswith("pl_") and not ("err" in col or "lim" in col):
        planet_params.append(col)

    # 항성 물리 파라미터
    elif col.startswith("st_") and not ("err" in col or "lim" in col):
        stellar_params.append(col)

    # Kepler 고유 파라미터
    elif col.startswith("kepler_") and col not in identifier_cols:
        mission_specific_cols.append(col)

    # K2 고유 파라미터
    elif col.startswith("k2_") and col not in identifier_cols:
        mission_specific_cols.append(col)

    # TESS 고유 파라미터
    elif col.startswith("tess_") and col not in identifier_cols:
        mission_specific_cols.append(col)

    else:
        other_cols.append(col)

# ============================================================================
# 결과 출력
# ============================================================================

print("\n" + "=" * 100)
print("1️⃣  식별자/메타데이터 컬럼 - ❌ 모델 학습 불필요")
print("=" * 100)
print("   (데이터 추적용, 학습에 사용하면 overfitting 위험)")
print(f"\n총 {len(identifier_cols)}개:")
for col in sorted(identifier_cols):
    sample_values = merged_df[col].dropna().unique()[:3]
    print(f"  - {col:30s} 예: {list(sample_values)}")

print("\n" + "=" * 100)
print("2️⃣  에러/불확실성 컬럼 - ⚠️  min/max로 변환 후 제거 가능")
print("=" * 100)
print("   (err1, err2, lim은 min/max 계산에 사용 후 제거)")
print(f"\n총 {len(error_cols)}개:")

# 그룹별로 정리
err_groups = {}
for col in error_cols:
    base = col.replace("err1", "").replace("err2", "").replace("lim", "")
    if base not in err_groups:
        err_groups[base] = []
    err_groups[base].append(col)

for base, cols in sorted(err_groups.items()):
    print(f"  {base}:")
    for col in sorted(cols):
        print(f"    - {col}")

print("\n" + "=" * 100)
print("3️⃣  위치/시간 정보 - ❌ 모델 학습 불필요")
print("=" * 100)
print("   (관측 위치/시각은 행성 특성과 무관)")
print(f"\n총 {len(position_time_cols)}개:")
for col in sorted(position_time_cols):
    if merged_df[col].notna().sum() > 0:
        print(
            f"  - {col:30s} (완성도: {merged_df[col].notna().sum()/len(merged_df)*100:.1f}%)"
        )

print("\n" + "=" * 100)
print("4️⃣  행성 물리 파라미터 - ✅ 모델 학습 필요")
print("=" * 100)
print("   (행성의 실제 특성, 중요!)")
print(f"\n총 {len(planet_params)}개:")

# 중요도 순으로 정렬
important_planet = [
    "pl_orbper",
    "pl_rade",
    "pl_trandep",
    "pl_trandurh",
    "pl_insol",
    "pl_eqt",
]

print("\n  🔥 핵심 파라미터 (우선순위 높음):")
for col in important_planet:
    if col in planet_params:
        completeness = merged_df[col].notna().sum() / len(merged_df) * 100
        print(f"    - {col:20s} (완성도: {completeness:5.1f}%)")

print("\n  📊 기타 행성 파라미터:")
for col in sorted(planet_params):
    if col not in important_planet:
        completeness = merged_df[col].notna().sum() / len(merged_df) * 100
        print(f"    - {col:20s} (완성도: {completeness:5.1f}%)")

print("\n" + "=" * 100)
print("5️⃣  항성 물리 파라미터 - ✅ 모델 학습 필요")
print("=" * 100)
print("   (모항성 특성, 행성 특성에 영향)")
print(f"\n총 {len(stellar_params)}개:")

important_stellar = ["st_teff", "st_logg", "st_rad"]

print("\n  🔥 핵심 파라미터 (우선순위 높음):")
for col in important_stellar:
    if col in stellar_params:
        completeness = merged_df[col].notna().sum() / len(merged_df) * 100
        print(f"    - {col:20s} (완성도: {completeness:5.1f}%)")

print("\n  📊 기타 항성 파라미터:")
for col in sorted(stellar_params):
    if col not in important_stellar:
        completeness = merged_df[col].notna().sum() / len(merged_df) * 100
        print(f"    - {col:20s} (완성도: {completeness:5.1f}%)")

print("\n" + "=" * 100)
print("6️⃣  미션별 고유 파라미터 - ⚠️  선택적 사용")
print("=" * 100)
print("   (미션 특정 정보, 일부는 유용할 수 있음)")
print(f"\n총 {len(mission_specific_cols)}개:")

print("\n  Kepler 고유:")
kepler_cols = [c for c in mission_specific_cols if c.startswith("kepler_")]
for col in sorted(kepler_cols)[:10]:  # 처음 10개만
    completeness = merged_df[col].notna().sum() / len(merged_df) * 100
    desc = ""
    if "score" in col:
        desc = " (행성 신뢰도 점수 - 유용!)"
    elif "fpflag" in col:
        desc = " (False Positive 플래그 - 유용!)"
    elif "snr" in col:
        desc = " (Signal-to-Noise Ratio - 유용!)"
    elif "quarters" in col:
        desc = " (관측 분기 정보)"
    print(f"    - {col:30s} (완성도: {completeness:5.1f}%){desc}")
if len(kepler_cols) > 10:
    print(f"    ... 외 {len(kepler_cols)-10}개")

print("\n  K2 고유:")
k2_cols = [c for c in mission_specific_cols if c.startswith("k2_")]
for col in sorted(k2_cols):
    completeness = merged_df[col].notna().sum() / len(merged_df) * 100
    print(f"    - {col:30s} (완성도: {completeness:5.1f}%)")

print("\n  TESS 고유:")
tess_cols = [c for c in mission_specific_cols if c.startswith("tess_")]
for col in sorted(tess_cols):
    completeness = merged_df[col].notna().sum() / len(merged_df) * 100
    print(f"    - {col:30s} (완성도: {completeness:5.1f}%)")

if other_cols:
    print("\n" + "=" * 100)
    print("7️⃣  기타 컬럼")
    print("=" * 100)
    print(f"\n총 {len(other_cols)}개:")
    for col in sorted(other_cols):
        print(f"  - {col}")

# ============================================================================
# 제거 권장 컬럼 요약
# ============================================================================

print("\n" + "=" * 100)
print("🗑️  제거 권장 컬럼 요약")
print("=" * 100)

remove_cols = (
    identifier_cols
    + error_cols
    + position_time_cols
    + [
        c
        for c in mission_specific_cols
        if not ("score" in c or "fpflag" in c or "snr" in c)
    ]
)

print(f"\n총 제거 권장: {len(remove_cols)}개 / {len(merged_df.columns)}개")
print(f"\n1. 식별자/메타데이터: {len(identifier_cols)}개")
print(f"2. 에러 컬럼 (min/max 변환 후): {len(error_cols)}개")
print(f"3. 위치/시간 정보: {len(position_time_cols)}개")
print(
    f"4. 미션별 고유 (유용하지 않은 것): {len([c for c in mission_specific_cols if not ('score' in c or 'fpflag' in c or 'snr' in c)])}개"
)

# ============================================================================
# 유지 권장 컬럼 요약
# ============================================================================

print("\n" + "=" * 100)
print("✅ 유지 권장 컬럼 요약")
print("=" * 100)

keep_cols = (
    ["mission", "label"]
    + planet_params
    + stellar_params
    + [
        c
        for c in mission_specific_cols
        if ("score" in c or "fpflag" in c or "snr" in c)
    ]
)

print(f"\n총 유지 권장: {len(keep_cols)}개 (min/max 추가 시 더 많아짐)")
print(f"\n1. 필수: mission, label (2개)")
print(f"2. 행성 파라미터: {len(planet_params)}개")
print(f"3. 항성 파라미터: {len(stellar_params)}개")
print(
    f"4. 유용한 미션별 파라미터: {len([c for c in mission_specific_cols if ('score' in c or 'fpflag' in c or 'snr' in c)])}개"
)

# ============================================================================
# 핵심 피처 리스트
# ============================================================================

print("\n" + "=" * 100)
print("🎯 추천: 최소 필수 피처 세트 (가장 중요한 것만)")
print("=" * 100)

essential_features = [
    "mission",  # 데이터 출처
    "label",  # 정답 레이블
    # 행성 핵심 6개
    "pl_orbper",  # 궤도 주기
    "pl_rade",  # 행성 반지름
    "pl_trandep",  # Transit depth
    "pl_trandurh",  # Transit duration
    "pl_insol",  # Insolation
    "pl_eqt",  # 평형 온도
    # 항성 핵심 3개
    "st_teff",  # 항성 온도
    "st_logg",  # 항성 중력
    "st_rad",  # 항성 반지름
]

print(
    f"\n총 {len(essential_features)}개 + min/max 버전 = 약 {len(essential_features) + 18}개"
)
print("\n기본 피처:")
for col in essential_features[:2]:
    print(f"  - {col}")

print("\n행성 피처:")
for col in essential_features[2:8]:
    completeness = (
        merged_df[col].notna().sum() / len(merged_df) * 100
        if col in merged_df.columns
        else 0
    )
    print(f"  - {col:15s} (완성도: {completeness:5.1f}%) → {col}_min, {col}_max 추가")

print("\n항성 피처:")
for col in essential_features[8:]:
    completeness = (
        merged_df[col].notna().sum() / len(merged_df) * 100
        if col in merged_df.columns
        else 0
    )
    print(f"  - {col:15s} (완성도: {completeness:5.1f}%) → {col}_min, {col}_max 추가")

# ============================================================================
# 추가 고려 피처
# ============================================================================

print("\n" + "=" * 100)
print("🔍 추가 고려 피처 (성능 향상 가능)")
print("=" * 100)

additional_features = [
    ("kepler_koi_score", "Kepler 행성 신뢰도 점수 (0-1)"),
    ("kepler_fpflag_nt", "Not Transit-Like 플래그"),
    ("kepler_fpflag_ss", "Stellar Eclipse 플래그"),
    ("kepler_fpflag_co", "Centroid Offset 플래그"),
    ("kepler_fpflag_ec", "Ephemeris Match 플래그"),
    ("kepler_model_snr", "Transit Signal-to-Noise"),
]

print(f"\n총 {len(additional_features)}개:")
for col, desc in additional_features:
    if col in merged_df.columns:
        completeness = merged_df[col].notna().sum() / len(merged_df) * 100
        print(f"  - {col:25s}: {desc} (완성도: {completeness:5.1f}%)")

print("\n" + "=" * 100)
print("💡 결론")
print("=" * 100)
print(
    """
1. 최소 필수: 11개 피처 + min/max (총 ~29개)
   → 빠른 프로토타입, 높은 완성도

2. 권장: 필수 + Kepler 고유 피처 (총 ~35개)
   → 더 나은 성능, Kepler 데이터 활용

3. 제거할 것:
   - 식별자 (kepid, toi, epic 등)
   - 위치/시간 (ra, dec, tranmid)
   - 에러 컬럼 (min/max 변환 후)
   - 미션별 이름 컬럼
"""
)

# 제거 권장 컬럼 리스트 저장
print("\n📝 제거 권장 컬럼 리스트를 파일로 저장합니다...")

remove_list_path = DATA_DIR.parent / "jh" / "columns_to_remove.txt"
with open(remove_list_path, "w", encoding="utf-8") as f:
    f.write("# 모델 학습에 불필요한 컬럼 리스트\n\n")
    f.write("## 1. 식별자/메타데이터\n")
    for col in sorted(identifier_cols):
        f.write(f"{col}\n")
    f.write("\n## 2. 에러 컬럼 (min/max 변환 후 제거)\n")
    for col in sorted(error_cols):
        f.write(f"{col}\n")
    f.write("\n## 3. 위치/시간 정보\n")
    for col in sorted(position_time_cols):
        f.write(f"{col}\n")
    f.write("\n## 4. 미션별 고유 (유용하지 않은 것)\n")
    for col in sorted(
        [
            c
            for c in mission_specific_cols
            if not ("score" in c or "fpflag" in c or "snr" in c)
        ]
    ):
        f.write(f"{col}\n")

print(f"✅ 저장 완료: {remove_list_path}")
