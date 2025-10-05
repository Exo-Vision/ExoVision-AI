"""
모델 학습용 데이터셋 준비 - 옵션 1: 최소 필수 피처 (29개)
- 기본 2개: mission, label
- 행성 파라미터 6개 + min/max (18개)
- 항성 파라미터 3개 + min/max (9개)
"""

from pathlib import Path

import numpy as np
import pandas as pd

# 데이터 로드
DATA_DIR = Path(__file__).parent.parent / "datasets"
INPUT_PATH = DATA_DIR / "all_missions_with_ranges.csv"
OUTPUT_PATH = DATA_DIR / "model_ready_data.csv"

print("=" * 100)
print("모델 학습용 데이터셋 준비 - 옵션 1: 최소 필수 피처")
print("=" * 100)

df = pd.read_csv(INPUT_PATH, low_memory=False)

print(f"\n원본 데이터: {len(df)} 행, {len(df.columns)} 컬럼")

# ============================================================================
# 1. 필수 피처 선택
# ============================================================================

print("\n" + "=" * 100)
print("1️⃣  필수 피처 선택")
print("=" * 100)

# 기본 피처
basic_features = ["mission", "label"]

# 행성 파라미터 (원본 + min/max)
planet_features = [
    "pl_orbper",
    "pl_orbper_min",
    "pl_orbper_max",
    "pl_rade",
    "pl_rade_min",
    "pl_rade_max",
    "pl_trandep",
    "pl_trandep_min",
    "pl_trandep_max",
    "pl_trandurh",
    "pl_trandurh_min",
    "pl_trandurh_max",
    "pl_insol",
    "pl_insol_min",
    "pl_insol_max",
    "pl_eqt",
    "pl_eqt_min",
    "pl_eqt_max",
]

# 항성 파라미터 (원본 + min/max)
stellar_features = [
    "st_teff",
    "st_teff_min",
    "st_teff_max",
    "st_logg",
    "st_logg_min",
    "st_logg_max",
    "st_rad",
    "st_rad_min",
    "st_rad_max",
]

# 전체 피처 리스트
selected_features = basic_features + planet_features + stellar_features

print(f"\n선택된 피처: {len(selected_features)}개")
print(f"  - 기본: {len(basic_features)}개")
print(f"  - 행성: {len(planet_features)}개 (원본 6개 + min/max 12개)")
print(f"  - 항성: {len(stellar_features)}개 (원본 3개 + min/max 6개)")

# 데이터 필터링
model_df = df[selected_features].copy()

print(f"\n필터링 후: {len(model_df)} 행, {len(model_df.columns)} 컬럼")

# ============================================================================
# 2. 데이터 품질 확인
# ============================================================================

print("\n" + "=" * 100)
print("2️⃣  데이터 품질 확인")
print("=" * 100)

print("\n미션별 분포:")
mission_counts = model_df["mission"].value_counts()
for mission, count in mission_counts.items():
    pct = (count / len(model_df)) * 100
    print(f"  {mission:10s}: {count:6d} ({pct:5.2f}%)")

print("\nLabel 분포:")
label_counts = model_df["label"].value_counts()
for label, count in label_counts.items():
    pct = (count / len(model_df)) * 100
    desc = {
        "PC": "Planet Candidate",
        "CP": "Confirmed Planet",
        "FP": "False Positive",
    }.get(label, label)
    print(f"  {label} ({desc:20s}): {count:6d} ({pct:5.2f}%)")

# ============================================================================
# 3. 결측치 분석
# ============================================================================

print("\n" + "=" * 100)
print("3️⃣  결측치 분석")
print("=" * 100)

# 원본 값 컬럼만 (min/max는 원본과 동일하므로 제외)
original_features = [
    "pl_orbper",
    "pl_rade",
    "pl_trandep",
    "pl_trandurh",
    "pl_insol",
    "pl_eqt",
    "st_teff",
    "st_logg",
    "st_rad",
]

print("\n주요 피처별 결측치:")
for col in original_features:
    missing = model_df[col].isna().sum()
    pct = (missing / len(model_df)) * 100
    status = "✅" if pct < 10 else "⚠️" if pct < 30 else "❌"
    print(f"  {status} {col:15s}: {missing:6d} ({pct:5.2f}%)")

# 행별 결측치 개수
missing_per_row = model_df[original_features].isna().sum(axis=1)

print("\n행별 결측치 분포:")
for i in range(10):
    count = (missing_per_row == i).sum()
    if count > 0:
        pct = (count / len(model_df)) * 100
        print(f"  {i}개 결측: {count:6d} ({pct:5.2f}%)")

# 완전한 행 (결측치 없는 행)
complete_rows = (missing_per_row == 0).sum()
print(
    f"\n✅ 완전한 행 (결측치 0개): {complete_rows:6d} ({complete_rows/len(model_df)*100:5.2f}%)"
)

# ============================================================================
# 4. 결측치 처리 전략
# ============================================================================

print("\n" + "=" * 100)
print("4️⃣  결측치 처리 전략")
print("=" * 100)

print("\n옵션:")
print("  A. 완전한 행만 사용 (결측치 0개) - 가장 안전, 데이터 손실 있음")
print("  B. 결측치 1-2개까지 허용 - 중간")
print("  C. 모든 데이터 사용 + 결측치 채우기 - 데이터 최대 활용")

# 옵션별 데이터 개수
option_a = (missing_per_row == 0).sum()
option_b = (missing_per_row <= 2).sum()
option_c = len(model_df)

print(f"\n  A 선택 시: {option_a:6d} 행 ({option_a/len(model_df)*100:5.2f}%)")
print(f"  B 선택 시: {option_b:6d} 행 ({option_b/len(model_df)*100:5.2f}%)")
print(f"  C 선택 시: {option_c:6d} 행 ({option_c/len(model_df)*100:5.2f}%)")

# 미션별 완전한 행 비율
print("\n미션별 완전한 행 비율:")
for mission in model_df["mission"].unique():
    mission_mask = model_df["mission"] == mission
    mission_complete = (missing_per_row[mission_mask] == 0).sum()
    mission_total = mission_mask.sum()
    pct = (mission_complete / mission_total) * 100
    print(f"  {mission:10s}: {mission_complete:5d}/{mission_total:5d} ({pct:5.2f}%)")

# ============================================================================
# 5. 옵션 C 선택 - 미션별 중앙값으로 결측치 채우기
# ============================================================================

print("\n" + "=" * 100)
print("5️⃣  결측치 처리 - 옵션 C (미션별 중앙값)")
print("=" * 100)

model_df_filled = model_df.copy()

# 미션별로 중앙값 계산 및 채우기
for col in original_features:
    before_na = model_df_filled[col].isna().sum()

    if before_na > 0:
        # 각 미션별 중앙값 계산
        for mission in model_df_filled["mission"].unique():
            mission_mask = model_df_filled["mission"] == mission
            mission_median = model_df_filled.loc[mission_mask, col].median()

            # 해당 미션의 결측치를 중앙값으로 채우기
            na_mask = mission_mask & model_df_filled[col].isna()
            if na_mask.sum() > 0 and pd.notna(mission_median):
                model_df_filled.loc[na_mask, col] = mission_median

                # min/max도 동일하게 채우기
                if f"{col}_min" in model_df_filled.columns:
                    model_df_filled.loc[na_mask, f"{col}_min"] = mission_median
                if f"{col}_max" in model_df_filled.columns:
                    model_df_filled.loc[na_mask, f"{col}_max"] = mission_median

        after_na = model_df_filled[col].isna().sum()
        filled = before_na - after_na

        if filled > 0:
            print(f"  {col:15s}: {before_na:5d} → {after_na:5d} ({filled:5d}개 채움)")

# 여전히 NA인 경우 전체 중앙값으로 채우기
print("\n전체 중앙값으로 남은 결측치 채우기:")
for col in original_features:
    before_na = model_df_filled[col].isna().sum()

    if before_na > 0:
        overall_median = model_df_filled[col].median()
        model_df_filled[col].fillna(overall_median, inplace=True)

        # min/max도 동일하게
        if f"{col}_min" in model_df_filled.columns:
            model_df_filled[f"{col}_min"].fillna(overall_median, inplace=True)
        if f"{col}_max" in model_df_filled.columns:
            model_df_filled[f"{col}_max"].fillna(overall_median, inplace=True)

        after_na = model_df_filled[col].isna().sum()
        filled = before_na - after_na

        if filled > 0:
            print(f"  {col:15s}: {filled}개 추가 채움")

# 최종 확인
total_na = model_df_filled[original_features].isna().sum().sum()
print(f"\n✅ 최종 결측치: {total_na}개")

# ============================================================================
# 6. 최종 데이터 통계
# ============================================================================

print("\n" + "=" * 100)
print("6️⃣  최종 데이터 통계")
print("=" * 100)

print(f"\n데이터 크기: {len(model_df_filled)} 행 × {len(model_df_filled.columns)} 컬럼")

print("\n피처별 통계 (원본 값):")
for col in original_features:
    stats = model_df_filled[col].describe()
    print(f"\n  {col}:")
    print(f"    평균: {stats['mean']:12.2f}")
    print(f"    중앙값: {stats['50%']:10.2f}")
    print(f"    최소: {stats['min']:12.2f}")
    print(f"    최대: {stats['max']:12.2f}")
    print(f"    표준편차: {stats['std']:8.2f}")

# ============================================================================
# 7. 데이터 저장
# ============================================================================

print("\n" + "=" * 100)
print("7️⃣  데이터 저장")
print("=" * 100)

# mission을 숫자로 인코딩 (선택적)
print("\nmission 인코딩:")
mission_mapping = {"Kepler": 0, "K2": 1, "TESS": 2}
model_df_filled["mission_encoded"] = model_df_filled["mission"].map(mission_mapping)
for mission, code in mission_mapping.items():
    print(f"  {mission}: {code}")

# label을 숫자로 인코딩
print("\nlabel 인코딩:")
label_mapping = {"FP": 0, "PC": 1, "CP": 2}
model_df_filled["label_encoded"] = model_df_filled["label"].map(label_mapping)
for label, code in label_mapping.items():
    desc = {"FP": "False Positive", "PC": "Planet Candidate", "CP": "Confirmed Planet"}[
        label
    ]
    print(f"  {label} ({desc:20s}): {code}")

# 저장
model_df_filled.to_csv(OUTPUT_PATH, index=False)
print(f"\n✅ 저장 완료: {OUTPUT_PATH}")

# ============================================================================
# 8. 최종 요약
# ============================================================================

print("\n" + "=" * 100)
print("🎉 최종 요약")
print("=" * 100)

print(
    f"""
📊 데이터셋 정보:
  - 총 데이터: {len(model_df_filled):,}개
  - 총 피처: {len(model_df_filled.columns)}개
    * 기본: 2개 (mission, label)
    * 행성: 18개 (6개 × 3: 원본, min, max)
    * 항성: 9개 (3개 × 3: 원본, min, max)
    * 인코딩: 2개 (mission_encoded, label_encoded)

📈 Label 분포:
  - False Positive (FP): {label_counts.get('FP', 0):,}개 ({label_counts.get('FP', 0)/len(model_df_filled)*100:.1f}%)
  - Planet Candidate (PC): {label_counts.get('PC', 0):,}개 ({label_counts.get('PC', 0)/len(model_df_filled)*100:.1f}%)
  - Confirmed Planet (CP): {label_counts.get('CP', 0):,}개 ({label_counts.get('CP', 0)/len(model_df_filled)*100:.1f}%)

🚀 다음 단계:
  1. 데이터 로드: pd.read_csv('{OUTPUT_PATH}')
  2. 피처와 레이블 분리
  3. Train/Test Split
  4. 모델 학습 (RandomForest, XGBoost, CatBoost 등)
  5. 평가 및 튜닝

💡 추천 피처 세트:
  - 기본 학습: 원본 9개 피처 (pl_orbper, pl_rade, ..., st_rad)
  - 고급 학습: 원본 + min/max 27개 피처 (불확실성 정보 포함)
  - mission_encoded: 선택적 사용 (미션별 특성 반영)
"""
)

# 컬럼 리스트 저장
print("\n📝 피처 리스트 저장...")
feature_list_path = DATA_DIR.parent / "jh" / "model_features.txt"
with open(feature_list_path, "w", encoding="utf-8") as f:
    f.write("# 모델 학습용 피처 리스트\n\n")
    f.write("## 기본 피처\n")
    for col in basic_features:
        f.write(f"{col}\n")
    f.write("\n## 행성 파라미터\n")
    for col in planet_features:
        f.write(f"{col}\n")
    f.write("\n## 항성 파라미터\n")
    for col in stellar_features:
        f.write(f"{col}\n")
    f.write("\n## 인코딩 피처\n")
    f.write("mission_encoded\n")
    f.write("label_encoded\n")

print(f"✅ 저장 완료: {feature_list_path}")

print("\n" + "=" * 100)
print("✅ 모델 학습용 데이터 준비 완료!")
print("=" * 100)
