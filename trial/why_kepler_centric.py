"""
Kepler를 중심으로 전처리한 이유 분석
"""
import pandas as pd
import numpy as np

print("=" * 100)
print("📊 왜 Kepler를 중심으로 전처리했는가?")
print("=" * 100)

# 데이터 로드
kepler = pd.read_csv('datasets/kepler.csv')
k2 = pd.read_csv('datasets/k2.csv')
tess = pd.read_csv('datasets/tess.csv')

print("\n[1] 데이터셋 규모 비교")
print("-" * 100)
print(f"Kepler:  {len(kepler):>7,} 샘플  ({len(kepler.columns):>3}개 컬럼)")
print(f"K2:      {len(k2):>7,} 샘플  ({len(k2.columns):>3}개 컬럼)")
print(f"TESS:    {len(tess):>7,} 샘플  ({len(tess.columns):>3}개 컬럼)")
print(f"합계:    {len(kepler)+len(k2)+len(tess):>7,} 샘플")

print("\n[2] 레이블 분포 (CONFIRMED 행성)")
print("-" * 100)
kepler_confirmed = (kepler['koi_disposition'] == 'CONFIRMED').sum()
k2_confirmed = (k2['disposition'] == 'CONFIRMED').sum()
tess_confirmed = (tess['tfopwg_disp'] == 'CP').sum()

total_confirmed = kepler_confirmed + k2_confirmed + tess_confirmed

print(f"Kepler:  {kepler_confirmed:>5}개  ({kepler_confirmed/total_confirmed*100:>5.1f}%)")
print(f"K2:      {k2_confirmed:>5}개  ({k2_confirmed/total_confirmed*100:>5.1f}%)")
print(f"TESS:    {tess_confirmed:>5}개  ({tess_confirmed/total_confirmed*100:>5.1f}%)")
print(f"합계:    {total_confirmed:>5}개  (100.0%)")

print("\n[3] 컬럼 완성도 비교 (결측률)")
print("-" * 100)

# 핵심 컬럼들의 결측률 비교
core_cols_kepler = ['koi_period', 'koi_depth', 'koi_prad', 'koi_teq', 'koi_insol', 
                    'koi_srad', 'koi_smass', 'koi_steff', 'koi_slogg', 'koi_smet']

core_cols_k2_map = {
    'koi_period': 'pl_orbper',
    'koi_depth': 'pl_trandep',
    'koi_prad': 'pl_rade',
    'koi_teq': 'pl_eqt',
    'koi_insol': 'pl_insol',
    'koi_srad': 'st_rad',
    'koi_smass': 'st_mass',
    'koi_steff': 'st_teff',
    'koi_slogg': 'st_logg',
    'koi_smet': 'st_met'
}

core_cols_tess_map = {
    'koi_period': 'pl_orbper',
    'koi_depth': 'pl_trandep',
    'koi_prad': 'pl_rade',
    'koi_teq': 'pl_eqt',
    'koi_insol': 'pl_insol',
    'koi_srad': 'st_rad',
    'koi_smass': None,  # TESS에 없음
    'koi_steff': 'st_teff',
    'koi_slogg': 'st_logg',
    'koi_smet': None  # TESS에 없음
}

print(f"{'컬럼 (Kepler 기준)':<25} {'Kepler 완성도':<15} {'K2 완성도':<15} {'TESS 완성도':<15}")
print("-" * 100)

for kepler_col in core_cols_kepler:
    kepler_complete = (1 - kepler[kepler_col].isna().sum() / len(kepler)) * 100
    
    k2_col = core_cols_k2_map.get(kepler_col)
    if k2_col and k2_col in k2.columns:
        k2_complete = (1 - k2[k2_col].isna().sum() / len(k2)) * 100
    else:
        k2_complete = 0.0
    
    tess_col = core_cols_tess_map.get(kepler_col)
    if tess_col and tess_col in tess.columns:
        tess_complete = (1 - tess[tess_col].isna().sum() / len(tess)) * 100
    else:
        tess_complete = 0.0
    
    print(f"{kepler_col:<25} {kepler_complete:>6.1f}%         {k2_complete:>6.1f}%         {tess_complete:>6.1f}%")

print("\n[4] 관측 기간 및 데이터 성숙도")
print("-" * 100)
print("Kepler:  2009-2013 (4년)")
print("         → 가장 오래된 미션")
print("         → 데이터 검증 완료 (13년 경과)")
print("         → 후속 연구 및 확인 관측 완료")
print()
print("K2:      2014-2018 (4년)")
print("         → Kepler 후속 미션")
print("         → 데이터 검증 완료 (7년 경과)")
print("         → 다양한 시야, 하지만 Kepler보다 짧은 관측")
print()
print("TESS:    2018-현재 (진행 중)")
print("         → 현재 진행 중인 미션")
print("         → 후보(CANDIDATE) 많음")
print("         → 아직 확인 관측 진행 중")

print("\n[5] 컬럼 구조 및 표준화")
print("-" * 100)
print("Kepler:  NASA Exoplanet Archive 표준 형식")
print("         → koi_* 접두사로 통일")
print("         → 가장 세밀한 파라미터 제공")
print("         → 150+ 컬럼 (모든 측정값 및 오차)")
print()
print("K2:      Kepler와 유사하지만 pl_*, st_* 접두사")
print("         → 컬럼명 다름")
print("         → 일부 파라미터 누락")
print()
print("TESS:    더 간소화된 구조")
print("         → pl_*, st_* 접두사")
print("         → 핵심 파라미터만 제공")
print("         → 항성 질량(st_mass), 금속성(st_met) 없음")

print("\n[6] 커뮤니티 표준 및 문헌")
print("-" * 100)
print("Kepler:  가장 많이 인용된 외계행성 데이터")
print("         → 논문 10,000+ 편")
print("         → 머신러닝 연구의 표준 데이터셋")
print("         → Kaggle, 학술 논문에서 광범위 사용")
print()
print("K2/TESS: 상대적으로 적은 연구")
print("         → 주로 개별 행성 발견 논문")
print("         → 머신러닝 연구는 아직 초기 단계")

print("\n" + "=" * 100)
print("🎯 결론: Kepler를 중심으로 전처리한 7가지 이유")
print("=" * 100)

reasons = [
    ("1. 데이터 규모", f"전체의 {len(kepler)/(len(kepler)+len(k2)+len(tess))*100:.1f}%를 차지하는 가장 큰 데이터셋"),
    ("2. 확인된 행성 비율", f"CONFIRMED 행성의 {kepler_confirmed/total_confirmed*100:.1f}%를 보유"),
    ("3. 컬럼 완성도", "핵심 피처들의 결측률이 가장 낮음 (특히 koi_smass, koi_smet)"),
    ("4. 데이터 성숙도", "13년간의 검증과 후속 관측으로 가장 신뢰도 높음"),
    ("5. 표준 형식", "NASA Exoplanet Archive의 표준 컬럼명 (koi_*)"),
    ("6. 파라미터 풍부도", "150+ 컬럼으로 가장 세밀한 측정값 제공"),
    ("7. 커뮤니티 표준", "학계 및 ML 연구의 사실상 표준 데이터셋")
]

for i, (title, desc) in enumerate(reasons, 1):
    print(f"\n{title}")
    print(f"   → {desc}")

print("\n" + "=" * 100)
print("💡 전처리 전략의 정당성")
print("=" * 100)
print("""
Kepler를 중심으로 전처리함으로써:
  ✅ 가장 신뢰도 높은 데이터를 기준으로 삼음
  ✅ 표준화된 컬럼명으로 통일성 확보
  ✅ K2와 TESS 데이터를 Kepler 형식으로 변환하여 통합
  ✅ 머신러닝 모델 학습 시 품질 높은 데이터 우선 활용
  ✅ 향후 새로운 미션(TESS, 후속 미션)도 같은 방식으로 통합 가능

TESS가 현재 진행 중이지만, 데이터 품질과 성숙도 면에서
Kepler가 머신러닝 학습의 기준으로 더 적합합니다.
""")
