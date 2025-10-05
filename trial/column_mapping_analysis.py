"""
세 데이터셋(Kepler, K2, TESS)의 공통 컬럼 분석 및 단위 변환 정보 제공
"""

import pandas as pd

# 컬럼 매핑 정의 - Kepler 컬럼 기준으로 TESS와 K2 매핑
column_mapping = {
    # ===== 정답 컬럼 (외계행성 판별 결과) =====
    "disposition": {
        "kepler": "koi_disposition",
        "tess": "tfopwg_disp", 
        "k2": "disposition",
        "description": "외계행성 판별 결과 (정답)",
        "unit": "categorical",
        "unit_match": "✓",
        "notes": "Kepler: CONFIRMED/FALSE POSITIVE/CANDIDATE, TESS: CP/FP/KP/PC, K2: CONFIRMED/FALSE POSITIVE/CANDIDATE"
    },
    
    # ===== 궤도 주기 (Orbital Period) =====
    "orbital_period": {
        "kepler": "koi_period",
        "tess": "pl_orbper",
        "k2": "pl_orbper",
        "description": "궤도 주기",
        "unit": "days (일)",
        "unit_match": "✓",
        "notes": "모든 데이터셋에서 단위 동일"
    },
    
    # ===== 통과 시각 (Transit Time) =====
    "transit_time": {
        "kepler": "koi_time0",
        "tess": "pl_tranmid",
        "k2": "pl_tranmid",
        "description": "통과 중심 시각",
        "unit": "BJD (Barycentric Julian Date)",
        "unit_match": "✓",
        "notes": "모든 데이터셋에서 BJD 사용"
    },
    
    # ===== 이심률 (Eccentricity) =====
    "eccentricity": {
        "kepler": "koi_eccen",
        "tess": None,
        "k2": "pl_orbeccen",
        "description": "궤도 이심률",
        "unit": "dimensionless (무차원)",
        "unit_match": "✓",
        "notes": "TESS 데이터에는 없음"
    },
    
    # ===== 충돌 파라미터 (Impact Parameter) =====
    "impact_parameter": {
        "kepler": "koi_impact",
        "tess": None,
        "k2": "pl_imppar",
        "description": "충돌 파라미터",
        "unit": "dimensionless (무차원)",
        "unit_match": "✓",
        "notes": "TESS 데이터에는 없음"
    },
    
    # ===== 통과 지속시간 (Transit Duration) =====
    "transit_duration": {
        "kepler": "koi_duration",
        "tess": "pl_trandurh",
        "k2": "pl_trandur",
        "description": "통과 지속시간",
        "unit": "hours (시간)",
        "unit_match": "✓",
        "notes": "모든 데이터셋에서 hours 단위 사용"
    },
    
    # ===== 통과 깊이 (Transit Depth) =====
    "transit_depth": {
        "kepler": "koi_depth",
        "tess": "pl_trandep",
        "k2": "pl_trandep",
        "description": "통과 깊이",
        "unit": "Kepler: ppm, TESS: ppm, K2: %",
        "unit_match": "X",
        "notes": "**단위 변환 필요**: K2의 % 값을 ppm으로 변환 (% × 10,000 = ppm)"
    },
    
    # ===== 행성-항성 반지름 비율 (Planet-Star Radius Ratio) =====
    "radius_ratio": {
        "kepler": "koi_ror",
        "tess": None,
        "k2": "pl_ratror",
        "description": "행성-항성 반지름 비율",
        "unit": "dimensionless (무차원)",
        "unit_match": "✓",
        "notes": "TESS 데이터에는 없음"
    },
    
    # ===== 항성 밀도 (Stellar Density) =====
    "stellar_density": {
        "kepler": "koi_srho",
        "tess": None,
        "k2": "st_dens",
        "description": "항성 밀도",
        "unit": "g/cm³",
        "unit_match": "✓",
        "notes": "TESS 데이터에는 없음"
    },
    
    # ===== 행성 반지름 (Planet Radius) =====
    "planet_radius": {
        "kepler": "koi_prad",
        "tess": "pl_rade",
        "k2": "pl_rade",
        "description": "행성 반지름",
        "unit": "Earth radii (지구 반지름)",
        "unit_match": "✓",
        "notes": "모든 데이터셋에서 지구 반지름 단위 사용"
    },
    
    # ===== 궤도 장반경 (Semi-Major Axis) =====
    "semi_major_axis": {
        "kepler": "koi_sma",
        "tess": None,
        "k2": "pl_orbsmax",
        "description": "궤도 장반경",
        "unit": "au (천문단위)",
        "unit_match": "✓",
        "notes": "TESS 데이터에는 없음"
    },
    
    # ===== 궤도 경사각 (Inclination) =====
    "inclination": {
        "kepler": "koi_incl",
        "tess": None,
        "k2": "pl_orbincl",
        "description": "궤도 경사각",
        "unit": "degrees (도)",
        "unit_match": "✓",
        "notes": "TESS 데이터에는 없음"
    },
    
    # ===== 평형 온도 (Equilibrium Temperature) =====
    "equilibrium_temp": {
        "kepler": "koi_teq",
        "tess": "pl_eqt",
        "k2": "pl_eqt",
        "description": "평형 온도",
        "unit": "Kelvin (K)",
        "unit_match": "✓",
        "notes": "모든 데이터셋에서 K 단위 사용"
    },
    
    # ===== 복사 플럭스 (Insolation Flux) =====
    "insolation_flux": {
        "kepler": "koi_insol",
        "tess": "pl_insol",
        "k2": "pl_insol",
        "description": "복사 플럭스",
        "unit": "Earth flux (지구 플럭스)",
        "unit_match": "✓",
        "notes": "모든 데이터셋에서 지구 플럭스 단위 사용"
    },
    
    # ===== 행성-항성 거리/항성반지름 비율 =====
    "distance_over_radius": {
        "kepler": "koi_dor",
        "tess": None,
        "k2": "pl_ratdor",
        "description": "행성-항성 거리/항성반지름 비율",
        "unit": "dimensionless (무차원)",
        "unit_match": "✓",
        "notes": "TESS 데이터에는 없음"
    },
    
    # ===== 항성 유효 온도 (Stellar Effective Temperature) =====
    "stellar_temp": {
        "kepler": "koi_steff",
        "tess": "st_teff",
        "k2": "st_teff",
        "description": "항성 유효 온도",
        "unit": "Kelvin (K)",
        "unit_match": "✓",
        "notes": "모든 데이터셋에서 K 단위 사용"
    },
    
    # ===== 항성 표면 중력 (Stellar Surface Gravity) =====
    "stellar_gravity": {
        "kepler": "koi_slogg",
        "tess": "st_logg",
        "k2": "st_logg",
        "description": "항성 표면 중력",
        "unit": "log10(cm/s²)",
        "unit_match": "✓",
        "notes": "모든 데이터셋에서 log10(cm/s²) 단위 사용"
    },
    
    # ===== 항성 금속 함량 (Stellar Metallicity) =====
    "stellar_metallicity": {
        "kepler": "koi_smet",
        "tess": None,
        "k2": "st_met",
        "description": "항성 금속 함량",
        "unit": "dex",
        "unit_match": "✓",
        "notes": "TESS 데이터에는 없음"
    },
    
    # ===== 항성 반지름 (Stellar Radius) =====
    "stellar_radius": {
        "kepler": "koi_srad",
        "tess": "st_rad",
        "k2": "st_rad",
        "description": "항성 반지름",
        "unit": "Solar radii (태양 반지름)",
        "unit_match": "✓",
        "notes": "모든 데이터셋에서 태양 반지름 단위 사용"
    },
    
    # ===== 항성 질량 (Stellar Mass) =====
    "stellar_mass": {
        "kepler": "koi_smass",
        "tess": None,
        "k2": "st_mass",
        "description": "항성 질량",
        "unit": "Solar mass (태양 질량)",
        "unit_match": "✓",
        "notes": "TESS 데이터에는 없음"
    },
    
    # ===== 항성 나이 (Stellar Age) =====
    "stellar_age": {
        "kepler": "koi_sage",
        "tess": None,
        "k2": "st_age",
        "description": "항성 나이",
        "unit": "Gyr (기가년)",
        "unit_match": "✓",
        "notes": "TESS 데이터에는 없음"
    },
    
    # ===== 적경 (Right Ascension) =====
    "right_ascension": {
        "kepler": "ra",
        "tess": "ra",
        "k2": "ra",
        "description": "적경",
        "unit": "degrees (도)",
        "unit_match": "✓",
        "notes": "모든 데이터셋에서 degree 단위 사용"
    },
    
    # ===== 적위 (Declination) =====
    "declination": {
        "kepler": "dec",
        "tess": "dec",
        "k2": "dec",
        "description": "적위",
        "unit": "degrees (도)",
        "unit_match": "✓",
        "notes": "모든 데이터셋에서 degree 단위 사용"
    },
    
    # ===== J-band Magnitude =====
    "j_magnitude": {
        "kepler": "koi_jmag",
        "tess": None,
        "k2": "sy_jmag",
        "description": "J-band 등급",
        "unit": "magnitude (등급)",
        "unit_match": "✓",
        "notes": "TESS 데이터에는 없음"
    },
    
    # ===== H-band Magnitude =====
    "h_magnitude": {
        "kepler": "koi_hmag",
        "tess": None,
        "k2": "sy_hmag",
        "description": "H-band 등급",
        "unit": "magnitude (등급)",
        "unit_match": "✓",
        "notes": "TESS 데이터에는 없음"
    },
    
    # ===== K-band Magnitude =====
    "k_magnitude": {
        "kepler": "koi_kmag",
        "tess": None,
        "k2": "sy_kmag",
        "description": "K-band 등급",
        "unit": "magnitude (등급)",
        "unit_match": "✓",
        "notes": "TESS 데이터에는 없음"
    },
}


def print_analysis():
    """분석 결과를 한국어로 출력"""
    
    print("=" * 100)
    print("세 데이터셋(Kepler, K2, TESS) 공통 컬럼 분석 결과")
    print("=" * 100)
    print()
    
    # 통계 정보
    total_cols = len(column_mapping)
    all_three = sum(1 for v in column_mapping.values() if all([v["kepler"], v["tess"], v["k2"]]))
    kepler_k2 = sum(1 for v in column_mapping.values() if v["kepler"] and v["k2"] and not v["tess"])
    kepler_tess = sum(1 for v in column_mapping.values() if v["kepler"] and v["tess"] and not v["k2"])
    unit_mismatch = sum(1 for v in column_mapping.values() if v["unit_match"] == "X")
    
    print(f"📊 전체 공통 컬럼 수: {total_cols}개")
    print(f"   - 세 데이터셋 모두 존재: {all_three}개")
    print(f"   - Kepler & K2만 존재: {kepler_k2}개")
    print(f"   - Kepler & TESS만 존재: {kepler_tess}개")
    print(f"   - 단위 변환 필요: {unit_mismatch}개")
    print()
    print("=" * 100)
    print()
    
    # 카테고리별로 분류
    categories = {
        "정답": ["disposition"],
        "궤도 파라미터": ["orbital_period", "transit_time", "eccentricity", "semi_major_axis", 
                      "inclination", "impact_parameter", "distance_over_radius"],
        "통과(Transit) 파라미터": ["transit_duration", "transit_depth", "radius_ratio"],
        "행성 물리량": ["planet_radius", "equilibrium_temp", "insolation_flux"],
        "항성 물리량": ["stellar_temp", "stellar_gravity", "stellar_metallicity", 
                     "stellar_radius", "stellar_mass", "stellar_age", "stellar_density"],
        "위치 정보": ["right_ascension", "declination"],
        "측광 등급": ["j_magnitude", "h_magnitude", "k_magnitude"]
    }
    
    for category, col_list in categories.items():
        print(f"\n{'='*100}")
        print(f"📁 [{category}]")
        print(f"{'='*100}\n")
        
        for col_key in col_list:
            if col_key not in column_mapping:
                continue
                
            col_info = column_mapping[col_key]
            
            print(f"🔹 {col_info['description']}")
            print(f"   단위: {col_info['unit']}")
            print(f"   단위 일치: {col_info['unit_match']}")
            print()
            
            # 각 데이터셋별 컬럼명
            print(f"   컬럼명:")
            print(f"      Kepler: {col_info['kepler'] if col_info['kepler'] else '❌ 없음'}")
            print(f"      TESS:   {col_info['tess'] if col_info['tess'] else '❌ 없음'}")
            print(f"      K2:     {col_info['k2'] if col_info['k2'] else '❌ 없음'}")
            print()
            
            if col_info["notes"]:
                print(f"   📝 참고사항: {col_info['notes']}")
                print()
            
            # 단위 변환이 필요한 경우 강조
            if col_info["unit_match"] == "X":
                print(f"   ⚠️  단위 변환 필요!")
                print()
            
            print("-" * 100)
    
    # 단위 변환 요약
    print(f"\n{'='*100}")
    print("⚠️  단위 변환이 필요한 컬럼 요약")
    print(f"{'='*100}\n")
    
    for col_key, col_info in column_mapping.items():
        if col_info["unit_match"] == "X":
            print(f"🔹 {col_info['description']} ({col_key})")
            print(f"   현재 단위: {col_info['unit']}")
            print(f"   변환 방법: {col_info['notes']}")
            print()


if __name__ == "__main__":
    print_analysis()
    
    # 매핑 정보를 CSV로 저장
    df_mapping = pd.DataFrame(column_mapping).T
    df_mapping.to_csv("column_mapping.csv", encoding="utf-8-sig")
    print(f"\n✅ 컬럼 매핑 정보가 'column_mapping.csv' 파일로 저장되었습니다.")
