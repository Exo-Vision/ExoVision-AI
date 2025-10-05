"""
Kepler 데이터셋 기준으로 K2, TESS 컬럼 매핑 및 단위 분석
"""

import pandas as pd
import numpy as np

def analyze_kepler_based_mapping():
    """Kepler 기준으로 세 데이터셋의 컬럼 매핑 분석"""
    
    print("=" * 120)
    print("Kepler 기준 데이터셋 통합 매핑 분석")
    print("=" * 120)
    print()
    
    # 데이터 로드
    try:
        df_kepler = pd.read_csv('datasets/kepler_merged.csv', nrows=5)
        df_k2 = pd.read_csv('datasets/k2_merged.csv', nrows=5)
        df_tess = pd.read_csv('datasets/tess_merged.csv', nrows=5)
        
        print(f"✅ 데이터 로드 완료")
        print(f"   - Kepler: {len(pd.read_csv('datasets/kepler_merged.csv'))} 행 × {len(df_kepler.columns)} 컬럼")
        print(f"   - K2: {len(pd.read_csv('datasets/k2_merged.csv'))} 행 × {len(df_k2.columns)} 컬럼")
        print(f"   - TESS: {len(pd.read_csv('datasets/tess_merged.csv'))} 행 × {len(df_tess.columns)} 컬럼")
        print()
    except Exception as e:
        print(f"❌ 데이터 로드 실패: {e}")
        return
    
    # Kepler → K2/TESS 컬럼 매핑 정의
    # 이전 분석에서 확인한 공통 컬럼 기준
    column_mapping = {
        # ========== 기본 정보 ==========
        "kepoi_name": {
            "kepler": "kepoi_name",
            "k2": "epic_candname",  # K2의 행성 이름
            "tess": "toi",  # TESS의 TOI 번호
            "meaning": "행성 후보 이름",
            "unit": "-",
            "kepler_unit": "-",
            "k2_unit": "-",
            "tess_unit": "-",
            "conversion": "없음"
        },
        
        # ========== 정답 컬럼 (타겟) ==========
        "koi_disposition": {
            "kepler": "koi_disposition",
            "k2": "pl_k2_disposition",  # K2의 판별 결과
            "tess": "tfopwg_disp",  # TESS의 판별 결과
            "meaning": "행성 판별 결과 (정답 레이블)",
            "unit": "categorical",
            "kepler_unit": "CONFIRMED / FALSE POSITIVE / CANDIDATE",
            "k2_unit": "CONFIRMED / FALSE POSITIVE / CANDIDATE",
            "tess_unit": "CP / FP / KP / PC",
            "conversion": "TESS → Kepler: CP/KP→CONFIRMED, FP→FALSE POSITIVE, PC→CANDIDATE"
        },
        
        # ========== 위치 정보 ==========
        "ra": {
            "kepler": "ra",
            "k2": "ra",
            "tess": "ra",
            "meaning": "적경",
            "unit": "degrees",
            "kepler_unit": "degrees",
            "k2_unit": "degrees",
            "tess_unit": "degrees",
            "conversion": "없음"
        },
        "dec": {
            "kepler": "dec",
            "k2": "dec",
            "tess": "dec",
            "meaning": "적위",
            "unit": "degrees",
            "kepler_unit": "degrees",
            "k2_unit": "degrees",
            "tess_unit": "degrees",
            "conversion": "없음"
        },
        
        # ========== 궤도 파라미터 ==========
        "koi_period": {
            "kepler": "koi_period",
            "k2": "pl_orbper",
            "tess": "pl_orbper",
            "meaning": "궤도 주기",
            "unit": "days",
            "kepler_unit": "days",
            "k2_unit": "days",
            "tess_unit": "days",
            "conversion": "없음"
        },
        "koi_time0bk": {
            "kepler": "koi_time0bk",
            "k2": "pl_tranmid",
            "tess": "pl_tranmid",
            "meaning": "통과 중심 시각",
            "unit": "BJD",
            "kepler_unit": "BJD - 2454833.0",
            "k2_unit": "BJD",
            "tess_unit": "BJD",
            "conversion": "Kepler BJD + 2454833.0 = 표준 BJD"
        },
        "koi_eccen": {
            "kepler": "koi_eccen",
            "k2": "pl_orbeccen",
            "tess": None,  # TESS에 없음
            "meaning": "궤도 이심률",
            "unit": "dimensionless",
            "kepler_unit": "0~1",
            "k2_unit": "0~1",
            "tess_unit": "N/A",
            "conversion": "없음 (TESS는 누락)"
        },
        "koi_longp": {
            "kepler": "koi_longp",
            "k2": "pl_orblper",
            "tess": None,
            "meaning": "근점 인수",
            "unit": "degrees",
            "kepler_unit": "degrees",
            "k2_unit": "degrees",
            "tess_unit": "N/A",
            "conversion": "없음 (TESS는 누락)"
        },
        "koi_incl": {
            "kepler": "koi_incl",
            "k2": "pl_orbincl",
            "tess": None,
            "meaning": "궤도 경사각",
            "unit": "degrees",
            "kepler_unit": "degrees",
            "k2_unit": "degrees",
            "tess_unit": "N/A",
            "conversion": "없음 (TESS는 누락)"
        },
        "koi_impact": {
            "kepler": "koi_impact",
            "k2": "pl_imppar",
            "tess": None,
            "meaning": "충격 계수",
            "unit": "R_star",
            "kepler_unit": "R_star",
            "k2_unit": "R_star",
            "tess_unit": "N/A",
            "conversion": "없음 (TESS는 누락)"
        },
        "koi_sma": {
            "kepler": "koi_sma",
            "k2": "pl_orbsmax",
            "tess": None,
            "meaning": "궤도 반장축",
            "unit": "AU",
            "kepler_unit": "AU",
            "k2_unit": "AU",
            "tess_unit": "N/A",
            "conversion": "없음 (TESS는 누락)"
        },
        
        # ========== 통과 파라미터 ==========
        "koi_duration": {
            "kepler": "koi_duration",
            "k2": "pl_trandur",
            "tess": "pl_trandurh",
            "meaning": "통과 지속시간",
            "unit": "hours",
            "kepler_unit": "hours",
            "k2_unit": "days",
            "tess_unit": "hours",
            "conversion": "K2: days × 24 = hours"
        },
        "koi_depth": {
            "kepler": "koi_depth",
            "k2": "pl_trandep",
            "tess": "pl_trandep",
            "meaning": "통과 깊이",
            "unit": "ppm",
            "kepler_unit": "ppm",
            "k2_unit": "% (percent)",
            "tess_unit": "ppm",
            "conversion": "⚠️ K2: % × 10,000 = ppm"
        },
        "koi_ingress": {
            "kepler": "koi_ingress",
            "k2": None,
            "tess": None,
            "meaning": "진입 시간",
            "unit": "hours",
            "kepler_unit": "hours",
            "k2_unit": "N/A",
            "tess_unit": "N/A",
            "conversion": "없음 (K2, TESS 누락)"
        },
        
        # ========== 행성 물리량 ==========
        "koi_prad": {
            "kepler": "koi_prad",
            "k2": "pl_rade",
            "tess": "pl_rade",
            "meaning": "행성 반지름",
            "unit": "Earth radii",
            "kepler_unit": "R_Earth",
            "k2_unit": "R_Earth",
            "tess_unit": "R_Earth",
            "conversion": "없음"
        },
        "koi_srad": {
            "kepler": "koi_srad",
            "k2": "st_rad",
            "tess": "st_rad",
            "meaning": "항성 반지름",
            "unit": "Solar radii",
            "kepler_unit": "R_Sun",
            "k2_unit": "R_Sun",
            "tess_unit": "R_Sun",
            "conversion": "없음"
        },
        "koi_smass": {
            "kepler": "koi_smass",
            "k2": "st_mass",
            "tess": None,
            "meaning": "항성 질량",
            "unit": "Solar masses",
            "kepler_unit": "M_Sun",
            "k2_unit": "M_Sun",
            "tess_unit": "N/A",
            "conversion": "없음 (TESS 누락)"
        },
        "koi_sage": {
            "kepler": "koi_sage",
            "k2": "st_age",
            "tess": None,
            "meaning": "항성 나이",
            "unit": "Gyr",
            "kepler_unit": "Gyr",
            "k2_unit": "Gyr",
            "tess_unit": "N/A",
            "conversion": "없음 (TESS 누락)"
        },
        "koi_steff": {
            "kepler": "koi_steff",
            "k2": "st_teff",
            "tess": "st_teff",
            "meaning": "항성 유효 온도",
            "unit": "Kelvin",
            "kepler_unit": "K",
            "k2_unit": "K",
            "tess_unit": "K",
            "conversion": "없음"
        },
        "koi_slogg": {
            "kepler": "koi_slogg",
            "k2": "st_logg",
            "tess": "st_logg",
            "meaning": "항성 표면 중력",
            "unit": "log10(cm/s²)",
            "kepler_unit": "log10(cm/s²)",
            "k2_unit": "log10(cm/s²)",
            "tess_unit": "log10(cm/s²)",
            "conversion": "없음"
        },
        "koi_smet": {
            "kepler": "koi_smet",
            "k2": "st_met",
            "tess": None,
            "meaning": "항성 금속성",
            "unit": "dex",
            "kepler_unit": "[Fe/H]",
            "k2_unit": "[Fe/H]",
            "tess_unit": "N/A",
            "conversion": "없음 (TESS 누락)"
        },
        "koi_insol": {
            "kepler": "koi_insol",
            "k2": "pl_insol",
            "tess": "pl_insol",
            "meaning": "행성 복사 플럭스",
            "unit": "Earth flux",
            "kepler_unit": "F_Earth",
            "k2_unit": "F_Earth",
            "tess_unit": "F_Earth",
            "conversion": "없음"
        },
        "koi_teq": {
            "kepler": "koi_teq",
            "k2": "pl_eqt",
            "tess": "pl_eqt",
            "meaning": "행성 평형 온도",
            "unit": "Kelvin",
            "kepler_unit": "K",
            "k2_unit": "K",
            "tess_unit": "K",
            "conversion": "없음"
        },
        
        # ========== 광도 측정 ==========
        "koi_kepmag": {
            "kepler": "koi_kepmag",
            "k2": "sy_kepmag",
            "tess": None,
            "meaning": "Kepler 등급",
            "unit": "magnitude",
            "kepler_unit": "mag",
            "k2_unit": "mag",
            "tess_unit": "N/A",
            "conversion": "없음 (TESS는 TESS mag 사용)"
        },
    }
    
    # 매핑 테이블 출력
    print("\n" + "=" * 120)
    print("📊 Kepler 기준 컬럼 매핑 테이블")
    print("=" * 120)
    print()
    
    # 카테고리별로 그룹화
    categories = {
        "기본 정보": ["kepoi_name"],
        "정답 레이블 (타겟)": ["koi_disposition"],
        "위치 정보": ["ra", "dec"],
        "궤도 파라미터": ["koi_period", "koi_time0bk", "koi_eccen", "koi_longp", 
                          "koi_incl", "koi_impact", "koi_sma"],
        "통과 파라미터": ["koi_duration", "koi_depth", "koi_ingress"],
        "행성 물리량": ["koi_prad", "koi_insol", "koi_teq"],
        "항성 물리량": ["koi_srad", "koi_smass", "koi_sage", "koi_steff", 
                        "koi_slogg", "koi_smet"],
        "광도 측정": ["koi_kepmag"]
    }
    
    for category, columns in categories.items():
        print(f"\n{'=' * 120}")
        print(f"📌 {category}")
        print(f"{'=' * 120}\n")
        
        # 테이블 헤더
        print(f"{'Kepler 컬럼':<25} {'K2 컬럼':<25} {'TESS 컬럼':<20} {'의미':<25} {'단위 통일':<15}")
        print("-" * 120)
        
        for col in columns:
            if col in column_mapping:
                info = column_mapping[col]
                kepler_col = info['kepler']
                k2_col = info['k2'] if info['k2'] else "❌ 없음"
                tess_col = info['tess'] if info['tess'] else "❌ 없음"
                meaning = info['meaning']
                
                # 단위 통일 여부 확인
                if info['conversion'] == "없음":
                    unit_status = "✅ 동일"
                elif "누락" in info['conversion']:
                    unit_status = "⚠️ 누락"
                else:
                    unit_status = "🔄 변환 필요"
                
                print(f"{kepler_col:<25} {k2_col:<25} {tess_col:<20} {meaning:<25} {unit_status:<15}")
    
    # 단위 변환이 필요한 컬럼 상세 설명
    print("\n\n" + "=" * 120)
    print("🔄 단위 변환이 필요한 컬럼 상세")
    print("=" * 120)
    print()
    
    conversion_needed = []
    for col, info in column_mapping.items():
        if info['conversion'] not in ["없음", ""] and "누락" not in info['conversion']:
            conversion_needed.append((col, info))
    
    if conversion_needed:
        for i, (col, info) in enumerate(conversion_needed, 1):
            print(f"{i}. {col} ({info['meaning']})")
            print(f"   Kepler 단위: {info['kepler_unit']}")
            print(f"   K2 단위: {info['k2_unit']}")
            print(f"   TESS 단위: {info['tess_unit']}")
            print(f"   변환 방법: {info['conversion']}")
            print()
    else:
        print("✅ 단위 변환이 필요한 컬럼이 없습니다.")
    
    # 누락된 컬럼 정리
    print("\n" + "=" * 120)
    print("⚠️ 데이터셋별 누락 컬럼")
    print("=" * 120)
    print()
    
    k2_missing = []
    tess_missing = []
    
    for col, info in column_mapping.items():
        if info['k2'] is None:
            k2_missing.append(f"{col} ({info['meaning']})")
        if info['tess'] is None:
            tess_missing.append(f"{col} ({info['meaning']})")
    
    print(f"📊 K2 데이터셋에 없는 Kepler 컬럼: {len(k2_missing)}개")
    if k2_missing:
        for item in k2_missing:
            print(f"   - {item}")
    print()
    
    print(f"📊 TESS 데이터셋에 없는 Kepler 컬럼: {len(tess_missing)}개")
    if tess_missing:
        for item in tess_missing:
            print(f"   - {item}")
    print()
    
    # 통계 요약
    print("\n" + "=" * 120)
    print("📈 매핑 통계 요약")
    print("=" * 120)
    print()
    
    total_kepler_cols = len(column_mapping)
    k2_mapped = sum(1 for info in column_mapping.values() if info['k2'] is not None)
    tess_mapped = sum(1 for info in column_mapping.values() if info['tess'] is not None)
    
    print(f"총 Kepler 기준 컬럼: {total_kepler_cols}개")
    print(f"K2 매핑 가능: {k2_mapped}개 ({k2_mapped/total_kepler_cols*100:.1f}%)")
    print(f"TESS 매핑 가능: {tess_mapped}개 ({tess_mapped/total_kepler_cols*100:.1f}%)")
    print()
    
    fully_mapped = sum(1 for info in column_mapping.values() 
                       if info['k2'] is not None and info['tess'] is not None)
    print(f"세 데이터셋 모두 존재: {fully_mapped}개 ({fully_mapped/total_kepler_cols*100:.1f}%)")
    print()
    
    # 매핑 딕셔너리 저장 (나중에 스크립트에서 사용)
    print("\n" + "=" * 120)
    print("💾 매핑 정보 저장")
    print("=" * 120)
    print()
    
    import json
    with open('kepler_column_mapping.json', 'w', encoding='utf-8') as f:
        json.dump(column_mapping, f, indent=2, ensure_ascii=False)
    
    print("✅ 매핑 정보가 'kepler_column_mapping.json'에 저장되었습니다.")
    print()
    
    # 권장 통합 전략
    print("\n" + "=" * 120)
    print("💡 권장 통합 전략")
    print("=" * 120)
    print()
    
    strategies = [
        "1. 정답 레이블 통일: TESS의 CP/KP/FP/PC를 Kepler 형식으로 변환",
        "2. 단위 변환:",
        "   - K2 통과 깊이: % × 10,000 = ppm",
        "   - K2 통과 지속시간: days × 24 = hours",
        "   - Kepler BJD: BJD + 2454833.0 = 표준 BJD",
        "3. 누락 컬럼 처리: TESS/K2에 없는 컬럼은 NaN으로 채우기",
        "4. 에러 컬럼 통일: *_error 형식으로 통합",
        "5. Limit Flag 보존: *_limit_flag 유지",
        "6. 데이터셋 출처 표시: 'source' 컬럼 추가 (Kepler/K2/TESS)"
    ]
    
    for strategy in strategies:
        print(strategy)
    print()


if __name__ == "__main__":
    analyze_kepler_based_mapping()
    print("\n✅ Kepler 기준 매핑 분석 완료!")
