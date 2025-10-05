"""
TESS에 누락된 컬럼의 중요도 분석 및 복원 전략
"""

import pandas as pd
import numpy as np

def analyze_missing_columns_importance():
    """TESS에 누락된 컬럼의 중요도와 복원 가능성 분석"""
    
    print("=" * 120)
    print("TESS 누락 컬럼 중요도 분석 및 복원 전략")
    print("=" * 120)
    print()
    
    # TESS에 누락된 10개 컬럼 정의
    missing_columns = {
        # ========== 궤도 파라미터 (5개) ==========
        "koi_eccen": {
            "name": "궤도 이심률 (Orbital Eccentricity)",
            "k2_available": True,
            "k2_column": "pl_orbeccen",
            "importance": "⭐⭐⭐ 높음",
            "ml_impact": "궤도 형태 결정, 거주 가능성과 관련",
            "can_calculate": True,
            "calculation_method": "RV 측정 또는 통과 지속시간 변화 분석",
            "alternative": "통계적 추정 (대부분 행성은 낮은 이심률)",
            "recommended_fill": "0.0 (원형 궤도 가정) 또는 통계적 분포 샘플링"
        },
        "koi_longp": {
            "name": "근점 인수 (Argument of Periastron)",
            "k2_available": True,
            "k2_column": "pl_orblper",
            "importance": "⭐ 낮음",
            "ml_impact": "궤도 방향, 대부분 모델에서 중요도 낮음",
            "can_calculate": False,
            "calculation_method": "RV 측정 또는 광도 곡선 분석 필요",
            "alternative": "임의 값 사용",
            "recommended_fill": "90° (기본값) 또는 균일 분포 샘플링"
        },
        "koi_incl": {
            "name": "궤도 경사각 (Orbital Inclination)",
            "k2_available": True,
            "k2_column": "pl_orbincl",
            "importance": "⭐⭐⭐⭐⭐ 매우 높음",
            "ml_impact": "통과 기하학, 행성 반지름 계산에 필수",
            "can_calculate": True,
            "calculation_method": "통과 지속시간 + 항성 반지름 + 궤도 반장축",
            "formula": "cos(i) = (a/R_star) * sqrt((1-δ) / (1+δ)) * (1-b²)",
            "alternative": "통과 파라미터로부터 계산",
            "recommended_fill": "⭐ 계산 가능 (최우선)"
        },
        "koi_impact": {
            "name": "충격 계수 (Impact Parameter)",
            "k2_available": True,
            "k2_column": "pl_imppar",
            "importance": "⭐⭐⭐⭐ 높음",
            "ml_impact": "통과 중심성, 깊이 해석에 중요",
            "can_calculate": True,
            "calculation_method": "통과 지속시간 + 경사각",
            "formula": "b = a*cos(i)/R_star",
            "alternative": "통과 깊이와 지속시간으로부터 추정",
            "recommended_fill": "⭐ 계산 가능"
        },
        "koi_sma": {
            "name": "궤도 반장축 (Semi-major Axis)",
            "k2_available": True,
            "k2_column": "pl_orbsmax",
            "importance": "⭐⭐⭐⭐⭐ 매우 높음",
            "ml_impact": "거주 가능 영역 계산, 복사 플럭스 결정",
            "can_calculate": True,
            "calculation_method": "케플러 제3법칙",
            "formula": "a³ = (G*M_star*P²) / (4π²)  또는  a = ((P²*M_star)^(1/3)",
            "alternative": "궤도 주기 + 항성 질량으로 계산",
            "recommended_fill": "⭐ 계산 가능 (최우선)"
        },
        
        # ========== 통과 파라미터 (1개) ==========
        "koi_ingress": {
            "name": "진입 시간 (Ingress Duration)",
            "k2_available": False,
            "k2_column": None,
            "importance": "⭐⭐ 중간",
            "ml_impact": "통과 형태, 정밀 분석에 유용",
            "can_calculate": True,
            "calculation_method": "통과 지속시간의 일부로 추정",
            "formula": "t_ingress ≈ 0.1 * t_duration (경험적)",
            "alternative": "통과 지속시간 비율로 추정",
            "recommended_fill": "계산 가능하나 Kepler/K2에도 없음"
        },
        
        # ========== 항성 파라미터 (3개) ==========
        "koi_smass": {
            "name": "항성 질량 (Stellar Mass)",
            "k2_available": True,
            "k2_column": "st_mass",
            "importance": "⭐⭐⭐⭐⭐ 매우 높음",
            "ml_impact": "궤도 반장축 계산, 행성 질량 추정",
            "can_calculate": True,
            "calculation_method": "질량-광도 관계 또는 항성 모델",
            "formula": "M/M_sun ≈ (R/R_sun)² * 10^(logg_star - logg_sun)",
            "alternative": "항성 반지름 + 표면 중력으로 계산",
            "recommended_fill": "⭐ 계산 가능 (최우선)"
        },
        "koi_sage": {
            "name": "항성 나이 (Stellar Age)",
            "k2_available": True,
            "k2_column": "st_age",
            "importance": "⭐⭐ 중간",
            "ml_impact": "항성 진화 단계, 행성 형성 이력",
            "can_calculate": False,
            "calculation_method": "등시선 분석 또는 회전 주기 (복잡)",
            "formula": "복잡 - 항성 진화 모델 필요",
            "alternative": "중앙값 사용 또는 제외",
            "recommended_fill": "5 Gyr (태양 나이) 또는 NaN"
        },
        "koi_smet": {
            "name": "항성 금속성 (Stellar Metallicity [Fe/H])",
            "k2_available": True,
            "k2_column": "st_met",
            "importance": "⭐⭐⭐ 높음",
            "ml_impact": "행성 형성 확률, 암석 행성 vs 가스 행성",
            "can_calculate": False,
            "calculation_method": "스펙트럼 분석 필요",
            "formula": "스펙트럼 분석 필요",
            "alternative": "태양 금속성 (0.0) 가정",
            "recommended_fill": "0.0 (태양 금속성) 또는 통계적 분포"
        },
        
        # ========== 광도 측정 (1개) ==========
        "koi_kepmag": {
            "name": "Kepler 등급 (Kepler Magnitude)",
            "k2_available": True,
            "k2_column": "sy_kepmag",
            "importance": "⭐ 낮음",
            "ml_impact": "관측 품질 지표, TESS는 TESS mag 사용",
            "can_calculate": False,
            "calculation_method": "측광 변환 필요",
            "formula": "복잡 - 필터 변환",
            "alternative": "TESS magnitude 사용",
            "recommended_fill": "NaN 또는 TESS mag 사용"
        }
    }
    
    # 1. 중요도별 정리
    print("📊 1. 외계행성 판별 중요도 순위")
    print("=" * 120)
    print()
    
    # 중요도별 그룹화
    importance_groups = {
        "⭐⭐⭐⭐⭐ 매우 높음": [],
        "⭐⭐⭐⭐ 높음": [],
        "⭐⭐⭐ 높음": [],
        "⭐⭐ 중간": [],
        "⭐ 낮음": []
    }
    
    for col, info in missing_columns.items():
        importance_groups[info['importance']].append((col, info))
    
    for importance, cols in importance_groups.items():
        if cols:
            print(f"\n{importance}")
            print("-" * 120)
            for col, info in cols:
                k2_status = "✅ K2 있음" if info['k2_available'] else "❌ K2 없음"
                calc_status = "🔢 계산 가능" if info['can_calculate'] else "❌ 계산 불가"
                print(f"  • {info['name']} ({col})")
                print(f"    - ML 영향: {info['ml_impact']}")
                print(f"    - K2 데이터: {k2_status}")
                print(f"    - 복원: {calc_status}")
                print()
    
    # 2. 계산 가능한 컬럼 상세
    print("\n" + "=" * 120)
    print("🔢 2. 계산으로 복원 가능한 컬럼 (최우선)")
    print("=" * 120)
    print()
    
    calculable = [(col, info) for col, info in missing_columns.items() if info['can_calculate']]
    
    for i, (col, info) in enumerate(calculable, 1):
        print(f"{i}. {info['name']} ({col})")
        print(f"   중요도: {info['importance']}")
        print(f"   계산 방법: {info['calculation_method']}")
        if 'formula' in info and info['formula']:
            print(f"   수식: {info['formula']}")
        print(f"   권장 처리: {info['recommended_fill']}")
        print()
    
    # 3. Python 구현 코드
    print("\n" + "=" * 120)
    print("💻 3. 계산 가능한 컬럼 복원 코드")
    print("=" * 120)
    print()
    
    print("```python")
    print("import pandas as pd")
    print("import numpy as np")
    print()
    print("def restore_missing_tess_columns(df_tess):")
    print('    """TESS 데이터셋의 누락 컬럼을 계산으로 복원"""')
    print("    ")
    print("    df = df_tess.copy()")
    print("    ")
    print("    # ========== 1. 항성 질량 계산 (최우선) ==========")
    print("    # M = R² * 10^(logg_star - logg_sun)")
    print("    # logg_sun = 4.44")
    print("    ")
    print("    if 'st_rad' in df.columns and 'st_logg' in df.columns:")
    print("        logg_sun = 4.44")
    print("        df['koi_smass'] = (df['st_rad'] ** 2) * 10 ** (df['st_logg'] - logg_sun)")
    print("        print('✅ 항성 질량 계산 완료')")
    print("    ")
    print("    # ========== 2. 궤도 반장축 계산 (최우선) ==========")
    print("    # 케플러 제3법칙: a³ = G*M*P²/(4π²)")
    print("    # 단순 공식: a [AU] = (P[days]²/365.25² * M[M_sun])^(1/3)")
    print("    ")
    print("    if 'pl_orbper' in df.columns and 'koi_smass' in df.columns:")
    print("        # a = ((P²*M_star)^(1/3)  [P in years, M in solar masses]")
    print("        P_years = df['pl_orbper'] / 365.25")
    print("        df['koi_sma'] = (P_years ** 2 * df['koi_smass']) ** (1/3)")
    print("        print('✅ 궤도 반장축 계산 완료')")
    print("    ")
    print("    # ========== 3. 궤도 경사각 계산 ==========")
    print("    # 통과하는 행성은 경사각이 ~90도에 가까움")
    print("    # 정확한 계산: cos(i) ≈ b*R_star/a")
    print("    # 또는 통과 지속시간으로부터 계산")
    print("    ")
    print("    if 'koi_sma' in df.columns and 'st_rad' in df.columns:")
    print("        # 단순 추정: 통과하는 행성은 대부분 i ≈ 90도")
    print("        # 더 정확한 계산은 impact parameter 필요")
    print("        # b = a*cos(i)/R_star → i = arccos(b*R_star/a)")
    print("        ")
    print("        # 통과 행성 가정: impact parameter b ≈ 0~0.9")
    print("        # 단순화: i ≈ 89도 (대부분 통과 행성)")
    print("        df['koi_incl'] = 89.0  # degrees")
    print("        print('⚠️ 궤도 경사각 기본값 설정 (89도)')")
    print("    ")
    print("    # ========== 4. 충격 계수 계산 ==========")
    print("    # b = (a/R_star) * cos(i)")
    print("    ")
    print("    if 'koi_sma' in df.columns and 'st_rad' in df.columns and 'koi_incl' in df.columns:")
    print("        # AU to Solar radii conversion: 1 AU = 215.032 R_sun")
    print("        AU_to_Rsun = 215.032")
    print("        a_in_Rstar = (df['koi_sma'] * AU_to_Rsun) / df['st_rad']")
    print("        df['koi_impact'] = a_in_Rstar * np.cos(np.radians(df['koi_incl']))")
    print("        print('✅ 충격 계수 계산 완료')")
    print("    ")
    print("    # ========== 5. 궤도 이심률 (통계적 추정) ==========")
    print("    # 대부분 행성은 낮은 이심률 (< 0.1)")
    print("    # 보수적 접근: 0.0 (원형 궤도)")
    print("    ")
    print("    df['koi_eccen'] = 0.0")
    print("    print('⚠️ 궤도 이심률 기본값 설정 (0.0 - 원형 궤도)')")
    print("    ")
    print("    # ========== 6. 근점 인수 (기본값) ==========")
    print("    df['koi_longp'] = 90.0  # degrees")
    print("    print('⚠️ 근점 인수 기본값 설정 (90도)')")
    print("    ")
    print("    # ========== 7. 항성 금속성 (태양 값 가정) ==========")
    print("    df['koi_smet'] = 0.0  # Solar metallicity")
    print("    print('⚠️ 항성 금속성 기본값 설정 (0.0 - 태양 금속성)')")
    print("    ")
    print("    # ========== 8. 항성 나이 (중앙값 사용) ==========")
    print("    df['koi_sage'] = 5.0  # Gyr (태양 나이)")
    print("    print('⚠️ 항성 나이 기본값 설정 (5 Gyr)')")
    print("    ")
    print("    # ========== 9. 진입 시간 (Kepler/K2에도 없음) ==========")
    print("    # 일반적으로 통과 시간의 10% 정도")
    print("    if 'pl_trandurh' in df.columns:")
    print("        df['koi_ingress'] = df['pl_trandurh'] * 0.1")
    print("        print('✅ 진입 시간 추정 완료 (10% of duration)')")
    print("    ")
    print("    # ========== 10. Kepler 등급 (NaN) ==========")
    print("    df['koi_kepmag'] = np.nan")
    print("    print('⚠️ Kepler 등급은 NaN으로 설정')")
    print("    ")
    print("    return df")
    print("```")
    print()
    
    # 4. K2 데이터 확인
    print("\n" + "=" * 120)
    print("📊 4. K2 데이터 가용성 확인")
    print("=" * 120)
    print()
    
    try:
        df_k2 = pd.read_csv('datasets/k2_merged.csv')
        
        print("K2 데이터셋에서 TESS 누락 컬럼 확인:\n")
        
        for col, info in missing_columns.items():
            if info['k2_available']:
                k2_col = info['k2_column']
                if k2_col in df_k2.columns:
                    non_null = df_k2[k2_col].notna().sum()
                    total = len(df_k2)
                    pct = non_null / total * 100
                    print(f"✅ {info['name']}")
                    print(f"   K2 컬럼: {k2_col}")
                    print(f"   데이터 있음: {non_null}/{total} ({pct:.1f}%)")
                    print()
                else:
                    print(f"❌ {info['name']}: K2 컬럼 {k2_col} 없음")
                    print()
    except Exception as e:
        print(f"K2 데이터 로드 실패: {e}")
    
    # 5. 최종 권장사항
    print("\n" + "=" * 120)
    print("💡 5. 최종 권장사항")
    print("=" * 120)
    print()
    
    recommendations = [
        {
            "priority": "1. 최우선 복원 (계산 가능 + 매우 중요)",
            "columns": [
                "koi_smass (항성 질량): 항성 반지름 + 표면 중력으로 계산",
                "koi_sma (궤도 반장축): 궤도 주기 + 항성 질량으로 계산",
                "koi_incl (궤도 경사각): 통과 기하학으로 추정 (89도)"
            ]
        },
        {
            "priority": "2. 중요 복원 (계산 가능 + 중요)",
            "columns": [
                "koi_impact (충격 계수): 반장축 + 경사각으로 계산",
                "koi_ingress (진입 시간): 통과 지속시간의 10%로 추정"
            ]
        },
        {
            "priority": "3. 통계적 추정 (계산 불가 + 중요)",
            "columns": [
                "koi_eccen (이심률): 0.0 (원형 궤도 가정)",
                "koi_smet (금속성): 0.0 (태양 금속성 가정)"
            ]
        },
        {
            "priority": "4. 기본값 또는 제외 (중요도 낮음)",
            "columns": [
                "koi_longp (근점 인수): 90도 또는 제외",
                "koi_sage (항성 나이): 5 Gyr 또는 제외",
                "koi_kepmag (Kepler 등급): NaN 또는 제외"
            ]
        }
    ]
    
    for rec in recommendations:
        print(f"\n{rec['priority']}")
        print("-" * 120)
        for col in rec['columns']:
            print(f"  • {col}")
    
    print()
    print()
    
    # 6. 데이터 품질 주의사항
    print("\n" + "=" * 120)
    print("⚠️ 6. 데이터 품질 주의사항")
    print("=" * 120)
    print()
    
    warnings = [
        "계산된 값은 실제 측정값보다 정확도가 낮음",
        "항성 질량 계산은 단성 모델에 의존 (젊은/늙은 항성에서 오차 증가)",
        "궤도 경사각 89도는 근사값 (실제는 87~90도 범위)",
        "이심률 0.0은 보수적 가정 (일부 행성은 높은 이심률)",
        "금속성 0.0은 태양 가정 (실제 별마다 다름)",
        "머신러닝 모델 학습 시 계산된 컬럼에 표시 필요",
        "Kepler/K2 데이터는 실측값, TESS 데이터는 계산값 혼재"
    ]
    
    for i, warning in enumerate(warnings, 1):
        print(f"{i}. {warning}")
    
    print()


if __name__ == "__main__":
    analyze_missing_columns_importance()
    print("\n✅ TESS 누락 컬럼 중요도 분석 완료!")
