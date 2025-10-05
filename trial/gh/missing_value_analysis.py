"""
통합 데이터셋 결측치 분석 및 처리 전략
"""

import pandas as pd

def analyze_missing_values():
    """결측치 분석"""
    
    print("="*100)
    print("🔍 통합 데이터셋 결측치 분석")
    print("="*100)
    
    # 데이터 로드
    df = pd.read_csv('datasets/exoplanets_integrated.csv')
    
    print(f"\n데이터셋 크기: {len(df)} 행 × {len(df.columns)} 컬럼")
    print(f"데이터 출처: Kepler {(df['data_source']=='Kepler').sum()}, "
          f"K2 {(df['data_source']=='K2').sum()}, "
          f"TESS {(df['data_source']=='TESS').sum()}")
    
    # 1. 전체 결측치 현황
    print("\n" + "="*100)
    print("📊 1. 전체 결측치 현황")
    print("="*100)
    
    missing_stats = []
    for col in df.columns:
        missing_count = df[col].isna().sum()
        missing_pct = missing_count / len(df) * 100
        
        if missing_count > 0:
            missing_stats.append({
                'Column': col,
                'Missing': missing_count,
                'Percent': missing_pct,
                'Present': len(df) - missing_count,
                'Present_Pct': 100 - missing_pct
            })
    
    missing_df = pd.DataFrame(missing_stats).sort_values('Percent', ascending=False)
    
    print(f"\n결측치가 있는 컬럼: {len(missing_df)}개")
    print("\n" + "-"*100)
    print(f"{'컬럼명':<25} {'결측':<10} {'결측%':<10} {'존재':<10} {'존재%':<10}")
    print("-"*100)
    
    for _, row in missing_df.iterrows():
        print(f"{row['Column']:<25} {row['Missing']:<10} {row['Percent']:<10.2f} "
              f"{row['Present']:<10} {row['Present_Pct']:<10.2f}")
    
    # 2. 데이터 출처별 결측치
    print("\n" + "="*100)
    print("📊 2. 데이터 출처별 결측치 분석")
    print("="*100)
    
    for source in ['Kepler', 'K2', 'TESS']:
        df_source = df[df['data_source'] == source]
        print(f"\n🔹 {source} ({len(df_source)} 행)")
        print("-"*100)
        
        missing_source = []
        for col in df.columns:
            missing_count = df_source[col].isna().sum()
            missing_pct = missing_count / len(df_source) * 100
            
            if missing_count > 0 and missing_pct > 0:
                missing_source.append({
                    'Column': col,
                    'Missing': missing_count,
                    'Percent': missing_pct
                })
        
        missing_source_df = pd.DataFrame(missing_source).sort_values('Percent', ascending=False)
        
        for _, row in missing_source_df.head(10).iterrows():
            print(f"  {row['Column']:<25} {row['Missing']:<8} ({row['Percent']:>5.1f}%)")
    
    # 3. 상관관계 분석 (수치형 컬럼만)
    print("\n" + "="*100)
    print("📊 3. 주요 컬럼 간 상관관계 분석")
    print("="*100)
    
    numeric_cols = [
        'koi_period', 'koi_time0bk', 'koi_sma', 'koi_incl', 'koi_eccen',
        'koi_longp', 'koi_impact', 'koi_duration', 'koi_depth', 'koi_ingress',
        'koi_prad', 'koi_insol', 'koi_teq', 'koi_srad', 'koi_smass',
        'koi_sage', 'koi_steff', 'koi_slogg', 'koi_smet', 'koi_kepmag'
    ]
    
    # 존재하는 컬럼만 선택
    numeric_cols = [col for col in numeric_cols if col in df.columns]
    
    # 상관관계 계산
    corr_matrix = df[numeric_cols].corr()
    
    # 높은 상관관계 찾기 (절대값 > 0.5, 자기 자신 제외)
    print("\n⭐ 높은 상관관계 (|r| > 0.5):")
    print("-"*100)
    
    high_corr = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.5:
                high_corr.append({
                    'Col1': corr_matrix.columns[i],
                    'Col2': corr_matrix.columns[j],
                    'Correlation': corr_val
                })
    
    high_corr_df = pd.DataFrame(high_corr).sort_values('Correlation', 
                                                         key=abs, 
                                                         ascending=False)
    
    for _, row in high_corr_df.iterrows():
        print(f"  {row['Col1']:<20} ↔ {row['Col2']:<20}  r = {row['Correlation']:>6.3f}")
    
    # 4. 컬럼별 상관관계 매핑 (결측치 채우기용)
    print("\n" + "="*100)
    print("📊 4. 결측치 채우기 전략")
    print("="*100)
    
    strategies = {
        # 궤도 파라미터
        'koi_sma': {
            'missing': df['koi_sma'].isna().sum(),
            'missing_pct': df['koi_sma'].isna().sum() / len(df) * 100,
            'method': '계산',
            'formula': 'a = (P² × M_star)^(1/3)  [케플러 제3법칙]',
            'required': ['koi_period', 'koi_smass'],
            'correlations': ['koi_period (r=0.95)', 'koi_smass (r=0.30)'],
            'priority': '⭐⭐⭐⭐⭐ 최우선'
        },
        'koi_incl': {
            'missing': df['koi_incl'].isna().sum(),
            'missing_pct': df['koi_incl'].isna().sum() / len(df) * 100,
            'method': '기본값',
            'formula': '89° (통과 행성 평균)',
            'required': [],
            'correlations': ['koi_impact (r=-0.8)'],
            'priority': '⭐⭐ 낮음 (이미 대부분 채워짐)'
        },
        'koi_impact': {
            'missing': df['koi_impact'].isna().sum(),
            'missing_pct': df['koi_impact'].isna().sum() / len(df) * 100,
            'method': '계산',
            'formula': 'b = (a/R_star) × cos(i)',
            'required': ['koi_sma', 'koi_srad', 'koi_incl'],
            'correlations': ['koi_incl (r=-0.8)'],
            'priority': '⭐⭐⭐ 높음'
        },
        'koi_eccen': {
            'missing': df['koi_eccen'].isna().sum(),
            'missing_pct': df['koi_eccen'].isna().sum() / len(df) * 100,
            'method': '통계적 추정',
            'formula': '중앙값 또는 0.0 (원형 궤도)',
            'required': [],
            'correlations': ['약함'],
            'priority': '⭐⭐ 중간'
        },
        'koi_longp': {
            'missing': df['koi_longp'].isna().sum(),
            'missing_pct': df['koi_longp'].isna().sum() / len(df) * 100,
            'method': '기본값',
            'formula': '90° 또는 균일 분포',
            'required': [],
            'correlations': ['없음'],
            'priority': '⭐ 낮음'
        },
        
        # 통과 파라미터
        'koi_duration': {
            'missing': df['koi_duration'].isna().sum(),
            'missing_pct': df['koi_duration'].isna().sum() / len(df) * 100,
            'method': '회귀 모델',
            'formula': 'duration = f(period, rad, sma, incl)',
            'required': ['koi_period', 'koi_srad', 'koi_sma', 'koi_incl'],
            'correlations': ['koi_period (r=0.6)'],
            'priority': '⭐⭐⭐⭐ 매우 높음'
        },
        'koi_depth': {
            'missing': df['koi_depth'].isna().sum(),
            'missing_pct': df['koi_depth'].isna().sum() / len(df) * 100,
            'method': '계산',
            'formula': 'depth ≈ (R_planet/R_star)²',
            'required': ['koi_prad', 'koi_srad'],
            'correlations': ['koi_prad (r=0.7)'],
            'priority': '⭐⭐⭐⭐ 매우 높음'
        },
        'koi_ingress': {
            'missing': df['koi_ingress'].isna().sum(),
            'missing_pct': df['koi_ingress'].isna().sum() / len(df) * 100,
            'method': '비율 추정',
            'formula': 'ingress ≈ 0.1 × duration',
            'required': ['koi_duration'],
            'correlations': ['koi_duration (r=0.9)'],
            'priority': '⭐⭐ 중간'
        },
        
        # 행성 물리량
        'koi_prad': {
            'missing': df['koi_prad'].isna().sum(),
            'missing_pct': df['koi_prad'].isna().sum() / len(df) * 100,
            'method': '회귀 모델',
            'formula': 'prad = f(depth, rad_star, period)',
            'required': ['koi_depth', 'koi_srad'],
            'correlations': ['koi_depth (r=0.7)', 'koi_teq (r=-0.5)'],
            'priority': '⭐⭐⭐⭐⭐ 최우선'
        },
        'koi_insol': {
            'missing': df['koi_insol'].isna().sum(),
            'missing_pct': df['koi_insol'].isna().sum() / len(df) * 100,
            'method': '계산',
            'formula': 'insol = L_star / (4π × a²)',
            'required': ['koi_sma', 'koi_srad', 'koi_steff'],
            'correlations': ['koi_sma (r=-0.95)', 'koi_teq (r=0.9)'],
            'priority': '⭐⭐⭐⭐ 매우 높음'
        },
        'koi_teq': {
            'missing': df['koi_teq'].isna().sum(),
            'missing_pct': df['koi_teq'].isna().sum() / len(df) * 100,
            'method': '계산',
            'formula': 'teq = T_star × (R_star/(2a))^0.5',
            'required': ['koi_steff', 'koi_srad', 'koi_sma'],
            'correlations': ['koi_insol (r=0.9)', 'koi_sma (r=-0.8)'],
            'priority': '⭐⭐⭐⭐ 매우 높음'
        },
        
        # 항성 물리량
        'koi_smass': {
            'missing': df['koi_smass'].isna().sum(),
            'missing_pct': df['koi_smass'].isna().sum() / len(df) * 100,
            'method': '계산',
            'formula': 'M = R² × 10^(logg - 4.44)',
            'required': ['koi_srad', 'koi_slogg'],
            'correlations': ['koi_srad (r=0.9)', 'koi_slogg (r=-0.5)'],
            'priority': '⭐⭐⭐⭐⭐ 최우선'
        },
        'koi_srad': {
            'missing': df['koi_srad'].isna().sum(),
            'missing_pct': df['koi_srad'].isna().sum() / len(df) * 100,
            'method': '회귀 모델',
            'formula': 'rad = f(teff, logg)',
            'required': ['koi_steff', 'koi_slogg'],
            'correlations': ['koi_smass (r=0.9)'],
            'priority': '⭐⭐⭐⭐⭐ 최우선'
        },
        'koi_steff': {
            'missing': df['koi_steff'].isna().sum(),
            'missing_pct': df['koi_steff'].isna().sum() / len(df) * 100,
            'method': '회귀 모델',
            'formula': 'teff = f(kepmag, color)',
            'required': ['koi_kepmag'],
            'correlations': ['koi_srad (r=0.3)'],
            'priority': '⭐⭐⭐⭐ 매우 높음'
        },
        'koi_slogg': {
            'missing': df['koi_slogg'].isna().sum(),
            'missing_pct': df['koi_slogg'].isna().sum() / len(df) * 100,
            'method': '계산',
            'formula': 'logg = log10(G×M/R²) + 4.44',
            'required': ['koi_smass', 'koi_srad'],
            'correlations': ['koi_srad (r=-0.7)'],
            'priority': '⭐⭐⭐ 높음'
        },
        'koi_smet': {
            'missing': df['koi_smet'].isna().sum(),
            'missing_pct': df['koi_smet'].isna().sum() / len(df) * 100,
            'method': '통계적 추정',
            'formula': '중앙값 또는 0.0 (태양 금속성)',
            'required': [],
            'correlations': ['약함'],
            'priority': '⭐⭐ 중간'
        },
        'koi_sage': {
            'missing': df['koi_sage'].isna().sum(),
            'missing_pct': df['koi_sage'].isna().sum() / len(df) * 100,
            'method': '통계적 추정',
            'formula': '중앙값 또는 5 Gyr',
            'required': [],
            'correlations': ['약함'],
            'priority': '⭐ 낮음'
        },
        
        # 기타
        'koi_kepmag': {
            'missing': df['koi_kepmag'].isna().sum(),
            'missing_pct': df['koi_kepmag'].isna().sum() / len(df) * 100,
            'method': '제외',
            'formula': 'TESS는 측정 불가',
            'required': [],
            'correlations': ['없음'],
            'priority': '❌ 제외 권장'
        },
        'koi_time0bk': {
            'missing': df['koi_time0bk'].isna().sum(),
            'missing_pct': df['koi_time0bk'].isna().sum() / len(df) * 100,
            'method': '제외',
            'formula': '관측 시각, 예측 불가',
            'required': [],
            'correlations': ['없음'],
            'priority': '❌ 제외 권장'
        }
    }
    
    # 우선순위별 정렬
    priority_order = {
        '⭐⭐⭐⭐⭐ 최우선': 1,
        '⭐⭐⭐⭐ 매우 높음': 2,
        '⭐⭐⭐ 높음': 3,
        '⭐⭐ 중간': 4,
        '⭐ 낮음': 5,
        '❌ 제외 권장': 6
    }
    
    sorted_strategies = sorted(strategies.items(), 
                               key=lambda x: priority_order.get(x[1]['priority'], 99))
    
    print("\n" + "-"*100)
    print(f"{'컬럼':<20} {'결측':<8} {'결측%':<8} {'처리방법':<15} {'우선순위':<25}")
    print("-"*100)
    
    for col, strategy in sorted_strategies:
        if strategy['missing'] > 0:
            print(f"{col:<20} {strategy['missing']:<8} {strategy['missing_pct']:<8.2f} "
                  f"{strategy['method']:<15} {strategy['priority']:<25}")
    
    # 5. 상세 전략 출력
    print("\n" + "="*100)
    print("📊 5. 컬럼별 상세 처리 전략")
    print("="*100)
    
    for col, strategy in sorted_strategies:
        if strategy['missing'] > 0 and strategy['priority'] != '❌ 제외 권장':
            print(f"\n🔹 {col}")
            print(f"   결측: {strategy['missing']:,} ({strategy['missing_pct']:.1f}%)")
            print(f"   우선순위: {strategy['priority']}")
            print(f"   방법: {strategy['method']}")
            print(f"   공식: {strategy['formula']}")
            if strategy['required']:
                print(f"   필요 컬럼: {', '.join(strategy['required'])}")
            print(f"   상관관계: {', '.join(strategy['correlations'])}")
    
    return df, strategies


if __name__ == "__main__":
    df, strategies = analyze_missing_values()
    print("\n" + "="*100)
    print("✅ 분석 완료!")
    print("="*100)
